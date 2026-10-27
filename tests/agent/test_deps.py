"""Tests for AgentDeps construction, build_deps, and write_scope/read_scope lifecycle behavior."""
import asyncio
import time
from datetime import datetime
from unittest.mock import MagicMock

import pytest
from pytest_mock import MockerFixture
from sqlalchemy.ext.asyncio import AsyncSession

from conftest import SAMPLE_AGENT_CONFIG, make_deps, make_ro_engine
from agent.deps import AgentDeps, build_deps
from agent.factory import get_or_create_agent_app_state
from agent.types import AgentAppState, AgentLockedError, AgentNotFoundError, BlockSettings
from db.models import AgentRecord
from memory.block_crud import create_block, update_block, ContentExceedsLimitError


@pytest.mark.asyncio
class TestWriteScopeExceptionRecovery:
    """A write_scope that exits via exception must leave the record in a known-good,
    readable state. Routine validation errors (e.g. ContentExceedsLimitError on the
    ModelRetry path) are normal operation — they must not poison subsequent property
    reads with DetachedInstanceError."""

    async def test_record_readable_after_body_exception(self, test_engine, agent_record):
        deps = make_deps(test_engine, agent_record)
        await create_block(deps, BlockSettings(label="probe", char_limit=10), content="12345")

        with pytest.raises(ContentExceedsLimitError):
            await update_block(deps, "probe", "this content is definitely too long")

        assert deps.compaction_warning_fired is False

    async def test_uncommitted_record_mutation_discarded_after_body_exception(
        self, test_engine, agent_record
    ):
        deps = make_deps(test_engine, agent_record)
        async with deps.write_scope():
            deps.compiled_system_prompt = "committed value"

        with pytest.raises(RuntimeError, match="deliberate"):
            async with deps.write_scope():
                deps.compiled_system_prompt = "uncommitted value"
                raise RuntimeError("deliberate")

        assert deps.compiled_system_prompt == "committed value"


# --- AgentDeps construction ---

@pytest.mark.parametrize("missing_field", ["rw_engine", "ro_engine", "agent_record"])
def test_agentdeps_requires_field(missing_field: str):
    """AgentDeps should raise TypeError when a required constructor argument is missing."""
    all_fields = {
        "rw_engine": object(),
        "ro_engine": object(),
        "agent_record": MagicMock(),
        "bypass_factory_guard": True,
    }
    del all_fields[missing_field]

    with pytest.raises(TypeError):
        AgentDeps(**all_fields)


def test_agentdeps_factory_guard_blocks_direct_construction():
    """AgentDeps must be constructed via build_deps or with explicit bypass."""
    with pytest.raises(RuntimeError, match="must be constructed via build_deps"):
        AgentDeps(object(), object(), MagicMock())


async def test_agentdeps_holds_expected_fields(agent_deps, agent_record):
    """AgentDeps properties should delegate to the underlying AgentRecord."""
    assert agent_deps.agent_id == agent_record.id
    assert agent_deps.config is agent_record.agent_config
    assert agent_deps.name == agent_record.name
    assert agent_deps.system_instructions == agent_record.system_instructions


async def test_agentdeps_setters_all_require_write_scope(agent_with_blocks: dict, test_engine):
    """All AgentDeps setters require write_scope context."""
    agent_record = agent_with_blocks["agent"]  # already detached (conftest expunges)

    deps = make_deps(test_engine, agent_record)

    setters_to_test = [
        ("config", agent_record.agent_config),
        ("system_instructions", "new instructions"),
        ("compiled_system_prompt", "new prompt"),
        ("sys_prompt_compiled_at", datetime.now()),
        ("context_window_start", 100),
        ("compaction_warning_fired", True),
    ]

    for prop_name, test_value in setters_to_test:
        with pytest.raises(RuntimeError, match="write_scope"):
            setattr(deps, prop_name, test_value)


# --- build_deps helpers ---

NONEXISTENT_AGENT_ID = "nonexistent-agent-id-12345"


def create_spied_lock(agent_id: str, agent_app_state_reg: dict, mocker: MockerFixture) -> asyncio.Lock:
    """Create a lock, register it in an AgentAppState slot, and spy on acquire/release."""
    lock = asyncio.Lock()
    agent_app_state_reg[agent_id] = AgentAppState(lock=lock)
    mocker.spy(lock, "acquire")
    mocker.spy(lock, "release")
    return lock


def assert_lock_acquired_and_released(lock: asyncio.Lock) -> None:
    """Assert acquire and release were each called exactly once."""
    assert lock.acquire.call_count == 1
    assert lock.release.call_count == 1


@pytest.fixture
def agent_app_state(agent_record: AgentRecord, agent_app_state_reg: dict) -> AgentAppState:
    """Get or create the AgentAppState for the test agent."""
    return get_or_create_agent_app_state(agent_app_state_reg, agent_record.id)


# --- build_deps tests ---

@pytest.mark.asyncio
async def test_build_deps_yields_deps_with_expected_fields(
    agent_record: AgentRecord,
    agent_app_state: AgentAppState,
    test_engine,
):
    """build_deps should yield AgentDeps with agent_id and config populated."""
    async with build_deps(agent_record.id, agent_app_state, test_engine, make_ro_engine(test_engine)) as deps:
        assert isinstance(deps, AgentDeps)
        assert deps.agent_id == agent_record.id
        assert deps.config == agent_record.agent_config
        assert deps.name == agent_record.name


@pytest.mark.asyncio
async def test_build_deps_creates_acquires_and_releases_lock(
    agent_record: AgentRecord,
    agent_app_state: AgentAppState,
    test_engine,
):
    """build_deps should acquire lock before yield, release after exit."""
    lock = agent_app_state.lock
    assert not lock.locked()

    async with build_deps(agent_record.id, agent_app_state, test_engine, make_ro_engine(test_engine)) as deps:
        assert lock.locked()

    # After exiting, lock should be released
    assert not lock.locked()


class TestBuildDepsLockAndCancelBehavior:
    """Verifies lock acquire/release (via spy) and cancel_requested clearing on teardown.

    Spy slot is pre-populated in the registry before build_deps is called.
    Tests cover both normal and exception exit paths.
    """

    @pytest.fixture(autouse=True)
    def _setup(
        self,
        agent_record: AgentRecord,
        agent_app_state_reg: dict[str, AgentAppState],
        test_engine,
        mocker: MockerFixture,
    ):
        self.lock = create_spied_lock(agent_record.id, agent_app_state_reg, mocker)
        self.agent_app_state = agent_app_state_reg[agent_record.id]
        self.cancel_requested = self.agent_app_state.cancel_requested
        self.agent_id = agent_record.id
        self.test_engine = test_engine

    @pytest.mark.asyncio
    async def test_normal_exit_releases_lock_and_clears_cancel(self):
        """build_deps should release the lock and clear a pending cancel on normal exit."""
        self.cancel_requested.set()

        async with build_deps(self.agent_id, self.agent_app_state, self.test_engine, make_ro_engine(self.test_engine)):
            assert self.lock.locked()

        assert_lock_acquired_and_released(self.lock)
        assert not self.cancel_requested.is_set()

    @pytest.mark.asyncio
    async def test_exception_exit_releases_lock_and_clears_cancel(self):
        """build_deps should release the lock and clear a pending cancel even if an exception is raised."""
        self.cancel_requested.set()

        with pytest.raises(RuntimeError):
            async with build_deps(self.agent_id, self.agent_app_state, self.test_engine, make_ro_engine(self.test_engine)):
                raise RuntimeError("Intentional test error")

        assert_lock_acquired_and_released(self.lock)
        assert not self.cancel_requested.is_set()


@pytest.mark.asyncio
async def test_build_deps_raises_releases_lock_and_clears_cancel_on_fetch_failure(
    agent_app_state_reg: dict,
    test_engine,
    mocker: MockerFixture,
):
    """build_deps should raise for unknown agent_id AND release lock and clear cancel on failure.

    Design decision: lock-then-fetch. The lock must be acquired BEFORE the DB fetch
    to prevent concurrent runs from seeing stale state. This means if the fetch fails,
    the lock was already acquired and must be released via try/finally.
    Uses NONEXISTENT_AGENT_ID (no agent_record fixture) — kept standalone for that reason.
    """
    lock = create_spied_lock(NONEXISTENT_AGENT_ID, agent_app_state_reg, mocker)
    app_state = agent_app_state_reg[NONEXISTENT_AGENT_ID]

    cancel_requested = app_state.cancel_requested
    cancel_requested.set()

    with pytest.raises(AgentNotFoundError):
        async with build_deps(NONEXISTENT_AGENT_ID, app_state, test_engine, make_ro_engine(test_engine)) as deps:
            pass

    assert_lock_acquired_and_released(lock)
    assert not cancel_requested.is_set()


# --- build_deps concurrency tests ---

@pytest.mark.asyncio
async def test_build_deps_concurrent_same_agent_blocks(
    agent_record: AgentRecord,
    agent_app_state: AgentAppState,
    test_engine,
):
    """Second concurrent call on same agent_id should block until first exits."""
    execution_order = []

    async def first_caller():
        async with build_deps(agent_record.id, agent_app_state, test_engine, make_ro_engine(test_engine)):
            execution_order.append("first_entered")
            await asyncio.sleep(0.05)  # Hold lock briefly
            execution_order.append("first_exiting")

    async def second_caller():
        await asyncio.sleep(0.01)  # Ensure first_caller enters first
        async with build_deps(agent_record.id, agent_app_state, test_engine, make_ro_engine(test_engine)):
            execution_order.append("second_entered")

    await asyncio.gather(first_caller(), second_caller())

    # Second should only enter after first exits
    assert execution_order == ["first_entered", "first_exiting", "second_entered"]


@pytest.mark.asyncio
async def test_build_deps_concurrent_different_agents_no_block(
    agent_app_state_reg: dict,
    session: AsyncSession,
    test_engine,
):
    """
    Concurrent calls on different agent_ids should not block each other.
    """
    # Create two agents for this test
    agent_a = AgentRecord(
        name="agent-a",
        agent_config=SAMPLE_AGENT_CONFIG,
        system_instructions="Agent A",
    )
    agent_b = AgentRecord(
        name="agent-b",
        agent_config=SAMPLE_AGENT_CONFIG,
        system_instructions="Agent B",
    )
    session.add_all([agent_a, agent_b])
    await session.commit()
    await session.refresh(agent_a)
    await session.refresh(agent_b)

    app_state_a = get_or_create_agent_app_state(agent_app_state_reg, agent_a.id)
    app_state_b = get_or_create_agent_app_state(agent_app_state_reg, agent_b.id)

    execution_order = []

    async def caller_a():
        async with build_deps(agent_a.id, app_state_a, test_engine, make_ro_engine(test_engine)):
            execution_order.append("a_entered")
            await asyncio.sleep(0.05)
            execution_order.append("a_exiting")

    async def caller_b():
        await asyncio.sleep(0.01)  # Small delay so A enters first
        async with build_deps(agent_b.id, app_state_b, test_engine, make_ro_engine(test_engine)):
            execution_order.append("b_entered")
            await asyncio.sleep(0.01)
            execution_order.append("b_exiting")

    await asyncio.gather(caller_a(), caller_b())

    # B should enter while A is still holding its lock (different agents, no blocking)
    assert execution_order.index("b_entered") < execution_order.index("a_exiting")


@pytest.mark.asyncio
@pytest.mark.parametrize("timeout,min_elapsed,max_elapsed", [
    (0.01, 0.0, 0.5),   # fast — just assert it doesn't block
    (1.0,  0.8, 2.5),   # roughly a second
])
async def test_build_deps_timeout_raises_agent_locked_error(
    agent_app_state: AgentAppState,
    agent_record: AgentRecord,
    test_engine,
    timeout: float,
    min_elapsed: float,
    max_elapsed: float,
):
    """build_deps raises AgentLockedError after approximately the requested timeout."""
    # Hold the lock so the second acquire times out
    await agent_app_state.lock.acquire()

    start = time.monotonic()
    with pytest.raises(AgentLockedError):
        async with build_deps(agent_record.id, agent_app_state, test_engine, make_ro_engine(test_engine), timeout=timeout):
            pass  # should not reach here
    elapsed = time.monotonic() - start

    assert elapsed >= min_elapsed, f"Timed out too fast: {elapsed:.3f}s < {min_elapsed}s"
    assert elapsed < max_elapsed, f"Timed out too slow: {elapsed:.3f}s >= {max_elapsed}s"

# --- end build_deps tests ---
