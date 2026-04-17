"""
Tests for agent/factory.py

Agent factory and dependency management:
- AgentFactory: Per-request factory with session + lock_reg bound
  - _get_lock: Per-agent lock registry management
  - get_deps: Async context manager yielding AgentDeps with lock acquisition
  - build_agent_and_deps: Async context manager yielding (Agent, AgentDeps)
- get_model: Module-level function mapping model name strings to Pydantic AI model instances
"""
import asyncio
from unittest.mock import MagicMock, patch

import pytest
from pytest_mock import MockerFixture
from pydantic_ai import Agent
from pydantic_ai.models.anthropic import AnthropicModel
from sqlalchemy.ext.asyncio import AsyncSession

from agent.factory import AgentFactory, get_model
from agent.types import AgentConfig, AgentDeps
from conftest import SAMPLE_AGENT_CONFIG
from db.models import AgentRecord


# --- Constants ---

NONEXISTENT_AGENT_ID = "nonexistent-agent-id-12345"


# --- Helpers ---

def create_spied_lock(agent_id: str, lock_reg: dict, mocker: MockerFixture) -> asyncio.Lock:
    """Create a lock, register it, and spy on acquire/release."""
    lock = asyncio.Lock()
    lock_reg[agent_id] = lock
    mocker.spy(lock, "acquire")
    mocker.spy(lock, "release")
    return lock


def assert_lock_acquired_and_released(lock: asyncio.Lock) -> None:
    """Assert acquire and release were each called exactly once."""
    assert lock.acquire.call_count == 1
    assert lock.release.call_count == 1


# --- Fixtures ---

@pytest.fixture
def lock_reg() -> dict[str, asyncio.Lock]:
    """Fresh lock registry for each test."""
    return {}


@pytest.fixture
def agent_factory(lock_reg: dict, session: AsyncSession) -> AgentFactory:
    """Per-request AgentFactory with lock_reg and session bound."""
    return AgentFactory(lock_reg, session)


# --- get_model tests (module-level function) ---

def test_get_model_returns_model_for_valid_name():
    """get_model should return an AnthropicModel instance for a valid model name."""
    model_str = "claude-sonnet-4-20250514"
    model = get_model(model_str)
    
    assert isinstance(model, AnthropicModel)
    assert model.model_name == model_str


@pytest.mark.parametrize("invalid_name", [
    "not-a-real-model",
    "gpt-4",  # Wrong provider
    "",
    "claude-unknown-version",
])
def test_get_model_raises_for_invalid_name(invalid_name: str):
    """get_model should raise for unknown/unsupported model names."""
    # TODO: Match exception in implementation when set
    with pytest.raises((ValueError)):
        get_model(invalid_name)


# --- AgentFactory._get_lock tests ---

def test_get_lock_returns_same_lock_for_same_id(agent_factory: AgentFactory):
    """_get_lock should return the same Lock instance for the same agent_id."""
    lock1 = agent_factory._get_lock("agent-123")
    lock2 = agent_factory._get_lock("agent-123")
    
    assert lock1 is lock2
    assert isinstance(lock1, asyncio.Lock)


def test_get_lock_returns_different_locks_for_different_ids(agent_factory: AgentFactory):
    """_get_lock should return different Lock instances for different agent_ids."""
    lock_a = agent_factory._get_lock("agent-aaa")
    lock_b = agent_factory._get_lock("agent-bbb")
    
    assert lock_a is not lock_b
    assert isinstance(lock_a, asyncio.Lock)
    assert isinstance(lock_b, asyncio.Lock)


# --- AgentFactory.get_deps tests ---

@pytest.mark.asyncio
async def test_get_deps_yields_deps_with_expected_fields(
    agent_factory: AgentFactory,
    agent_record: AgentRecord,
):
    """get_deps should yield AgentDeps with agent_id, session, and config populated."""
    async with agent_factory.get_deps(agent_record.id) as deps:
        assert isinstance(deps, AgentDeps)
        assert deps.agent_id == agent_record.id
        assert deps.session is agent_factory._session
        assert deps.config == agent_record.agent_config


@pytest.mark.asyncio
async def test_get_deps_creates_acquires_and_releases_lock(
    agent_factory: AgentFactory,
    agent_record: AgentRecord,
    lock_reg: dict,
):
    """get_deps should create lock if needed, acquire before yield, release after exit."""
    # Verify lock doesn't exist yet
    assert agent_record.id not in lock_reg
    
    async with agent_factory.get_deps(agent_record.id) as deps:
        # Lock should have been created and held
        assert agent_record.id in lock_reg
        lock = lock_reg[agent_record.id]
        assert lock.locked()
    
    # After exiting, lock should be released
    assert not lock.locked()


@pytest.mark.asyncio
async def test_get_deps_acquires_and_releases_existing_lock(
    agent_factory: AgentFactory,
    agent_record: AgentRecord,
    lock_reg: dict,
    mocker: MockerFixture,
):
    """get_deps should acquire and release an existing lock (verified via spy)."""
    lock = create_spied_lock(agent_record.id, lock_reg, mocker)

    async with agent_factory.get_deps(agent_record.id) as deps:
        assert lock.locked()

    assert_lock_acquired_and_released(lock)


@pytest.mark.asyncio
async def test_get_deps_releases_lock_on_exception(
    agent_factory: AgentFactory,
    agent_record: AgentRecord,
    lock_reg: dict,
    mocker: MockerFixture,
):
    """get_deps should release the lock even if an exception is raised inside the context."""
    lock = create_spied_lock(agent_record.id, lock_reg, mocker)

    with pytest.raises(RuntimeError):
        async with agent_factory.get_deps(agent_record.id) as deps:
            raise RuntimeError("Intentional test error")

    assert_lock_acquired_and_released(lock)


@pytest.mark.asyncio
async def test_get_deps_raises_and_releases_lock_on_fetch_failure(
    agent_factory: AgentFactory,
    lock_reg: dict,
    mocker: MockerFixture,
):
    """get_deps should raise for unknown agent_id AND release lock on failure.

    Design decision: lock-then-fetch. The lock must be acquired BEFORE the DB fetch
    to prevent concurrent runs from seeing stale state. This means if the fetch fails,
    the lock was already acquired and must be released via try/finally.
    """
    lock = create_spied_lock(NONEXISTENT_AGENT_ID, lock_reg, mocker)

    # TODO: Narrow to specific NotFound or whatever exception once we define it
    with pytest.raises(ValueError):
        async with agent_factory.get_deps(NONEXISTENT_AGENT_ID) as deps:
            pass

    assert_lock_acquired_and_released(lock)


# --- get_deps concurrency tests ---

@pytest.mark.asyncio
async def test_get_deps_concurrent_same_agent_blocks(
    agent_factory: AgentFactory,
    agent_record: AgentRecord,
):
    """Second concurrent call on same agent_id should block until first exits."""
    execution_order = []
    
    async def first_caller():
        async with agent_factory.get_deps(agent_record.id):
            execution_order.append("first_entered")
            await asyncio.sleep(0.05)  # Hold lock briefly
            execution_order.append("first_exiting")
    
    async def second_caller():
        await asyncio.sleep(0.01)  # Ensure first_caller enters first
        async with agent_factory.get_deps(agent_record.id):
            execution_order.append("second_entered")
    
    await asyncio.gather(first_caller(), second_caller())
    
    # Second should only enter after first exits
    assert execution_order == ["first_entered", "first_exiting", "second_entered"]


@pytest.mark.asyncio
async def test_get_deps_concurrent_different_agents_no_block(
    agent_factory: AgentFactory,
    session: AsyncSession,
):
    """
    Concurrent calls on different agent_ids should not block each other.
    Note: technically the behavior in usage will be distinct agent factories per request so
    we will have seperate agent factories handling concurrent diff agents, but this test is close enough
    """
    # Create two agents for this test
    agent_a = AgentRecord(
        name="agent-a", 
        agent_config=SAMPLE_AGENT_CONFIG.model_copy(),
        system_instructions="Agent A",
    )
    agent_b = AgentRecord(
        name="agent-b",
        agent_config=SAMPLE_AGENT_CONFIG.model_copy(), 
        system_instructions="Agent B",
    )
    session.add_all([agent_a, agent_b])
    await session.flush()
    
    execution_order = []
    
    async def caller_a():
        async with agent_factory.get_deps(agent_a.id):
            execution_order.append("a_entered")
            await asyncio.sleep(0.05)
            execution_order.append("a_exiting")
    
    async def caller_b():
        await asyncio.sleep(0.01)  # Small delay so A enters first
        async with agent_factory.get_deps(agent_b.id):
            execution_order.append("b_entered")
            await asyncio.sleep(0.01)
            execution_order.append("b_exiting")
    
    await asyncio.gather(caller_a(), caller_b())
    
    # B should enter while A is still holding its lock (different agents, no blocking)
    # Expected: a_entered, b_entered, b_exiting, a_exiting
    assert execution_order.index("b_entered") < execution_order.index("a_exiting")


# --- AgentFactory.build_agent_and_deps tests ---

@pytest.mark.asyncio
async def test_build_agent_and_deps_yields_tuple(
    agent_factory: AgentFactory,
    agent_record: AgentRecord,
):
    """build_agent_and_deps should yield a valid (agent, deps) tuple."""
    async with agent_factory.build_agent_and_deps(agent_record.id) as (agent, deps):
        assert isinstance(agent, Agent)
        assert isinstance(deps, AgentDeps)


@pytest.mark.asyncio
async def test_build_agent_and_deps_holds_lock(
    agent_factory: AgentFactory,
    agent_record: AgentRecord,
    lock_reg: dict,
):
    """build_agent_and_deps should hold the lock for the duration of the context."""
    async with agent_factory.build_agent_and_deps(agent_record.id) as (agent, deps):
        lock = lock_reg[agent_record.id]
        assert lock.locked()
    
    # Released after exit
    assert not lock_reg[agent_record.id].locked()


@pytest.mark.asyncio
async def test_build_agent_and_deps_uses_correct_model(
    agent_factory: AgentFactory,
    agent_record: AgentRecord,
):
    """Constructed agent should use the model from agent_config.model_name."""
    agent_record.agent_config.model_name = "claude-sonnet-4-20250514"
    async with agent_factory.build_agent_and_deps(agent_record.id) as (agent, deps):
        # Agent.model is the model instance used for this agent
        assert isinstance(agent.model, AnthropicModel)
        assert agent.model.model_name == agent_record.agent_config.model_name


@pytest.mark.asyncio
async def test_build_agent_and_deps_has_cache_settings(
    agent_factory: AgentFactory,
    agent_record: AgentRecord,
):
    """Constructed agent should have Anthropic prompt caching enabled in model_settings.

    model_settings is passed at Agent construction and is directly inspectable via
    agent.model_settings. All three cache flags should be set to enable caching on
    system instructions, tool definitions, and the last user message.
    """
    async with agent_factory.build_agent_and_deps(agent_record.id) as (agent, deps):
        settings = agent.model_settings
        assert settings.get("anthropic_cache_instructions") == True, "System prompt caching should be enabled with default TTL (5m)"
        assert settings.get("anthropic_cache_tool_definitions") == True, "Tool definition caching should be enabled with default TTL (5m)"
        assert settings.get("anthropic_cache_messages") == True, "Message caching should be enabled with default TTL (5m)"


@pytest.mark.asyncio
async def test_build_agent_and_deps_agent_has_correct_tools(
    agent_factory: AgentFactory,
    agent_record: AgentRecord,
):
    """Constructed agent should have tools matching agent_config.tool_names.

    get_tools_for_agent (Section 3.2) is mocked with real dummy callables so
    Pydantic AI can register them. We verify both that get_tools_for_agent was
    called with the correct tool_names, and that the returned tools are actually
    present on the constructed agent.

    Note: inspects agent._function_toolset.tools (private API) — stable enough
    for tests, but worth revisiting if Pydantic AI changes internals.
    """
    def memory_replace(x: str) -> str:
        """Stub for memory_replace."""
        return x

    def memory_insert(x: str) -> str:
        """Stub for memory_insert."""
        return x

    stub_tools = [memory_replace, memory_insert]
    expected_tool_names = {"memory_replace", "memory_insert"}

    with patch("agent.factory.get_tools_for_agent", return_value=stub_tools) as mock_get_tools:
        async with agent_factory.build_agent_and_deps(agent_record.id) as (agent, deps):
            mock_get_tools.assert_called_once_with(agent_record.agent_config.tool_names)
            actual_tool_names = set(agent._function_toolset.tools.keys())
            assert actual_tool_names == expected_tool_names


"""
TODO: Test agent has expected:
- instructions getting function

"""
def test_more_agent_properties():
    pytest.fail()