"""
Tests for agent/factory.py

Agent factory and dependency management:
- AgentFactory: Per-agent, per-request factory with agent_id, session, and agent app state bound
  - _get_or_create_agent_app_state: Per-agent state registry management (static)
  - build_deps: Async context manager yielding AgentDeps with lock acquisition
  - build_agent_and_deps: Async context manager yielding (Agent, AgentDeps)
- get_model: Module-level function mapping model name strings to Pydantic AI model instances

NOTE: This got a little ugly with the move from a simple lock reg to AgentAppState. Since we might move to a different OO design with a 
StatefulAgent class, I don't think its worth cleaning this up right now though.
In particular, the lock testing, create_spied_lock should be rethought if we stick with this pattern. The agent factory creating AgentAppState
if nonexistent during *construction* as opposed to during deps building caused issues with how these tests were previously written and the patches
are not ideal
"""
import asyncio
from unittest.mock import patch

import pytest
import pytest_asyncio
from pytest_mock import MockerFixture
from pydantic_ai import Agent
from pydantic_ai.models.anthropic import AnthropicModel
from sqlalchemy.ext.asyncio import AsyncSession

from agent.factory import AgentFactory, AgentNotFoundError, get_model
from agent.types import AgentAppState, AgentDeps
from memory.system_prompt_compilation import get_system_prompt
from conftest import SAMPLE_AGENT_CONFIG
from db.models import AgentRecord


# --- Constants ---

NONEXISTENT_AGENT_ID = "nonexistent-agent-id-12345"


# --- Helpers ---

def create_spied_lock(agent_id: str, agent_app_states: dict, mocker: MockerFixture) -> asyncio.Lock:
    """Create a lock, register it in an AgentAppState slot, and spy on acquire/release."""
    lock = asyncio.Lock()
    agent_app_states[agent_id] = AgentAppState(lock, asyncio.Event())
    mocker.spy(lock, "acquire")
    mocker.spy(lock, "release")
    return lock


def assert_lock_acquired_and_released(lock: asyncio.Lock) -> None:
    """Assert acquire and release were each called exactly once."""
    assert lock.acquire.call_count == 1
    assert lock.release.call_count == 1


# --- Fixtures ---

@pytest.fixture
def agent_app_states() -> dict[str, AgentAppState]:
    """
    Fresh agent state registry for each test.
    A fixture that returns an empty dict is a bit ridiculous but it helps with documentation
    (IE communicates what this dict is meant to be.)
    It also gives us a single common dict obj across other fixtures to inspect within a single test
    NOTE: This fixture MUST be an empty dict
    """
    return {}


@pytest.fixture
def agent_factory(agent_record: AgentRecord, agent_app_states: dict, session: AsyncSession) -> AgentFactory:
    """Per-agent, per-request AgentFactory with agent_id, agent_app_states, and session bound."""
    return AgentFactory(agent_record.id, agent_app_states, session)


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


# --- AgentFactory._get_or_create_agent_app_state tests ---

# NOTE: We're testing internals too much here. If we stick with the AgentFactory pattern, just test the construction and its side effects
def test_get_or_create_agent_app_state_returns_same_agent_app_state_for_same_id(agent_app_states: dict):
    """_get_or_create_agent_app_state should return the same AgentAppState for the same agent_id."""
    # testing static method so no need to use the fixture which gives an object
    slot1 = AgentFactory._get_or_create_agent_app_state(agent_app_states, "agent-123")
    slot2 = AgentFactory._get_or_create_agent_app_state(agent_app_states, "agent-123")

    assert slot1 is slot2
    assert isinstance(slot1, AgentAppState)
    assert isinstance(slot1.lock, asyncio.Lock)
    assert isinstance(slot1.cancel_requested, asyncio.Event)
    assert not slot1.lock.locked()
    assert not slot1.cancel_requested.is_set()


def test_get_or_create_agent_app_state_returns_different_agent_app_states_for_different_ids(agent_app_states: dict):
    """_get_or_create_agent_app_state should return different AgentAppState instances for different agent_ids."""
    slot_a = AgentFactory._get_or_create_agent_app_state(agent_app_states, "agent-aaa")
    slot_b = AgentFactory._get_or_create_agent_app_state(agent_app_states, "agent-bbb")

    assert slot_a is not slot_b
    assert isinstance(slot_a, AgentAppState)
    assert isinstance(slot_b, AgentAppState)


# --- AgentFactory.build_deps tests ---

@pytest.mark.asyncio
async def test_build_deps_yields_deps_with_expected_fields(
    agent_factory: AgentFactory,
    agent_record: AgentRecord,
):
    """build_deps should yield AgentDeps with agent_id, session, and config populated."""
    async with agent_factory.build_deps() as deps:
        assert isinstance(deps, AgentDeps)
        assert deps.agent_id == agent_record.id
        assert deps.session is agent_factory._session
        assert deps.config == agent_record.agent_config
        assert deps.name == agent_record.name


@pytest.mark.asyncio
async def test_build_deps_creates_acquires_and_releases_lock(
    agent_factory: AgentFactory,
    agent_record: AgentRecord,
    agent_app_states: dict,
):
    """build_deps should acquire lock before yield, release after exit.

    agent_app_state for particular agent is created at AgentFactory construction, so the "creates" part of this test is really
    testing the AgentFactory constructor
    """
    assert agent_record.id in agent_app_states
    lock = agent_app_states[agent_record.id].lock
    assert not lock.locked()

    async with agent_factory.build_deps() as deps:
        assert lock.locked()

    # After exiting, lock should be released
    assert not lock.locked()


@pytest.mark.asyncio
async def test_build_deps_acquires_and_releases_existing_lock(
    agent_record: AgentRecord,
    agent_app_states: dict,
    session: AsyncSession,
    mocker: MockerFixture,
):
    """build_deps should acquire and release the lock (verified via spy).

    Spy slot is pre-populated before factory construction so _get_or_create_agent_app_state
    returns it as self._agent_app_state.
    """
    lock = create_spied_lock(agent_record.id, agent_app_states, mocker)
    factory = AgentFactory(agent_record.id, agent_app_states, session)

    async with factory.build_deps() as deps:
        assert lock.locked()

    assert_lock_acquired_and_released(lock)


@pytest.mark.asyncio
async def test_build_deps_releases_lock_on_exception(
    agent_record: AgentRecord,
    agent_app_states: dict,
    session: AsyncSession,
    mocker: MockerFixture,
):
    """build_deps should release the lock even if an exception is raised inside the context."""
    lock = create_spied_lock(agent_record.id, agent_app_states, mocker)
    factory = AgentFactory(agent_record.id, agent_app_states, session)

    with pytest.raises(RuntimeError):
        async with factory.build_deps() as deps:
            raise RuntimeError("Intentional test error")

    assert_lock_acquired_and_released(lock)


@pytest.mark.asyncio
async def test_build_deps_raises_and_releases_lock_on_fetch_failure(
    agent_app_states: dict,
    session: AsyncSession,
    mocker: MockerFixture,
):
    """build_deps should raise for unknown agent_id AND release lock on failure.

    Design decision: lock-then-fetch. The lock must be acquired BEFORE the DB fetch
    to prevent concurrent runs from seeing stale state. This means if the fetch fails,
    the lock was already acquired and must be released via try/finally.
    """
    lock = create_spied_lock(NONEXISTENT_AGENT_ID, agent_app_states, mocker)
    factory = AgentFactory(NONEXISTENT_AGENT_ID, agent_app_states, session)

    with pytest.raises(AgentNotFoundError):
        async with factory.build_deps() as deps:
            pass

    assert_lock_acquired_and_released(lock)


# --- build_deps concurrency tests ---

@pytest.mark.asyncio
async def test_build_deps_concurrent_same_agent_blocks(
    agent_factory: AgentFactory,
    agent_record: AgentRecord,
):
    """Second concurrent call on same agent_id should block until first exits."""
    execution_order = []
    
    async def first_caller():
        async with agent_factory.build_deps():
            execution_order.append("first_entered")
            await asyncio.sleep(0.05)  # Hold lock briefly
            execution_order.append("first_exiting")

    async def second_caller():
        await asyncio.sleep(0.01)  # Ensure first_caller enters first
        async with agent_factory.build_deps():
            execution_order.append("second_entered")
    
    await asyncio.gather(first_caller(), second_caller())
    
    # Second should only enter after first exits
    assert execution_order == ["first_entered", "first_exiting", "second_entered"]


@pytest.mark.asyncio
async def test_build_deps_concurrent_different_agents_no_block(
    agent_app_states: dict,
    session: AsyncSession,
):
    """
    Concurrent calls on different agent_ids should not block each other.
    Each agent gets its own factory (matching real usage — distinct factories per request).
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

    factory_a = AgentFactory(agent_a.id, agent_app_states, session)
    factory_b = AgentFactory(agent_b.id, agent_app_states, session)

    execution_order = []

    async def caller_a():
        async with factory_a.build_deps():
            execution_order.append("a_entered")
            await asyncio.sleep(0.05)
            execution_order.append("a_exiting")

    async def caller_b():
        await asyncio.sleep(0.01)  # Small delay so A enters first
        async with factory_b.build_deps():
            execution_order.append("b_entered")
            await asyncio.sleep(0.01)
            execution_order.append("b_exiting")

    await asyncio.gather(caller_a(), caller_b())
    
    # B should enter while A is still holding its lock (different agents, no blocking)
    # Expected: a_entered, b_entered, b_exiting, a_exiting
    assert execution_order.index("b_entered") < execution_order.index("a_exiting")


# --- AgentFactory.build_agent_and_deps tests ---

class TestBuildAgentAndDeps:
    """Tests for AgentFactory.build_agent_and_deps.

    All tests patch get_tools_for_agent → [] to isolate agent construction from
    tool behavior. Tool injection is tested separately below.
    """

    @pytest_asyncio.fixture(autouse=True)
    async def _setup(
        self,
        agent_factory: AgentFactory,
        agent_record: AgentRecord,
        agent_app_states: dict,
    ):
        self.factory = agent_factory
        self.agent_record = agent_record
        self.agent_app_states = agent_app_states
        with patch("agent.factory.get_tools_for_agent", return_value=[]):
            yield

    async def test_yields_tuple(self):
        """build_agent_and_deps should yield a valid (agent, deps) tuple."""
        async with self.factory.build_agent_and_deps() as (agent, deps):
            assert isinstance(agent, Agent)
            assert isinstance(deps, AgentDeps)

    async def test_holds_lock(self):
        """build_agent_and_deps should hold the lock for the duration of the context."""
        async with self.factory.build_agent_and_deps() as (agent, deps):
            assert self.agent_app_states[self.agent_record.id].lock.locked()

        assert not self.agent_app_states[self.agent_record.id].lock.locked()

    async def test_uses_correct_model(self):
        """Constructed agent should use the model from agent_config.model_name."""
        self.agent_record.agent_config.model_name = "claude-sonnet-4-20250514"
        async with self.factory.build_agent_and_deps() as (agent, deps):
            assert isinstance(agent.model, AnthropicModel)
            assert agent.model.model_name == self.agent_record.agent_config.model_name

    async def test_has_cache_settings(self):
        """Constructed agent should have Anthropic prompt caching enabled in model_settings.

        model_settings is passed at Agent construction and is directly inspectable via
        agent.model_settings. All three cache flags should be set to enable caching on
        system instructions, tool definitions, and the last user message.
        """
        async with self.factory.build_agent_and_deps() as (agent, deps):
            settings = agent.model_settings
            assert settings.get("anthropic_cache_instructions") == True, "System prompt caching should be enabled with default TTL (5m)"
            assert settings.get("anthropic_cache_tool_definitions") == True, "Tool definition caching should be enabled with default TTL (5m)"
            assert settings.get("anthropic_cache_messages") == True, "Message caching should be enabled with default TTL (5m)"

    async def test_retries_set_from_config(self):
        """Constructed agent should use retries from agent_config.retries."""
        async with self.factory.build_agent_and_deps() as (agent, deps):
            assert agent._max_tool_retries == deps.config.retries

    async def test_misc_agent_settings(self):
        """Constructed agent should have name, deps_type, instructions, and output_type correctly set.

        These settings are not derived from per-agent config (except name) but are critical
        for correct agent behavior. Uses private pydantic_ai internals (_deps_type,
        _instructions, _output_schema) — consistent with the tools test.
        """
        async with self.factory.build_agent_and_deps() as (agent, deps):
            assert agent.name == self.agent_record.name, "Agent name should come from the agent record"
            assert agent._deps_type is AgentDeps, "deps_type must be AgentDeps for tool functions to receive correct deps"
            assert get_system_prompt in agent._instructions, "get_system_prompt must be registered as the instructions function"
            assert agent._output_schema.allows_deferred_tools, "output_type must include DeferredToolRequests for the tool approval flow"

    async def test_thinking_disabled_by_default(self):
        """When thinking_enabled=False (default), no anthropic_thinking setting and deferred tools allowed."""
        assert self.agent_record.agent_config.thinking_enabled is False
        async with self.factory.build_agent_and_deps() as (agent, deps):
            assert "anthropic_thinking" not in agent.model_settings
            assert agent._output_schema.allows_deferred_tools is True

    async def test_thinking_enabled_sets_anthropic_thinking(self):
        """When thinking_enabled=True, anthropic_thinking and max_tokens are set.

        DeferredToolRequests is kept regardless of thinking mode — with str also in the output_type union,
        pydantic-ai uses tool_choice='auto' (not 'required'), which Anthropic accepts with thinking.
        Also requires max_tokens > budget_tokens.
        """
        self.agent_record.agent_config.thinking_enabled = True
        async with self.factory.build_agent_and_deps() as (agent, deps):
            assert agent.model_settings.get("anthropic_thinking") == {
                "type": "enabled",
                "budget_tokens": 10000,
            }
            assert agent.model_settings.get("max_tokens") == 16000
            assert agent._output_schema.allows_deferred_tools is True


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
        async with agent_factory.build_agent_and_deps() as (agent, deps):
            mock_get_tools.assert_called_once_with(agent_record.agent_config.tool_names)
            actual_tool_names = set(agent._function_toolset.tools.keys())
            assert actual_tool_names == expected_tool_names
