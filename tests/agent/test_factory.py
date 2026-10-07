"""
Tests for agent/factory.py

Agent factory and dependency management:
- AgentFactory: Per-agent, per-request factory with agent_id, session, and agent app state bound
  - _get_or_create_agent_app_state: Per-agent state registry management (static)
  - build_deps: Async context manager yielding AgentDeps with lock acquisition
  - build_agent_and_deps: Async context manager yielding (Agent, AgentDeps)

NOTE: This got a little ugly with the move from a simple lock reg to AgentAppState. Since we might move to a different OO design with a 
StatefulAgent class, I don't think its worth cleaning this up right now though.
In particular, the lock testing, create_spied_lock should be rethought if we stick with this pattern. The agent factory creating AgentAppState
if nonexistent during *construction* as opposed to during deps building caused issues with how these tests were previously written and the patches
are not ideal
"""
import asyncio
import time
from unittest.mock import patch, MagicMock

import pytest
import pytest_asyncio
from pytest_mock import MockerFixture
from pydantic_ai import Agent
from pydantic_ai.mcp import MCPToolset
from pydantic_ai.messages import ModelResponse, RetryPromptPart, ThinkingPart, ToolCallPart
from pydantic_ai.models.function import FunctionModel
from pydantic_ai_harness.tool_output_limits import Band, LocalFileStore, Spill, ToolOutputLimits, Truncate
from sqlalchemy.ext.asyncio import AsyncSession

from agent.compaction_warner import CompactionWarner
from agent.factory import (
    AgentFactory,
    TOOL_OUTPUT_SPILL_CLEANUP_AFTER,
    TOOL_OUTPUT_SPILL_THRESHOLD_CHARS,
    _build_capabilities,
    _build_model_settings,
)
from agent.types import AgentAppState, AgentConfig, AgentDeps, AgentLockedError, AgentNotFoundError, AgentOutput
from memory.system_prompt_compilation import get_system_prompt
from conftest import SAMPLE_AGENT_CONFIG, _ScriptedFunction, local_dummy_tool
from db.models import AgentRecord


# --- Constants ---

NONEXISTENT_AGENT_ID = "nonexistent-agent-id-12345"


# --- Helpers ---

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


# --- Fixtures ---

@pytest.fixture(autouse=True)
def _fake_provider_keys(fake_provider_keys):
    """Fake keys for every factory test — all construct real provider clients
    via infer_model() in build_agent_and_deps / _build_model_settings."""


@pytest.fixture
def agent_app_state_reg() -> dict[str, AgentAppState]:
    """
    Fresh agent state registry for each test.
    A fixture that returns an empty dict is a bit ridiculous but it helps with documentation
    (IE communicates what this dict is meant to be.)
    It also gives us a single common dict obj across other fixtures to inspect within a single test
    NOTE: This fixture MUST be an empty dict
    """
    return {}


@pytest.fixture
def agent_factory(agent_record: AgentRecord, agent_app_state_reg: dict, session: AsyncSession) -> AgentFactory:
    """Per-agent, per-request AgentFactory with agent_id, agent_app_state_reg, and session bound."""
    return AgentFactory(agent_record.id, agent_app_state_reg, session)


# --- AgentFactory._get_or_create_agent_app_state tests ---

# NOTE: We're testing internals too much here. If we stick with the AgentFactory pattern, just test the construction and its side effects
def test_get_or_create_agent_app_state_returns_same_agent_app_state_for_same_id(agent_app_state_reg: dict):
    """_get_or_create_agent_app_state should return the same AgentAppState for the same agent_id."""
    # testing static method so no need to use the fixture which gives an object
    slot1 = AgentFactory._get_or_create_agent_app_state(agent_app_state_reg, "agent-123")
    slot2 = AgentFactory._get_or_create_agent_app_state(agent_app_state_reg, "agent-123")

    assert slot1 is slot2
    assert isinstance(slot1, AgentAppState)
    assert isinstance(slot1.lock, asyncio.Lock)
    assert isinstance(slot1.cancel_requested, asyncio.Event)
    assert not slot1.lock.locked()
    assert not slot1.cancel_requested.is_set()


def test_get_or_create_agent_app_state_returns_different_agent_app_state_reg_for_different_ids(agent_app_state_reg: dict):
    """_get_or_create_agent_app_state should return different AgentAppState instances for different agent_ids."""
    slot_a = AgentFactory._get_or_create_agent_app_state(agent_app_state_reg, "agent-aaa")
    slot_b = AgentFactory._get_or_create_agent_app_state(agent_app_state_reg, "agent-bbb")

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
    agent_app_state_reg: dict,
):
    """build_deps should acquire lock before yield, release after exit.

    agent_app_state for particular agent is created at AgentFactory construction, so the "creates" part of this test is really
    testing the AgentFactory constructor
    """
    assert agent_record.id in agent_app_state_reg
    lock = agent_app_state_reg[agent_record.id].lock
    assert not lock.locked()

    async with agent_factory.build_deps() as deps:
        assert lock.locked()

    # After exiting, lock should be released
    assert not lock.locked()


class TestBuildDepsLockAndCancelBehavior:
    """Verifies lock acquire/release (via spy) and cancel_requested clearing on teardown.

    Spy slot is pre-populated before factory construction so _get_or_create_agent_app_state
    returns it as self._agent_app_state. Tests cover both normal and exception exit paths.
    """

    @pytest.fixture(autouse=True)
    def _setup(
        self,
        agent_record: AgentRecord,
        agent_app_state_reg: dict[str, AgentAppState],
        session: AsyncSession,
        mocker: MockerFixture,
    ):
        self.lock = create_spied_lock(agent_record.id, agent_app_state_reg, mocker)
        self.factory = AgentFactory(agent_record.id, agent_app_state_reg, session)
        self.cancel_requested = agent_app_state_reg[agent_record.id].cancel_requested

    @pytest.mark.asyncio
    async def test_normal_exit_releases_lock_and_clears_cancel(self):
        """build_deps should release the lock and clear a pending cancel on normal exit."""
        self.cancel_requested.set()

        async with self.factory.build_deps():
            assert self.lock.locked()

        assert_lock_acquired_and_released(self.lock)
        assert not self.cancel_requested.is_set()

    @pytest.mark.asyncio
    async def test_exception_exit_releases_lock_and_clears_cancel(self):
        """build_deps should release the lock and clear a pending cancel even if an exception is raised."""
        self.cancel_requested.set()

        with pytest.raises(RuntimeError):
            async with self.factory.build_deps():
                raise RuntimeError("Intentional test error")

        assert_lock_acquired_and_released(self.lock)
        assert not self.cancel_requested.is_set()


@pytest.mark.asyncio
async def test_build_deps_raises_releases_lock_and_clears_cancel_on_fetch_failure(
    agent_app_state_reg: dict,
    session: AsyncSession,
    mocker: MockerFixture,
):
    """build_deps should raise for unknown agent_id AND release lock and clear cancel on failure.

    Design decision: lock-then-fetch. The lock must be acquired BEFORE the DB fetch
    to prevent concurrent runs from seeing stale state. This means if the fetch fails,
    the lock was already acquired and must be released via try/finally.
    Uses NONEXISTENT_AGENT_ID (no agent_record fixture) — kept standalone for that reason.
    """
    lock = create_spied_lock(NONEXISTENT_AGENT_ID, agent_app_state_reg, mocker)
    factory = AgentFactory(NONEXISTENT_AGENT_ID, agent_app_state_reg, session)

    cancel_requested = agent_app_state_reg[NONEXISTENT_AGENT_ID].cancel_requested
    cancel_requested.set()

    with pytest.raises(AgentNotFoundError):
        async with factory.build_deps() as deps:
            pass

    assert_lock_acquired_and_released(lock)
    assert not cancel_requested.is_set()


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
    agent_app_state_reg: dict,
    session: AsyncSession,
):
    """
    Concurrent calls on different agent_ids should not block each other.
    Each agent gets its own factory (matching real usage — distinct factories per request).
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
    await session.flush()

    factory_a = AgentFactory(agent_a.id, agent_app_state_reg, session)
    factory_b = AgentFactory(agent_b.id, agent_app_state_reg, session)

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


@pytest.mark.asyncio
@pytest.mark.parametrize("timeout,min_elapsed,max_elapsed", [
    (0.01, 0.0, 0.5),   # fast — just assert it doesn't block
    (1.0,  0.8, 2.5),   # roughly a second
])
async def test_build_deps_timeout_raises_agent_locked_error(
    agent_factory: AgentFactory,
    agent_app_state_reg: dict,
    agent_record: AgentRecord,
    timeout: float,
    min_elapsed: float,
    max_elapsed: float,
):
    """build_deps raises AgentLockedError after approximately the requested timeout."""
    # Hold the lock so the second acquire times out
    await agent_app_state_reg[agent_record.id].lock.acquire()

    start = time.monotonic()
    with pytest.raises(AgentLockedError):
        async with agent_factory.build_deps(timeout=timeout):
            pass  # should not reach here
    elapsed = time.monotonic() - start

    assert elapsed >= min_elapsed, f"Timed out too fast: {elapsed:.3f}s < {min_elapsed}s"
    assert elapsed < max_elapsed, f"Timed out too slow: {elapsed:.3f}s >= {max_elapsed}s"


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
        agent_app_state_reg: dict,
    ):
        self.factory = agent_factory
        self.agent_record = agent_record
        self.agent_app_state_reg = agent_app_state_reg
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
            assert self.agent_app_state_reg[self.agent_record.id].lock.locked()

        assert not self.agent_app_state_reg[self.agent_record.id].lock.locked()

    async def test_uses_correct_model(self):
        """Constructed agent should use the model from agent_config.model_name."""
        async with self.factory.build_agent_and_deps() as (agent, deps):
            # pydantic-ai resolves 'provider:model' strings to a model instance;
            # model_name on the resolved object is the model part only (after the colon).
            # NOTE: This assumes pydantic-ai exposes model_name on resolved models consistently.
            # If pydantic-ai changes how it reports model names, this assertion may need updating.
            assert agent.model.model_name == self.agent_record.agent_config.model_name.split(":", 1)[1]

    async def test_output_type_is_agent_output(self):
        """Constructed agent must use the AgentOutput union, including None.

        None opts into pydantic-ai's allows_none path so empty/thinking-only responses
        complete runs instead of triggering output retries. Behavioral tripwire below.
        """
        async with self.factory.build_agent_and_deps() as (agent, deps):
            assert agent.output_type == AgentOutput


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
            instructions = [si.instruction for si in agent._instructions]
            assert get_system_prompt in instructions, "get_system_prompt must be registered as the instructions function"
            assert agent._output_schema.allows_deferred_tools, "output_type must include DeferredToolRequests for the tool approval flow"

    async def test_model_settings_applied_from_helper(self):
        """build_agent_and_deps applies the result of _build_model_settings to the agent.

        Provider-specific correctness is covered by TestBuildModelSettings unit tests.
        Here we just verify the helper is wired up and its output reaches the agent.
        """
        with patch("agent.factory._build_model_settings", wraps=_build_model_settings) as mock_helper:
            async with self.factory.build_agent_and_deps() as (agent, deps):
                mock_helper.assert_called_once_with(self.agent_record.agent_config)

    async def test_capabilities_applied_from_helper(self):
        """build_agent_and_deps applies the result of _build_capabilities to the agent.

        Capability contents are covered by TestBuildCapabilities unit tests. Here we just
        verify the helper is wired up (same pattern as the model settings helper test —
        the Agent wraps capabilities into a CombinedCapability, so list-level inspection
        is not practical).
        """
        with patch("agent.factory._build_capabilities", wraps=_build_capabilities) as mock_helper:
            async with self.factory.build_agent_and_deps() as (agent, deps):
                mock_helper.assert_called_once()


# ---------------------------------------------------------------------------
# AgentOutput behavioral tripwire (pydantic-ai upgrade guard)
# ---------------------------------------------------------------------------

@pytest.mark.asyncio
@pytest.mark.parametrize(
    "final_parts",
    [
        pytest.param([], id="empty"),
        pytest.param([ThinkingPart(content="all done")], id="thinking-only"),
    ],
)
async def test_empty_final_response_completes_run(final_parts):
    """Tripwire on pydantic-ai behavior Agent Home depends on: with None in the
    output union (AgentOutput), an empty or thinking-only response after tool
    work completes the run with a None result — no output retry.

    We don't test the framework for its own sake — but our 1.97→2.54 upgrade
    silently changed empty-response behavior (upstream #6403 removed v1.97's
    text recovery), and our only detection was post-hoc forensics. This pins
    the behavior so the next pydantic-ai bump surfaces a regression in CI
    instead of in conversation history.

    Bare agent on AgentOutput (same output type the factory wires, pinned by
    test_output_type_is_agent_output above) driven by conftest's
    _ScriptedFunction. Asserts: None result, exactly two model calls (a third
    would mean a retry fired), no RetryPromptPart in history, and exact
    history shape — nothing extra injected.
    """
    scripted = _ScriptedFunction([
        ModelResponse(parts=[ToolCallPart(
            tool_name="local_dummy_tool", args='{"text": "ok"}', tool_call_id="tc-1",
        )]),
        ModelResponse(parts=final_parts),
    ])
    agent = Agent(FunctionModel(scripted), tools=[local_dummy_tool], output_type=AgentOutput)

    result = await agent.run("go")

    assert result.output is None
    assert scripted.invocation == 2, f"Model called {scripted.invocation}x — retry fired?"

    messages = result.all_messages()
    retry_parts = [
        part for message in messages
        for part in getattr(message, "parts", [])
        if isinstance(part, RetryPromptPart)
    ]
    assert not retry_parts, f"Unexpected retry prompts in history: {retry_parts}"

    # Structural fingerprint (message type, part kinds) — avoids equality
    # pitfalls with run-generated timestamps/usage while asserting exact shape.
    shape = [
        (type(m).__name__, tuple(type(p).__name__ for p in m.parts)) for m in messages
    ]
    assert shape == [
        ("ModelRequest", ("UserPromptPart",)),
        ("ModelResponse", ("ToolCallPart",)),
        ("ModelRequest", ("ToolReturnPart",)),
        ("ModelResponse", tuple(type(p).__name__ for p in final_parts)),
    ]


# =============================================================================
# _build_model_settings unit tests
# =============================================================================

class TestBuildModelSettings:
    """Unit tests for _build_model_settings — called directly as a pure function."""

    def _config(self, model_name: str, thinking_mode: "bool | str" = False) -> AgentConfig:
        return SAMPLE_AGENT_CONFIG.model_copy(update={"model_name": model_name, "thinking_mode": thinking_mode})

    # --- Base settings (common to all providers) ---

    @pytest.mark.parametrize("model_name", [
        "together:meta-llama/Llama-3.3-70B-Instruct-Turbo",
        "openai-chat:gpt-4o",
        "anthropic:claude-haiku-4-5",
        "openrouter:z-ai/glm-5.3",
    ])
    def test_base_settings_defaults(self, model_name: str):
        """All providers: parallel_tool_calls=False, thinking=False by default."""
        settings = _build_model_settings(self._config(model_name))
        assert settings.get("parallel_tool_calls") is False
        assert settings.get("thinking") is False

    @pytest.mark.parametrize("model_name", [
        "together:meta-llama/Llama-3.3-70B-Instruct-Turbo",
        "openai-chat:gpt-4o",
        "anthropic:claude-haiku-4-5",
        "openrouter:z-ai/glm-5.3",
    ])
    @pytest.mark.parametrize("thinking_mode", [True, "high", "low"])
    def test_base_settings_thinking_mode(self, model_name: str, thinking_mode: "bool | str"):
        """All providers: thinking passes through thinking_mode (bool or effort level)."""
        settings = _build_model_settings(self._config(model_name, thinking_mode=thinking_mode))
        assert settings.get("thinking") is thinking_mode

    # --- Anthropic-specific settings ---

    def test_anthropic_cache_flags(self):
        """Anthropic model → all three prompt cache flags enabled."""
        settings = _build_model_settings(self._config("anthropic:claude-haiku-4-5"))
        assert settings.get("anthropic_cache_instructions") is True
        assert settings.get("anthropic_cache_tool_definitions") is True
        assert settings.get("anthropic_cache_messages") is True

    def test_anthropic_thinking_sets_budget(self):
        """thinking_mode=True → anthropic_thinking with budget and max_tokens."""
        settings = _build_model_settings(self._config("anthropic:claude-haiku-4-5", thinking_mode=True))
        assert settings.get("anthropic_thinking") == {"type": "enabled", "budget_tokens": 10000}
        assert settings.get("max_tokens") == 16000

    def test_anthropic_thinking_disabled_no_budget(self):
        """thinking_mode=False → no anthropic_thinking set."""
        settings = _build_model_settings(self._config("anthropic:claude-haiku-4-5", thinking_mode=False))
        assert "anthropic_thinking" not in settings

    def test_non_anthropic_no_cache_flags(self):
        """Non-Anthropic providers → no Anthropic-specific cache fields."""
        for model_name in ("together:meta-llama/Llama-3.3-70B-Instruct-Turbo", "openai-chat:gpt-4o"):
            settings = _build_model_settings(self._config(model_name))
            assert "anthropic_cache_instructions" not in settings
            assert "anthropic_thinking" not in settings

    # --- OpenRouter-specific settings ---

    def test_openrouter_routing_prefs(self):
        """OpenRouter model → fp8-only routing, price sort (Auto Exacto off), data collection denied."""
        settings = _build_model_settings(self._config("openrouter:z-ai/glm-5.3"))
        provider = settings.get("openrouter_provider")
        assert provider == {"quantizations": ["fp8"], "sort": "price", "data_collection": "deny"}

    def test_openrouter_usage_included(self):
        """OpenRouter model → usage details included in responses."""
        settings = _build_model_settings(self._config("openrouter:z-ai/glm-5.3"))
        assert settings.get("openrouter_usage") == {"include": True}

    def test_non_openrouter_no_routing_prefs(self):
        """Non-OpenRouter providers → no openrouter fields."""
        for model_name in ("together:meta-llama/Llama-3.3-70B-Instruct-Turbo", "anthropic:claude-haiku-4-5"):
            settings = _build_model_settings(self._config(model_name))
            assert "openrouter_provider" not in settings
            assert "openrouter_usage" not in settings


# =============================================================================
# _build_capabilities unit tests
# =============================================================================

class TestBuildCapabilities:
    """Unit tests for _build_capabilities — called directly as a pure function.

    Behavioral coverage (spill actually happening) lives in test_tool_output_limits.py;
    these tests pin the *configuration* the factory injects.
    """

    def test_returns_compaction_warner_and_tool_output_limits(self):
        """Every agent gets the compaction warner and the tool output spill capability."""
        caps = _build_capabilities()
        assert [type(c) for c in caps] == [CompactionWarner, ToolOutputLimits]

    def test_tool_output_limits_config(self):
        """ToolOutputLimits is configured for straight spill with truncate fallback and a TTL'd local store."""
        tol = _build_capabilities()[1]

        assert isinstance(tol.store, LocalFileStore)
        assert tol.store.cleanup_after == TOOL_OUTPUT_SPILL_CLEANUP_AFTER

        assert len(tol.bands) == 1
        band = tol.bands[0]
        assert isinstance(band, Band)
        assert band.over == TOOL_OUTPUT_SPILL_THRESHOLD_CHARS
        assert isinstance(band.action, Spill)
        assert isinstance(band.action.then, Truncate)


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


# --- Toolset Conditional Attachment Tests ---

class TestToolsetConditionalAttachment:
    """Tests for conditional MCP toolset attachment based on AgentConfig.toolset_names.
    
    MCPToolset should only be instantiated when its name appears in toolset_names.
    This prevents connection attempts when MCP server isn't running.
    """
    
    @pytest.fixture(autouse=True)
    def patch_factory_deps(self):
        """Patch get_tools_for_agent and MCPToolset for all tests in this class."""
        with (
            patch("agent.factory.get_tools_for_agent", return_value=[]),
            patch("agent.factory.MCPToolset") as mock_mcp,
        ):
            # Configure mock to return something that passes pydantic-ai's toolset validation
            mock_mcp.return_value = MagicMock(spec=MCPToolset)
            self.mock_mcp = mock_mcp
            yield
    
    @pytest_asyncio.fixture
    async def agent_record_no_toolsets(self, session: AsyncSession) -> AgentRecord:
        """Agent record with empty toolset_names (default behavior)."""
        config = SAMPLE_AGENT_CONFIG.model_copy(update={"toolset_names": []})
        record = AgentRecord(
            name="no-toolsets-agent",
            agent_config=config,
            system_instructions="Test agent with no toolsets",
        )
        session.add(record)
        await session.flush()
        return record
    
    @pytest_asyncio.fixture
    async def agent_record_with_mcp(self, session: AsyncSession) -> AgentRecord:
        """Agent record with MCP filesystem toolset enabled."""
        config = SAMPLE_AGENT_CONFIG.model_copy(update={"toolset_names": ["mcp_filesystem"]})
        record = AgentRecord(
            name="mcp-enabled-agent",
            agent_config=config,
            system_instructions="Test agent with MCP toolset",
        )
        session.add(record)
        await session.flush()
        return record
    
    @pytest.mark.asyncio
    async def test_empty_toolset_names_no_mcp_instantiation(
        self,
        session: AsyncSession,
        agent_record_no_toolsets: AgentRecord,
        agent_app_state_reg: dict,
    ):
        """When toolset_names is empty, MCPToolset should not be instantiated."""
        factory = AgentFactory(agent_record_no_toolsets.id, agent_app_state_reg, session)
        
        async with factory.build_agent_and_deps() as (agent, deps):
            self.mock_mcp.assert_not_called()
            assert agent._user_toolsets == [], "No toolsets should be attached"
    
    @pytest.mark.asyncio
    async def test_mcp_in_toolset_names_triggers_instantiation(
        self,
        session: AsyncSession,
        agent_record_with_mcp: AgentRecord,
        agent_app_state_reg: dict,
    ):
        """When toolset_names includes 'mcp_filesystem', MCPToolset should be instantiated."""
        factory = AgentFactory(agent_record_with_mcp.id, agent_app_state_reg, session)
        
        async with factory.build_agent_and_deps() as (agent, deps):
            self.mock_mcp.assert_called_once_with("http://host.docker.internal:8080/mcp")
            assert len(agent._user_toolsets) == 1, "One toolset should be attached"
            # The toolset is the return value of our mocked MCPToolset constructor
            assert agent._user_toolsets[0] is self.mock_mcp.return_value
