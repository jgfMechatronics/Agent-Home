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
from unittest.mock import patch, MagicMock

import pytest
import pytest_asyncio
from pydantic_ai import Agent
from pydantic_ai.mcp import MCPToolset
from pydantic_ai.messages import (
    ModelRequest,
    ModelResponse,
    RetryPromptPart,
    ThinkingPart,
    ToolCallPart,
    ToolReturnPart,
    UserPromptPart,
)
from pydantic_ai.models.function import FunctionModel
from pydantic_ai.profiles.anthropic import ANTHROPIC_THINKING_BUDGET_MAP
from pydantic_ai_harness.tool_output_limits import Band, LocalFileStore, Spill, ToolOutputLimits, Truncate, indented_json
from sqlalchemy.ext.asyncio import AsyncSession

from agent.compaction_warner import CompactionWarner
from agent.parallel_tool_call_trimmer import ParallelToolCallTrimmer
from agent.timestamping import ToolReturnTimestamper
from agent.factory import (
    AgentFactory,
    TOOL_OUTPUT_SPILL_CLEANUP_AFTER,
    TOOL_OUTPUT_SPILL_THRESHOLD_CHARS,
    _build_capabilities,
    _build_model_settings,
    get_or_create_agent_app_state,
)
from agent.deps import AgentDeps
from agent.types import (
    THINKING_EFFORT_LEVELS,
    AgentAppState,
    AgentConfig,
    AgentOutput,
)
from memory.system_prompt_compilation import get_system_prompt
from conftest import (
    SAMPLE_AGENT_CONFIG,
    ScriptedFunction,
    assert_ModelMessage_list_eq,
    local_dummy_tool,
    commit_and_detach_fixture_record,
    make_ro_engine,
)
from db.models import AgentRecord


# --- Constants ---

# --- Helpers ---


# --- Fixtures ---

@pytest.fixture(autouse=True)
def _fake_provider_keys(fake_provider_keys):
    """Fake keys for every factory test — all construct real provider clients
    via infer_model() in build_agent_and_deps / _build_model_settings."""


@pytest.fixture
def agent_factory(agent_record: AgentRecord, agent_app_state_reg: dict, test_engine) -> AgentFactory:
    """Per-agent, per-request AgentFactory with agent_id, agent_app_state_reg, and engines bound."""
    return AgentFactory(agent_record.id, agent_app_state_reg, test_engine, make_ro_engine(test_engine))



# --- AgentFactory._get_or_create_agent_app_state tests ---

# NOTE: We're testing internals too much here. If we stick with the AgentFactory pattern, just test the construction and its side effects
def test_get_or_create_agent_app_state_returns_same_agent_app_state_for_same_id(agent_app_state_reg: dict):
    """_get_or_create_agent_app_state should return the same AgentAppState for the same agent_id."""
    # testing static method so no need to use the fixture which gives an object
    slot1 = get_or_create_agent_app_state(agent_app_state_reg, "agent-123")
    slot2 = get_or_create_agent_app_state(agent_app_state_reg, "agent-123")

    assert slot1 is slot2
    assert isinstance(slot1, AgentAppState)
    assert isinstance(slot1.lock, asyncio.Lock)
    assert isinstance(slot1.cancel_requested, asyncio.Event)
    assert not slot1.lock.locked()
    assert not slot1.cancel_requested.is_set()


def test_get_or_create_agent_app_state_returns_different_agent_app_state_reg_for_different_ids(agent_app_state_reg: dict):
    """_get_or_create_agent_app_state should return different AgentAppState instances for different agent_ids."""
    slot_a = get_or_create_agent_app_state(agent_app_state_reg, "agent-aaa")
    slot_b = get_or_create_agent_app_state(agent_app_state_reg, "agent-bbb")

    assert slot_a is not slot_b
    assert isinstance(slot_a, AgentAppState)
    assert isinstance(slot_b, AgentAppState)


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
    ScriptedFunction. Asserts: None result, exactly two model calls (a third
    would mean a retry fired), no RetryPromptPart in history, and exact
    history content via assert_ModelMessage_list_eq (semantic equality
    ignoring runtime fields).
    """
    DUMMY_TOOL_CALL = ToolCallPart(
    tool_name="local_dummy_tool", args='{"text": "ok"}', tool_call_id="tc-1",
    )
    DUMMY_TOOL_RETURN = ToolReturnPart(
        tool_name="local_dummy_tool", content="ok", tool_call_id="tc-1",
    )

    scripted = ScriptedFunction([
        ModelResponse(parts=[DUMMY_TOOL_CALL]),
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

    assert_ModelMessage_list_eq(messages, [
        ModelRequest(parts=[UserPromptPart(content="go")]),
        ModelResponse(parts=[DUMMY_TOOL_CALL]),
        ModelRequest(parts=[DUMMY_TOOL_RETURN]),
        ModelResponse(parts=final_parts),
    ])


# =============================================================================
# _build_model_settings unit tests
# =============================================================================

class TestBuildModelSettings:
    """Unit tests for _build_model_settings — called directly as a pure function."""

    EXPECTED_MAX_TOKENS = 36864
    """The provider-wide output cap set in the common ModelSettings base.

    36864 = xhigh thinking budget (32768) + 4096 answer room; must exceed
    every thinking level's budget (Anthropic requires max_tokens > budget).
    """

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
        """All providers: thinking=False, output capped at 32k, 180s inter-chunk timeout.

        The max_tokens cap bounds a degenerate generation (provider glitch → no stop
        token) to 32k tokens instead of an unbounded runaway stream. The timeout caps
        how long a hung provider can stall the run (pydantic-ai's default is 600s).
        """
        settings = _build_model_settings(self._config(model_name))
        assert settings.get("thinking") is False
        assert settings.get("max_tokens") == self.EXPECTED_MAX_TOKENS
        assert settings.get("timeout") == 180

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

    @pytest.mark.parametrize("thinking_mode", [True, *THINKING_EFFORT_LEVELS])
    def test_anthropic_thinking_unified_no_override(self, thinking_mode: "bool | str"):
        """Every thinking level → unified 'thinking' passes through; no override; cap fits the budget.

        pydantic-ai maps the unified field natively (True/effort levels → budget_tokens
        via ANTHROPIC_THINKING_BUDGET_MAP), so a hand-rolled anthropic_thinking dict
        is unnecessary. Anthropic requires max_tokens > budget_tokens, so the base
        cap must exceed the largest level's budget. Importing the budget map keeps
        this in sync with the library (loud tripwire if budgets change).
        """
        settings = _build_model_settings(self._config("anthropic:claude-haiku-4-5", thinking_mode=thinking_mode))
        assert settings.get("thinking") == thinking_mode
        assert "anthropic_thinking" not in settings
        assert settings.get("max_tokens") == self.EXPECTED_MAX_TOKENS
        assert self.EXPECTED_MAX_TOKENS > ANTHROPIC_THINKING_BUDGET_MAP[thinking_mode]

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
        """OpenRouter model → data collection denied, default routing (cache-adjusted effective price)."""
        settings = _build_model_settings(self._config("openrouter:z-ai/glm-5.3"))
        provider = settings.get("openrouter_provider")
        assert provider == {"data_collection": "deny"}

    def test_openrouter_provider_order_from_env(self, monkeypatch):
        """OPENROUTER_PROVIDER_ORDER env var → provider order list added."""
        monkeypatch.setenv("OPENROUTER_PROVIDER_ORDER", "anthropic, aws-bedrock")
        settings = _build_model_settings(self._config("openrouter:z-ai/glm-5.3"))
        provider = settings.get("openrouter_provider")
        assert provider == {
            "data_collection": "deny",
            "order": ["anthropic", "aws-bedrock"],
        }

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
        """Every agent gets the compaction warner, tool return stamper, spill capability, and
        parallel tool call trimmer (last, so its after_model_request runs first)."""
        caps = _build_capabilities()
        assert [type(c) for c in caps] == [
            CompactionWarner, ToolReturnTimestamper, ToolOutputLimits, ParallelToolCallTrimmer,
        ]

    def test_tool_output_limits_config(self):
        """ToolOutputLimits is configured for straight spill with truncate fallback and a TTL'd local store."""
        tol = _build_capabilities()[2]

        assert isinstance(tol.store, LocalFileStore)
        assert tol.store.cleanup_after == TOOL_OUTPUT_SPILL_CLEANUP_AFTER

        assert len(tol.bands) == 1
        band = tol.bands[0]
        assert isinstance(band, Band)
        assert band.over == TOOL_OUTPUT_SPILL_THRESHOLD_CHARS
        assert isinstance(band.action, Spill)
        assert isinstance(band.action.then, Truncate)

        # indented_json serializer ensures structured tool returns spill with newlines,
        # making read_tool_result line-based pagination work correctly
        assert tol.serializer is indented_json


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
        return await commit_and_detach_fixture_record(session, record)
    
    @pytest_asyncio.fixture
    async def agent_record_with_mcp(self, session: AsyncSession) -> AgentRecord:
        """Agent record with MCP filesystem toolset enabled."""
        config = SAMPLE_AGENT_CONFIG.model_copy(update={"toolset_names": ["mcp_filesystem"]})
        record = AgentRecord(
            name="mcp-enabled-agent",
            agent_config=config,
            system_instructions="Test agent with MCP toolset",
        )
        return await commit_and_detach_fixture_record(session, record)
    
    @pytest.mark.asyncio
    async def test_empty_toolset_names_no_mcp_instantiation(
        self,
        test_engine,
        agent_record_no_toolsets: AgentRecord,
        agent_app_state_reg: dict,
    ):
        """When toolset_names is empty, MCPToolset should not be instantiated."""
        factory = AgentFactory(agent_record_no_toolsets.id, agent_app_state_reg, test_engine, make_ro_engine(test_engine))
        
        async with factory.build_agent_and_deps() as (agent, deps):
            self.mock_mcp.assert_not_called()
            assert agent._user_toolsets == [], "No toolsets should be attached"
    
    @pytest.mark.asyncio
    async def test_mcp_in_toolset_names_triggers_instantiation(
        self,
        test_engine,
        agent_record_with_mcp: AgentRecord,
        agent_app_state_reg: dict,
    ):
        """When toolset_names includes 'mcp_filesystem', MCPToolset should be instantiated."""
        factory = AgentFactory(agent_record_with_mcp.id, agent_app_state_reg, test_engine, make_ro_engine(test_engine))
        
        async with factory.build_agent_and_deps() as (agent, deps):
            self.mock_mcp.assert_called_once_with("http://host.docker.internal:8080/mcp")
            assert len(agent._user_toolsets) == 1, "One toolset should be attached"
            # The toolset is the return value of our mocked MCPToolset constructor
            assert agent._user_toolsets[0] is self.mock_mcp.return_value
