"""
Agent factory and dependency management — Section 3.1

Provides per-request AgentFactory for acquiring agent locks, loading deps, and building agents.

TODO: Consider the StatefulAgent class pattern, which would significantly shake the AgentFactory up
StatefulAgent Pattern:
  - Facade class owning a private pydantic-ai agent + member objects (MemorySystem, AgentRunner, etc.)
  - "One object = one agent" — easier to reason about than scattered free functions + deps
  - Related: ReadOnlyStatefulAgent for read-only routes (get_blocks, etc.) — no AgentRunner, read-only DB connection
  - Could own lifespan. Lock acquisition/release and such. AgentFactory could then be an object which only exists long enough to construct a StatefulAgent, or could just be a free function
"""
import asyncio
import logging
from contextlib import asynccontextmanager
from datetime import timedelta
from typing import AsyncIterator

from pydantic_ai import Agent
from pydantic_ai.capabilities import AgentCapability
from pydantic_ai.mcp import MCPToolset
from pydantic_ai.models import infer_model
from pydantic_ai.models.anthropic import AnthropicModel, AnthropicModelSettings
from pydantic_ai.models.openrouter import OpenRouterModel, OpenRouterModelSettings
from pydantic_ai.settings import ModelSettings
from pydantic_ai_harness.tool_output_limits import Band, LocalFileStore, Spill, ToolOutputLimits, Truncate
from sqlalchemy.ext.asyncio import AsyncSession

from agent.compaction_warner import CompactionWarner
from agent.crud import get_agent_record
from agent.types import AgentAppState, AgentConfig, AgentDeps, AgentLockedError, AgentNotFoundError, AgentOutput
from memory.system_prompt_compilation import get_system_prompt
from agent.tools import get_tools_for_agent


LOCK_TIMEOUT_SECONDS: int = 60
LOCK_TIMEOUT_FAST: int = 2
_MCP_FILESYSTEM_URL = "http://host.docker.internal:8080/mcp"

logger = logging.getLogger(__name__)


def _build_model_settings(config: "AgentConfig") -> ModelSettings:
    """Return provider-appropriate ModelSettings for the given agent config.

    Uses infer_model() to resolve the provider type, then dispatches on
    AnthropicModel for Anthropic-specific settings and OpenRouterModel for
    OpenRouter routing preferences. All other providers receive base
    ModelSettings with the unified 'thinking' field, which pydantic-ai maps
    to each provider's native reasoning parameter.

    parallel_tool_calls=False is applied for all providers: our orphan remover
    is not compatible with parallel tool calls.
    """
    m = infer_model(config.model_name)

    # Common settings for all providers. 'thinking' is the unified field pydantic-ai
    # maps to each provider's native reasoning parameter. For Anthropic, anthropic_thinking
    # takes precedence over this field (set below), so both can coexist safely.
    settings = ModelSettings(
        parallel_tool_calls=False,
        thinking=config.thinking_mode,
    )

    if isinstance(m, AnthropicModel):
        # Unpack common settings, then add Anthropic-specific fields.
        # anthropic_thinking takes precedence over the unified 'thinking' field above.
        settings = AnthropicModelSettings(
            **settings,
            anthropic_cache_instructions=True,
            anthropic_cache_tool_definitions=True,
            anthropic_cache_messages=True,
            # Anthropic requires max_tokens > budget_tokens when thinking is enabled
            **({"anthropic_thinking": {"type": "enabled", "budget_tokens": 10000},
                "max_tokens": 16000} if config.thinking_mode else {}),
        )
    elif isinstance(m, OpenRouterModel):
        # Routing preferences for OpenRouter-served models:
        # - quantizations: fp8 matches Z.AI's native serving quant — avoids fp4 routes
        #   with observable quality loss on our workload
        # - sort: price — cheapest qualifying provider; also opts out of Auto Exacto
        #   (provider reordering per tool-calling request), which would bust prompt
        #   caches mid-conversation. With price sort, sticky routing holds.
        # - data_collection: deny — no provider-side training on our data
        settings = OpenRouterModelSettings(
            **settings,
            openrouter_provider={
                "quantizations": ["fp8"],
                "sort": "price",
                "data_collection": "deny",
            },
            openrouter_usage={"include": True},
        )

    return settings


def _construct_toolsets(toolset_names: list[str]) -> list:
    """Construct toolset instances from a list of toolset names.

    Maps toolset names to their constructors and builds instances.

    Args:
        toolset_names: List of toolset identifiers (e.g., ["mcp_filesystem"])

    Returns:
        List of constructed toolset instances ready for Agent consumption.
    """
    # TODO: Consider module-level instances for connection reuse
    toolsets = []
    for name in toolset_names:
        if name == "mcp_filesystem":
            toolsets.append(MCPToolset(_MCP_FILESYSTEM_URL))
        else:
            logger.warning("Unknown toolset name %r — skipping. Check agent config for typos.", name)
    return toolsets


# --- Auto-injected capability constants ---

# Tool returns at or above this size (chars) are spilled to the overflow store instead of
# persisting in message history. ~500 lines of code: whole-file reads of typical source
# files pass through; larger sweeps, verbose search output, and context bombs spill.
# Chosen from read-distribution data + a feel-test (Oct 5); tune with live experience.
TOOL_OUTPUT_SPILL_THRESHOLD_CHARS: int = 20_000

# Spilled payloads are pruned (best-effort, background) after this age. Containers get no
# system-level temp cleanup, so this bounds disk growth between server container rebuilds.
TOOL_OUTPUT_SPILL_CLEANUP_AFTER: timedelta = timedelta(days=14)


def _build_capabilities() -> list[AgentCapability[AgentDeps]]:
    """Auto-injected capabilities applied to every constructed agent.

    - CompactionWarner: warns when the context window approaches the compaction threshold.
    - ToolOutputLimits: spills oversized tool returns to a store, replacing them in history
      with a handle + preview the model can page back through via the read_tool_result tool.
      Spill is lossless; Truncate is the fallback when the store cannot accept the write.
    """
    return [
        CompactionWarner(),
        ToolOutputLimits(
            bands=[Band(over=TOOL_OUTPUT_SPILL_THRESHOLD_CHARS, action=Spill(then=Truncate()))],
            store=LocalFileStore(cleanup_after=TOOL_OUTPUT_SPILL_CLEANUP_AFTER),
        ),
    ]


class AgentFactory:
    """Per-agent, per-request factory for building agents with locking.

    Constructed via FastAPI Depends with agent_id, session, and agent_app_state bound.
    Resolves (or creates) the agent's AppState entry at construction time, so the registry
    reference can be discarded after __init__. Routes call build_agent_and_deps() or
    build_deps() for clean interface.

    Read-only operations can use session directly without factory.
    Write operations require build_deps() or build_agent_and_deps(),
    which proves the caller holds the lock.
    """

    def __init__(self, agent_id: str, agent_app_state_reg: dict[str, AgentAppState], session: AsyncSession):
        """Resolve (or create) the agent slot from the registry, then discard the registry ref."""
        self._agent_id = agent_id
        self._agent_app_state = self._get_or_create_agent_app_state(agent_app_state_reg, agent_id)
        self._session = session  # TODO: session also lives and is passed around in deps. ref spaghetti?

    @staticmethod
    def _get_or_create_agent_app_state(agent_app_state_reg: dict[str, AgentAppState], agent_id: str) -> AgentAppState:
        """Get or create an AgentAppState entry for the given agent_id."""
        # TODO: Memory leak on garbage agent_IDs or if there are a TON of registered agents being invoked
        if agent_id not in agent_app_state_reg:
            agent_app_state_reg[agent_id] = AgentAppState()
        return agent_app_state_reg[agent_id]

    @asynccontextmanager
    async def build_deps(self, timeout: float = LOCK_TIMEOUT_SECONDS) -> AsyncIterator[AgentDeps]:
        """Async context manager that acquires the agent lock and yields AgentDeps.

        Lock-then-fetch: acquires lock BEFORE fetching from DB to prevent stale state.
        Releases lock on exit (normal or exception) via try/finally.
        """
        try:
            await asyncio.wait_for(self._agent_app_state.lock.acquire(), timeout=timeout)
        except asyncio.TimeoutError:
            raise AgentLockedError(f"Agent {self._agent_id!r} did not become available within {timeout}s")

        try:
            agent_record = await get_agent_record(self._session, self._agent_id)
            if agent_record is None:
                raise AgentNotFoundError(f"Agent {self._agent_id!r} not found")

            deps = AgentDeps(self._session, agent_record)
            yield deps
        finally:
            # Clear cancel_requested on exit to prevent a stale signal (e.g. a cancel that arrived
            # during teardown) from leaking into the next run. Tradeoff: if a second client is waiting
            # on the lock and sent a cancel for their queued run, that cancel is silently lost here.
            # Acceptable for now — our "queue" is just lock waiters with no ordering guarantees and no
            # run attribution. When we add a real queue, cancels should be associated with a specific
            # run so this ambiguity goes away.
            self._agent_app_state.cancel_requested.clear()
            self._agent_app_state.lock.release()


    @asynccontextmanager
    async def build_agent_and_deps(self) -> AsyncIterator[tuple[Agent[AgentDeps, AgentOutput], AgentDeps]]:
        """Async context manager that yields a configured (Agent, AgentDeps) tuple.
        
        Wraps build_deps and constructs the Pydantic AI Agent with correct model and tools.
        TODO: Sanity check the DeferredToolRequests thing, JF doesn't really understand whats going on there anymore
        Could be useful for client side tool execution.
        TODO: This doesn't actually manage context properly. It might release the lock but that seems to be pretty much ALL
        it does, it doesn't null out the resources actually associated with the lock!!!! Oops.
        """
        async with self.build_deps() as deps:
            model_settings = _build_model_settings(deps.config)
            toolsets = _construct_toolsets(deps.config.toolset_names)
            
            agent = Agent(deps.config.model_name,
                          instructions=get_system_prompt,
                          deps_type=AgentDeps,
                          name=deps.name,
                          tools=get_tools_for_agent(deps.config.tool_names),
                          toolsets=toolsets,
                          retries=deps.config.retries,
                          output_type=AgentOutput,
                          model_settings=model_settings,
                          capabilities=_build_capabilities())
            
            yield (agent, deps)

