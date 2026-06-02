"""
Agent factory and dependency management — Section 3.1

Provides per-request AgentFactory for acquiring agent locks, loading deps, and building agents.
"""
import asyncio
from contextlib import asynccontextmanager
from typing import AsyncIterator, Literal, get_args, get_origin

from pydantic_ai import Agent, DeferredToolRequests
from pydantic_ai.mcp import MCPToolset
from pydantic_ai.models.anthropic import AnthropicModel, AnthropicModelName, AnthropicModelSettings
from sqlalchemy.ext.asyncio import AsyncSession

from agent.crud import get_agent_record
from agent.types import AgentDeps
from memory.system_prompt_compilation import get_system_prompt
from agent.tools import get_tools_for_agent


# --- Domain Exceptions ---
# Routes translate these to HTTP status codes (404, 503)

class AgentNotFoundError(Exception):
    """Raised when agent_id doesn't exist in DB."""
    pass


class AgentLockedError(Exception):
    """Raised when agent is already in use by another request."""
    pass


class AgentFactory:
    """Per-request factory for building agents with locking.
    
    Constructed via FastAPI Depends with session + lock_reg bound.
    Routes call agent_factory.build_agent_and_deps(agent_id) for clean interface.
    
    Read-only operations can use session directly without factory.
    Write operations require build_deps() or build_agent_and_deps(),
    which proves the caller holds the lock.
    """

    LOCK_TIMEOUT_SECONDS: int = 60
    
    def __init__(self, lock_reg: dict[str, asyncio.Lock], session: AsyncSession):
        """Initialize factory with shared lock registry and per-request session."""
        self._lock_reg = lock_reg
        self._session = session # TODO: session also lives and is passed around in deps. ref spaghetti?
    
    def _get_lock(self, agent_id: str) -> asyncio.Lock:
        """Get or create a lock for the given agent_id from the registry."""
        # TODO: Memory leak on garbage agent_IDs or if there are a TON of registered agents being invoked
        if agent_id not in self._lock_reg:
            self._lock_reg[agent_id] = asyncio.Lock()
        return self._lock_reg[agent_id]
    
    @asynccontextmanager
    async def build_deps(self, agent_id: str) -> AsyncIterator[AgentDeps]:
        """Async context manager that acquires the agent lock and yields AgentDeps.
        
        Lock-then-fetch: acquires lock BEFORE fetching from DB to prevent stale state.
        Releases lock on exit (normal or exception) via try/finally.
        """
        lock = self._get_lock(agent_id)
        try:
            await asyncio.wait_for(lock.acquire(), timeout=self.LOCK_TIMEOUT_SECONDS)
        except asyncio.TimeoutError:
            raise AgentLockedError(f"Agent {agent_id!r} did not become available within {self.LOCK_TIMEOUT_SECONDS}s")

        try:
            agent_record = await get_agent_record(self._session, agent_id)
            if agent_record is None:
                raise AgentNotFoundError(f"Agent {agent_id!r} not found")
                        
            deps = AgentDeps(self._session, agent_record)
            yield deps
        finally:
            lock.release()
            

    @asynccontextmanager
    async def build_agent_and_deps(self, agent_id: str) -> AsyncIterator[tuple[Agent[AgentDeps, DeferredToolRequests | str], AgentDeps]]:
        """Async context manager that yields a configured (Agent, AgentDeps) tuple.
        
        Wraps build_deps and constructs the Pydantic AI Agent with correct model and tools.
        TODO: Sanity check the DeferredToolRequests thing, JF doesn't really understand whats going on there anymore
        Could be useful for client side tool execution.
        TODO: This doesn't actually manage context properly. It might release the lock but that seems to be pretty much ALL
        it does, it doesn't null out the resources actually associated with the lock!!!! Oops.
        """
        async with self.build_deps(agent_id) as deps:
            model = get_model(deps.config.model_name)
            
            model_settings = AnthropicModelSettings(
                anthropic_cache_instructions=True,
                anthropic_cache_tool_definitions=True,
                anthropic_cache_messages=True,
                # Anthropic requires max_tokens > budget_tokens when thinking is enabled
                **({"anthropic_thinking": {"type": "enabled", "budget_tokens": 10000},
                    "max_tokens": 16000}
                   if deps.config.thinking_enabled else {}),
            )
            # MCP filesystem tools via Streamable HTTP (requires fs_proxy running)
            # TODO: Make URL configurable via deps.config or env
            # TODO: Consider moving to module/app level for connection reuse (currently per-request)
            mcp_fs_server = MCPToolset("http://localhost:8080/mcp")
            
            agent = Agent(model,
                          instructions=get_system_prompt,
                          deps_type=AgentDeps,
                          name=deps.name,
                          tools=get_tools_for_agent(deps.config.tool_names),
                          toolsets=[mcp_fs_server],
                          retries=deps.config.retries,
                          output_type=[str, DeferredToolRequests],
                          model_settings=model_settings)
            
            yield (agent, deps)


# AnthropicModelName is defined as str | Literal['claude-...', ...]. The str union arm is an
# escape hatch for forward compatibility — we want only the known Literal values for validation.
_literal_type = next(arg for arg in get_args(AnthropicModelName) if get_origin(arg) is Literal)
_VALID_MODEL_NAMES: frozenset[str] = frozenset(get_args(_literal_type))


def get_model(model_name: str) -> AnthropicModel:
    """Map a model name string to a Pydantic AI model instance.
    
    Raises ValueError for unknown or unsupported model names.
    """
    #TODO: JF Review
    if model_name not in _VALID_MODEL_NAMES:
        raise ValueError(f"Unsupported model name: {model_name!r}. Must be one of: {sorted(_VALID_MODEL_NAMES)}")
    return AnthropicModel(model_name)
