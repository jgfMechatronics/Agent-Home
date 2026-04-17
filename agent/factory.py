"""
Agent factory and dependency management — Section 3.1

Provides per-request AgentFactory for acquiring agent locks, loading deps, and building agents.
"""
import asyncio
from contextlib import asynccontextmanager
from typing import AsyncIterator

from pydantic_ai import Agent, DeferredToolRequests
from pydantic_ai.models.anthropic import AnthropicModel, AnthropicModelSettings
from sqlalchemy.ext.asyncio import AsyncSession

from agent.types import AgentDeps
from db.models import AgentRecord
from memory.system_prompt_compilation import get_system_prompt
from tools import get_tools_for_agent

class AgentFactory:
    """Per-request factory for building agents with locking.
    
    Constructed via FastAPI Depends with session + lock_reg bound.
    Routes call agent_factory.build_agent_and_deps(agent_id) for clean interface.
    
    Read-only operations can use session directly without factory.
    Write operations require get_deps() or build_agent_and_deps(),
    which proves the caller holds the lock.
    """
    
    def __init__(self, lock_reg: dict[str, asyncio.Lock], session: AsyncSession):
        """Initialize factory with shared lock registry and per-request session."""
        self._lock_reg = lock_reg
        self._session = session # TODO: session also lives and is passed around in deps. ref spaghetti?
    
    def _get_lock(self, agent_id: str) -> asyncio.Lock:
        """Get or create a lock for the given agent_id from the registry."""
        pass
    
    @asynccontextmanager
    async def get_deps(self, agent_id: str) -> AsyncIterator[AgentDeps]:
        """Async context manager that acquires the agent lock and yields AgentDeps.
        
        Lock-then-fetch: acquires lock BEFORE fetching from DB to prevent stale state.
        Releases lock on exit (normal or exception) via try/finally.
        """
        lock = self._get_lock(agent_id)
        await lock.acquire()

        try:
            agent_record = await self._session.get(AgentRecord, agent_id)
            if agent_record is None:
                raise ValueError
                        
            config = agent_record.agent_config
            deps = AgentDeps(self._session, agent_id, config)
            yield deps
        finally:
            lock.release()
            

    @asynccontextmanager
    async def build_agent_and_deps(self, agent_id: str) -> AsyncIterator[tuple[Agent, AgentDeps]]:
        """Async context manager that yields a configured (Agent, AgentDeps) tuple.
        
        Wraps get_deps and constructs the Pydantic AI Agent with correct model and tools.
        """
        async with self.get_deps(agent_id) as deps:
            model = get_model(deps.config.model_name)
            agent = Agent(model,
                          instructions=get_system_prompt,
                          deps_type=AgentDeps,
                          name=agent_record.name,
                          tools=get_tools_for_agent(deps.config.tool_names),
                          output_type=[str, DeferredToolRequests],
                          model_settings=AnthropicModelSettings(
                              anthropic_cache_instructions=True,
                              anthropic_cache_tool_definitions=True,
                              anthropic_cache_messages=True,
                          ))


def get_model(model_name: str) -> AnthropicModel:
    """Map a model name string to a Pydantic AI model instance."""
    pass
