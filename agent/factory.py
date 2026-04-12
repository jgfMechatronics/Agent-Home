"""
Agent factory and dependency management — Section 3.1

Provides context managers for acquiring agent locks, loading deps, and building agents.
"""
import asyncio
from contextlib import asynccontextmanager
from typing import AsyncIterator

from pydantic_ai import Agent
from pydantic_ai.models.anthropic import AnthropicModel
from sqlalchemy.ext.asyncio import AsyncSession

from agent.types import AgentDeps


def get_agent_lock(agent_id: str, lock_reg: dict[str, asyncio.Lock]) -> asyncio.Lock:
    """Get or create a lock for the given agent_id from the registry."""
    pass


def get_model(model_name: str) -> AnthropicModel:
    """Map a model name string to a Pydantic AI model instance."""
    pass


@asynccontextmanager
async def get_deps(
    session: AsyncSession,
    agent_id: str,
    lock_reg: dict[str, asyncio.Lock],
) -> AsyncIterator[AgentDeps]:
    """Async context manager that acquires the agent lock and yields AgentDeps.
    
    Lock-then-fetch: acquires lock BEFORE fetching from DB to prevent stale state.
    Releases lock on exit (normal or exception) via try/finally.
    """
    pass
    yield  # type: ignore


@asynccontextmanager
async def build_agent_and_deps(
    session: AsyncSession,
    agent_id: str,
    lock_reg: dict[str, asyncio.Lock],
) -> AsyncIterator[tuple[Agent, AgentDeps]]:
    """Async context manager that yields a configured (Agent, AgentDeps) tuple.
    
    Wraps get_deps and constructs the Pydantic AI Agent with correct model and tools.
    """
    pass
    yield  # type: ignore
