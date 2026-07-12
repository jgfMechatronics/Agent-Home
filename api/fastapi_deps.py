"""FastAPI dependencies — Section 4.3.

All FastAPI dependency functions live here. Routes import from this module;
app.py and db/connection.py remain free of FastAPI route concerns.

Domain exceptions (AgentNotFoundError, AgentLockedError) are translated to HTTP responses
by app-level exception handlers registered in api/app.py — not caught here.
"""
from typing import AsyncIterator

from fastapi import Depends, Request
from sqlalchemy.ext.asyncio import AsyncSession

from agent.factory import AgentFactory, LOCK_TIMEOUT_FAST, LOCK_TIMEOUT_SECONDS
from agent.types import AgentAppState, AgentDeps
from db.connection import get_session
from pydantic_ai import Agent


def get_agent_app_state_reg(request: Request) -> dict[str, AgentAppState]:
    """FastAPI dependency: returns the app-wide agent state registry from app.state."""
    return request.app.state.agent_app_state_reg


async def get_session_dep(request: Request) -> AsyncIterator[AsyncSession]:
    """FastAPI dependency: yields a session bound to app.state.engine."""
    async with get_session(request.app.state.engine) as session:
        yield session


async def get_agent_and_deps(
    agent_id: str,
    session: AsyncSession = Depends(get_session_dep),
    agent_app_state_reg: dict[str, AgentAppState] = Depends(get_agent_app_state_reg),
) -> AsyncIterator[tuple[Agent, AgentDeps]]:
    """FastAPI yield dependency: acquires agent lock, yields (Agent, AgentDeps).

    Domain exceptions propagate to app-level handlers in api/app.py:
      AgentNotFoundError → 404
      AgentLockedError   → 423

    Lock is released on exit regardless of outcome (normal, exception, or client disconnect).
    """
    factory = AgentFactory(agent_id, agent_app_state_reg, session)
    async with factory.build_agent_and_deps() as (agent, deps):
        yield (agent, deps)


def get_deps_dep(timeout: float = LOCK_TIMEOUT_SECONDS):
    """Factory for a FastAPI yield dependency that acquires the agent lock and yields AgentDeps.

    Call with a timeout to get a dep function with that timeout baked in:
        Depends(get_deps_dep())                          # default 60s timeout
        Depends(get_deps_dep(timeout=LOCK_TIMEOUT_FAST)) # fast 2s timeout for write routes

    Has the best function name in the entire codebase.

    Use for write routes that need the lock but don't need a configured Agent instance.
    Domain exceptions propagate to app-level handlers in api/app.py:
      AgentNotFoundError → 404
      AgentLockedError   → 423

    Lock is released on exit regardless of outcome (normal, exception, or client disconnect).
    """
    async def _dep(
        agent_id: str,
        session: AsyncSession = Depends(get_session_dep),
        agent_app_state_reg: dict[str, AgentAppState] = Depends(get_agent_app_state_reg),
    ) -> AsyncIterator[AgentDeps]:
        factory = AgentFactory(agent_id, agent_app_state_reg, session)
        async with factory.build_deps(timeout=timeout) as deps:
            yield deps

    return _dep


# Module-level dep instances — use these in routes and as dependency_overrides keys in tests.
# Routes that need the default 60s timeout use deps_dep; write routes that should
# yield quickly (avoiding long waits on a locked agent) use deps_dep_fast.
deps_dep = get_deps_dep()
deps_dep_fast = get_deps_dep(timeout=LOCK_TIMEOUT_FAST)
