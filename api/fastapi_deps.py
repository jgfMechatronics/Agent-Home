"""FastAPI dependencies — Section 4.3.

All FastAPI dependency functions live here. Routes import from this module;
app.py and db/connection.py remain free of FastAPI route concerns.

Domain exceptions (AgentNotFoundError, AgentLockedError) are translated to HTTP responses
by app-level exception handlers registered in api/app.py — not caught here.
"""
from typing import AsyncIterator

from fastapi import Depends, Request
from sqlalchemy.ext.asyncio import AsyncSession

from agent.factory import AgentFactory
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
      AgentLockedError   → 503

    Lock is released on exit regardless of outcome (normal, exception, or client disconnect).
    """
    factory = AgentFactory(agent_id, agent_app_state_reg, session)
    async with factory.build_agent_and_deps() as (agent, deps):
        yield (agent, deps)


async def get_deps_dep(
    agent_id: str,
    session: AsyncSession = Depends(get_session_dep),
    agent_app_state_reg: dict[str, AgentAppState] = Depends(get_agent_app_state_reg),
) -> AsyncIterator[AgentDeps]:
    """
    FastAPI yield dependency: acquires agent lock, yields AgentDeps (without building Agent).

    Use for write routes that need the lock but don't need a configured Agent instance.
    Domain exceptions propagate to app-level handlers in api/app.py:
      AgentNotFoundError → 404
      AgentLockedError   → 503

    Lock is released on exit regardless of outcome (normal, exception, or client disconnect).
    
    Has the best function name in the entire codebase
    """
    factory = AgentFactory(agent_id, agent_app_state_reg, session)
    async with factory.build_deps() as deps:
        yield deps
