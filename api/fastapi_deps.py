"""FastAPI dependencies — Section 4.3.

All FastAPI dependency functions live here. Routes import from this module;
app.py and db/connection.py remain free of FastAPI route concerns.
"""
import asyncio
from typing import AsyncIterator

from fastapi import Depends, HTTPException, Request
from sqlalchemy.ext.asyncio import AsyncSession

# Pre-placed for get_agent_and_deps implementation — used when James implements this module
from agent.factory import AgentFactory, AgentLockedError, AgentNotFoundError
from agent.types import AgentDeps
from db.connection import get_session
from pydantic_ai import Agent


def get_lock_reg(request: Request) -> dict[str, asyncio.Lock]:
    """FastAPI dependency: returns the app-wide agent lock registry from app.state."""
    return request.app.state.agent_lock_reg


async def get_session_dep(request: Request) -> AsyncIterator[AsyncSession]:
    """FastAPI dependency: yields a session bound to app.state.engine."""
    async with get_session(request.app.state.engine) as session:
        yield session


async def get_agent_and_deps(
    agent_id: str,
    session: AsyncSession = Depends(get_session_dep),
    lock_reg: dict[str, asyncio.Lock] = Depends(get_lock_reg),
) -> AsyncIterator[tuple[Agent, AgentDeps]]:
    """FastAPI yield dependency: acquires agent lock, yields (Agent, AgentDeps).

    Translates domain exceptions to HTTP errors:
      AgentNotFoundError → 404
      AgentLockedError   → 503

    Lock is released on exit regardless of outcome (normal, exception, or client disconnect).
    """

    try:
        factory = AgentFactory(lock_reg, session)
        async with factory.build_agent_and_deps(agent_id) as (agent, deps):
            yield (agent, deps)
    except AgentNotFoundError as e:
        raise HTTPException(404, detail=str(e))
    except AgentLockedError as e:
        raise HTTPException(503, detail=str(e))
    except Exception as e:
        raise HTTPException(500, detail=f"Unexpected error during agent + dependency construction: {e!r}")
