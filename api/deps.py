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
from pydantic_ai import Agent


def get_lock_reg(request: Request) -> dict[str, asyncio.Lock]:
    """FastAPI dependency: returns the app-wide agent lock registry from app.state.

    Stub — implementation will return request.app.state.agent_lock_reg.
    """
    raise NotImplementedError("get_lock_reg not implemented")


async def get_session_dep(request: Request) -> AsyncIterator[AsyncSession]:
    """FastAPI dependency: yields a session bound to app.state.engine.

    Stub — implementation will use get_session(request.app.state.engine).
    """
    raise NotImplementedError("get_session_dep not implemented")
    yield  # type: ignore — makes this a generator for type checking


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
    raise NotImplementedError("get_agent_and_deps not implemented")
    yield  # type: ignore — makes this a generator for type checking
