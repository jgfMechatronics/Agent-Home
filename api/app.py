"""FastAPI application and lifespan — Section 4.3 (stub)."""
from typing import AsyncIterator

from fastapi import Request
from sqlalchemy.ext.asyncio import AsyncSession


async def get_session_dep(request: Request) -> AsyncIterator[AsyncSession]:
    """FastAPI dependency: yields session from app.state.engine.
    
    Stub — implementation will use get_session(request.app.state.engine).
    """
    raise NotImplementedError("get_session_dep not implemented")
    yield  # type: ignore — makes this a generator for type checking
