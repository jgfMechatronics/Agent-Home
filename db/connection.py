from contextlib import asynccontextmanager
from typing import AsyncIterator

from fastapi import Request
from sqlalchemy import event
from sqlalchemy.ext.asyncio import AsyncEngine, AsyncSession, create_async_engine
from sqlalchemy.pool import NullPool

from db.models import Base


def create_sqlite_engine(db_path: str) -> AsyncEngine:
    url = f"sqlite+aiosqlite:///{db_path}"
    engine = create_async_engine(url=url, poolclass=NullPool)

    @event.listens_for(engine.sync_engine, "connect")
    def enable_foreign_keys(dbapi_conn, _):
        cursor = dbapi_conn.cursor()
        cursor.execute("PRAGMA foreign_keys=ON")
        cursor.close()

    return engine


async def init_db(engine: AsyncEngine) -> None:
    async with engine.begin() as conn:
        await conn.run_sync(Base.metadata.create_all)


@asynccontextmanager
async def get_session(engine: AsyncEngine) -> AsyncIterator[AsyncSession]:
    session = AsyncSession(engine)
    try:
        yield session
    finally:
        await session.close()


async def get_session_dep(request: Request) -> AsyncIterator[AsyncSession]:
    """FastAPI dependency: yields session from app.state.engine.

    Stub — implementation will use get_session(request.app.state.engine).
    """
    raise NotImplementedError("get_session_dep not implemented")
    yield  # type: ignore — makes this a generator for type checking

