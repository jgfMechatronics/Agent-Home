from contextlib import asynccontextmanager
from typing import AsyncIterator, NewType

from sqlalchemy import event
from sqlalchemy.ext.asyncio import AsyncEngine, AsyncSession, create_async_engine
from sqlalchemy.pool import NullPool

from db.models import Base

ReadOnlyEngine = NewType("ReadOnlyEngine", AsyncEngine)
"""Nominal type for an engine whose connections are read-only. Created only by
create_readonly_sqlite_engine(); lets static analysis catch rw/ro wiring mistakes."""

ReadOnlySession = NewType("ReadOnlySession", AsyncSession)
"""Type alias for sessions bound to a read-only engine. Type-level only — communicates
intent in function signatures and lets type checkers catch passing a rw session where
read-only is expected. Yielded by AgentDeps.read_scope() and get_session()."""

_DB_BUSY_TIMEOUT_MS = 5000


def _configure_sqlite_conn(dbapi_conn, _) -> None:
    """SQLAlchemy 'connect' event handler that applies standard SQLite PRAGMAs.

    Attach to any SQLite engine via:
        event.listens_for(engine.sync_engine, "connect")(_configure_sqlite_conn)
    """
    cursor = dbapi_conn.cursor()
    cursor.execute("PRAGMA foreign_keys=ON")
    cursor.execute(f"PRAGMA busy_timeout={_DB_BUSY_TIMEOUT_MS}")
    cursor.execute("PRAGMA journal_mode=WAL")
    cursor.close()


def _create_sqlite_engine_from_url(url: str) -> AsyncEngine:
    engine = create_async_engine(url=url, poolclass=NullPool)
    event.listens_for(engine.sync_engine, "connect")(_configure_sqlite_conn)
    return engine


def create_sqlite_engine(db_path: str) -> AsyncEngine:
    return _create_sqlite_engine_from_url(f"sqlite+aiosqlite:///{db_path}")


def create_readonly_sqlite_engine(db_path: str) -> ReadOnlyEngine:
    url = f"sqlite+aiosqlite:///file:{db_path}?mode=ro&uri=true"
    return ReadOnlyEngine(_create_sqlite_engine_from_url(url))


async def init_db(engine: AsyncEngine) -> None:
    # TODO: Do we need to pass checkfirst or whatever = True or otherwise check if db already init? This is init_db's responsibility
    async with engine.begin() as conn:
        await conn.run_sync(Base.metadata.create_all)


@asynccontextmanager
async def get_session(engine: ReadOnlyEngine) -> AsyncIterator[ReadOnlySession]:
    """Yield a read-only session. No commit/rollback — callers must not write."""
    session = AsyncSession(engine)
    try:
        yield ReadOnlySession(session)
    finally:
        await session.close()
