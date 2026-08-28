from contextlib import asynccontextmanager
from typing import AsyncIterator

from sqlalchemy import event
from sqlalchemy.ext.asyncio import AsyncEngine, AsyncSession, create_async_engine
from sqlalchemy.pool import NullPool

from db.models import Base

_DB_BUSY_TIMEOUT_MS = 7000


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


def create_sqlite_engine(db_path: str) -> AsyncEngine:
    url = f"sqlite+aiosqlite:///{db_path}"
    engine = create_async_engine(url=url, poolclass=NullPool)
    event.listens_for(engine.sync_engine, "connect")(_configure_sqlite_conn)
    return engine


async def init_db(engine: AsyncEngine) -> None:
    # TODO: Do we need to pass checkfirst or whatever = True or otherwise check if db already init? This is init_db's responsibility
    async with engine.begin() as conn:
        await conn.run_sync(Base.metadata.create_all)


@asynccontextmanager
async def get_session(engine: AsyncEngine) -> AsyncIterator[AsyncSession]:
    session = AsyncSession(engine)
    try:
        yield session
        await session.commit()
    except Exception:
        await session.rollback()
        raise
    finally:
        await session.close()
