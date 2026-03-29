from contextlib import asynccontextmanager

from sqlalchemy import event
from sqlalchemy.ext.asyncio import AsyncEngine, create_async_engine, AsyncSession
from sqlalchemy.pool import NullPool


def create_sqlite_engine(db_path: str) -> AsyncEngine:
    url = f"sqlite+aiosqlite:///{db_path}"
    engine = create_async_engine(url = url, poolclass = NullPool)

    @event.listens_for(engine.sync_engine, "connect")
    def enable_foreign_keys(dbapi_conn, _):
        cursor = dbapi_conn.cursor()
        cursor.execute("PRAGMA foreign_keys=ON")
        cursor.close()
    
    return engine


async def init_db(engine: AsyncEngine) -> None:
    pass


@asynccontextmanager
async def get_session(engine: AsyncEngine) -> AsyncSession:
    pass

