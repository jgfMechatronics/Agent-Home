from contextlib import asynccontextmanager
from dataclasses import dataclass
from typing import AsyncIterator, NewType

from sqlalchemy import event
from sqlalchemy.ext.asyncio import AsyncEngine, AsyncSession, create_async_engine
from sqlalchemy.pool import NullPool
from starlette.requests import Request

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
    # WAL is persistent in the DB file header. On a mode=ro connection this PRAGMA is a no-op if
    # the file is already WAL, but raises "attempt to write a readonly database" if it isn't. Fine
    # in practice: the app's lifespan runs init_db on the rw engine (first connection, sets WAL)
    # before anything touches the ro engine, and the utils runners read the existing (WAL) prod DB.
    # A ro engine pointed at a never-WAL'd file (e.g. a foreign SQLite file) will fail here.
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


@dataclass(frozen=True)
class Engines:
    """Typed container for rw/ro engine pair. Frozen to prevent accidental mutation.

    Runtime check in __post_init__ verifies ro engine is actually read-only.
    This catches wiring mistakes that static analysis can't (NewType is erased at runtime).
    """

    rw: AsyncEngine
    ro: ReadOnlyEngine

    def __post_init__(self) -> None:
        # ro engine URL should have mode=ro in query string
        # URL is like: sqlite+aiosqlite:///file:/path/to/db?mode=ro&uri=true
        url_str = str(self.ro.url)
        if "mode=ro" not in url_str:
            raise RuntimeError(
                f"Engines.ro is not a mode=ro engine. URL: {url_str}"
            )


def get_engines(request: Request) -> Engines:
    """Single typed accessor for engine pair. All engine access should go through this.

    Using a dedicated accessor (vs direct app.state access) ensures type checking works:
    app.state.__getattr__ returns Any, bypassing static analysis.
    """
    return request.app.state._engines


@asynccontextmanager
async def get_session(engine: ReadOnlyEngine) -> AsyncIterator[ReadOnlySession]:
    """Yield a read-only session. No commit/rollback — callers must not write."""
    async with AsyncSession(engine) as session:
        yield ReadOnlySession(session)
