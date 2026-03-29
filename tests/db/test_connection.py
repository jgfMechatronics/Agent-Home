import asyncio

import pytest
import pytest_asyncio
from sqlalchemy import text
from sqlalchemy.ext.asyncio import AsyncEngine, AsyncSession
from sqlalchemy.pool import NullPool

from db.connection import create_sqlite_engine, get_session, init_db
from db.models import AgentRecord, MessageRecord, MemoryBlockRecord

TABLE_NAMES = [AgentRecord.__tablename__, MessageRecord.__tablename__, MemoryBlockRecord.__tablename__]


@pytest_asyncio.fixture
async def raw_engine(tmp_path):
    """Engine created via create_sqlite_engine, no schema initialised. File-based SQLite
    so that multiple connections share the same database (unlike :memory:)."""
    db_path = str(tmp_path / "test.db")
    engine = create_sqlite_engine(db_path)
    assert str(engine.url) == f"sqlite+aiosqlite:///{db_path}"
    yield engine
    await engine.dispose()


@pytest_asyncio.fixture
async def initialized_engine(raw_engine):
    """raw_engine with the full schema created via init_db."""
    await init_db(raw_engine)
    yield raw_engine


# --- create_sqlite_engine ---

async def test_create_sqlite_engine_returns_async_engine(raw_engine):
    assert isinstance(raw_engine, AsyncEngine)


async def test_create_sqlite_engine_uses_null_pool(raw_engine):
    """NullPool is correct for async SQLite — avoids file-locking issues from connection reuse."""
    assert isinstance(raw_engine.pool, NullPool)


async def test_create_sqlite_engine_enables_foreign_keys(raw_engine):
    """create_sqlite_engine must set PRAGMA foreign_keys=ON — SQLite disables it by default."""
    async with raw_engine.connect() as conn:
        result = await conn.execute(text("PRAGMA foreign_keys"))
        assert result.scalar() == 1


# --- init_db ---

async def test_init_db_creates_tables(raw_engine):
    """Tables don't exist before init_db is called; they do after."""
    async with raw_engine.connect() as conn:
        for table_name in TABLE_NAMES:
            with pytest.raises(Exception, match="no such table"):
                await conn.execute(text("SELECT 1 FROM " + table_name))

    await init_db(raw_engine)

    async with raw_engine.connect() as conn:
        for table_name in TABLE_NAMES:
            await conn.execute(text("SELECT 1 FROM " + table_name))  # no error = table exists


# --- get_session ---

async def test_get_session_yields_async_session(initialized_engine):
    async with get_session(initialized_engine) as session:
        assert isinstance(session, AsyncSession)


async def test_get_session_bound_to_provided_engine(tmp_path):
    """Sessions are bound to the engine they were created from, not a global engine."""
    engine_a = create_sqlite_engine(str(tmp_path / "a.db"))
    engine_b = create_sqlite_engine(str(tmp_path / "b.db"))
    try:
        await init_db(engine_a)
        await init_db(engine_b)

        async with get_session(engine_a) as session:
            session.add(AgentRecord(name="agent-a", agent_config={}))
            await session.commit()

        async with get_session(engine_b) as session:
            result = await session.execute(text("SELECT COUNT(*) FROM agent"))
            assert result.scalar() == 0
    finally:
        await engine_a.dispose()
        await engine_b.dispose()


async def test_get_session_closes_on_context_exit(initialized_engine : AsyncEngine):
    checkedOutCounter = initialized_engine.sync_engine.pool.checkedout

    async with get_session(initialized_engine) as session:
        assert checkedOutCounter() == 1
    assert checkedOutCounter() == 0


# --- concurrent sessions ---

async def test_concurrent_sessions_function_independently(initialized_engine):
    """Two sessions can write and commit concurrently without deadlock or error.
    Catches misconfigured pool settings (e.g. StaticPool)."""
    async def write_agent(name: str):
        async with get_session(initialized_engine) as session:
            session.add(AgentRecord(name=name, agent_config={}))
            await session.commit()

    await asyncio.gather(write_agent("agent-1"), write_agent("agent-2"))

    async with get_session(initialized_engine) as session:
        result = await session.execute(text("SELECT COUNT(*) FROM agent"))
        assert result.scalar() == 2
