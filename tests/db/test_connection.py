import asyncio

import pytest
import pytest_asyncio
from sqlalchemy import text
from sqlalchemy.ext.asyncio import AsyncEngine, AsyncSession

from db.connection import create_db_engine, get_session, init_db
from db.models import AgentRecord


@pytest_asyncio.fixture
async def raw_engine(tmp_path):
    """Engine created via create_db_engine, no schema initialised. File-based SQLite
    so that multiple connections share the same database (unlike :memory:)."""
    engine = create_db_engine(f"sqlite+aiosqlite:///{tmp_path}/test.db")
    yield engine
    await engine.dispose()


@pytest_asyncio.fixture
async def initialized_engine(raw_engine):
    """raw_engine with the full schema created via init_db."""
    await init_db(raw_engine)
    yield raw_engine


async def test_create_db_engine_returns_async_engine(raw_engine):
    assert isinstance(raw_engine, AsyncEngine)


async def test_init_db_creates_tables(raw_engine):
    """Tables don't exist before init_db is called; they do after."""
    async with raw_engine.connect() as conn:
        with pytest.raises(Exception, match="no such table"):
            await conn.execute(text("SELECT 1 FROM agent"))

    await init_db(raw_engine)

    async with raw_engine.connect() as conn:
        await conn.execute(text("SELECT 1 FROM agent"))  # no error = table exists


async def test_get_session_yields_async_session(initialized_engine):
    async with get_session(initialized_engine) as session:
        assert isinstance(session, AsyncSession)


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
