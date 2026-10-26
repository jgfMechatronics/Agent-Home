import asyncio

import pytest
import pytest_asyncio
from sqlalchemy import text
from sqlalchemy.ext.asyncio import AsyncEngine, AsyncSession
from sqlalchemy.pool import NullPool

from conftest import SAMPLE_AGENT_CONFIG
from db.connection import create_readonly_sqlite_engine, create_sqlite_engine, get_session, init_db
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


# --- create_readonly_sqlite_engine ---

class TestCreateReadonlySqliteEngine:
    """Read-only engines use SQLite mode=ro: reads succeed and writes are rejected."""

    @pytest_asyncio.fixture(autouse=True)
    async def setup(self, tmp_path):
        """Create and initialise a writable DB, then open a readonly engine against it."""
        db_path = str(tmp_path / "test.db")
        writable = create_sqlite_engine(db_path)
        await init_db(writable)
        await writable.dispose()

        self.engine = create_readonly_sqlite_engine(db_path)
        yield
        await self.engine.dispose()

    async def test_uses_readonly_sqlite_uri(self):
        assert self.engine.url.query["mode"] == "ro"
        assert self.engine.url.query["uri"] == "true"

    async def test_can_read(self):
        async with self.engine.connect() as conn:
            result = await conn.execute(text("SELECT 1"))
            assert result.scalar() == 1

    async def test_cannot_write(self):
        with pytest.raises(Exception, match="attempt to write a readonly database"):
            async with self.engine.begin() as conn:
                await conn.execute(text("CREATE TABLE _readonly_test (id INTEGER)"))


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


# --- get_session (read-only) ---

@pytest_asyncio.fixture
async def ro_engine(tmp_path):
    """Read-only engine with schema initialised. File-based SQLite with mode=ro."""
    db_path = str(tmp_path / "test.db")
    # Create and init with a rw engine first (readonly can't create schema)
    rw = create_sqlite_engine(db_path)
    await init_db(rw)
    await rw.dispose()
    engine = create_readonly_sqlite_engine(db_path)
    yield engine
    await engine.dispose()


async def test_get_session_yields_async_session(ro_engine):
    async with get_session(ro_engine) as session:
        assert isinstance(session, AsyncSession)


async def test_get_session_bound_to_provided_engine(tmp_path):
    """Sessions are bound to the engine they were created from, not a global engine."""
    # Init two DBs with different data
    for name, agent_name in [("a.db", "agent-a"), ("b.db", None)]:
        rw = create_sqlite_engine(str(tmp_path / name))
        await init_db(rw)
        if agent_name:
            async with rw.begin() as conn:
                await conn.execute(
                    AgentRecord.__table__.insert().values(
                        id="test-id", name=agent_name, agent_config=SAMPLE_AGENT_CONFIG,
                        system_instructions="test", compiled_system_prompt="",
                    )
                )
        await rw.dispose()

    ro_a = create_readonly_sqlite_engine(str(tmp_path / "a.db"))
    ro_b = create_readonly_sqlite_engine(str(tmp_path / "b.db"))
    try:
        async with get_session(ro_a) as session:
            result = await session.execute(text("SELECT COUNT(*) FROM agent"))
            assert result.scalar() == 1

        async with get_session(ro_b) as session:
            result = await session.execute(text("SELECT COUNT(*) FROM agent"))
            assert result.scalar() == 0
    finally:
        await ro_a.dispose()
        await ro_b.dispose()


async def test_get_session_closes_on_context_exit(ro_engine):
    async with get_session(ro_engine) as session:
        await session.execute(text("SELECT 1"))  # triggers autobegin
        assert session.in_transaction()
    assert not session.in_transaction()


# --- concurrent sessions ---

async def test_concurrent_sessions_function_independently(ro_engine):
    """Two read-only sessions can query concurrently without deadlock or error."""
    async def read_count():
        async with get_session(ro_engine) as session:
            result = await session.execute(text("SELECT COUNT(*) FROM agent"))
            return result.scalar()

    results = await asyncio.gather(read_count(), read_count())
    assert all(r == 0 for r in results)
