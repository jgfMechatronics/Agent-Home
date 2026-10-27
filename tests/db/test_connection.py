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


@pytest.mark.parametrize("make_engine", [create_sqlite_engine, create_readonly_sqlite_engine], ids=["rw", "ro"])
async def test_sqlite_engine_applies_pragmas(make_engine, tmp_path):
    """Both rw and ro engines apply WAL, foreign_keys, and busy_timeout on every connection."""
    db_path = str(tmp_path / "test.db")
    # Initialise the file so create_readonly_sqlite_engine can open it (harmless for rw)
    seed = create_sqlite_engine(db_path)
    async with seed.connect():
        pass
    await seed.dispose()
    engine = make_engine(db_path)
    try:
        async with engine.connect() as conn:
            assert (await conn.execute(text("PRAGMA journal_mode"))).scalar() == "wal"
            assert (await conn.execute(text("PRAGMA foreign_keys"))).scalar() == 1
            assert (await conn.execute(text("PRAGMA busy_timeout"))).scalar() == 7000
    finally:
        await engine.dispose()


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


async def test_get_session_bound_to_provided_engine(ro_engine):
    """Sessions are bound to the engine they were created from, not a global engine."""
    async with get_session(ro_engine) as session:
        assert session.bind is ro_engine


async def test_get_session_closes_on_context_exit(ro_engine):
    async with get_session(ro_engine) as session:
        await session.execute(text("SELECT 1"))  # triggers autobegin
        assert session.in_transaction()
    assert not session.in_transaction()


