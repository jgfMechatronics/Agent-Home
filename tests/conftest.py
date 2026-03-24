import pytest_asyncio
from sqlalchemy import event
from sqlalchemy.ext.asyncio import AsyncSession, create_async_engine, AsyncEngine

from db.models import AgentRecord, Base


@pytest_asyncio.fixture(scope="session")
async def engine():
    """Session-scoped async engine backed by SQLite in-memory database."""
    engine = create_async_engine("sqlite+aiosqlite:///:memory:")

    @event.listens_for(engine.sync_engine, "connect")
    def enable_foreign_keys(dbapi_conn, _):
        cursor = dbapi_conn.cursor()
        cursor.execute("PRAGMA foreign_keys=ON")
        cursor.close()

    async with engine.begin() as conn:
        await conn.run_sync(Base.metadata.create_all)

    yield engine
    await engine.dispose()


@pytest_asyncio.fixture
async def session(engine : AsyncEngine):
    """Function-scoped session. Each test runs in a savepoint that rolls back after."""
    async with engine.connect() as conn:
        async with conn.begin() as trans:
            async_session = AsyncSession(bind=conn, join_transaction_mode="create_savepoint")
            yield async_session
            await async_session.close()
            await trans.rollback()


@pytest_asyncio.fixture
async def sample_agent_record(session : AsyncSession):
    """A persisted AgentRecord for use in tests that require an existing agent."""
    agent = AgentRecord(
        name="test-agent",
        agent_config={
            "model_name": "claude-sonnet-4-20250514",
            "tool_names": ["memory_replace", "memory_insert"],
            "soft_limit": 10000,
        },
        system_instructions="You are a test agent.",
    )
    session.add(agent)
    await session.flush()
    return agent
