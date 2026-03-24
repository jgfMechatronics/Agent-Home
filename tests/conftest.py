import pytest_asyncio
from sqlalchemy import event
from sqlalchemy.ext.asyncio import AsyncSession, create_async_engine
from sqlalchemy.pool import StaticPool

from db.models import AgentRecord, Base

SAMPLE_AGENT_CONFIG = {
    "model_name": "claude-sonnet-4-20250514",
    "tool_names": ["memory_replace", "memory_insert"],
    "soft_limit": 10000,
}


@pytest_asyncio.fixture
async def session():
    """Fresh in-memory SQLite database per test. StaticPool ensures create_all and
    the session share the same connection; engine disposal destroys the DB."""
    engine = create_async_engine(
        "sqlite+aiosqlite:///:memory:",
        poolclass=StaticPool,
    )

    @event.listens_for(engine.sync_engine, "connect")
    def enable_foreign_keys(dbapi_conn, _):
        cursor = dbapi_conn.cursor()
        cursor.execute("PRAGMA foreign_keys=ON")
        cursor.close()

    try:
        async with engine.begin() as conn:
            await conn.run_sync(Base.metadata.create_all)
        
        async with AsyncSession(engine) as async_session:
            yield async_session
    finally:
        await engine.dispose()


@pytest_asyncio.fixture
async def sample_agent_record(session: AsyncSession) -> AgentRecord:
    """A persisted AgentRecord for use in tests that require an existing agent."""
    agent = AgentRecord(
        name="test-agent",
        agent_config=SAMPLE_AGENT_CONFIG,
        system_instructions="You are a test agent.",
    )
    session.add(agent)
    await session.flush()
    return agent
