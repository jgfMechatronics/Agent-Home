import pytest_asyncio
from sqlalchemy import event
from sqlalchemy.ext.asyncio import AsyncSession, create_async_engine
from sqlalchemy.pool import StaticPool

from agent.types import AgentConfig, AgentDeps
from db.models import AgentRecord, Base, MemoryBlockRecord



SAMPLE_AGENT_CONFIG = {
    "model_name": "claude-sonnet-4-20250514",
    "tool_names": ["memory_replace", "memory_insert"],
    "soft_compaction_limit": 10000,
}

def make_deps(session: AsyncSession, agent: AgentRecord) -> AgentDeps:
    """Construct AgentDeps from a session and agent record with default config."""
    return AgentDeps(session=session, agent_id=agent.id, config=AgentConfig(**SAMPLE_AGENT_CONFIG))



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
async def agent_record(session: AsyncSession) -> AgentRecord:
    """
    A persisted AgentRecord for use in tests that require an existing agent.
    The underlying session can be used by dependents by seperately requesting that fixture. Pytest will
    cache it resulting in session pointing to the temp DB which contains the persisted agent
    """
    agent = AgentRecord(
        name="test-agent",
        agent_config=SAMPLE_AGENT_CONFIG,
        system_instructions="You are a test agent.",
    )
    session.add(agent)
    await session.flush()
    return agent


@pytest_asyncio.fixture
async def agent_with_blocks(session: AsyncSession):
    """
    Agent with system_instructions and three memory blocks in known positions.
    
    Blocks have descriptions (for XML formatting tests) and varied char_limits
    (for limit enforcement tests). Created out of position order to verify sorting.
    
    Returns dict with agent and blocks for test access.
    """
    agent = AgentRecord(
        name="agent-with-blocks",
        agent_config=SAMPLE_AGENT_CONFIG,
        system_instructions="You are a helpful assistant.",
    )
    session.add(agent)
    await session.flush()
    
    block_persona = MemoryBlockRecord(
        agent_id=agent.id,
        label="persona",
        description="The agent's identity",
        content="I am a test agent.",
        char_limit=1000,
        position=0,
    )
    block_human = MemoryBlockRecord(
        agent_id=agent.id,
        label="human",
        description="Information about the user",
        content="The user's name is Alice.",
        char_limit=500,
        position=1,
    )
    block_notes = MemoryBlockRecord(
        agent_id=agent.id,
        label="notes",
        description="Scratch space",
        content="Remember to be helpful.",
        char_limit=2000,
        position=2,
    )
    
    # Insert out of position order to verify queries sort by position, not insertion order
    session.add_all([block_human, block_notes, block_persona])
    blocks = [block_persona, block_human, block_notes]  # position order for test assertions
    await session.flush()
    
    return {"agent": agent, "blocks": blocks}
