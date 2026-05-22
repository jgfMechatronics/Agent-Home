"""
Documents SQLAlchemy async ORM expiry behavior under our session model.

The scenario: a tool call commits the session (which expires _agent_record attributes),
then code reads a mutable deps property that goes through the ORM directly.

CONCLUSIONS (confirmed May 8, 2026):
- READ of an expired column attribute in async context → MissingGreenlet. The greenlet
  bridge does NOT save you in regular async code between awaits. It only activates inside
  SQLAlchemy's own await session.execute() machinery.
- WRITE to an expired attribute → works fine. No lazy load is triggered; SQLAlchemy just
  records the new value as a pending change. A subsequent read of that same attribute also
  works (writing clears the expired state).

IMPLICATION: After any session.commit(), any code that reads mutable ORM attributes must
explicitly refresh first. Fix: mutating tools (which hold deps/lock) call
`await session.refresh(deps._agent_record)` after commit.
"""

import pytest

import pytest
import pytest_asyncio

pytestmark = pytest.mark.skip(reason="Documentation of ORM expiry behavior — conclusions in module docstring.")
from sqlalchemy.ext.asyncio import AsyncSession, create_async_engine
from sqlalchemy.pool import StaticPool
from sqlalchemy import event

from agent.types import AgentConfig, AgentDeps
from db.models import AgentRecord, Base

SAMPLE_CONFIG = AgentConfig(
    model_name="claude-sonnet-4-20250514",
    tool_names=["memory_replace"],
    soft_compaction_limit=10000,
)


@pytest_asyncio.fixture
async def async_session():
    engine = create_async_engine("sqlite+aiosqlite:///:memory:", poolclass=StaticPool)

    @event.listens_for(engine.sync_engine, "connect")
    def enable_fk(dbapi_conn, _):
        cursor = dbapi_conn.cursor()
        cursor.execute("PRAGMA foreign_keys=ON")
        cursor.close()

    async with engine.begin() as conn:
        await conn.run_sync(Base.metadata.create_all)

    async with AsyncSession(engine) as session:
        yield session

    await engine.dispose()


@pytest_asyncio.fixture
async def deps_with_mutable_data(async_session: AsyncSession) -> AgentDeps:
    """AgentRecord with compiled_system_prompt set, so we have a mutable field to read."""
    agent = AgentRecord(
        name="test-agent",
        agent_config=SAMPLE_CONFIG,
        system_instructions="Be helpful.",
        compiled_system_prompt="<system>Be helpful.</system>",
    )
    async_session.add(agent)
    await async_session.flush()
    return AgentDeps(session=async_session, agent_record=agent)


async def test_mutable_property_after_commit(deps_with_mutable_data: AgentDeps, async_session: AsyncSession):
    """
    After session.commit() expires _agent_record, can we still read mutable deps properties
    in async context? If yes: SQLAlchemy's greenlet bridge handles it transparently.
    If MissingGreenlet: we need explicit refresh after commits.
    """
    deps = deps_with_mutable_data

    # Sanity check — readable before commit
    assert deps.compiled_system_prompt == "<system>Be helpful.</system>"

    # Simulate a tool call that commits (the behavior we're moving to with commit=True)
    await async_session.commit()

    # _agent_record is now expired. Can we read the mutable property?
    # This is the question: does async context save us, or do we get MissingGreenlet?
    result = deps.compiled_system_prompt

    assert result == "<system>Be helpful.</system>"


async def test_mutable_property_after_commit_then_write(deps_with_mutable_data: AgentDeps, async_session: AsyncSession):
    """
    After commit (expiry), can we also WRITE to a mutable property and read it back?
    Tests the write-through setter path too.
    """
    deps = deps_with_mutable_data

    await async_session.commit()

    # Write through the setter (goes to expired ORM attr — does this work?)
    deps.compiled_system_prompt = "<system>Updated.</system>"
    await async_session.flush()

    # Read it back
    result = deps.compiled_system_prompt
    assert result == "<system>Updated.</system>"
