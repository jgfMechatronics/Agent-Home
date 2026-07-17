import re
import uuid
from typing import Any

from datetime import datetime, timedelta, timezone

import pytest
from sqlalchemy.exc import IntegrityError, StatementError
from sqlalchemy.ext.asyncio import AsyncSession

from agent.types import AgentConfig
from conftest import SAMPLE_AGENT_CONFIG
from db.models import AgentRecord, MemoryBlockRecord, MessageRecord, utcnow

_UUID_RE = re.compile(r"^[0-9a-f]{8}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{12}$")


PARTIAL_MEMORY_BLOCK_FIELDS = {
    "description": "",
    "char_limit": 2000,
    "position": 0,
}

PARTIAL_MESSAGE_FIELDS = {
    "type": "ModelRequest",
    "content": "{}",
    "total_tokens": None,
}


@pytest.fixture
def memory_block_record(agent_record: AgentRecord) -> MemoryBlockRecord:
    """An unpersisted MemoryBlockRecord for use in tests that need an existing block."""
    return MemoryBlockRecord(agent_id=agent_record.id, label="persona", content="x", **PARTIAL_MEMORY_BLOCK_FIELDS)


@pytest.fixture
def message_record(agent_record: AgentRecord) -> MessageRecord:
    """An unpersisted MessageRecord for use in tests that need an existing message."""
    return MessageRecord(agent_id=agent_record.id, seq_id=0, timestamp=datetime(2026, 1, 1, 12, 0, 0), **PARTIAL_MESSAGE_FIELDS)


async def assert_round_trips(session: AsyncSession, record: Any, expected_fields: dict):
    """Add a record, flush, refresh from DB, and assert each expected field matches."""
    session.add(record)
    await session.flush()
    await session.refresh(record)
    for field, expected in expected_fields.items():
        actual = getattr(record, field)
        assert actual == expected, f"Field '{field}': expected {expected!r}, got {actual!r}"


async def assert_timestamps_auto_populated(session: AsyncSession, record: Any):
    """Verify created_at and updated_at are set automatically on creation and within 5 minutes of now (rough sanity check on accuracy)."""
    await session.refresh(record)
    now = utcnow()
    five_minutes = timedelta(minutes=5)
    assert record.created_at is not None, "created_at should be auto-populated on creation"
    assert record.updated_at is not None, "updated_at should be auto-populated on creation"
    assert abs(now - record.created_at) < five_minutes, f"created_at {record.created_at!r} is not within 5 minutes of now"
    assert abs(now - record.updated_at) < five_minutes, f"updated_at {record.updated_at!r} is not within 5 minutes of now"


async def assert_updated_at_bumps_on_modify(session: AsyncSession, record: Any, modify_fn):
    """Verify updated_at bumps when record is modified (tests onupdate=datetime.now behavior).

    onupdate is evaluated Python-side with microsecond precision, so no sleep needed.
    modify_fn: Callable that mutates the record (e.g., lambda r: setattr(r, "name", "new")).
    """
    session.add(record)  # no-op if already in session
    await session.flush()
    await session.refresh(record)

    original_updated_at = record.updated_at

    modify_fn(record)
    await session.commit()
    await session.refresh(record)

    assert record.updated_at > original_updated_at, (
        f"updated_at should bump on modify: was {original_updated_at}, still {record.updated_at}"
    )


# --- UUID auto-generation ---

@pytest.mark.parametrize("make_record", [
    lambda agent_id: AgentRecord(name="uuid-test", agent_config=SAMPLE_AGENT_CONFIG, system_instructions=""),
    lambda agent_id: MemoryBlockRecord(agent_id=agent_id, label="uuid-test", content="", **PARTIAL_MEMORY_BLOCK_FIELDS),
    lambda agent_id: MessageRecord(agent_id=agent_id, seq_id=0, timestamp=datetime.now(timezone.utc), **PARTIAL_MESSAGE_FIELDS),
])
async def test_id_auto_generated_as_uuid_string(session: AsyncSession, agent_record: AgentRecord, make_record: Any):
    """All models auto-generate a UUID string id on insert — not required at construction."""
    record = make_record(agent_record.id)
    assert record.id is None, "id should not be set before add and flush"
    session.add(record)
    await session.flush()
    await session.refresh(record)
    assert record.id is not None, "id should be populated after flush"
    assert isinstance(record.id, str), "id should be stored as a string"
    assert _UUID_RE.match(record.id), f"id should be a valid UUID string, got: {record.id!r}"


# --- AgentRecord ---

async def test_agent_record_stores_all_fields(session: AsyncSession):
    # Use naive datetimes: SQLAlchemy DateTime stores TEXT in SQLite and may strip
    # timezone info depending on the timezone= flag, making tz-aware comparisons brittle.
    fields = {
        "name": "my-agent",
        "agent_config": SAMPLE_AGENT_CONFIG,
        "system_instructions": "Be helpful.",
        "compiled_system_prompt": "<compiled>Be helpful.</compiled>",
        "sys_prompt_compiled_at": datetime(2026, 1, 1, 12, 0, 0),
        "context_window_start": 42,
    }
    await assert_round_trips(session, AgentRecord(**fields), fields)


async def test_agent_config_roundtrip_via_type_decorator(session: AsyncSession):
    """AgentConfig round-trips through the DB: stored as JSON, loaded back as an AgentConfig instance."""
    config = SAMPLE_AGENT_CONFIG
    agent = AgentRecord(name="roundtrip-test", agent_config=config.model_copy())
    session.add(agent)
    await session.flush()
    await session.refresh(agent)
    assert isinstance(agent.agent_config, AgentConfig)
    assert agent.agent_config == config


async def test_agent_config_type_decorator_rejects_wrong_type(session: AsyncSession):
    agent = AgentRecord(name="wrong_type_agent_config", agent_config="not a config")
    with pytest.raises(StatementError):
        session.add(agent)
        await session.flush()


async def test_agent_record_defaults(session: AsyncSession):
    """Verify default values on a freshly created agent (no optional fields provided)."""
    agent = AgentRecord(name="defaults-test", agent_config=SAMPLE_AGENT_CONFIG)
    session.add(agent)
    await session.flush()
    await session.refresh(agent)
    assert agent.context_window_start == 0
    assert agent.sys_prompt_compiled_at is None
    assert agent.system_instructions == ""


async def test_agent_record_timestamps_auto_populated(session: AsyncSession, agent_record: AgentRecord):
    """created_at and updated_at are automatically set when an agent is created."""
    await assert_timestamps_auto_populated(session, agent_record)


async def test_agent_record_updated_at_bumps_on_modify(session: AsyncSession, agent_record: AgentRecord):
    """AgentRecord.updated_at should bump when modified (onupdate=func.now())."""
    await assert_updated_at_bumps_on_modify(session, agent_record, lambda r: setattr(r, "name", "modified"))


# --- MemoryBlockRecord ---

async def test_memory_block_stores_all_fields(session: AsyncSession, memory_block_record: MemoryBlockRecord):
    fields = {
        "agent_id": memory_block_record.agent_id,
        "label": memory_block_record.label,
        "content": memory_block_record.content,
        "description": memory_block_record.description,
        "char_limit": memory_block_record.char_limit,
        "position": memory_block_record.position
    }
    await assert_round_trips(session, memory_block_record, fields)


async def test_memory_block_timestamps_auto_populated(session: AsyncSession, memory_block_record: MemoryBlockRecord):
    """created_at and updated_at are automatically set when a memory block is created."""
    session.add(memory_block_record)
    await session.flush()
    await assert_timestamps_auto_populated(session, memory_block_record)


async def test_memory_block_updated_at_bumps_on_modify(session: AsyncSession, memory_block_record: MemoryBlockRecord):
    """MemoryBlockRecord.updated_at should bump when modified (onupdate=func.now())."""
    await assert_updated_at_bumps_on_modify(session, memory_block_record, lambda r: setattr(r, "content", "modified"))


@pytest.mark.parametrize("overrides", [
    pytest.param({"label": "persona", "position": 1}, id="duplicate_label"),
    pytest.param({"label": "other",   "position": 0}, id="duplicate_position"),
])
async def test_memory_block_unique_constraints_per_agent(session: AsyncSession, memory_block_record: MemoryBlockRecord, overrides: dict):
    """Duplicate (agent_id, label) or (agent_id, position) under the same agent violates unique constraints."""
    session.add(memory_block_record)
    session.add(MemoryBlockRecord(agent_id=memory_block_record.agent_id, content="x", description="", char_limit=2000, **overrides))
    with pytest.raises(IntegrityError):
        await session.flush()


# --- MessageRecord ---

async def test_message_record_stores_all_fields(session: AsyncSession, message_record: MessageRecord):
    message_record.total_tokens = 150  # use a non-null value to verify integer persistence
    fields = {
        "agent_id": message_record.agent_id,
        "seq_id": message_record.seq_id,
        "type": message_record.type,
        "content": message_record.content,
        "total_tokens": message_record.total_tokens,
        "timestamp": message_record.timestamp,
    }
    await assert_round_trips(session, message_record, fields)


async def test_message_total_tokens_nullable(session: AsyncSession, message_record: MessageRecord):
    """total_tokens may be NULL — only set on the final response row that closes a run."""
    message_record.total_tokens = None
    await assert_round_trips(session, message_record, {"total_tokens": None})


async def test_message_content_stores_nested_json(session: AsyncSession, message_record: MessageRecord):
    """Message content preserves nested JSON structures including unicode escapes."""
    message_record.content = '{"parts": [{"type": "user-prompt", "content": "hello \\u2603"}]}'
    await assert_round_trips(session, message_record, {"content": message_record.content})


# --- FK enforcement ---

@pytest.mark.parametrize("fixture_name", ["memory_block_record", "message_record"])
async def test_fk_enforced(
    session: AsyncSession,
    agent_record: AgentRecord,  # pre-resolved so getfixturevalue can resolve sync fixtures that depend on it without needing to spin up a new Runner inside the running event loop
    request: pytest.FixtureRequest,
    fixture_name: str,
):
    """MemoryBlockRecord and MessageRecord cannot reference a nonexistent agent."""
    record = request.getfixturevalue(fixture_name)
    record.agent_id = str(uuid.uuid4())
    session.add(record)
    with pytest.raises(IntegrityError):
        await session.flush()


# --- Cascade delete / isolation ---

@pytest.mark.parametrize("delete_fixture, survive_fixture", [
    ("memory_block_record", "message_record"),
    ("message_record", "memory_block_record"),
])
async def test_child_delete_is_isolated(
    session: AsyncSession,
    agent_record: AgentRecord,  # pre-resolved so getfixturevalue can resolve sync fixtures that depend on it without needing to spin up a new Runner inside the running event loop
    memory_block_record: MemoryBlockRecord,
    message_record: MessageRecord,
    request: pytest.FixtureRequest,
    delete_fixture: str,
    survive_fixture: str,
):
    """Deleting a child record removes only that record — agent and sibling survive."""
    session.add(memory_block_record)
    session.add(message_record)
    await session.flush()

    to_delete = request.getfixturevalue(delete_fixture)
    survivor = request.getfixturevalue(survive_fixture)

    await session.delete(to_delete)
    await session.flush()

    assert await session.get(AgentRecord, agent_record.id) is not None
    assert await session.get(type(survivor), survivor.id) is not None


async def test_cascade_delete_removes_blocks_and_messages(session: AsyncSession, agent_record: AgentRecord, memory_block_record: MemoryBlockRecord, message_record: MessageRecord):
    """Deleting an agent cascades to all associated blocks and messages."""
    session.add(memory_block_record)
    session.add(message_record)
    await session.flush()

    block_id, message_id = memory_block_record.id, message_record.id

    await session.delete(agent_record)
    await session.flush()

    assert await session.get(MemoryBlockRecord, block_id) is None
    assert await session.get(MessageRecord, message_id) is None
