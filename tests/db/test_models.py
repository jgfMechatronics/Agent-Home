import re
import uuid
from typing import Any

from datetime import datetime, timedelta, timezone

import pytest
from sqlalchemy.exc import IntegrityError
from sqlalchemy.ext.asyncio import AsyncSession

from conftest import SAMPLE_AGENT_CONFIG
from db.models import AgentRecord, MemoryBlockRecord, MessageRecord

_UUID_RE = re.compile(r"^[0-9a-f]{8}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{12}$")


PARTIAL_MEMORY_BLOCK_FIELDS = {
    "description": "",
    "char_limit": 2000,
    "position": 0,
}

PARTIAL_MESSAGE_FIELDS = {
    "type": "ModelRequest",
    "content": "{}",
    "input_tokens": None,
}


@pytest.fixture
def memory_block_record(agent_record: AgentRecord) -> MemoryBlockRecord:
    """An unpersisted MemoryBlockRecord for use in tests that need an existing block."""
    return MemoryBlockRecord(agent_id=agent_record.id, label="persona", content="x", **PARTIAL_MEMORY_BLOCK_FIELDS)


@pytest.fixture
def message_record(agent_record: AgentRecord) -> MessageRecord:
    """An unpersisted MessageRecord for use in tests that need an existing message."""
    return MessageRecord(agent_id=agent_record.id, timestamp=datetime(2026, 1, 1, 12, 0, 0), **PARTIAL_MESSAGE_FIELDS)


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
    now = datetime.now(timezone.utc).replace(tzinfo=None)
    five_minutes = timedelta(minutes=5)
    assert record.created_at is not None, "created_at should be auto-populated on insert"
    assert record.updated_at is not None, "updated_at should be auto-populated on update"
    assert abs(now - record.created_at) < five_minutes, f"created_at {record.created_at!r} is not within 5 minutes of now"
    assert abs(now - record.updated_at) < five_minutes, f"updated_at {record.updated_at!r} is not within 5 minutes of now"


# --- UUID auto-generation ---

@pytest.mark.parametrize("make_record", [
    lambda agent_id: AgentRecord(name="uuid-test", agent_config=SAMPLE_AGENT_CONFIG, system_instructions=""),
    lambda agent_id: MemoryBlockRecord(agent_id=agent_id, label="uuid-test", content="", **PARTIAL_MEMORY_BLOCK_FIELDS),
    lambda agent_id: MessageRecord(agent_id=agent_id, timestamp=datetime.now(timezone.utc), **PARTIAL_MESSAGE_FIELDS),
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
        "context_window_start": datetime(2026, 1, 1, 13, 0, 0),
    }
    await assert_round_trips(session, AgentRecord(**fields), fields)


async def test_agent_config_structure(session: AsyncSession, agent_record: AgentRecord):
    """AgentConfig JSON contains required keys with correct types. Validation responsibility lies with AgentConfig its self
    So this test is really just a sanity check that we are validating storage and retrieval of an AgentConfig like obj
    TODO: consider delete once AgentConfig implemented and usage included in agent_record"""

    config = agent_record.agent_config
    assert isinstance(config["model_name"], str)
    assert isinstance(config["tool_names"], list)
    assert all(isinstance(t, str) for t in config["tool_names"])
    assert isinstance(config["soft_compaction_limit"], int)
    assert isinstance(config["is_deletable"], bool)


@pytest.mark.xfail(reason="AgentConfig Pydantic model not yet implemented", strict=True)
def test_agent_config_typed_model_todo():
    """TODO: AgentRecord.agent_config should serialize/deserialize through a typed AgentConfig
    Pydantic model rather than a plain dict. When implemented, replace test_agent_config_structure
    with a round-trip test against an actual AgentConfig instance."""
    pytest.fail("AgentConfig Pydantic model not yet implemented")


async def test_agent_record_defaults(session: AsyncSession):
    """Verify default values on a freshly created agent (no optional fields provided)."""
    agent = AgentRecord(name="defaults-test", agent_config=SAMPLE_AGENT_CONFIG)
    session.add(agent)
    await session.flush()
    await session.refresh(agent)
    assert agent.context_window_start is None
    assert agent.sys_prompt_compiled_at is None
    assert agent.system_instructions == ""


async def test_agent_record_timestamps_auto_populated(session: AsyncSession, agent_record: AgentRecord):
    """created_at and updated_at are automatically set when an agent is created."""
    await assert_timestamps_auto_populated(session, agent_record)


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


async def test_memory_block_unique_label_per_agent(session: AsyncSession, memory_block_record: MemoryBlockRecord):
    """Two blocks with the same label under the same agent violate the unique constraint."""
    session.add(memory_block_record)
    session.add(MemoryBlockRecord(agent_id=memory_block_record.agent_id, label=memory_block_record.label, content="different content", **PARTIAL_MEMORY_BLOCK_FIELDS))
    with pytest.raises(IntegrityError):
        await session.flush()


# --- MessageRecord ---

async def test_message_record_stores_all_fields(session: AsyncSession, message_record: MessageRecord):
    message_record.input_tokens = 150  # use a non-null value to verify integer persistence
    fields = {
        "agent_id": message_record.agent_id,
        "type": message_record.type,
        "content": message_record.content,
        "input_tokens": message_record.input_tokens,
        "timestamp": message_record.timestamp,
    }
    await assert_round_trips(session, message_record, fields)


async def test_message_input_tokens_nullable(session: AsyncSession, message_record: MessageRecord):
    """input_tokens may be NULL — only set on the final response row that closes a run."""
    message_record.input_tokens = None
    await assert_round_trips(session, message_record, {"input_tokens": None})


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


# --- Cascade delete ---

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
