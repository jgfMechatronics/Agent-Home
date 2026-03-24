import re
import uuid

from datetime import datetime, timezone

import pytest
from sqlalchemy.exc import IntegrityError

from db.models import AgentRecord, MemoryBlockRecord, MessageRecord

_UUID_RE = re.compile(r"^[0-9a-f]{8}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{12}$")


async def assert_round_trips(session, record, expected_fields: dict):
    """Add a record, flush, refresh from DB, and assert each expected field matches."""
    session.add(record)
    await session.flush()
    await session.refresh(record)
    for field, expected in expected_fields.items():
        actual = getattr(record, field)
        assert actual == expected, f"Field '{field}': expected {expected!r}, got {actual!r}"


async def assert_timestamps_auto_populated(session, record):
    """Verify created_at and updated_at are set automatically on creation."""
    await session.refresh(record)
    assert record.created_at is not None, "created_at should be auto-populated on insert"
    assert record.updated_at is not None, "updated_at should be auto-populated on insert"


# --- UUID auto-generation ---

@pytest.mark.parametrize("make_record", [
    lambda agent_id: AgentRecord(name="uuid-test", agent_config={"model_name": "m", "tool_names": [], "soft_limit": 1}, system_instructions=""),
    lambda agent_id: MemoryBlockRecord(agent_id=agent_id, label="uuid-test", description="", content="", char_limit=100, position=0),
    lambda agent_id: MessageRecord(agent_id=agent_id, type="ModelRequest", content="{}", input_tokens=None, timestamp=datetime.now(timezone.utc)),
])
async def test_id_auto_generated_as_uuid_string(session, sample_agent, make_record):
    """All models auto-generate a UUID string id on insert — not required at construction."""
    record = make_record(sample_agent.id)
    assert record.id is None, "id should not be set before flush"
    session.add(record)
    await session.flush()
    await session.refresh(record)
    assert record.id is not None, "id should be populated after flush"
    assert isinstance(record.id, str), "id should be stored as a string"
    assert _UUID_RE.match(record.id), f"id should be a valid UUID string, got: {record.id!r}"


# --- AgentRecord ---

async def test_agent_record_stores_all_fields(session):
    agent_config = {"model_name": "claude-sonnet-4-20250514", "tool_names": ["tool_a"], "soft_limit": 8000}
    # Use naive datetimes: SQLAlchemy DateTime stores TEXT in SQLite and may strip
    # timezone info depending on the timezone= flag, making tz-aware comparisons brittle.
    compiled_at = datetime(2026, 1, 1, 12, 0, 0)
    context_window_start = datetime(2026, 1, 1, 13, 0, 0)
    await assert_round_trips(
        session,
        AgentRecord(
            name="my-agent",
            agent_config=agent_config,
            system_instructions="Be helpful.",
            compiled_system_prompt="<compiled>Be helpful.</compiled>",
            compiled_at=compiled_at,
            context_window_start=context_window_start,
        ),
        {
            "name": "my-agent",
            "agent_config": agent_config,
            "system_instructions": "Be helpful.",
            "compiled_system_prompt": "<compiled>Be helpful.</compiled>",
            "compiled_at": compiled_at,
            "context_window_start": context_window_start,
        },
    )


async def test_agent_config_structure(session, sample_agent):
    """AgentConfig JSON contains required keys with correct types."""
    config = sample_agent.agent_config
    assert isinstance(config["model_name"], str)
    assert isinstance(config["tool_names"], list)
    assert all(isinstance(t, str) for t in config["tool_names"])
    assert isinstance(config["soft_limit"], int)


async def test_agent_record_null_defaults(session, sample_agent):
    """context_window_start and compiled_at are both NULL on a freshly created agent."""
    await session.refresh(sample_agent)
    assert sample_agent.context_window_start is None
    assert sample_agent.compiled_at is None


async def test_agent_record_timestamps_auto_populated(session, sample_agent):
    """created_at and updated_at are automatically set when an agent is created."""
    await assert_timestamps_auto_populated(session, sample_agent)


# --- MemoryBlockRecord ---

async def test_memory_block_stores_all_fields(session, sample_agent):
    await assert_round_trips(
        session,
        MemoryBlockRecord(
            agent_id=sample_agent.id,
            label="persona",
            description="The agent's persona.",
            content="I am a helpful assistant.",
            char_limit=2000,
            position=0,
        ),
        {
            "agent_id": sample_agent.id,
            "label": "persona",
            "description": "The agent's persona.",
            "content": "I am a helpful assistant.",
            "char_limit": 2000,
            "position": 0,
        },
    )


async def test_memory_block_timestamps_auto_populated(session, sample_agent):
    """created_at and updated_at are automatically set when a memory block is created."""
    block = MemoryBlockRecord(
        agent_id=sample_agent.id, label="auto-ts", description="", content="x", char_limit=2000, position=0,
    )
    session.add(block)
    await session.flush()
    await assert_timestamps_auto_populated(session, block)


async def test_memory_block_fk_enforced(session):
    """Cannot create a MemoryBlockRecord referencing a nonexistent agent."""
    block = MemoryBlockRecord(
        agent_id=str(uuid.uuid4()),
        label="persona",
        description="",
        content="",
        char_limit=2000,
        position=0,
    )
    session.add(block)
    with pytest.raises(IntegrityError):
        await session.flush()


async def test_memory_block_unique_label_per_agent(session, sample_agent):
    """Two blocks with the same label under the same agent violate the unique constraint."""
    session.add(MemoryBlockRecord(
        agent_id=sample_agent.id, label="persona", description="", content="first", char_limit=2000, position=0,
    ))
    session.add(MemoryBlockRecord(
        agent_id=sample_agent.id, label="persona", description="", content="second", char_limit=2000, position=1,
    ))
    with pytest.raises(IntegrityError):
        await session.flush()


# --- MessageRecord ---

async def test_message_record_stores_all_fields(session, sample_agent):
    content = '{"parts": [{"type": "text", "content": "Hello"}]}'
    ts = datetime(2026, 1, 1, 12, 0, 0)  # naive — avoids timezone round-trip brittleness
    await assert_round_trips(
        session,
        MessageRecord(
            agent_id=sample_agent.id,
            type="ModelRequest",
            content=content,
            input_tokens=150,
            timestamp=ts,
        ),
        {
            "agent_id": sample_agent.id,
            "type": "ModelRequest",
            "content": content,
            "input_tokens": 150,
            "timestamp": ts,
        },
    )


async def test_message_fk_enforced(session):
    """Cannot create a MessageRecord referencing a nonexistent agent."""
    message = MessageRecord(
        agent_id=str(uuid.uuid4()),
        type="ModelRequest",
        content="{}",
        input_tokens=None,
        timestamp=datetime.now(timezone.utc),
    )
    session.add(message)
    with pytest.raises(IntegrityError):
        await session.flush()


async def test_message_input_tokens_nullable(session, sample_agent):
    """input_tokens may be NULL — only set on the final response row that closes a run."""
    await assert_round_trips(
        session,
        MessageRecord(
            agent_id=sample_agent.id,
            type="ModelResponse",
            content="{}",
            input_tokens=None,
            timestamp=datetime.now(timezone.utc),
        ),
        {"input_tokens": None},
    )


# --- Cascade delete ---

async def test_cascade_delete_removes_blocks_and_messages(session, sample_agent):
    """Deleting an agent cascades to all associated blocks and messages."""
    block = MemoryBlockRecord(agent_id=sample_agent.id, label="persona", description="", content="x", char_limit=2000, position=0)
    message = MessageRecord(agent_id=sample_agent.id, type="ModelRequest", content="{}", input_tokens=None, timestamp=datetime.now(timezone.utc))
    session.add(block)
    session.add(message)
    await session.flush()

    block_id, message_id = block.id, message.id

    await session.delete(sample_agent)
    await session.flush()

    assert await session.get(MemoryBlockRecord, block_id) is None
    assert await session.get(MessageRecord, message_id) is None


# --- JSON round-trip ---

async def test_json_fields_round_trip(session, sample_agent):
    """Nested JSON structures in AgentConfig and message content survive a write-read cycle."""
    complex_config = {
        "model_name": "claude-opus-4",
        "tool_names": ["a", "b", "c"],
        "soft_limit": 99999,
        "extra_flag": True,
    }
    agent = AgentRecord(name="json-test", agent_config=complex_config, system_instructions="")
    session.add(agent)
    await session.flush()
    await session.refresh(agent)
    assert agent.agent_config == complex_config

    content = '{"parts": [{"type": "user-prompt", "content": "hello \\u2603"}]}'
    message = MessageRecord(agent_id=agent.id, type="ModelRequest", content=content, input_tokens=None, timestamp=datetime.now(timezone.utc))
    session.add(message)
    await session.flush()
    await session.refresh(message)
    assert message.content == content
