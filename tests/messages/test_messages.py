"""
Tests for messages/messages.py — persist_messages, load_messages, deserialize_messages.
"""
import time
from datetime import datetime, timezone

import pytest
import pytest_asyncio
from pydantic_ai.messages import (
    ModelMessage,
    ModelRequest,
    ModelResponse,
    TextPart,
    ToolCallPart,
    ToolReturnPart,
    UserPromptPart,
)
from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from agent.types import AgentDeps
from db.models import AgentRecord, MessageRecord
from messages.messages import deserialize_messages, load_messages, persist_messages

# make_deps is a plain helper (not a pytest fixture) — import directly for use in test bodies
from conftest import make_deps


# ---------------------------------------------------------------------------
# Message factories
# ---------------------------------------------------------------------------

def make_request(content: str = "hello") -> ModelRequest:
    """Minimal ModelRequest with a single UserPromptPart."""
    return ModelRequest(parts=[UserPromptPart(content=content)])


def make_response(content: str = "hi") -> ModelResponse:
    """Minimal ModelResponse with a single TextPart."""
    return ModelResponse(parts=[TextPart(content=content)])


def make_tool_pair() -> tuple[ModelResponse, ModelRequest]:
    """A matched tool-call / tool-return pair.

    Returns (response_with_tool_call, request_with_tool_return).
    Use element [0] alone to simulate an orphaned tool call.
    """
    call_part = ToolCallPart(tool_name="mem_replace", args='{"label":"x"}', tool_call_id="tc1")
    return_part = ToolReturnPart(tool_name="mem_replace", content="ok", tool_call_id="tc1")
    return (
        ModelResponse(parts=[call_part]),
        ModelRequest(parts=[return_part]),
    )


def make_messages_batch(n: int) -> list[ModelMessage]:
    """Generate n alternating request/response pairs for performance tests."""
    messages: list[ModelMessage] = []
    for i in range(n):
        messages.append(make_request(f"msg {i}"))
        messages.append(make_response(f"resp {i}"))
    return messages


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

async def fetch_all_records(session: AsyncSession, agent_id: str) -> list[MessageRecord]:
    """Load all MessageRecords for an agent, ordered by timestamp."""
    result = await session.execute(
        select(MessageRecord)
        .where(MessageRecord.agent_id == agent_id)
        .order_by(MessageRecord.timestamp)
    )
    return list(result.scalars().all())


def naive_now() -> datetime:
    """Current UTC time as a naive datetime (matches MessageRecord.timestamp storage)."""
    return datetime.now(timezone.utc).replace(tzinfo=None)


# ---------------------------------------------------------------------------
# TestPersistMessages
# ---------------------------------------------------------------------------

@pytest.mark.asyncio
class TestPersistMessages:
    """Tests for persist_messages(deps, messages, input_tokens).

    Uses real in-memory DB. All tests share a session fixture from conftest.
    """

    @pytest_asyncio.fixture(autouse=True)
    async def setup(self, session: AsyncSession, agent_record: AgentRecord):
        """Wire up deps for all tests in this class."""
        self.session = session
        self.agent = agent_record
        self.deps = make_deps(session, agent_record)

    async def test_creates_one_record_per_message(self):
        messages = [make_request(), make_response()]
        await persist_messages(self.deps, messages, input_tokens=100)
        records = await fetch_all_records(self.session, self.agent.id)
        assert len(records) == 2

    async def test_sets_type_from_message_class(self):
        messages = [make_request(), make_response()]
        await persist_messages(self.deps, messages, input_tokens=100)
        records = await fetch_all_records(self.session, self.agent.id)
        assert records[0].type == "ModelRequest"
        assert records[1].type == "ModelResponse"

    async def test_serializes_content_as_json(self):
        req = make_request("round-trip content")
        await persist_messages(self.deps, [req], input_tokens=10)
        records = await fetch_all_records(self.session, self.agent.id)
        # Content must be valid JSON that round-trips to the original message
        from pydantic_ai.messages import ModelMessagesTypeAdapter
        restored = ModelMessagesTypeAdapter.validate_json(f"[{records[0].content}]")
        assert len(restored) == 1
        assert isinstance(restored[0], ModelRequest)

    async def test_input_tokens_on_final_row_only(self):
        messages = [make_request(), make_response(), make_request(), make_response()]
        await persist_messages(self.deps, messages, input_tokens=42)
        records = await fetch_all_records(self.session, self.agent.id)
        # Only the last record gets input_tokens
        for r in records[:-1]:
            assert r.input_tokens is None
        assert records[-1].input_tokens == 42

    async def test_timestamp_set_on_all_records(self):
        messages = [make_request(), make_response()]
        await persist_messages(self.deps, messages, input_tokens=10)
        records = await fetch_all_records(self.session, self.agent.id)
        for r in records:
            assert r.timestamp is not None
            assert isinstance(r.timestamp, datetime)

    async def test_agent_id_set_on_all_records(self):
        messages = [make_request(), make_response()]
        await persist_messages(self.deps, messages, input_tokens=10)
        records = await fetch_all_records(self.session, self.agent.id)
        for r in records:
            assert r.agent_id == self.agent.id

    async def test_empty_messages_list_is_noop(self):
        await persist_messages(self.deps, [], input_tokens=0)
        records = await fetch_all_records(self.session, self.agent.id)
        assert records == []

    async def test_persists_tool_call_and_return(self):
        response_with_call, request_with_return = make_tool_pair()
        await persist_messages(self.deps, [response_with_call, request_with_return], input_tokens=20)
        records = await fetch_all_records(self.session, self.agent.id)
        assert len(records) == 2
        assert records[0].type == "ModelResponse"
        assert records[1].type == "ModelRequest"

    async def test_orphaned_tool_call_replaced_with_error_response(self):
        """A ModelResponse with a ToolCallPart not followed by a matching ToolReturnPart
        should be replaced with an error ModelResponse (not stored as-is)."""
        orphan_response, _ = make_tool_pair()  # discard the matching return
        await persist_messages(self.deps, [orphan_response], input_tokens=5)
        records = await fetch_all_records(self.session, self.agent.id)
        assert len(records) == 1
        assert records[0].type == "ModelResponse"
        # Stored content must not contain a ToolCallPart — it should be the error replacement
        from pydantic_ai.messages import ModelMessagesTypeAdapter
        restored = ModelMessagesTypeAdapter.validate_json(f"[{records[0].content}]")
        assert not any(isinstance(p, ToolCallPart) for p in restored[0].parts)

    async def test_serialization_failure_injects_error_response(self):
        """If dump_json raises for a message, an error ModelResponse is stored and
        remaining messages continue to be persisted."""
        from unittest.mock import patch
        from pydantic_ai.messages import ModelMessagesTypeAdapter as MTA

        good = make_request("before")
        bad = make_response("problem")
        good2 = make_request("after")

        # Save original before patching — error path also calls dump_json to serialize the
        # injected error response, so we need to pass through for all but the targeted call.
        original_dump = MTA.dump_json
        call_count = 0

        def controlled_dump(messages_arg):
            nonlocal call_count
            call_count += 1
            # TODO: call_count == 2 assumes 'bad' is the second dump_json call in the loop.
            # If persist_messages ever adds a pre-loop dump_json call, this ordinal shifts.
            # Consider keying on messages_arg content instead.
            if call_count == 2:  # the 'bad' message is the second call
                raise ValueError("sim failure")
            return original_dump(messages_arg)

        with patch.object(MTA, "dump_json", side_effect=controlled_dump):
            await persist_messages(self.deps, [good, bad, good2], input_tokens=10)

        records = await fetch_all_records(self.session, self.agent.id)
        assert len(records) == 3
        # Middle record should be an error ModelResponse
        assert records[1].type == "ModelResponse"
        assert "sim failure" in records[1].content

    async def test_timestamp_ordering_preserved_when_new_messages_are_older(self):
        """If a new message's timestamp is older than the last DB record, it should be
        bumped forward to preserve chronological order."""
        # Persist a first message normally
        await persist_messages(self.deps, [make_request("first")], input_tokens=5)

        # Force that record's timestamp into the far future
        records = await fetch_all_records(self.session, self.agent.id)
        far_future = datetime(9999, 12, 31, 23, 59, 58)
        records[0].timestamp = far_future
        await self.session.flush()

        # Persist a second message — its natural timestamp will be far older
        await persist_messages(self.deps, [make_response("second")], input_tokens=5)

        records = await fetch_all_records(self.session, self.agent.id)
        assert len(records) == 2
        assert records[1].timestamp > records[0].timestamp


# ---------------------------------------------------------------------------
# TestLoadMessages
# ---------------------------------------------------------------------------

@pytest.mark.asyncio
class TestLoadMessages:
    """Tests for load_messages(session, agent_id, start_timestamp=None).

    Pre-seeds DB via persist_messages to ensure realistic records.
    """

    @pytest_asyncio.fixture(autouse=True)
    async def setup(self, session: AsyncSession, agent_record: AgentRecord):
        self.session = session
        self.agent = agent_record
        self.deps = make_deps(session, agent_record)

    async def test_returns_all_messages_when_no_start_timestamp(self):
        messages = [make_request(), make_response(), make_request(), make_response()]
        await persist_messages(self.deps, messages, input_tokens=50)
        records = await load_messages(self.session, self.agent.id)
        assert len(records) == 4

    async def test_returns_empty_list_when_no_messages(self):
        records = await load_messages(self.session, self.agent.id)
        assert records == []

    async def test_start_timestamp_filters_inclusive(self):
        early = [make_request("early"), make_response("early reply")]
        await persist_messages(self.deps, early, input_tokens=10)

        # Record the cutoff timestamp before persisting the second batch
        cutoff_record = (await fetch_all_records(self.session, self.agent.id))[-1]
        cutoff = cutoff_record.timestamp

        late = [make_request("late"), make_response("late reply")]
        await persist_messages(self.deps, late, input_tokens=10)

        records = await load_messages(self.session, self.agent.id, start_timestamp=cutoff)
        # Should include the cutoff record and everything after (inclusive)
        assert len(records) == 3

    async def test_results_in_chronological_order(self):
        messages = [make_request("first"), make_response("second"), make_request("third")]
        await persist_messages(self.deps, messages, input_tokens=10)
        records = await load_messages(self.session, self.agent.id)
        timestamps = [r.timestamp for r in records]
        assert timestamps == sorted(timestamps)

    async def test_returns_only_records_for_given_agent(self):
        """Records from other agents must not appear in results."""
        other_agent = AgentRecord(
            name="other-agent",
            agent_config=self.agent.agent_config,
            system_instructions="Other agent.",
        )
        self.session.add(other_agent)
        await self.session.flush()
        other_deps = make_deps(self.session, other_agent)

        await persist_messages(self.deps, [make_request(), make_response()], input_tokens=10)
        await persist_messages(other_deps, [make_request(), make_response()], input_tokens=10)

        records = await load_messages(self.session, self.agent.id)
        assert len(records) == 2
        assert all(r.agent_id == self.agent.id for r in records)

    async def test_returns_list_of_message_records(self):
        await persist_messages(self.deps, [make_request()], input_tokens=5)
        records = await load_messages(self.session, self.agent.id)
        assert len(records) == 1
        assert isinstance(records[0], MessageRecord)

    async def test_start_timestamp_ahead_of_all_messages_returns_empty(self):
        """When start_timestamp is later than every message, the result is empty."""
        await persist_messages(self.deps, [make_request(), make_response()], input_tokens=10)
        far_future = datetime(9999, 12, 31, 23, 59, 59)
        records = await load_messages(self.session, self.agent.id, start_timestamp=far_future)
        assert records == []


# ---------------------------------------------------------------------------
# TestDeserializeMessages
# ---------------------------------------------------------------------------

class TestDeserializeMessages:
    """Tests for deserialize_messages(records) — pure function, no DB needed."""

    def _make_record(self, message: ModelMessage, agent_id: str = "test-agent") -> MessageRecord:
        """Build a MessageRecord from a ModelMessage without touching the DB."""
        from pydantic_ai.messages import ModelMessagesTypeAdapter
        serialized = ModelMessagesTypeAdapter.dump_json([message])
        # Extract the single-message JSON object (strip the outer array brackets)
        content = serialized.decode()[1:-1]
        return MessageRecord(
            agent_id=agent_id,
            type=type(message).__name__,
            content=content,
            input_tokens=None,
            timestamp=naive_now(),
        )

    def test_deserializes_request(self):
        req = make_request("hello")
        record = self._make_record(req)
        result = deserialize_messages([record])
        assert len(result) == 1
        assert isinstance(result[0], ModelRequest)

    def test_deserializes_response(self):
        resp = make_response("hi")
        record = self._make_record(resp)
        result = deserialize_messages([record])
        assert len(result) == 1
        assert isinstance(result[0], ModelResponse)

    def test_deserializes_tool_pair(self):
        response_with_call, request_with_return = make_tool_pair()
        records = [self._make_record(response_with_call), self._make_record(request_with_return)]
        result = deserialize_messages(records)
        assert len(result) == 2

    def test_invalid_content_injects_error_response(self):
        """Invalid JSON content should produce an error ModelResponse, not raise."""
        bad_record = MessageRecord(
            agent_id="test-agent",
            type="ModelRequest",
            content="not valid json at all",
            input_tokens=None,
            timestamp=naive_now(),
        )
        result = deserialize_messages([bad_record])
        assert len(result) == 1
        assert isinstance(result[0], ModelResponse)
        # Error response should contain some error context
        assert result[0].parts

    def test_summary_record_passthrough(self):
        """Summary records (ModelRequest with XML content) round-trip cleanly."""
        summary_content = "<summary>Prior conversation summary here.</summary>"
        summary_msg = make_request(summary_content)
        record = self._make_record(summary_msg)
        record.type = "Summary"
        result = deserialize_messages([record])
        assert len(result) == 1
        assert isinstance(result[0], ModelRequest)

    def test_performance_1000_messages(self):
        """Deserialization of 1000 messages should be fast (< 1s budget)."""
        messages = make_messages_batch(500)  # 500 pairs = 1000 messages
        records = [self._make_record(m) for m in messages]

        start = time.perf_counter()
        result = deserialize_messages(records)
        elapsed = time.perf_counter() - start

        assert len(result) == 1000
        print(f"\nDeserialize 1000 messages: {elapsed:.3f}s ({elapsed/1000*1000:.3f}ms/msg)")
        assert elapsed < 1.0, f"Deserialization too slow: {elapsed:.3f}s for 1000 messages"


# ---------------------------------------------------------------------------
# TestRoundTrip
# ---------------------------------------------------------------------------

@pytest.mark.asyncio
class TestRoundTrip:
    """Integration tests: persist → load → deserialize produces equivalent messages."""

    @pytest_asyncio.fixture(autouse=True)
    async def setup(self, session: AsyncSession, agent_record: AgentRecord):
        self.session = session
        self.agent = agent_record
        self.deps = make_deps(session, agent_record)

    async def test_request_response_round_trip(self):
        original = [make_request("round-trip me"), make_response("got it")]
        await persist_messages(self.deps, original, input_tokens=20)
        records = await load_messages(self.session, self.agent.id)
        restored = deserialize_messages(records)

        assert len(restored) == 2
        assert isinstance(restored[0], ModelRequest)
        assert isinstance(restored[1], ModelResponse)

    async def test_tool_pair_round_trip(self):
        response_with_call, request_with_return = make_tool_pair()
        await persist_messages(self.deps, [response_with_call, request_with_return], input_tokens=15)
        records = await load_messages(self.session, self.agent.id)
        restored = deserialize_messages(records)

        assert len(restored) == 2
        call_parts = [p for p in restored[0].parts if isinstance(p, ToolCallPart)]
        return_parts = [p for p in restored[1].parts if isinstance(p, ToolReturnPart)]
        assert len(call_parts) == 1
        assert len(return_parts) == 1
        assert call_parts[0].tool_call_id == return_parts[0].tool_call_id
