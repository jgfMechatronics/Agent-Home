"""
Tests for messages/messages.py — persist_messages, load_messages, deserialize_messages.
"""
import logging
import time
from datetime import datetime

import pytest
import pytest_asyncio
import asyncio
from unittest.mock import patch
from pydantic_ai.messages import (
    ModelMessage,
    ModelMessagesTypeAdapter,
    ModelRequest,
    ModelResponse,
    RetryPromptPart,
    ToolCallPart,
    ToolReturnPart,
)
from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from agent.types import AgentDeps
from db.models import AgentRecord, MessageRecord, utcnow
from messages.messages import deserialize_messages, load_messages, persist_messages

# Plain helpers (not fixtures) — import directly for use in test bodies
from conftest import (
    SAMPLE_AGENT_CONFIG,
    make_deps,
    make_request,
    make_response,
    make_retry_pair,
    make_tool_pair,
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


# ---------------------------------------------------------------------------
# TestPersistMessages
# ---------------------------------------------------------------------------

class DBTestBase:
    """Base class for test classes that need a database session and agent."""

    @pytest_asyncio.fixture(autouse=True)
    async def setup(self, session: AsyncSession, agent_record: AgentRecord):
        self.session = session
        self.agent = agent_record
        self.deps = make_deps(session, agent_record)


@pytest.mark.asyncio
class TestPersistMessages(DBTestBase):
    """Tests for persist_messages(deps, messages, total_tokens).

    Uses real in-memory DB. All tests share a session fixture from conftest.
    """

    async def _persist_and_fetch(self, messages, total_tokens=10) -> list[MessageRecord]:
        await persist_messages(self.deps, messages, total_tokens=total_tokens)
        return await fetch_all_records(self.session, self.agent.id)

    async def test_creates_one_record_per_message(self):
        records = await self._persist_and_fetch([make_request(), make_response()], total_tokens=100)
        assert len(records) == 2

    async def test_sets_type_from_message_class(self):
        records = await self._persist_and_fetch([make_request(), make_response()], total_tokens=100)
        assert records[0].type == "ModelRequest"
        assert records[1].type == "ModelResponse"

    async def test_serializes_content_as_json(self):
        req = make_request("round-trip content")
        records = await self._persist_and_fetch([req])
        # Content must be valid JSON that round-trips to the original message
        restored = ModelMessagesTypeAdapter.validate_json(f"[{records[0].content}]")
        assert len(restored) == 1
        assert restored[0] == req

    async def test_total_tokens_on_final_row_only(self):
        records = await self._persist_and_fetch(
            [make_request(), make_response(), make_request(), make_response()], total_tokens=42
        )
        # Only the last record gets total_tokens
        for r in records[:-1]:
            assert r.total_tokens is None
        assert records[-1].total_tokens == 42

    async def test_timestamp_set_on_all_records(self):
        request = make_request()
        response = make_response()
        records = await self._persist_and_fetch([request, response])
        
        # Pydantic-AI's handling of timestamps across ModelRequests and ModelResponses is inconsistent
        assert records[0].timestamp == request.parts[0].timestamp.replace(tzinfo=None)
        assert records[1].timestamp == response.timestamp.replace(tzinfo=None)

    async def test_agent_id_set_on_all_records(self):
        records = await self._persist_and_fetch([make_request(), make_response()])
        for r in records:
            assert r.agent_id == self.agent.id

    async def test_empty_messages_list_is_noop(self):
        records = await self._persist_and_fetch([], total_tokens=0)
        assert records == []

    @pytest.mark.parametrize("pair_fn", [make_tool_pair, make_retry_pair])
    async def test_persists_tool_call_pair(self, pair_fn):
        """A matched tool-call / tool-response pair should survive persist unchanged,
        whether the response is a ToolReturnPart (success) or RetryPromptPart (ModelRetry)."""
        response_with_call, request_with_response = pair_fn()

        records = await self._persist_and_fetch([response_with_call, request_with_response], total_tokens=20)

        assert len(records) == 2
        assert records[0].type == "ModelResponse"
        assert records[1].type == "ModelRequest"
        restored = [ModelMessagesTypeAdapter.validate_json(f"[{r.content}]")[0] for r in records]
        assert restored == [response_with_call, request_with_response]

    async def test_records_isolated_per_agent(self):
        """
        Messages persisted for one agent must not affect another agent's records,
        and _get_last_timestamp must not cross agent boundaries for ordering.
        Here we also test that new messages for a given agent do not affect same agent's old msgs
        """
        other_agent = AgentRecord(
            name="other-agent",
            agent_config=SAMPLE_AGENT_CONFIG,
            system_instructions="Other agent.",
        )
        self.session.add(other_agent)
        await self.session.flush()
        other_deps = make_deps(self.session, other_agent)

        expected_other_msg = make_request("other msg")
        my_first_expected_msg = make_request("my msg")
        my_second_expected_msg = make_request("my second msg")
        
        await persist_messages(other_deps, [expected_other_msg], total_tokens=5)
        await persist_messages(self.deps, [my_first_expected_msg], total_tokens=5)
        await persist_messages(self.deps, [my_second_expected_msg], total_tokens=5)

        my_records = await fetch_all_records(self.session, self.agent.id)
        other_records = await fetch_all_records(self.session, other_agent.id)

        assert len(my_records) == 2
        my_restored = [ModelMessagesTypeAdapter.validate_json(f"[{r.content}]")[0] for r in my_records]
        assert my_restored[0] == my_first_expected_msg
        assert my_restored[1] == my_second_expected_msg

        assert len(other_records) == 1
        assert ModelMessagesTypeAdapter.validate_json(f"[{other_records[0].content}]")[0] == expected_other_msg


    # ------------Tests for handling non-persistable messages -----------------------
    
    # ------------ Helpers -------------------------
    def _assert_summary_warning_appended(self, records, error_text, original_timestamp):
        """Assert the last record in records is a summary warning matching the expected format."""
        record = records[-1]
        assert record.type == "ModelResponse"
        restored = ModelMessagesTypeAdapter.validate_json(f"[{record.content}]")

        expected = (
            f"WARNING: A problem was encountered while persisting messages from the last turn: "
            f"'{error_text}'. A warning was injected in place of the problematic message, "
            f"problematic message timestamp was {original_timestamp}"
        )
        assert restored[0].parts[0].content == expected

    async def _assert_orphan_replaced(self, orphan_msg, orphaned_part_type, expected_error):
        """Persist a single orphaned tool message and assert it was replaced with the expected
        error record, with a summary warning appended at the end of the chain."""
        records = await self._persist_and_fetch([orphan_msg], total_tokens=5)
        assert len(records) == 2  # positional error + summary warning

        # Original orphaned message was dropped — no record should contain the orphaned part type
        for record in records:
            deserialized = ModelMessagesTypeAdapter.validate_json(f"[{record.content}]")
            assert not any(isinstance(p, orphaned_part_type) for p in deserialized[0].parts)

        # Positional error record has the expected error text
        assert records[0].type == "ModelResponse"
        restored = ModelMessagesTypeAdapter.validate_json(f"[{records[0].content}]")
        assert restored[0].parts[0].content == expected_error

        # Summary warning appended at end
        ts = orphan_msg.timestamp
        original_timestamp = ts.replace(tzinfo=None) if (ts is not None and ts.tzinfo is not None) else ts
        self._assert_summary_warning_appended(records, expected_error, original_timestamp)

    # -------------- Tests -------------------
    # TODO: Check for multiple orphaned tool messages in a single list of msgs,
    # and check for behavior around multiple tool call/return parts in a single req/resp (after verifying that is even a sensical situation
    # , parallel tool calls perhaps?)
    async def test_orphaned_tool_call_replaced_with_error_response(self):
        """A ModelResponse with a ToolCallPart not followed by a matching ToolReturnPart
        should be replaced with an error ModelResponse (not stored as-is)."""
        orphan_response, _ = make_tool_pair()  # discard the matching return
        await self._assert_orphan_replaced(
            orphan_response, ToolCallPart, "[Orphaned tool call(s) dropped: mem_replace]"
        )

    @pytest.mark.parametrize("pair_fn,orphaned_part_type,expected_error", [
        (make_tool_pair, ToolReturnPart, "[Orphaned tool return(s) dropped: mem_replace]"),
        (make_retry_pair, RetryPromptPart, "[Orphaned tool retry(s) dropped: mem_replace]"),
    ])
    async def test_orphaned_tool_response_replaced_with_error_response(self, pair_fn, orphaned_part_type, expected_error):
        """A ModelRequest with a tool response part (ToolReturnPart or RetryPromptPart) not
        preceded by a matching ToolCallPart should be replaced with an error ModelResponse."""
        _, orphan_request = pair_fn()  # discard the matching call
        await self._assert_orphan_replaced(orphan_request, orphaned_part_type, expected_error)

    async def test_serialization_failure_injects_error_response(self, caplog):
        """
        If dump_json raises for a message, an error ModelResponse is stored in place of
        the bad message, remaining messages continue to be persisted, and a summary warning
        is appended at the end of the chain so the model/user can't miss it.
        TODO: It would be nice to have an explicit error designator for model messages or responses.
        Possibly we should add a subtype to ModelRecord for stuff like easy tracking of "Tool Call", "Tool Return", "Error"
        """
        good = make_request("before")
        bad = make_response("problem")
        good2 = make_request("after")

        # Save original before patching — error path also calls dump_json to serialize the
        # injected error response, so we need to pass through for all but the targeted call.
        original_dump = ModelMessagesTypeAdapter.dump_json

        def controlled_dump(messages_arg):
            if messages_arg[0] is bad:
                raise ValueError("sim failure")
            return original_dump(messages_arg)

        with patch.object(ModelMessagesTypeAdapter, "dump_json", side_effect=controlled_dump):
            await persist_messages(self.deps, [good, bad, good2], total_tokens=10)

        records = await fetch_all_records(self.session, self.agent.id)
        assert len(records) == 4  # good, positional error, good2, summary warning

        error_text = "[persist_messages serialization error]: ValueError: sim failure"

        # Good messages before and after the bad one are stored intact
        assert ModelMessagesTypeAdapter.validate_json(f"[{records[0].content}]")[0] == good
        assert ModelMessagesTypeAdapter.validate_json(f"[{records[2].content}]")[0] == good2

        # Positional error record replaces bad in-place
        assert records[1].type == "ModelResponse"
        positional = ModelMessagesTypeAdapter.validate_json(f"[{records[1].content}]")
        assert positional[0].parts[0].content == error_text

        # Summary warning appended at end, referencing the error and original timestamp
        expected_ts = bad.timestamp.replace(tzinfo=None)
        self._assert_summary_warning_appended(records, error_text, expected_ts)

        # Exception logged
        assert any("sim failure" in r.message for r in caplog.records)

    async def test_timestamp_ordering_preserved_when_new_messages_are_older(self, caplog):
        """If a new message's timestamp is older than the last DB record, it should be
        bumped forward to preserve chronological order, and a warning should be logged."""
        # Persist a first message normally
        await persist_messages(self.deps, [make_request("first")], total_tokens=5)

        # Force that record's timestamp into the far future
        records = await fetch_all_records(self.session, self.agent.id)
        far_future = datetime(9999, 12, 31, 23, 59, 58)
        records[0].timestamp = far_future
        await self.session.flush()

        # Persist a second message — its natural timestamp will be far older
        with caplog.at_level(logging.WARNING):
            await persist_messages(self.deps, [make_response("second")], total_tokens=5)

        records = await fetch_all_records(self.session, self.agent.id)
        assert len(records) == 2
        assert records[1].timestamp > records[0].timestamp
        assert any("timestamp ordering violation" in r.message for r in caplog.records)


# ---------------------------------------------------------------------------
# TestLoadMessages
# ---------------------------------------------------------------------------

@pytest.mark.asyncio
class TestLoadMessages(DBTestBase):
    """Tests for load_messages(session, agent_id, start_timestamp=None).

    Pre-seeds DB via persist_messages to ensure realistic records.
    """

    async def test_returns_all_messages_when_no_start_timestamp(self):
        messages = [make_request(), make_response(), make_request(), make_response()]
        await persist_messages(self.deps, messages, total_tokens=50)
        records = await load_messages(self.session, self.agent.id)
        assert deserialize_messages(records) == messages

    async def test_returns_empty_list_when_no_messages(self):
        records = await load_messages(self.session, self.agent.id)
        assert records == []

    async def test_start_timestamp_filters_inclusive(self):
        early = [make_request("early"), make_response("early reply")]
        await persist_messages(self.deps, early, total_tokens=10)

        # Record the cutoff timestamp before persisting the second batch
        cutoff_record = (await fetch_all_records(self.session, self.agent.id))[-1]
        cutoff = cutoff_record.timestamp
        await asyncio.sleep(0.1) # ensure timestamp unique from early
        late = [make_request("late"), make_response("late reply")]
        await persist_messages(self.deps, late, total_tokens=10)

        records = await load_messages(self.session, self.agent.id, start_timestamp=cutoff)
        # Should include the cutoff record and everything after (inclusive)
        assert deserialize_messages(records) == [early[1]] + late

    async def test_results_in_chronological_order(self):
        msg1 = make_request("first")
        await asyncio.sleep(0.005) # Ensure distinct timestamps
        msg2 = make_response("second")
        await asyncio.sleep(0.005)
        msg3 = make_request("third")
        messages = [msg1, msg2, msg3]
        await persist_messages(self.deps, messages, total_tokens=10)
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

        await persist_messages(self.deps, [make_request(), make_response()], total_tokens=10)
        await persist_messages(other_deps, [make_request(), make_response()], total_tokens=10)

        records = await load_messages(self.session, self.agent.id)
        assert len(records) == 2
        assert all(r.agent_id == self.agent.id for r in records)

    async def test_returns_list_of_message_records(self):
        await persist_messages(self.deps, [make_request()], total_tokens=5)
        records = await load_messages(self.session, self.agent.id)
        assert len(records) == 1
        assert isinstance(records[0], MessageRecord)

    async def test_start_timestamp_ahead_of_all_messages_returns_empty(self):
        """When start_timestamp is later than every message, the result is empty."""
        await persist_messages(self.deps, [make_request(), make_response()], total_tokens=10)
        far_future = datetime(9999, 12, 31, 23, 59, 59)
        records = await load_messages(self.session, self.agent.id, start_timestamp=far_future)
        assert records == []


# ---------------------------------------------------------------------------
# TestDeserializeMessages
# ---------------------------------------------------------------------------

class TestDeserializeMessages:
    """Tests for deserialize_messages(records) — pure function, no DB needed."""

    def _make_record(self, message: ModelMessage, agent_id: str = "test-agent") -> MessageRecord:
        """Build a MessageRecord from a ModelMessage without touching the DB. This is 'serializing'"""
        serialized = ModelMessagesTypeAdapter.dump_json([message])
        # Extract the single-message JSON object (strip the outer array brackets)
        content = serialized.decode()[1:-1]
        return MessageRecord(
            agent_id=agent_id,
            type=type(message).__name__,
            content=content,
            total_tokens=None,
            timestamp=utcnow(),
        )

    @pytest.mark.parametrize("messages", [
        [make_request("hello")],
        [make_response("hi")],
        list(make_tool_pair()),
    ])
    def test_deserializes_messages(self, messages):
        records = [self._make_record(m) for m in messages]
        result = deserialize_messages(records)
        assert result == messages

    def test_empty_list_returns_empty(self):
        assert deserialize_messages([]) == []

    def test_invalid_content_raises(self):
        """Invalid JSON content should raise ValueError, not silently inject an error response."""
        bad_record = MessageRecord(
            agent_id="test-agent",
            type="ModelRequest",
            content="not valid json at all",
            total_tokens=None,
            timestamp=utcnow(),
        )
        with pytest.raises(ValueError, match=f"Deserialization error for record {bad_record.id}"):
            deserialize_messages([bad_record])

    def test_performance_1000_messages(self):
        """Deserialization of 1000 messages should be fast (< 1s budget)."""
        n_messages = 1000
        n_pairs = n_messages//2
        messages = make_messages_batch(n_pairs)
        records = [self._make_record(m) for m in messages]

        start = time.perf_counter()
        result = deserialize_messages(records)
        elapsed = time.perf_counter() - start

        assert len(result) == n_messages
        print(f"\nDeserialize {n_messages} messages: {elapsed:.3f}s ({elapsed/(1000*n_messages):.3f}ms/msg)")
        max_time_sec = 0.5
        assert elapsed < max_time_sec, f"Deserialization too slow: {elapsed:.3f}s for {n_messages} messages"
        
        # while we're here, may as well check bulk deserialization accuracy
        assert result == messages


# ---------------------------------------------------------------------------
# TestRoundTrip
# ---------------------------------------------------------------------------

@pytest.mark.asyncio
class TestRoundTrip(DBTestBase):
    """Integration tests: persist → load → deserialize produces equivalent messages."""

    async def test_request_response_round_trip(self):
        original = [make_request("round-trip me"), make_response("got it")]
        await persist_messages(self.deps, original, total_tokens=20)
        records = await load_messages(self.session, self.agent.id)
        restored = deserialize_messages(records)

        assert restored == original

    async def test_tool_pair_round_trip(self):
        response_with_call, request_with_return = make_tool_pair()
        await persist_messages(self.deps, [response_with_call, request_with_return], total_tokens=15)
        records = await load_messages(self.session, self.agent.id)
        restored = deserialize_messages(records)

        assert restored[0] == response_with_call
        assert restored[1] == request_with_return
