"""
Tests for messages/messages.py — persist_messages, load_messages, deserialize_messages.
"""
import json
import time

import pytest
import pytest_asyncio
from unittest.mock import patch
from pydantic_ai.messages import (
    ModelMessage,
    ModelMessagesTypeAdapter,
    ModelResponse,
    RetryPromptPart,
    TextPart,
    ToolCallPart,
    ToolReturnPart,
)
from pydantic_ai.tools import ToolDefinition
from pydantic_ai.usage import RequestUsage
from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from agent.types import AgentConfig
from db.models import AgentConfigSnapshot, AgentRecord, MessageRecord, SystemPromptSnapshot, ToolDefinitionSnapshot, utcnow
from messages.messages import deserialize_messages, load_messages, persist_messages

# Plain helpers (not fixtures) — import directly for use in test bodies
from conftest import (
    SAMPLE_AGENT_CONFIG,
    make_alternating_messages,
    make_deps,
    make_request,
    make_response,
    make_retry_pair,
    make_tool_pair,
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

SAMPLE_TOOL_SCHEMAS = [
    ToolDefinition(
        name="memory_replace",
        description="Replace text in a memory block.",
        parameters_json_schema={
            "type": "object",
            "properties": {
                "label": {"type": "string"},
                "old_string": {"type": "string"},
                "new_string": {"type": "string"},
            },
            "required": ["label", "old_string", "new_string"],
            "additionalProperties": False,
        },
    )
]

MUTATED_AGENT_CONFIG = AgentConfig(
    model_name="claude-haiku-4-5-20251001",
    tool_names=["memory_replace"],
    soft_compaction_limit=20000,
    thinking_enabled=True,
)

MUTATED_TOOL_SCHEMAS = [
    ToolDefinition(
        name="memory_insert",
        description="Insert text into a memory block.",
        parameters_json_schema={
            "type": "object",
            "properties": {
                "label": {"type": "string"},
                "new_string": {"type": "string"},
            },
            "required": ["label", "new_string"],
            "additionalProperties": False,
        },
    )
]


async def fetch_all_records(session: AsyncSession, agent_id: str) -> list[MessageRecord]:
    """Load all MessageRecords for an agent, in seq_id order."""
    result = await session.execute(
        select(MessageRecord)
        .where(MessageRecord.agent_id == agent_id)
        .order_by(MessageRecord.seq_id)
    )
    return list(result.scalars().all())


def make_messages_batch(n: int) -> list[ModelMessage]:
    """Generate n alternating request/response pairs for performance tests."""
    return make_alternating_messages(n * 2)


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

    async def _persist_and_fetch(self, messages, tool_schemas=None) -> list[MessageRecord]:
        schemas = tool_schemas if tool_schemas is not None else SAMPLE_TOOL_SCHEMAS
        await persist_messages(self.deps, messages, schemas)
        return await fetch_all_records(self.session, self.agent.id)


@pytest.mark.asyncio
class TestPersistMessages(DBTestBase):
    """Tests for persist_messages(deps, messages).

    Uses real in-memory DB. All tests share a session fixture from conftest.
    """

    @pytest.fixture
    def resp_with_usage(self):
        """A ModelResponse with known usage data (30 in + 12 out = 42 total)."""
        return ModelResponse(
            parts=[TextPart(content="with usage")],
            usage=RequestUsage(input_tokens=30, output_tokens=12),
        )

    async def test_creates_one_record_per_message(self):
        records = await self._persist_and_fetch([make_request(), make_response()])
        assert len(records) == 2

    async def test_sets_type_from_message_class(self):
        records = await self._persist_and_fetch([make_request(), make_response()])
        assert records[0].type == "ModelRequest"
        assert records[1].type == "ModelResponse"

    async def test_serializes_content_as_json(self):
        req = make_request("round-trip content")
        records = await self._persist_and_fetch([req])
        # Content must be valid JSON that round-trips to the original message
        restored = ModelMessagesTypeAdapter.validate_json(f"[{records[0].content}]")
        assert len(restored) == 1
        assert restored[0] == req

    async def test_total_tokens_persisted_per_model_response(self, resp_with_usage):
        """total_tokens is extracted from ModelResponse.usage for each response row.

        ModelRequests always get None. ModelResponses get usage.total_tokens when usage has values,
        None when usage is default/empty (e.g. make_response() with no token data).
        """
        resp_no_usage = make_response("second")  # default RequestUsage() — all zeros, has_values() False
        records = await self._persist_and_fetch(
            [make_request(), resp_with_usage, make_request(), resp_no_usage, make_request(), resp_with_usage]
        )
        assert records[0].total_tokens is None                                # ModelRequest
        assert records[1].total_tokens == resp_with_usage.usage.total_tokens  # ModelResponse with real usage
        assert records[2].total_tokens is None                                # ModelRequest
        assert records[3].total_tokens is None                                # ModelResponse with no usage values
        assert records[4].total_tokens is None                                # ModelRequest
        assert records[5].total_tokens == resp_with_usage.usage.total_tokens  # ModelResponse with real usage

    async def test_returns_last_seen_total_tokens_when_sequence_ends_with_nones(self, resp_with_usage):
        """When the sequence ends with messages that have no usage, the last seen non-None value is returned."""
        result = await persist_messages(
            self.deps,
            [resp_with_usage, make_request(), make_response()],
        )
        assert result == resp_with_usage.usage.total_tokens

    async def test_returns_last_total_tokens_when_multiple_responses_have_usage(self, resp_with_usage):
        """When multiple responses have usage data, the last one's value is returned, not the first."""
        resp_with_other_usage = ModelResponse(
            parts=[TextPart(content="later")],
            usage=RequestUsage(input_tokens=50, output_tokens=25),
        )
        result = await persist_messages(self.deps, [resp_with_usage, make_request(), resp_with_other_usage])
        assert result == resp_with_other_usage.usage.total_tokens

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
        records = await self._persist_and_fetch([])
        assert records == []

    @pytest.mark.parametrize("existing_count,new_count", [
        (0, 1),   # empty DB, single message → seq_id = 0
        (0, 3),   # empty DB, multiple messages → seq_ids = 0, 1, 2
        (1, 2),   # one existing (seq_id=0), add two → seq_ids = 1, 2
        (5, 3),   # five existing (seq_ids 0-4), add three → seq_ids = 5, 6, 7
    ])
    async def test_assigns_sequential_seq_ids(self, existing_count: int, new_count: int):
        """persist_messages assigns sequential seq_ids starting from MAX(existing) + 1."""
        # will be no op for the 0 case, helper returns empty list
        await persist_messages(self.deps, make_alternating_messages(existing_count, "existing"))
        await persist_messages(self.deps, make_alternating_messages(new_count, "new"))

        # Load all and verify seq_ids
        all_records = await load_messages(self.session, self.agent.id)
        assert len(all_records) == existing_count + new_count

        # All records should have sequential seq_ids starting from 0
        expected_seq_ids = list(range(existing_count + new_count))
        actual_seq_ids = [r.seq_id for r in all_records]
        assert actual_seq_ids == expected_seq_ids

    @pytest.mark.parametrize("pair_fn", [make_tool_pair, make_retry_pair])
    async def test_persists_tool_call_pair(self, pair_fn):
        """A matched tool-call / tool-response pair should survive persist unchanged,
        whether the response is a ToolReturnPart (success) or RetryPromptPart (ModelRetry)."""
        response_with_call, request_with_response = pair_fn()

        records = await self._persist_and_fetch([response_with_call, request_with_response])

        assert len(records) == 2
        assert records[0].type == "ModelResponse"
        assert records[1].type == "ModelRequest"
        restored = [ModelMessagesTypeAdapter.validate_json(f"[{r.content}]")[0] for r in records]
        assert restored == [response_with_call, request_with_response]

    async def test_records_isolated_per_agent(self):
        """
        Messages persisted for one agent must not affect another agent's records.
        Also verifies that new messages for a given agent do not affect that agent's old msgs.
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
        
        await persist_messages(other_deps, [expected_other_msg])
        await persist_messages(self.deps, [my_first_expected_msg])
        await persist_messages(self.deps, [my_second_expected_msg])

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
        records = await self._persist_and_fetch([orphan_msg])
        assert len(records) == 2  # positional error + summary warning
        assert [r.seq_id for r in records] == [0, 1]  # sequential including warning

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
            await persist_messages(self.deps, [good, bad, good2])

        records = await fetch_all_records(self.session, self.agent.id)
        assert len(records) == 4  # good, positional error, good2, summary warning
        assert [r.seq_id for r in records] == [0, 1, 2, 3]  # sequential including warning

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

# ---------------------------------------------------------------------------
# TestPersistMessagesSnapshots
# ---------------------------------------------------------------------------

@pytest.mark.asyncio
class TestPersistMessagesSnapshots(DBTestBase):
    """Tests for the snapshot and context-capture fields added to persist_messages."""

    async def _sys_snapshots(self):
        return (await self.session.execute(select(SystemPromptSnapshot))).scalars().all()

    async def _tool_snapshots(self):
        return (await self.session.execute(select(ToolDefinitionSnapshot))).scalars().all()

    async def _config_snapshots(self):
        return (await self.session.execute(select(AgentConfigSnapshot))).scalars().all()

    @pytest.mark.parametrize("prompt", ["You are a helpful test assistant.", ""])
    async def test_snapshot_stores_correct_system_prompt_content(self, prompt):
        """Snapshot row stores the exact system prompt string (including empty); record hash points to it."""
        self.agent.compiled_system_prompt = prompt
        records = await self._persist_and_fetch([make_request()])
        snap = (await self._sys_snapshots())[0]
        assert snap.content == prompt
        assert records[0].system_prompt_hash == snap.id

    @pytest.mark.parametrize("tool_schemas", [SAMPLE_TOOL_SCHEMAS, []])
    async def test_snapshot_stores_correct_tool_schema_content(self, tool_schemas):
        """Snapshot row stores the correct tool schemas (including empty list); record hash points to it."""
        records = await self._persist_and_fetch([make_request()], tool_schemas=tool_schemas)
        snap = (await self._tool_snapshots())[0]
        assert [ToolDefinition(**d) for d in json.loads(snap.content)] == tool_schemas
        assert records[0].tool_definition_hash == snap.id

    @pytest.mark.parametrize("second_prompt,expected_count", [
        ("prompt A", 1), ("prompt B", 2)
    ])
    async def test_system_prompt_snapshot_dedup(self, second_prompt, expected_count):
        """Same prompt reuses the existing snapshot row; a changed prompt creates a new one.
        Record hashes exactly match the set of snapshot IDs."""
        self.agent.compiled_system_prompt = "prompt A"
        await persist_messages(self.deps, [make_request()], SAMPLE_TOOL_SCHEMAS)
        self.agent.compiled_system_prompt = second_prompt
        await persist_messages(self.deps, [make_request()], SAMPLE_TOOL_SCHEMAS)
        snaps, records = await self._sys_snapshots(), await fetch_all_records(self.session, self.agent.id)
        assert len(snaps) == expected_count
        assert {r.system_prompt_hash for r in records} == {s.id for s in snaps}

    @pytest.mark.parametrize("second_schemas,expected_count", [
        (SAMPLE_TOOL_SCHEMAS, 1), (MUTATED_TOOL_SCHEMAS, 2)
    ])
    async def test_tool_schema_snapshot_dedup(self, second_schemas, expected_count):
        """Same schemas reuse the existing snapshot row; changed schemas create a new one.
        Record hashes exactly match the set of snapshot IDs."""
        await persist_messages(self.deps, [make_request()], SAMPLE_TOOL_SCHEMAS)
        await persist_messages(self.deps, [make_request()], second_schemas)
        snaps, records = await self._tool_snapshots(), await fetch_all_records(self.session, self.agent.id)
        assert len(snaps) == expected_count
        assert {r.tool_definition_hash for r in records} == {s.id for s in snaps}

    @pytest.mark.parametrize("n_messages", [1, 3])
    async def test_context_window_start_all_records_share_first_record_id(self, n_messages):
        """All records in one persist call share context_window_start_msg_id == first record's id.
        n=1 verifies the self-referential case; n=3 verifies the full-batch case."""
        records = await self._persist_and_fetch(make_alternating_messages(n_messages))
        assert all(r.context_window_start_msg_id == records[0].id for r in records)

    async def test_snapshot_stores_correct_agent_config_content(self):
        """Snapshot row stores the agent config as JSON that round-trips correctly; record hash points to it."""
        records = await self._persist_and_fetch([make_request()])
        snap = (await self._config_snapshots())[0]
        assert AgentConfig.model_validate_json(snap.content) == self.agent.agent_config
        assert records[0].agent_config_hash == snap.id

    @pytest.mark.parametrize("second_config,expected_count", [
        (SAMPLE_AGENT_CONFIG, 1), (MUTATED_AGENT_CONFIG, 2)
    ])
    async def test_agent_config_snapshot_dedup(self, second_config, expected_count):
        """Same config reuses the existing snapshot row; a changed config creates a new one.
        Record hashes exactly match the set of snapshot IDs."""
        await persist_messages(self.deps, [make_request()], SAMPLE_TOOL_SCHEMAS)
        self.agent.agent_config = second_config
        await persist_messages(self.deps, [make_request()], SAMPLE_TOOL_SCHEMAS)
        snaps, records = await self._config_snapshots(), await fetch_all_records(self.session, self.agent.id)
        assert len(snaps) == expected_count
        assert {r.agent_config_hash for r in records} == {s.id for s in snaps}

    async def test_context_window_start_stable_across_calls(self):
        """Subsequent persist calls all reference the UUID of the very first message ever persisted."""
        first_records = await self._persist_and_fetch([make_request(), make_response()])
        await persist_messages(self.deps, [make_request(), make_response()], SAMPLE_TOOL_SCHEMAS)
        all_records = await fetch_all_records(self.session, self.agent.id)
        assert all(r.context_window_start_msg_id == first_records[0].id for r in all_records)


# ---------------------------------------------------------------------------
# TestLoadMessages
# ---------------------------------------------------------------------------

@pytest.mark.asyncio
class TestLoadMessages(DBTestBase):
    """Tests for load_messages(session, agent_id, start_seq_id=0).

    Pre-seeds DB via persist_messages to ensure realistic records.
    TODO: Update with make_alternating fixture or whatever it was called
    """

    async def test_returns_all_messages_by_default(self):
        messages = [make_request(), make_response(), make_request(), make_response()]
        await persist_messages(self.deps, messages)

        records = await load_messages(self.session, self.agent.id)
        assert deserialize_messages(records) == messages

    async def test_returns_empty_list_when_no_messages(self):
        records = await load_messages(self.session, self.agent.id)
        assert records == []

    async def test_start_seq_id_filters_inclusive(self):
        first = [make_request("first"), make_response("first reply")]
        second = [make_request("second"), make_response("second reply")]
        await persist_messages(self.deps, first)
        await persist_messages(self.deps, second)

        all_records = await load_messages(self.session, self.agent.id)
        cutoff_seq_id = all_records[1].seq_id  # trim to second record (first reply) and on

        records = await load_messages(self.session, self.agent.id, start_seq_id=cutoff_seq_id)
        # Should include the cutoff record and everything after (inclusive)
        assert deserialize_messages(records) == [first[1]] + second

    async def test_end_seq_id_filters_exclusive(self):
        """end_seq_id excludes messages at or after that seq_id."""
        messages = [make_request("0"), make_response("1"), make_request("2"), make_response("3")]
        await persist_messages(self.deps, messages)

        all_records = await load_messages(self.session, self.agent.id)
        # Get messages 1 and 2 (exclusive of 0 and 3)
        records = await load_messages(
            self.session, self.agent.id,
            start_seq_id=all_records[1].seq_id,
            end_seq_id=all_records[3].seq_id,
        )
        assert deserialize_messages(records) == messages[1:3]

    async def test_results_in_seq_id_order(self):
        messages = [make_request("first"), make_response("second"), make_request("third")]
        await persist_messages(self.deps, messages)

        records = await load_messages(self.session, self.agent.id)
        seq_ids = [r.seq_id for r in records]
        assert seq_ids == sorted(seq_ids)

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

        await persist_messages(self.deps, [make_request(), make_response()])
        await persist_messages(other_deps, [make_request(), make_response()])

        records = await load_messages(self.session, self.agent.id)
        assert len(records) == 2
        assert all(r.agent_id == self.agent.id for r in records)

    async def test_returns_list_of_message_records(self):
        await persist_messages(self.deps, [make_request()])
        records = await load_messages(self.session, self.agent.id)
        assert len(records) == 1
        assert isinstance(records[0], MessageRecord)

    async def test_start_seq_id_ahead_of_all_returns_empty(self):
        """When start_seq_id is larger than every message's seq_id, the result is empty."""
        await persist_messages(self.deps, [make_request(), make_response()])

        records = await load_messages(self.session, self.agent.id, start_seq_id=999999)
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
        await persist_messages(self.deps, original)

        records = await load_messages(self.session, self.agent.id)
        restored = deserialize_messages(records)

        assert restored == original

    async def test_tool_pair_round_trip(self):
        response_with_call, request_with_return = make_tool_pair()
        await persist_messages(self.deps, [response_with_call, request_with_return])

        records = await load_messages(self.session, self.agent.id)
        restored = deserialize_messages(records)

        assert restored[0] == response_with_call
        assert restored[1] == request_with_return
