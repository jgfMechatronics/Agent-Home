"""
Agent Data Integrity Checker - Test Suite

Top-level function: check_agent_integrity(session, agent_id) -> list[IntegrityIssue]

=== CHECKS ===

1. SEQ_ID CONSECUTIVE ✓
   - seq_ids must be strictly consecutive integers (0, 1, 2, 3, 4...)
   - Gaps indicate lost messages, duplicate seq_ids indicate re-persistence bugs
   - First seq_id should be 0 (per persist_messages logic)
   - Out-of-order seq_ids (e.g., 0, 2, 1) indicate corruption
   - Severity: ERROR

2. TIMESTAMPS STRICTLY INCREASING ✓
   - Each MessageRecord timestamp must be > previous (by seq_id order)
   - Duplicate timestamps: likely re-persistence or clock issues
   - Out-of-order timestamps: definitely corruption
   - Severity: ERROR

(Item 3 removed)

4. DUPLICATE CONTENT DETECTION (hash-based, O(n)) ✓
   - Use hashlib.sha256 for stable canonical hashing (NOT hash(), which is process-unstable)
   - Track seen hashes with their seq_ids (don't overwrite after detection - keep original)
   
   KEY INSIGHT: Split adjacent vs non-adjacent duplicates (different failure modes):
   
   ADJACENT duplicates (consecutive seq_ids):
     - ALWAYS ERROR regardless of length
     - This is precisely the Haiku re-persistence pattern
     - "Got it" then "Got it" back-to-back is sus
   
   NON-ADJACENT duplicates:
     - Above LENGTH_THRESHOLD (35 chars): ERROR
     - Below threshold: WARN only if appears 3+ times (frequency threshold)
     - Same short response weeks apart is probably fine
   
   Note: Haiku experienced BOTH modes - first non-adjacent, then adjacent due to compounding bug

5. EMPTY/NULL CONTENT ✓
   - MessageRecord with null or empty content blob is always suspicious
   - Cheap to catch
   - Severity: ERROR
   - Include a deserializability check just for fun. Use our existing deserializer function which is....somewhere
   - NOTE: A ModelMessage with no parts, especially a ModelResopnse with no parts, is a known valid case. The concern here is a
   *MessageRecord* with empty or undeserializable content.

6. TOOL CALL/RETURN PAIRING ✓
   - ToolCallPart in ModelResponse should have matching ToolReturnPart by tool_call_id
   - Orphaned tool calls = incomplete turn
   - Matching on tool_call_id is more precise than just counting
   - Severity: ERROR for orphaned calls

7. MESSAGE ORDERING SANITY
   - ModelRequest -> ModelResponse is normal flow
   - Multiple ModelRequests in a row: actually normal (tool return + compaction resume, etc.)
   - Multiple ModelResponses in a row: unusual but can be valid (cancelled turn + retry)
   - Special cases: cancellation markers, compaction
   - May abandon if too noisy - part-level checks (tool pairing) may be more valuable
   - Severity: WARN for unusual patterns, INFO for multiple ModelRequests
   - James and Sonnet to do this one together after others are done, may be fraught with peril

8. CONTEXT_WINDOW_START_MSG_ID VALIDITY ✓
   - Must point to a message that exists
   - Must point to same seq_id or earlier (not forward references)
   - Severity: ERROR

(Removed check 9, 10)

=== TESTING APPROACH ===

Parametrized tests: each case is (poisoned_messages, expected_issues).
Test loads messages into DB via raw insert, calls top-level function, asserts exact equality.
"""

from typing import Callable
from uuid import uuid4

import pytest
import pytest_asyncio
from sqlalchemy.ext.asyncio import AsyncSession

from datetime import datetime

from pydantic_ai.messages import ModelRequest, ModelResponse, RetryPromptPart, TextPart, ThinkingPart, ToolCallPart, ToolReturnPart

from conftest import PARTIAL_MESSAGE_FIELDS, make_request, make_response, make_retry_pair, make_tool_pair
from messages.messages import dump_msg_json
from db.models import AgentRecord, MessageRecord
from utils.integrity_checker import CRITICAL, ERROR, WARN, IntegrityIssue, Severity, check_agent_integrity


# ---------------------------------------------------------------------------
# Message Record Factory (local to integrity tests)
# ---------------------------------------------------------------------------

def make_message_record(agent_id: str, *, seq_id: int, **overrides) -> MessageRecord:
    """Construct a MessageRecord with realistic defaults.

    Requires seed_stub_snapshots fixture to be active (for FK satisfaction on snapshot hash fields).
    - type alternates ModelRequest/ModelResponse by seq_id (even=request, odd=response)
    - content is a properly serialized ModelMessage with random UUID text to ensure uniqueness in default case
    - timestamp defaults to datetime(2026, 1, 1, 12, 0, seq_id)
    - context_window_start_msg_id defaults to the record's own id (self-reference, always valid
      for check 8). Override to test specific good/bad values; make_message_sequence overrides
      it to the first record's id for realistic multi-record sequences.
    """
    record_id = overrides.get("id", str(uuid4()))
    msg = make_request(str(uuid4())) if seq_id % 2 == 0 else make_response(str(uuid4()))
    defaults = {
        "id": record_id,
        "agent_id": agent_id,
        "seq_id": seq_id,
        "timestamp": datetime(2026, 1, 1, 12, 0, seq_id),
        **PARTIAL_MESSAGE_FIELDS,
        "context_window_start_msg_id": record_id,  # self-reference; overrides STUB_CTX_MSG_ID in PARTIAL_MESSAGE_FIELDS
        "type": type(msg).__name__,
        "content": dump_msg_json(msg),
    }
    return MessageRecord(**{**defaults, **overrides})


# Type alias for the callable that builds records given an agent_id
RecordBuilder = Callable[[str], list[MessageRecord]]


def make_message_sequence(agent_id: str, overrides_list: list[dict]) -> list[MessageRecord]:
    """Create messages with auto-assigned consecutive seq_ids (0, 1, 2...).

    Each record self-references as context_window_start_msg_id by default (always
    valid for ctx window start id checks). Override in individual entries to test specific values.
    """
    return [
        make_message_record(agent_id, seq_id=i, **overrides)
        for i, overrides in enumerate(overrides_list)
    ]


# ---------------------------------------------------------------------------
# Test Cases (parametrized)
# ---------------------------------------------------------------------------

# Each test case: (build_records callable, expected_issues list)
# build_records takes agent_id and returns list of MessageRecords to insert

# Universal happy path — shared across all check categories via CLEAN_TEST_CASES below.
# Verifies that a well-formed sequence produces no issues regardless of which check runs.
CLEAN_TEST_CASES = [
    pytest.param(
        lambda agent_id: make_message_sequence(agent_id, [{}, {}, {}, {}]),
        [],
        id="clean_no_issues",
    ),
    # Empty part content should never be flagged as a duplicate regardless of frequency
    pytest.param(
        lambda agent_id: make_message_sequence(agent_id, [
            {**_EMPTY_DUP},
            {**_EMPTY_DUP},  # adjacent "duplicate" — empty content, never flag
            {**_EMPTY_DUP},  # third occurrence — still never flag
        ]),
        [],
        id="empty_part_content_not_flagged",
    ),
]

# NOTE: No non-adjacent duplicate test for seq_id — after ORDER BY seq_id, duplicates
# are always adjacent. Non-adjacent duplicates in insertion order become adjacent after load.
SEQ_ID_TEST_CASES = [
    pytest.param(
        lambda agent_id: [],
        [],
        id="empty_list",
    ),
    pytest.param(
        lambda agent_id: [make_message_record(agent_id, seq_id=0)],
        [],
        id="single_record",
    ),
    pytest.param(
        lambda agent_id: [
            make_message_record(agent_id, seq_id=0),
            make_message_record(agent_id, seq_id=1),
            make_message_record(agent_id, seq_id=3),  # gap: missing 2
        ],
        [IntegrityIssue(
            check_type="seq_id_gap",
            severity=ERROR,
            seq_ids=[1, 3],
            details="Gap in seq_ids: expected 2, got 3 (missing 2 through 2)",
        )],
        id="seq_id_gap",
    ),
    pytest.param(
        lambda agent_id: [
            make_message_record(agent_id, seq_id=0, timestamp=datetime(2026, 1, 1, 12, 0, 0)),
            make_message_record(agent_id, seq_id=1, timestamp=datetime(2026, 1, 1, 12, 0, 1)),
            make_message_record(agent_id, seq_id=1, timestamp=datetime(2026, 1, 1, 12, 0, 2)),  # duplicate seq_id, distinct timestamp
            make_message_record(agent_id, seq_id=2, timestamp=datetime(2026, 1, 1, 12, 0, 3)),
        ],
        [IntegrityIssue(
            check_type="seq_id_duplicate",
            severity=ERROR,
            seq_ids=[1],
            details="Duplicate seq_id 1 at positions 1 and 2",
        )],
        id="seq_id_duplicate_adjacent",
    ),

    pytest.param(
        lambda agent_id: [
            make_message_record(agent_id, seq_id=1),  # missing 0
            make_message_record(agent_id, seq_id=2),
            make_message_record(agent_id, seq_id=3),
        ],
        [IntegrityIssue(
            check_type="seq_id_gap",
            severity=ERROR,
            seq_ids=[1],
            details="Gap in seq_ids: expected 0, got 1 (missing 0 through 0)",
        )],
        id="missing_initial_seq_id",
    ),
]


# ---------------------------------------------------------------------------
# Check 2: Timestamp Test Cases
# ---------------------------------------------------------------------------

TIMESTAMP_TEST_CASES = [
    pytest.param(
        lambda agent_id: make_message_sequence(agent_id, [
            {"timestamp": datetime(2026, 1, 1, 12, 0, 0)},
            {"timestamp": datetime(2026, 1, 1, 12, 0, 0)},  # duplicate timestamp
            {"timestamp": datetime(2026, 1, 1, 12, 0, 1)},
        ]),
        [IntegrityIssue(
            check_type="timestamp_duplicate",
            severity=ERROR,
            seq_ids=[0, 1],
            details="Duplicate timestamp at seq_ids 0 and 1",
        )],
        id="duplicate_timestamp_adjacent",
    ),
    pytest.param(
        lambda agent_id: make_message_sequence(agent_id, [
            {"timestamp": datetime(2026, 1, 1, 12, 0, 0)},
            {"timestamp": datetime(2026, 1, 1, 12, 0, 1)},
            {"timestamp": datetime(2026, 1, 1, 12, 0, 2)},
            {"timestamp": datetime(2026, 1, 1, 12, 0, 3)},
            {"timestamp": datetime(2026, 1, 1, 12, 0, 4)},
            {"timestamp": datetime(2026, 1, 1, 12, 0, 1)},  # non-adjacent duplicate — caught as out-of-order
        ]),
        [IntegrityIssue(
            check_type="timestamp_out_of_order",
            severity=ERROR,
            seq_ids=[4, 5],
            details=(
                "Timestamp out of order at seq_ids 4 → 5: "
                "2026-01-01 12:00:04 → 2026-01-01 12:00:01. "
                "Possible causes: insertion order bug, clock skew, or re-persisted duplicate."
            ),
        )],
        id="duplicate_timestamp_non_adjacent",
    ),
    pytest.param(
        lambda agent_id: make_message_sequence(agent_id, [
            {"timestamp": datetime(2026, 1, 1, 12, 0, 0)},
            {"timestamp": datetime(2026, 1, 1, 12, 0, 2)},
            {"timestamp": datetime(2026, 1, 1, 12, 0, 1)},  # goes backward
        ]),
        [IntegrityIssue(
            check_type="timestamp_out_of_order",
            severity=ERROR,
            seq_ids=[1, 2],
            details=(
                "Timestamp out of order at seq_ids 1 → 2: "
                "2026-01-01 12:00:02 → 2026-01-01 12:00:01. "
                "Possible causes: insertion order bug, clock skew, or re-persisted duplicate."
            ),
        )],
        id="timestamp_inversion",
    ),
]


# ---------------------------------------------------------------------------
# Check 3: Content Duplicate Test Cases
# ---------------------------------------------------------------------------

_CONTENT_LENGTH_THRESHOLD = 35  # must match the threshold in the impl

# Pre-baked overrides dicts for use in make_message_sequence.
# Each is computed once so both records get identical serialized content (simulating re-persistence).
# UserPromptPart duplicates (ModelRequest)
_EMPTY_DUP = {"type": "ModelRequest", "content": dump_msg_json(make_request(""))}
_SHORT_DUP = {"type": "ModelRequest", "content": dump_msg_json(make_request("ok"))}
_LONG_DUP  = {"type": "ModelRequest", "content": dump_msg_json(make_request("x" * (_CONTENT_LENGTH_THRESHOLD + 1)))}

# ThinkingPart duplicates (ModelResponse with a thinking block)
def _make_thinking_response(thinking: str) -> ModelResponse:
    return ModelResponse(parts=[ThinkingPart(content=thinking)])


def _make_thinking_and_text_response(thinking: str, text: str) -> ModelResponse:
    return ModelResponse(parts=[ThinkingPart(content=thinking), TextPart(content=text)])

_LONG_THINKING_DUP = {"type": "ModelResponse", "content": dump_msg_json(_make_thinking_response("x" * (_CONTENT_LENGTH_THRESHOLD + 1)))}


CONTENT_DUPLICATE_TEST_CASES = [
    # Adjacent duplicate — always ERROR, regardless of content length
    pytest.param(
        lambda agent_id: make_message_sequence(agent_id, [
            {**_LONG_DUP},
            {**_LONG_DUP},  # adjacent duplicate
            {},
        ]),
        [IntegrityIssue(
            check_type="content_duplicate",
            severity=ERROR,
            seq_ids=[0, 1],
            details=(
                "Duplicate content found in adjacent messages. "
                "Adjacent duplication is unlikely to naturally occur. "
                "Duplication occurred at seq_ids: [0, 1]"
            ),
        )],
        id="adjacent_duplicate_long",
    ),
    # Adjacent duplicate, short content — still ERROR (regardless of length)
    pytest.param(
        lambda agent_id: make_message_sequence(agent_id, [
            {**_SHORT_DUP},
            {**_SHORT_DUP},  # adjacent duplicate
            {},
        ]),
        [IntegrityIssue(
            check_type="content_duplicate",
            severity=ERROR,
            seq_ids=[0, 1],
            details=(
                "Duplicate content found in adjacent messages. "
                "Adjacent duplication is unlikely to naturally occur. "
                "Duplication occurred at seq_ids: [0, 1]"
            ),
        )],
        id="adjacent_duplicate_short",
    ),
    # Non-adjacent, long content — ERROR
    pytest.param(
        lambda agent_id: make_message_sequence(agent_id, [
            {**_LONG_DUP},
            {},  # unique x4
            {},
            {},
            {},
            {**_LONG_DUP},  # non-adjacent duplicate
        ]),
        [IntegrityIssue(
            check_type="content_duplicate",
            severity=ERROR,
            seq_ids=[0, 5],
            details=(
                "High length duplicate content detected. "
                "Higher length content is less likely to naturally recur. "
                "Duplication occurred at seq_ids: [0, 5]"
            ),
        )],
        id="non_adjacent_duplicate_long",
    ),
    # Non-adjacent, short content, 2 occurrences — no issue (threshold is 3+)
    pytest.param(
        lambda agent_id: make_message_sequence(agent_id, [
            {**_SHORT_DUP},
            {},              # unique middle
            {**_SHORT_DUP},  # 2nd occurrence — below frequency threshold
        ]),
        [],
        id="non_adjacent_duplicate_short_2x_no_issue",
    ),
    # Non-adjacent, short content, 3 occurrences — WARN
    pytest.param(
        lambda agent_id: make_message_sequence(agent_id, [
            {**_SHORT_DUP},
            {},              # unique
            {**_SHORT_DUP},  # 2nd
            {},              # unique
            {**_SHORT_DUP},  # 3rd — crosses frequency threshold
        ]),
        [IntegrityIssue(
            check_type="content_duplicate",
            severity=WARN,
            seq_ids=[0, 2, 4],
            details=(
                "Short length duplicate content detected with suspect frequency. "
                "Duplication occurred at seq_ids: [0, 2, 4]"
            ),
        )],
        id="non_adjacent_duplicate_short_3x_warn",
    ),
    # ThinkingPart: adjacent duplicate — verifies thinking block content is inspected at all
    pytest.param(
        lambda agent_id: make_message_sequence(agent_id, [
            {**_LONG_THINKING_DUP},
            {**_LONG_THINKING_DUP},  # adjacent duplicate
            {},
        ]),
        [IntegrityIssue(
            check_type="content_duplicate",
            severity=ERROR,
            seq_ids=[0, 1],
            details=(
                "Duplicate content found in adjacent messages. "
                "Adjacent duplication is unlikely to naturally occur. "
                "Duplication occurred at seq_ids: [0, 1]"
            ),
        )],
        id="adjacent_duplicate_thinking",
    ),
    # Duplicate thinking part across messages that also have unique text parts.
    # A full-message hash would see two different messages and miss this entirely —
    # verifies that part-by-part inspection is what's actually running.
    pytest.param(
        lambda agent_id: make_message_sequence(agent_id, [
            {
                "type": "ModelResponse",
                "content": dump_msg_json(_make_thinking_and_text_response(
                    "x" * (_CONTENT_LENGTH_THRESHOLD + 1), str(uuid4()),
                )),
            },
            {
                "type": "ModelResponse",
                "content": dump_msg_json(_make_thinking_and_text_response(
                    "x" * (_CONTENT_LENGTH_THRESHOLD + 1), str(uuid4()),
                )),
            },
        ]),
        [IntegrityIssue(
            check_type="content_duplicate",
            severity=ERROR,
            seq_ids=[0, 1],
            details=(
                "Duplicate content found in adjacent messages. "
                "Adjacent duplication is unlikely to naturally occur. "
                "Duplication occurred at seq_ids: [0, 1]"
            ),
        )],
        id="adjacent_duplicate_thinking_with_unique_text",
    ),
]


# ---------------------------------------------------------------------------
# Check 5: Empty / Null Content
# ---------------------------------------------------------------------------

EMPTY_CONTENT_TEST_CASES = [
    # Empty string content — always suspicious, regardless of position
    pytest.param(
        lambda agent_id: make_message_sequence(agent_id, [
            {},
            {"content": ""},  # empty content blob
            {},
        ]),
        [IntegrityIssue(
            check_type="empty_content",
            severity=ERROR,
            seq_ids=[1],
            details="MessageRecord at seq_id 1 has empty or null content",
        )],
        id="empty_content",
    ),
]


# ---------------------------------------------------------------------------
# Deserialization Failure (gating step in check_agent_integrity)
# ---------------------------------------------------------------------------

_UNDESERIALIZABLE_RECORD_ID = "bbbbbbbb-bbbb-bbbb-bbbb-bbbbbbbbbbbb"

DESERIALIZATION_FAILURE_TEST_CASES = [
    # Content present but not valid JSON — ID fixed so details string is deterministic
    pytest.param(
        lambda agent_id: make_message_sequence(agent_id, [
            {},
            {"content": "this is not valid json", "id": _UNDESERIALIZABLE_RECORD_ID},
            {},
        ]),
        [IntegrityIssue(
            check_type="deserialization_failure",
            severity=CRITICAL,
            seq_ids=[1],
            details=(
                f"Failed to deserialize message at seq_id 1 (id: {_UNDESERIALIZABLE_RECORD_ID}). "
                f"This is the first failure encountered — there may be more. "
                f"Checks requiring deserialization have been skipped. "
                f"Fix or remove record with undeserializable content and re-run."
            ),
        )],
        id="deserialization_failure",
    ),
]


# ---------------------------------------------------------------------------
# Check 6: Tool Call / Return Pairing
# ---------------------------------------------------------------------------

TOOL_PAIRING_TEST_CASES = [
    # Orphaned ToolCallPart — call made, no matching return in history
    pytest.param(
        lambda agent_id: make_message_sequence(agent_id, [
            {},
            {"type": "ModelResponse", "content": dump_msg_json(make_tool_pair()[0])},
            {},  # plain UserPromptPart, not a ToolReturnPart
        ]),
        [IntegrityIssue(
            check_type="orphaned_tool_call",
            severity=ERROR,
            seq_ids=[1],
            details="ModelResponse at seq_id 1 has ToolCallPart(s) with no matching return in the following message",
        )],
        id="orphaned_tool_call",
    ),
    # Orphaned ToolReturnPart — return present but no matching call in history
    pytest.param(
        lambda agent_id: make_message_sequence(agent_id, [
            {},
            {"type": "ModelRequest", "content": dump_msg_json(make_tool_pair()[1])},
            {},
        ]),
        [IntegrityIssue(
            check_type="orphaned_tool_return",
            severity=ERROR,
            seq_ids=[1],
            details="ModelRequest at seq_id 1 has ToolReturnPart(s) with no matching call in the preceding message",
        )],
        id="orphaned_tool_return",
    ),
    # Clean: matched tool call / return pair
    pytest.param(
        lambda agent_id: make_message_sequence(agent_id, [
            {},
            {"type": "ModelResponse", "content": dump_msg_json(make_tool_pair()[0])},
            {"type": "ModelRequest", "content": dump_msg_json(make_tool_pair()[1])},
            {},
        ]),
        [],
        id="matched_tool_pair",
    ),
    # Clean: retry pair (RetryPromptPart counts as a valid match for a ToolCallPart)
    pytest.param(
        lambda agent_id: make_message_sequence(agent_id, [
            {},
            {"type": "ModelResponse", "content": dump_msg_json(make_retry_pair()[0])},
            {"type": "ModelRequest", "content": dump_msg_json(make_retry_pair()[1])},
            {},
        ]),
        [],
        id="matched_retry_pair",
    ),
]


# ---------------------------------------------------------------------------
# Check 8: context_window_start_msg_id Validity
# ---------------------------------------------------------------------------

def _forward_ctx_ref_sequence(agent_id: str) -> list[MessageRecord]:
    """Seq_id 0 pointing to the ID of seq_id 1 — a forward reference."""
    later_id = str(uuid4())
    return make_message_sequence(agent_id, [
        {"context_window_start_msg_id": later_id},  # forward reference to seq_id 1
        {"id": later_id},
        {},
    ])


CTX_WINDOW_START_TEST_CASES = [
    # Nonexistent ID — points to a message that isn't in the agent's history
    pytest.param(
        lambda agent_id: make_message_sequence(agent_id, [
            {},
            {"context_window_start_msg_id": str(uuid4())},  # random UUID, not in records
            {},
        ]),
        [IntegrityIssue(
            check_type="invalid_context_window_start",
            severity=ERROR,
            seq_ids=[1],
            details="MessageRecord at seq_id 1 has context_window_start_msg_id pointing to a nonexistent message",
        )],
        id="nonexistent_context_window_start",
    ),
    # Forward reference — context_window_start points to a later message (seq_id 0 → seq_id 1)
    pytest.param(
        _forward_ctx_ref_sequence,
        [IntegrityIssue(
            check_type="invalid_context_window_start",
            severity=ERROR,
            seq_ids=[0],
            details="MessageRecord at seq_id 0 has context_window_start_msg_id pointing to a future message at seq_id 1",
        )],
        id="forward_context_window_start",
    ),
]


# ---------------------------------------------------------------------------
# TestCheckAgentIntegrity
# ---------------------------------------------------------------------------

@pytest.mark.asyncio
@pytest.mark.usefixtures("seed_stub_snapshots")
class TestCheckAgentIntegrity:
    """Top-down tests for check_agent_integrity(session, agent_id)."""

    @pytest_asyncio.fixture(autouse=True)
    async def setup(self, session: AsyncSession, agent_record: AgentRecord):
        self.session = session
        self.agent = agent_record

    async def _run_check(self, build_records: RecordBuilder, expected_issues: list[IntegrityIssue]):
        """Common test body for all check categories."""
        records = build_records(self.agent.id)
        self.session.add_all(records)
        await self.session.flush()
        issues = await check_agent_integrity(self.session, self.agent.id)
        assert issues == expected_issues

    # The below structure gives us a nice heirerchy in test explorer and isolates failures to particular param lists better
    @pytest.mark.parametrize("build_records,expected_issues", CLEAN_TEST_CASES)
    async def test_clean(self, build_records: RecordBuilder, expected_issues: list[IntegrityIssue]):
        await self._run_check(build_records, expected_issues)

    @pytest.mark.parametrize("build_records,expected_issues", SEQ_ID_TEST_CASES)
    async def test_seq_id(self, build_records: RecordBuilder, expected_issues: list[IntegrityIssue]):
        await self._run_check(build_records, expected_issues)

    @pytest.mark.parametrize("build_records,expected_issues", TIMESTAMP_TEST_CASES)
    async def test_timestamp(self, build_records: RecordBuilder, expected_issues: list[IntegrityIssue]):
        await self._run_check(build_records, expected_issues)

    @pytest.mark.parametrize("build_records,expected_issues", CONTENT_DUPLICATE_TEST_CASES)
    async def test_content_duplicate(self, build_records: RecordBuilder, expected_issues: list[IntegrityIssue]):
        await self._run_check(build_records, expected_issues)

    @pytest.mark.parametrize("build_records,expected_issues", EMPTY_CONTENT_TEST_CASES)
    async def test_empty_content(self, build_records: RecordBuilder, expected_issues: list[IntegrityIssue]):
        await self._run_check(build_records, expected_issues)

    @pytest.mark.parametrize("build_records,expected_issues", DESERIALIZATION_FAILURE_TEST_CASES)
    async def test_deserialization_failure(self, build_records: RecordBuilder, expected_issues: list[IntegrityIssue]):
        await self._run_check(build_records, expected_issues)

    @pytest.mark.parametrize("build_records,expected_issues", TOOL_PAIRING_TEST_CASES)
    async def test_tool_pairing(self, build_records: RecordBuilder, expected_issues: list[IntegrityIssue]):
        await self._run_check(build_records, expected_issues)

    @pytest.mark.parametrize("build_records,expected_issues", CTX_WINDOW_START_TEST_CASES)
    async def test_ctx_window_start(self, build_records: RecordBuilder, expected_issues: list[IntegrityIssue]):
        await self._run_check(build_records, expected_issues)


def test_checkers_units_do_not_mutate_input():
    # TODO: Once all internal checker functions exist, parametrize this test on them.
    # For each checker: create records, snapshot state, run checker, assert unchanged.
    # This enforces the invariant that checkers are pure functions.
    pytest.fail("Not yet implemented — add after all checkers exist")
