"""
Agent Data Integrity Checker - Test Suite
"""

from typing import Callable
from uuid import uuid4

import copy
from dataclasses import asdict
import json
import pytest
import pytest_asyncio
from sqlalchemy.ext.asyncio import AsyncSession

from datetime import datetime

from pydantic_ai.messages import ModelResponse, TextPart, ThinkingPart

from conftest import PARTIAL_MESSAGE_FIELDS, make_request, make_response, make_retry_pair, make_tool_pair
from messages.messages import dump_msg_json, format_system_alert
from db.models import AgentRecord, MessageRecord
from messages.messages import deserialize_messages
from utils.integrity_checker import (
    CRITICAL, ERROR, WARN, IntegrityIssue, Severity, check_agent_integrity,
    _check_seq_id_consecutive, _check_timestamps_increasing, _check_context_window_start_validity,
    _check_message_ordering, _check_for_empty_content,
    _check_tool_call_return_pairing, _check_for_duplicate_content,
    Dismissal, filter_dismissed_issues, load_dismissals,
)


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
    override_type = overrides.get("type")

    if override_type == "ModelRequest":
        msg = make_request(str(uuid4()))
    elif override_type == "ModelResponse":
        msg = make_response(str(uuid4()))
    else:
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
    pytest.param(
        lambda agent_id: make_message_sequence(agent_id, [{}]),
        [],
        id="clean_single_msg",
    ),
    pytest.param(
        lambda agent_id: [],
        [],
        id="clean_no_msg"
    )
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
            # gap: missing 2; type forced to avoid accidental adjacent responses
            make_message_record(agent_id, seq_id=3, type="ModelRequest"),
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
            make_message_record(agent_id, seq_id=1, timestamp=datetime(2026, 1, 1, 12, 0, 1), type="ModelRequest"),
            make_message_record(agent_id, seq_id=1, timestamp=datetime(2026, 1, 1, 12, 0, 2), type="ModelRequest"),  # duplicate seq_id, distinct timestamp; types forced to avoid accidental adjacent responses
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

_ALERT_TEXT = format_system_alert("Unexpected big chungus detected")
_SYSTEM_ALERT = {"type": "ModelRequest", "content": dump_msg_json(make_request(_ALERT_TEXT))}


# ThinkingPart duplicates (ModelResponse with a thinking block)
def _make_thinking_response(thinking: str) -> ModelResponse:
    return ModelResponse(parts=[ThinkingPart(content=thinking)])


def _make_thinking_and_text_response(thinking: str, text: str) -> ModelResponse:
    return ModelResponse(parts=[ThinkingPart(content=thinking), TextPart(content=text)])


_LONG_THINKING_DUP = {"type": "ModelResponse", "content": dump_msg_json(_make_thinking_response("x" * (_CONTENT_LENGTH_THRESHOLD + 1)))}

# Pre-built matched pairs — computed once so call and return share the same tool_call_id
_MATCHED_TOOL_CALL, _MATCHED_TOOL_RETURN = make_tool_pair()
_MATCHED_RETRY_CALL, _MATCHED_RETRY_RETURN = make_retry_pair()


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
        [
            # Hard to avoid tripping this one while testing the adjacent dupe detection
            IntegrityIssue(
                check_type="adjacent_model_responses",
                severity=ERROR,
                seq_ids=[0, 1],
                details="Adjacent ModelResponses at seq_ids 0 and 1 — consecutive responses should not occur",
            ),
            IntegrityIssue(
                check_type="content_duplicate",
                severity=ERROR,
                seq_ids=[0, 1],
                details=(
                    "Duplicate content found in adjacent messages. "
                    "Adjacent duplication is unlikely to naturally occur. "
                    "Duplication occurred at seq_ids: [0, 1]"
                ),
            ),
        ],
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
        [
            IntegrityIssue(
                check_type="adjacent_model_responses",
                severity=ERROR,
                seq_ids=[0, 1],
                details="Adjacent ModelResponses at seq_ids 0 and 1 — consecutive responses should not occur",
            ),
            IntegrityIssue(
                check_type="content_duplicate",
                severity=ERROR,
                seq_ids=[0, 1],
                details=(
                    "Duplicate content found in adjacent messages. "
                    "Adjacent duplication is unlikely to naturally occur. "
                    "Duplication occurred at seq_ids: [0, 1]"
                ),
            ),
        ],
        id="adjacent_duplicate_thinking_with_unique_text",
    ),
    pytest.param(
        lambda agent_id: make_message_sequence(agent_id, [{**_SYSTEM_ALERT}, {}, {**_SYSTEM_ALERT},{}]),
        [],
        id="check_system_alert_allowed"        
    )
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
            {"type": "ModelResponse", "content": dump_msg_json(_MATCHED_TOOL_CALL)},
            {"type": "ModelRequest", "content": dump_msg_json(_MATCHED_TOOL_RETURN)},
            {},
        ]),
        [],
        id="matched_tool_pair",
    ),
    # Clean: retry pair (RetryPromptPart counts as a valid match for a ToolCallPart)
    pytest.param(
        lambda agent_id: make_message_sequence(agent_id, [
            {},
            {"type": "ModelResponse", "content": dump_msg_json(_MATCHED_RETRY_CALL)},
            {"type": "ModelRequest", "content": dump_msg_json(_MATCHED_RETRY_RETURN)},
            {},
        ]),
        [],
        id="matched_retry_pair",
    ),
    # Structurally valid but mismatched tool_call_ids — call and return are from different pairs.
    # is_valid_tool_pair currently only checks structure, not IDs, so this slips through.
    pytest.param(
        lambda agent_id: make_message_sequence(agent_id, [
            {},
            {"type": "ModelResponse", "content": dump_msg_json(make_tool_pair()[0])},  # call id A
            {"type": "ModelRequest", "content": dump_msg_json(make_tool_pair()[1])},   # return id B — mismatch
            {},
        ]),
        [
            IntegrityIssue(
                check_type="orphaned_tool_call",
                severity=ERROR,
                seq_ids=[1],
                details="ModelResponse at seq_id 1 has ToolCallPart(s) with no matching return in the following message",
            ),
            IntegrityIssue(
                check_type="orphaned_tool_return",
                severity=ERROR,
                seq_ids=[2],
                details="ModelRequest at seq_id 2 has ToolReturnPart(s) with no matching call in the preceding message",
            ),
        ],
        id="mismatched_tool_call_ids",
    ),
]


# ---------------------------------------------------------------------------
# Check 7: Message Ordering
# ---------------------------------------------------------------------------

MESSAGE_ORDERING_TEST_CASES = [
    # Adjacent ModelResponses — never legitimate
    pytest.param(
        lambda agent_id: make_message_sequence(agent_id, [
            {"type": "ModelResponse"},
            {"type": "ModelResponse"},
            {},
        ]),
        [IntegrityIssue(
            check_type="adjacent_model_responses",
            severity=ERROR,
            seq_ids=[0, 1],
            details="Adjacent ModelResponses at seq_ids 0 and 1 — consecutive responses should not occur",
        )],
        id="adjacent_model_responses",
    ),
    # Consecutive ModelRequests — legitimate (compaction resume, cancel notice, etc.)
    pytest.param(
        lambda agent_id: make_message_sequence(agent_id, [
            {},
            {"type": "ModelRequest"},
            {},
        ]),
        [],
        id="consecutive_model_requests_clean",
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

_RECORD_ONLY_CHECKERS = [
    pytest.param(_check_seq_id_consecutive, id="check_seq_id_consecutive"),
    pytest.param(_check_timestamps_increasing, id="check_timestamps_increasing"),
    pytest.param(_check_context_window_start_validity, id="check_context_window_start_validity"),
    pytest.param(_check_message_ordering, id="check_message_ordering"),
    pytest.param(_check_for_empty_content, id="check_for_empty_content"),
]

_RECORD_AND_MESSAGE_CHECKERS = [
    pytest.param(_check_tool_call_return_pairing, id="check_tool_call_return_pairing"),
    pytest.param(_check_for_duplicate_content, id="check_for_duplicate_content"),
]


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

    # The below structure gives us a nice hierarchy in test explorer and isolates failures to particular param lists better
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

    @pytest.mark.parametrize("build_records,expected_issues", MESSAGE_ORDERING_TEST_CASES)
    async def test_message_ordering(self, build_records: RecordBuilder, expected_issues: list[IntegrityIssue]):
        await self._run_check(build_records, expected_issues)

    @pytest.mark.parametrize("build_records,expected_issues", CTX_WINDOW_START_TEST_CASES)
    async def test_ctx_window_start(self, build_records: RecordBuilder, expected_issues: list[IntegrityIssue]):
        await self._run_check(build_records, expected_issues)

    @pytest.mark.parametrize("checker", _RECORD_ONLY_CHECKERS)
    async def test_record_only_checkers_do_not_mutate_input(self, checker):
        records = make_message_sequence(self.agent.id, [{}, {}, {}, {}])
        snapshot = copy.deepcopy(records)
        checker(records)

        # No native __eq__ for Sqlalchemy ORM, have to do some funny stuff
        for r, s in zip(records, snapshot):
            s._sa_instance_state = r._sa_instance_state
        assert [vars(r) for r in records] == [vars(s) for s in snapshot]

    @pytest.mark.parametrize("checker", _RECORD_AND_MESSAGE_CHECKERS)
    async def test_record_and_message_checkers_do_not_mutate_input(self, checker):
        records = make_message_sequence(self.agent.id, [{}, {}, {}, {}])
        messages = deserialize_messages(records)
        records_snapshot, messages_snapshot = copy.deepcopy(records), copy.deepcopy(messages)
        checker(records, messages)
        for r, s in zip(records, records_snapshot):
            s._sa_instance_state = r._sa_instance_state
        assert [vars(r) for r in records] == [vars(s) for s in records_snapshot]
        assert messages == messages_snapshot


# Reusable fixtures for dismissal tests
_ISSUE_A = IntegrityIssue(check_type="adjacent_duplicate", severity=WARN, seq_ids=[111, 396], details="test A")
_ISSUE_B = IntegrityIssue(check_type="seq_id_gap", severity=ERROR, seq_ids=[5, 7], details="test B")
_DISMISSAL_A = Dismissal(check_type="adjacent_duplicate", seq_ids=[111, 396], reason="Known false positive")

_FILTER_TEST_CASES = [
    # input issues, dismissals, expected filter result
    pytest.param([_ISSUE_A], [_DISMISSAL_A], [], id="matching_dismissal_filters"),
    pytest.param([_ISSUE_A, _ISSUE_B], [_DISMISSAL_A], [_ISSUE_B], id="selective_filtering_keeps_unrelated"),
    pytest.param([_ISSUE_A, _ISSUE_B], [], [_ISSUE_A, _ISSUE_B], id="empty_dismissals_keeps_all"),
    pytest.param([], [_DISMISSAL_A], [], id="empty_issues_returns_empty"),
]

class TestFilterDismissedIssues:
    """Tests for filter_dismissed_issues()."""

    @pytest.mark.parametrize("issues,dismissals,expected", _FILTER_TEST_CASES)
    def test_filter_dismissed_issues(self, issues, dismissals, expected):
        """Filter removes matching issues and keeps others."""
        assert filter_dismissed_issues(issues, dismissals) == expected

    @pytest.mark.parametrize("dismissal", [
        pytest.param(
            Dismissal(check_type="wrong_type", seq_ids=[111, 396], reason="x"),
            id="wrong_check_type",
        ),
        pytest.param(
            Dismissal(check_type="adjacent_duplicate", seq_ids=[111, 397], reason="x"),
            id="wrong_seq_ids",
        ),
        pytest.param(
            Dismissal(check_type="adjacent_duplicate", seq_ids=[396, 111], reason="x"),
            id="reversed_seq_ids",
        ),
    ])
    def test_non_matching_dismissal_keeps_issue(self, dismissal: Dismissal):
        """Issues are kept when dismissal doesn't match exactly."""
        result = filter_dismissed_issues([_ISSUE_A], [dismissal])
        assert result == [_ISSUE_A]


class TestLoadDismissals:
    """Tests for load_dismissals()."""

    _AGENT_ID = "test-agent-id"

    def test_loads_valid_file(self, tmp_path):
        """Parses JSON file into Dismissal objects for the specified agent."""
        config = {self._AGENT_ID: [asdict(_DISMISSAL_A)]}
        path = tmp_path / "dismissals.json"
        path.write_text(json.dumps(config))
        
        result = load_dismissals(path, self._AGENT_ID)
        
        assert result == [_DISMISSAL_A]

    def test_missing_file_returns_empty(self, tmp_path):
        """Non-existent file returns empty list (not an error)."""
        result = load_dismissals(tmp_path / "nonexistent.json", self._AGENT_ID)
        assert result == []

    def test_agent_not_in_file_returns_empty(self, tmp_path):
        """Agent with no dismissals returns empty list."""
        config = {"other-agent": [asdict(_DISMISSAL_A)]}
        path = tmp_path / "dismissals.json"
        path.write_text(json.dumps(config))
        
        result = load_dismissals(path, self._AGENT_ID)
        
        assert result == []
