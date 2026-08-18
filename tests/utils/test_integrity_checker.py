"""
Agent Data Integrity Checker - Test Suite

Top-level function: check_agent_integrity(session, agent_id) -> list[IntegrityIssue]

=== CHECKS ===

1. SEQ_ID CONSECUTIVE
   - seq_ids must be strictly consecutive integers (1, 2, 3, 4, 5...)
   - Gaps indicate lost messages, duplicate seq_ids indicate re-persistence bugs
   - First seq_id should be 1 (or agent's known start)
   - Severity: ERROR

2. TIMESTAMPS STRICTLY INCREASING
   - Each MessageRecord timestamp must be > previous (by seq_id order)
   - Duplicate timestamps: likely re-persistence or clock issues
   - Out-of-order timestamps: definitely corruption
   - Severity: ERROR (major inversion), INFO (< 1s inversion, could be clock jitter)

3. PROVIDER TIMESTAMPS (in serialized ModelMessage content)
   - Same rules as MessageRecord timestamps
   - These come from the provider response, separate from our persistence timestamp

4. DUPLICATE CONTENT DETECTION (hash-based, O(n))
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

5. EMPTY/NULL CONTENT
   - MessageRecord with null or empty content blob is always suspicious
   - Cheap to catch
   - Severity: ERROR

6. TOOL CALL/RETURN PAIRING
   - ToolCallPart in ModelResponse should eventually have matching ToolReturnPart by tool_call_id
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

8. CONTEXT_WINDOW_START_MSG_ID VALIDITY
   - Must point to a message that exists
   - Must point to same seq_id or earlier (not forward references)
   - Flag if pointing to very recent messages (compaction misfire signal)
   - Severity: ERROR (nonexistent), WARN (too recent)

9. SNAPSHOT REFERENCE INTEGRITY
   - system_prompt_hash -> must exist in SystemPromptSnapshot table
   - tool_schema_hash -> must exist in ToolSchemaSnapshot table  
   - agent_config_hash -> must exist in AgentConfigSnapshot table
   - Orphan references indicate incomplete persistence
   - Severity: ERROR

10. PART-LEVEL SANITY
    - ToolReturnPart should only appear in ModelRequest (user returning tool results)
    - ToolCallPart should only appear in ModelResponse (model calling tools)
    - Wrong part types in wrong message kinds = corruption or serialization bug
    - Severity: ERROR

=== TESTING APPROACH ===

Parametrized tests: each case is (poisoned_messages, expected_issues).
Test loads messages into DB via raw insert, calls top-level function, asserts exact equality.
"""

from typing import Callable

import pytest
import pytest_asyncio
from sqlalchemy.ext.asyncio import AsyncSession

from conftest import make_message_record
from db.models import AgentRecord, MessageRecord
from utils.integrity_checker import IntegrityIssue, Severity, check_agent_integrity


# Type alias for the callable that builds records given an agent_id
RecordBuilder = Callable[[str], list[MessageRecord]]


# ---------------------------------------------------------------------------
# Test Cases (parametrized)
# ---------------------------------------------------------------------------

# Each test case: (build_records callable, expected_issues list)
# build_records takes agent_id and returns list of MessageRecords to insert

SEQ_ID_TEST_CASES = [
    pytest.param(
        lambda aid: [
            make_message_record(aid, 0),
            make_message_record(aid, 1),
            make_message_record(aid, 2),
        ],
        [],
        id="clean_sequence",
    ),
    pytest.param(
        lambda aid: [],
        [],
        id="empty_list",
    ),
    pytest.param(
        lambda aid: [make_message_record(aid, 0)],
        [],
        id="single_record",
    ),
    pytest.param(
        lambda aid: [
            make_message_record(aid, 0),
            make_message_record(aid, 1),
            make_message_record(aid, 3),  # gap: missing 2
        ],
        [IntegrityIssue(
            check_type="seq_id_gap",
            severity=Severity.ERROR,
            seq_ids=[1, 3],
            details="Gap in seq_ids: expected 2, got 3 (missing 2 through 2)",
        )],
        id="seq_id_gap",
    ),
    pytest.param(
        lambda aid: [
            make_message_record(aid, 0),
            make_message_record(aid, 1),
            make_message_record(aid, 1),  # duplicate
            make_message_record(aid, 2),
        ],
        [IntegrityIssue(
            check_type="seq_id_duplicate",
            severity=Severity.ERROR,
            seq_ids=[1],
            details="Duplicate seq_id 1 at positions 1 and 2",
        )],
        id="seq_id_duplicate",
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

    @pytest.mark.parametrize("build_records,expected_issues", SEQ_ID_TEST_CASES)
    async def test_check_agent_integrity(
        self,
        build_records: RecordBuilder,
        expected_issues: list[IntegrityIssue],
    ):
        """Parametrized test for check_agent_integrity"""
        records = build_records(self.agent.id)
        self.session.add_all(records)
        await self.session.flush()

        issues = await check_agent_integrity(self.session, self.agent.id)
        assert issues == expected_issues
