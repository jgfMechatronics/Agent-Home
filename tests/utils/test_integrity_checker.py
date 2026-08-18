"""
Agent Data Integrity Checker - Test Suite

Planning docstring: captures intended checks. Remove/simplify as tests capture behaviors.

=== CHECKS TO IMPLEMENT ===

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

=== ARCHITECTURE ===

- Pure functions: data in, findings out (list of IntegrityIssue with severity)
- Each check is independent, can run individually or as suite
- Thin HTTP wrapper for runtime use (later)
- Parametrized tests for each check type
- Snapshot integrity requires DB access (session parameter)

=== SEVERITY LEVELS ===

ERROR: seq_id gap/duplicate, adjacent duplicate content, orphan tool call,
       context_window_start nonexistent, empty content, snapshot orphan,
       part-level violations, non-adjacent long duplicate
WARN:  multiple adjacent ModelResponses, non-adjacent short duplicate (3+ times),
       context_window_start very recent
INFO:  multiple adjacent ModelRequests, timestamp inversion < 1s

=== CONSTANTS ===

LENGTH_THRESHOLD = 35  # chars - conservative, "ToolReturnPart should only appear in" is 36
FREQUENCY_THRESHOLD = 3  # for short non-adjacent duplicates

=== ALGORITHM NOTES ===

Duplicate detection (adjacent-aware):
```python
seen_hashes = {}  # hash -> (first_seq_id, count)
prev_hash = None
prev_seq_id = None

for msg in messages:
    content_hash = sha256(canonical_serialize(msg.content))
    content_len = len(msg.content)
    
    # Adjacent check (always error regardless of length)
    if content_hash == prev_hash and msg.seq_id == prev_seq_id + 1:
        yield IntegrityIssue(ADJACENT_DUPLICATE, ERROR, original=prev_seq_id, duplicate=msg.seq_id)
    
    # Non-adjacent check
    elif content_hash in seen_hashes:
        first_seq_id, count = seen_hashes[content_hash]
        if content_len >= LENGTH_THRESHOLD:
            yield IntegrityIssue(NON_ADJACENT_DUPLICATE, ERROR, ...)
        else:
            new_count = count + 1
            if new_count >= FREQUENCY_THRESHOLD:
                yield IntegrityIssue(FREQUENT_SHORT_DUPLICATE, WARN, ...)
            seen_hashes[content_hash] = (first_seq_id, new_count)
    else:
        seen_hashes[content_hash] = (msg.seq_id, 1)
    
    prev_hash = content_hash
    prev_seq_id = msg.seq_id
```

"""

from datetime import datetime

import pytest

from db.models import MessageRecord
from utils.integrity_checker import IntegrityIssue, Severity, check_seq_id_consecutive


# ---------------------------------------------------------------------------
# Test Helpers
# ---------------------------------------------------------------------------

def _make_record(seq_id: int, content: str = "msg") -> MessageRecord:
    """Build a minimal MessageRecord for integrity checking tests.
    
    Only populates fields relevant to integrity checks. FK fields use dummy values
    since we're testing pure functions, not DB operations.
    """
    return MessageRecord(
        agent_id="test-agent",
        type="ModelRequest",
        content=f'{{"parts": [{{"type": "user-prompt", "content": "{content} {seq_id}"}}]}}',
        total_tokens=None,
        seq_id=seq_id,
        timestamp=datetime(2026, 1, 1, 12, 0, seq_id),  # increments by 1 second per seq_id
        system_prompt_hash="fake-hash",
        tool_definition_hash="fake-hash",
        agent_config_hash="fake-hash",
        context_window_start_msg_id="fake-id",
    )


def _make_records(*seq_ids: int) -> list[MessageRecord]:
    """Build MessageRecords with the given seq_ids."""
    return [_make_record(seq_id) for seq_id in seq_ids]


# ---------------------------------------------------------------------------
# TestSeqIdConsecutive
# ---------------------------------------------------------------------------

class TestSeqIdConsecutive:
    """Tests for check_seq_id_consecutive(records) — pure function."""

    def test_clean_sequence_returns_no_issues(self):
        """Consecutive seq_ids (1,2,3,4) should pass with no issues."""
        records = _make_records(1, 2, 3, 4)
        issues = check_seq_id_consecutive(records)
        assert issues == []

    def test_empty_list_returns_no_issues(self):
        """Empty records list is valid (nothing to check)."""
        issues = check_seq_id_consecutive([])
        assert issues == []

    def test_single_record_returns_no_issues(self):
        """Single record is valid (no gaps possible)."""
        records = _make_records(1)
        issues = check_seq_id_consecutive(records)
        assert issues == []

    def test_gap_detected(self):
        """Gap in seq_ids (1,2,4,5) should return an ERROR issue."""
        records = _make_records(1, 2, 4, 5)
        issues = check_seq_id_consecutive(records)
        
        assert len(issues) == 1
        issue = issues[0]
        assert issue.check_type == "seq_id_gap"
        assert issue.severity == Severity.ERROR
        assert 2 in issue.seq_ids  # before gap
        assert 4 in issue.seq_ids  # after gap
        assert "gap" in issue.details.lower() or "3" in issue.details

    def test_duplicate_detected(self):
        """Duplicate seq_id (1,2,2,3) should return an ERROR issue."""
        records = _make_records(1, 2, 2, 3)
        issues = check_seq_id_consecutive(records)
        
        assert len(issues) == 1
        issue = issues[0]
        assert issue.check_type == "seq_id_duplicate"
        assert issue.severity == Severity.ERROR
        assert 2 in issue.seq_ids

    def test_multiple_issues_detected(self):
        """Multiple problems (gap + duplicate) should all be reported."""
        records = _make_records(1, 2, 2, 5, 6)  # duplicate at 2, gap 3-4
        issues = check_seq_id_consecutive(records)
        
        # Should find both the duplicate and the gap
        assert len(issues) >= 2
        check_types = {i.check_type for i in issues}
        assert "seq_id_duplicate" in check_types
        assert "seq_id_gap" in check_types
