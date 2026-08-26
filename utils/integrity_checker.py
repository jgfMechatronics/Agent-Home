"""Agent Data Integrity Checker.

Detects database corruption patterns in agent message history.
Top-level function: check_agent_integrity(session, agent_id) -> list[IntegrityIssue]
"""

from collections.abc import Sequence
from dataclasses import dataclass
from enum import Enum
import hashlib
import json
from pathlib import Path

from sqlalchemy.ext.asyncio import AsyncSession
from pydantic import TypeAdapter
from pydantic_ai import ToolCallPart, ToolReturnPart, RetryPromptPart, ModelRequestPart, ModelResponsePart
from pydantic_ai.messages import ModelMessage, ModelRequest, ModelResponse

from db.models import MessageRecord
from messages.messages import load_messages, deserialize_messages, is_valid_tool_pair, is_system_alert

class Severity(Enum):
    """Severity levels for integrity issues."""
    CRITICAL = "critical"  # Blocks further checks — must be resolved before re-running
    ERROR = "error"
    WARN = "warn"
    INFO = "info"
    NO_ERROR = "no error"

CRITICAL, ERROR, WARN, INFO, NO_ERROR = Severity.CRITICAL, Severity.ERROR, Severity.WARN, Severity.INFO, Severity.NO_ERROR


@dataclass
class IntegrityIssue:
    """A detected integrity problem in the message history."""
    check_type: str  # e.g., "seq_id_gap", "seq_id_duplicate", "adjacent_duplicate"
    severity: Severity
    seq_ids: list[int]  # seq_ids involved in the issue
    details: str  # human-readable description


def _check_seq_id_consecutive(records: Sequence[MessageRecord]) -> list[IntegrityIssue]:
    """Check that seq_ids are strictly consecutive starting at 0 (no gaps, no duplicates).
    
    Args:
        records: MessageRecords sorted by seq_id (caller's responsibility. Handled by load_messages typically)
    
    Returns:
        List of IntegrityIssues for any gaps or duplicates found.
    """
    
    issues: list[IntegrityIssue] = []
    
    # Check first seq_id is 0
    if records[0].seq_id != 0:
        first_seq = records[0].seq_id
        issues.append(IntegrityIssue(
            check_type="seq_id_gap",
            severity=ERROR,
            seq_ids=[first_seq],
            details=f"Gap in seq_ids: expected 0, got {first_seq} (missing 0 through {first_seq - 1})",
        ))
    
    # Check consecutive pairs
    for i in range(1, len(records)):
        prev_seq = records[i - 1].seq_id
        curr_seq = records[i].seq_id
        
        if curr_seq == prev_seq:
            # Duplicate seq_id
            issues.append(IntegrityIssue(
                check_type="seq_id_duplicate",
                severity=ERROR,
                seq_ids=[curr_seq],
                details=f"Duplicate seq_id {curr_seq} at positions {i-1} and {i}",
            ))
        elif curr_seq != prev_seq + 1:
            # Gap in seq_ids
            expected = prev_seq + 1
            issues.append(IntegrityIssue(
                check_type="seq_id_gap",
                severity=ERROR,
                seq_ids=[prev_seq, curr_seq],
                details=f"Gap in seq_ids: expected {expected}, got {curr_seq} (missing {expected} through {curr_seq - 1})",
            ))
    
    return issues


def _check_timestamps_increasing(records: Sequence[MessageRecord]) -> list[IntegrityIssue]:
    """Check that timestamps are monotonically increasing in seq_id order.
    
    Detects clock issues or re-persistence bugs where a later message has an earlier timestamp.
    
    Args:
        records: MessageRecords in seq_id order (as returned by load_messages)
    
    Returns:
        List of IntegrityIssues for any timestamp inversions or duplicates.
    """
    if len(records) <= 1:
        return []
    
    issues: list[IntegrityIssue] = []
    
    for i in range(1, len(records)):
        prev = records[i - 1]
        curr = records[i]
        
        if curr.timestamp < prev.timestamp:
            issues.append(IntegrityIssue(
                check_type="timestamp_out_of_order",
                severity=ERROR,
                seq_ids=[prev.seq_id, curr.seq_id],
                details=(
                    f"Timestamp out of order at seq_ids {prev.seq_id} → {curr.seq_id}: "
                    f"{prev.timestamp} → {curr.timestamp}. "
                    "Possible causes: insertion order bug, clock skew, or re-persisted duplicate."
                ),
            ))
        elif curr.timestamp == prev.timestamp:
            issues.append(IntegrityIssue(
                check_type="timestamp_duplicate",
                severity=ERROR,
                seq_ids=[prev.seq_id, curr.seq_id],
                details=f"Duplicate timestamp at seq_ids {prev.seq_id} and {curr.seq_id}",
            ))
    
    return issues


# ---------------------------------------------------------------------------
# Duplicate Content Detection
# ---------------------------------------------------------------------------

@dataclass
class PartAndMetadata:
    part: ModelRequestPart | ModelResponsePart 
    idx: int # index of the msg where this part originated
    seq_id: int # seq_id of the msg where this part originated


def _build_part_hash_table(
    messages: list,
    records: Sequence[MessageRecord],
) -> tuple[dict[str, list[PartAndMetadata]], list[str]]:
    part_hash_table: dict[str, list[PartAndMetadata]] = {}
    part_hashes_suspected_of_duplication: list[str] = []

    for i, message in enumerate(messages):
        for part in message.parts:
            # These situations too ripe for "valid" duplication. Short circuiting should narrow type of part
            # enough that len comparison is valid
            if isinstance(part, (ToolCallPart, ToolReturnPart, RetryPromptPart)) or (len(part.content) == 0):
                continue
            # System alerts (compaction warnings, etc.) are expected to repeat
            if is_system_alert(part.content):
                continue

            # Avoid declaring duplicate across part type
            part_type_and_content = str(part.content) + part.part_kind
            part_hash = hashlib.sha256(part_type_and_content.encode("utf-8")).hexdigest()
            part_and_metadata = PartAndMetadata(part, i, records[i].seq_id)

            if part_hash in part_hash_table:
                if len(part_hash_table[part_hash]) == 1:
                    # first time we've suspected this hash, record it as such
                    # We have further checks later to determine if its truly bad
                    part_hashes_suspected_of_duplication.append(part_hash)
                part_hash_table[part_hash].append(part_and_metadata)
            else:
                part_hash_table[part_hash] = [part_and_metadata]
                # since this is new its not suspect...yet

    return part_hash_table, part_hashes_suspected_of_duplication


def _find_issues_in_suspect_parts(
    part_hashes_suspected_of_duplication: list[str],
    part_hash_table: dict[str, list[PartAndMetadata]],
) -> list[IntegrityIssue]:
    CONTENT_LENGTH_THRESHOLD = 35
    SHORT_CONTENT_FREQ_THRESHOLD = 3
    integrity_issues = []

    for suspect_hash in part_hashes_suspected_of_duplication:
        suspect_part_and_meta_list = part_hash_table[suspect_hash]

        # looking for ANY adjacent suspect messages
        suspect_msgs_adjacent = any(
            a.idx + 1 == b.idx
            for a, b in zip(suspect_part_and_meta_list, suspect_part_and_meta_list[1:])
        )

        # we already know these parts have identical type and content, now determine how much we care
        if suspect_msgs_adjacent:
            severity = ERROR
            detail_preamble = "Duplicate content found in adjacent messages. Adjacent duplication is unlikely to naturally occur."
        elif len(suspect_part_and_meta_list[0].part.content) > CONTENT_LENGTH_THRESHOLD:
            # content is long enough that legit repetition by chance is very unlikely
            severity = ERROR
            detail_preamble = "High length duplicate content detected. Higher length content is less likely to naturally recur."
        elif len(suspect_part_and_meta_list) >= SHORT_CONTENT_FREQ_THRESHOLD:
            # Even though its short, this much repetition is suspicious
            # NOTE: we may find we need an intermediate threshold or regex for stuff like "ok"
            severity = WARN
            detail_preamble = "Short length duplicate content detected with suspect frequency."
        else:
            # Short content that didn't occur many times or adjacently. Not that sus
            severity = NO_ERROR

        if severity != NO_ERROR:
            bad_seq_ids = [p.seq_id for p in suspect_part_and_meta_list]
            integrity_issues.append(IntegrityIssue(
                check_type="content_duplicate",
                severity=severity,
                seq_ids=bad_seq_ids,
                details=detail_preamble + f" Duplication occurred at seq_ids: {bad_seq_ids}",
            ))

    return integrity_issues

def _check_for_duplicate_content(records: Sequence[MessageRecord], messages: Sequence[ModelMessage]) -> list[IntegrityIssue]:
    part_hash_table, part_hashes_suspected_of_duplication = _build_part_hash_table(messages, records)
    return _find_issues_in_suspect_parts(part_hashes_suspected_of_duplication, part_hash_table)


def _check_context_window_start_validity(records: Sequence[MessageRecord]) -> list[IntegrityIssue]:
    """Check that each context_window_start_msg_id points to an existing, non-forward message.

    A valid context_window_start_msg_id must:
    - Correspond to a MessageRecord that exists in the agent's history
    - Have a seq_id <= the record that references it (no forward references)
    """
    id_to_seq: dict[str, int] = {r.id: r.seq_id for r in records}

    issues: list[IntegrityIssue] = []
    for record in records:
        ctx_id = record.context_window_start_msg_id
        if ctx_id not in id_to_seq:
            issues.append(IntegrityIssue(
                check_type="invalid_context_window_start",
                severity=ERROR,
                seq_ids=[record.seq_id],
                details=(
                    f"MessageRecord at seq_id {record.seq_id} has "
                    f"context_window_start_msg_id pointing to a nonexistent message"
                ),
            ))
        elif id_to_seq[ctx_id] > record.seq_id:
            issues.append(IntegrityIssue(
                check_type="invalid_context_window_start",
                severity=ERROR,
                seq_ids=[record.seq_id],
                details=(
                    f"MessageRecord at seq_id {record.seq_id} has "
                    f"context_window_start_msg_id pointing to a future message at seq_id {id_to_seq[ctx_id]}"
                ),
            ))

    return issues


def _check_message_ordering(records: Sequence[MessageRecord]) -> list[IntegrityIssue]:
    """Check for adjacent ModelResponse records, which should never occur.

    Consecutive ModelRequests are legitimate (compaction resume notices, cancel notices,
    tool returns adjacent to user prompts). Consecutive ModelResponses are not — there is
    no code path that produces them in normal operation.
    """
    return [
        IntegrityIssue(
            check_type="adjacent_model_responses",
            severity=ERROR,
            seq_ids=[records[i - 1].seq_id, records[i].seq_id],
            details=(
                f"Adjacent ModelResponses at seq_ids {records[i - 1].seq_id} and {records[i].seq_id}"
                f" — consecutive responses should not occur"
            ),
        )
        for i in range(1, len(records))
        if records[i].type == "ModelResponse" and records[i - 1].type == "ModelResponse"
    ]


def _check_for_empty_content(records: Sequence[MessageRecord]) -> list[IntegrityIssue]:
    """Check for MessageRecords with empty/null content blobs.

    NOTE: A ModelMessage with no parts (including ModelResponse) is a known valid case.
    This check is about the MessageRecord.content field being empty — not about the message
    having zero parts. Undeserializable content is handled separately in check_agent_integrity.
    """
    return [
        IntegrityIssue(
            check_type="empty_content",
            severity=ERROR,
            seq_ids=[record.seq_id],
            details=f"MessageRecord at seq_id {record.seq_id} has empty or null content",
        )
        for record in records
        if not record.content
    ]


def _check_tool_call_return_pairing(records: Sequence[MessageRecord], messages: Sequence[ModelMessage]) -> list[IntegrityIssue]:
    """Check that every ToolCallPart has a matching ToolReturnPart or RetryPromptPart, and vice versa.

    Uses adjacency-based pairing: a tool call must be immediately followed by its return, and
    a tool return must be immediately preceded by its call. Mirrors the sanitization logic in
    persist_messages._replace_orphaned_tool_messages.
    """
    issues: list[IntegrityIssue] = []
    for i, (record, msg) in enumerate(zip(records, messages)):
        if isinstance(msg, ModelResponse) and any(isinstance(p, ToolCallPart) for p in msg.parts):
            next_msg = messages[i + 1] if i + 1 < len(messages) else None
            if not is_valid_tool_pair(msg, next_msg):
                issues.append(IntegrityIssue(
                    check_type="orphaned_tool_call",
                    severity=ERROR,
                    seq_ids=[record.seq_id],
                    details=f"ModelResponse at seq_id {record.seq_id} has ToolCallPart(s) with no matching return in the following message",
                ))
        elif isinstance(msg, ModelRequest) and any(isinstance(p, (ToolReturnPart, RetryPromptPart)) for p in msg.parts):
            prev_msg = messages[i - 1] if i > 0 else None
            if not is_valid_tool_pair(prev_msg, msg):
                issues.append(IntegrityIssue(
                    check_type="orphaned_tool_return",
                    severity=ERROR,
                    seq_ids=[record.seq_id],
                    details=f"ModelRequest at seq_id {record.seq_id} has ToolReturnPart(s) with no matching call in the preceding message",
                ))
    return issues


async def check_agent_integrity(
    session: AsyncSession,
    agent_id: str,
) -> list[IntegrityIssue]:
    """Check agent's message history for integrity issues.
    
    Args:
        session: Database session
        agent_id: Agent to check
    
    Returns:
        List of all IntegrityIssues found across all checks.
    """
    # Load all messages once (in seq_id order)
    records = await load_messages(session, agent_id)

    if not records:
        return []

    issues: list[IntegrityIssue] = []

    # Non-deserializing checks — always run
    issues.extend(_check_seq_id_consecutive(records))
    issues.extend(_check_timestamps_increasing(records))
    issues.extend(_check_context_window_start_validity(records))
    issues.extend(_check_message_ordering(records))
    issues.extend(_check_for_empty_content(records))

    # Gating deserialization step — on failure, report CRITICAL and skip remaining checks
    id_to_seq = {r.id: r.seq_id for r in records}
    try:
        messages = deserialize_messages(records)
    except ValueError as e:
        failed_id = str(e).removeprefix("[Deserialization error for record ").split("]")[0]
        seq_id = id_to_seq.get(failed_id)
        issues.append(IntegrityIssue(
            check_type="deserialization_failure",
            severity=CRITICAL,
            seq_ids=[seq_id] if seq_id is not None else [],
            details=(
                f"Failed to deserialize message at seq_id {seq_id} (id: {failed_id}). "
                f"This is the first failure encountered — there may be more. "
                f"Checks requiring deserialization have been skipped. "
                f"Fix or remove record with undeserializable content and re-run."
            ),
        ))
        return issues

    # Deserialization-dependent checks
    issues.extend(_check_tool_call_return_pairing(records, messages))
    issues.extend(_check_for_duplicate_content(records, messages))

    return issues


# ---------------------------------------------------------------------------
# Blacklist based issue filtering for known false positives or minor issues
# ---------------------------------------------------------------------------

@dataclass
class Dismissal:
    """A user-acknowledged false positive to filter from results."""
    check_type: str
    seq_ids: list[int]
    reason: str  # Why this was dismissed (for future reference)


def filter_dismissed_issues(
    issues: list[IntegrityIssue],
    dismissals: list[Dismissal],
) -> list[IntegrityIssue]:
    """Remove issues that match a dismissal entry."""
    return [
        issue for issue in issues
        if not any(
            issue.check_type == d.check_type and issue.seq_ids == d.seq_ids
            for d in dismissals
        )
    ]


def load_dismissals(path: Path, agent_id: str) -> list[Dismissal]:
    """Load dismissals for a specific agent from a JSON file.
    
    File structure: dict keyed by agent_id, values are lists of dismissals.
    Returns empty list if file doesn't exist or agent has no dismissals.
    """
    if not path.exists():
        return []
    data = json.loads(path.read_text())
    return [TypeAdapter(Dismissal).validate_python(d) for d in data.get(agent_id, [])]
