"""Agent Data Integrity Checker.

Detects database corruption patterns in agent message history.
Top-level function: check_agent_integrity(session, agent_id) -> list[IntegrityIssue]
"""

from collections.abc import Sequence
from dataclasses import dataclass
from enum import Enum
import hashlib

from sqlalchemy.ext.asyncio import AsyncSession
from pydantic_ai import ToolCallPart, ToolReturnPart, RetryPromptPart, ModelRequestPart, ModelResponsePart
from pydantic_ai.messages import ModelRequest, ModelResponse

from sqlalchemy import select

from db.models import AgentConfigSnapshot, MessageRecord, SystemPromptSnapshot, ToolDefinitionSnapshot
from messages.messages import load_messages, deserialize_messages

class Severity(Enum):
    """Severity levels for integrity issues."""
    ERROR = "error"
    WARN = "warn"
    INFO = "info"
    NO_ERROR = "no error"

ERROR, WARN, INFO, NO_ERROR = Severity.ERROR, Severity.WARN, Severity.INFO, Severity.NO_ERROR


@dataclass
class IntegrityIssue:
    """A detected integrity problem in the message history."""
    check_type: str  # e.g., "seq_id_gap", "seq_id_duplicate", "adjacent_duplicate"
    severity: Severity
    seq_ids: list[int]  # seq_ids involved in the issue
    details: str  # human-readable description


def check_seq_id_consecutive(records: Sequence[MessageRecord]) -> list[IntegrityIssue]:
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


def check_timestamps_increasing(records: Sequence[MessageRecord]) -> list[IntegrityIssue]:
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


def check_context_window_start_validity(records: Sequence[MessageRecord]) -> list[IntegrityIssue]:
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


def check_for_empty_content(records: Sequence[MessageRecord]) -> list[IntegrityIssue]:
    """Check for MessageRecords with empty/null content blobs or undeserializable content.

    NOTE: A ModelMessage with no parts (including ModelResponse) is a known valid case.
    This check is about the MessageRecord.content field being empty or unparseable — not
    about the message having zero parts.
    """
    issues: list[IntegrityIssue] = []

    for record in records:
        if not record.content:
            issues.append(IntegrityIssue(
                check_type="empty_content",
                severity=ERROR,
                seq_ids=[record.seq_id],
                details=f"MessageRecord at seq_id {record.seq_id} has empty or null content",
            ))
            continue

        try:
            deserialize_messages([record])
        except ValueError as e:
            issues.append(IntegrityIssue(
                check_type="undeserializable_content",
                severity=ERROR,
                seq_ids=[record.seq_id],
                details=f"MessageRecord at seq_id {record.seq_id} has undeserializable content: {e}",
            ))

    return issues


def check_tool_call_return_pairing(records: Sequence[MessageRecord]) -> list[IntegrityIssue]:
    """Check that every ToolCallPart has a matching ToolReturnPart or RetryPromptPart, and vice versa.

    Matches by tool_call_id. Orphaned calls or returns indicate incomplete turns — a persistence
    bug, cancellation mid-turn, or history truncation error.
    """
    messages = deserialize_messages(records)

    # Collect all tool_call_ids from ToolCallParts, with the seq_id they appeared in
    calls: dict[str, int] = {}  # tool_call_id -> seq_id
    for record, message in zip(records, messages):
        if isinstance(message, ModelResponse):
            for part in message.parts:
                if isinstance(part, ToolCallPart):
                    calls[part.tool_call_id] = record.seq_id

    # Collect all tool_call_ids that have been returned (ToolReturnPart or RetryPromptPart)
    returns: set[str] = set()
    return_orphans: dict[str, int] = {}  # tool_call_id -> seq_id (for ToolReturnParts with no matching call)
    for record, message in zip(records, messages):
        if isinstance(message, ModelRequest):
            for part in message.parts:
                if isinstance(part, (ToolReturnPart, RetryPromptPart)):
                    returns.add(part.tool_call_id)
                    if part.tool_call_id not in calls:
                        return_orphans[part.tool_call_id] = record.seq_id

    issues: list[IntegrityIssue] = []

    for tool_call_id, seq_id in calls.items():
        if tool_call_id not in returns:
            issues.append(IntegrityIssue(
                check_type="orphaned_tool_call",
                severity=ERROR,
                seq_ids=[seq_id],
                details=(
                    f"ToolCallPart with tool_call_id {tool_call_id!r} at seq_id {seq_id} "
                    f"has no matching ToolReturnPart or RetryPromptPart"
                ),
            ))

    for tool_call_id, seq_id in return_orphans.items():
        issues.append(IntegrityIssue(
            check_type="orphaned_tool_return",
            severity=ERROR,
            seq_ids=[seq_id],
            details=(
                f"ToolReturnPart with tool_call_id {tool_call_id!r} at seq_id {seq_id} "
                f"has no matching ToolCallPart"
            ),
        ))

    return issues


def check_for_duplicate_content(records: Sequence[MessageRecord]) -> list[IntegrityIssue]:
    messages = deserialize_messages(records)
    part_hash_table, part_hashes_suspected_of_duplication = _build_part_hash_table(messages, records)
    return _find_issues_in_suspect_parts(part_hashes_suspected_of_duplication, part_hash_table)


async def check_snapshot_references(session: AsyncSession, records: Sequence[MessageRecord]) -> list[IntegrityIssue]:
    """Check that every snapshot hash referenced by a MessageRecord exists in its snapshot table.

    This is a belt-and-suspenders check: FK constraints normally prevent orphaned references.
    Failure here indicates the DB was written outside normal ORM paths (import, migration,
    direct SQL) with FK enforcement disabled.
    """
    sys_hashes = {r.system_prompt_hash for r in records}
    tool_hashes = {r.tool_definition_hash for r in records}
    config_hashes = {r.agent_config_hash for r in records}

    existing_sys = {row for (row,) in (await session.execute(
        select(SystemPromptSnapshot.id).where(SystemPromptSnapshot.id.in_(sys_hashes))
    ))}
    existing_tool = {row for (row,) in (await session.execute(
        select(ToolDefinitionSnapshot.id).where(ToolDefinitionSnapshot.id.in_(tool_hashes))
    ))}
    existing_config = {row for (row,) in (await session.execute(
        select(AgentConfigSnapshot.id).where(AgentConfigSnapshot.id.in_(config_hashes))
    ))}

    issues: list[IntegrityIssue] = []
    for record in records:
        for hash_val, existing, field_name in [
            (record.system_prompt_hash, existing_sys, "system_prompt_hash"),
            (record.tool_definition_hash, existing_tool, "tool_definition_hash"),
            (record.agent_config_hash, existing_config, "agent_config_hash"),
        ]:
            if hash_val not in existing:
                issues.append(IntegrityIssue(
                    check_type="missing_snapshot",
                    severity=ERROR,
                    seq_ids=[record.seq_id],
                    details=(
                        f"MessageRecord at seq_id {record.seq_id} references "
                        f"{field_name} {hash_val!r} which has no corresponding snapshot row"
                    ),
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
    issues.extend(check_seq_id_consecutive(records))
    issues.extend(check_timestamps_increasing(records))
    issues.extend(check_context_window_start_validity(records))

    empty_content_issues = check_for_empty_content(records)
    issues.extend(empty_content_issues)

    # Skip records already flagged as empty/undeserializable for checks that require deserialization
    bad_seq_ids = {sid for issue in empty_content_issues for sid in issue.seq_ids}
    good_records = [r for r in records if r.seq_id not in bad_seq_ids]

    issues.extend(check_tool_call_return_pairing(good_records))
    issues.extend(check_for_duplicate_content(good_records))
    issues.extend(await check_snapshot_references(session, records))

    return issues
