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

from db.models import MessageRecord
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
            seq_ids=[None, first_seq],
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
                details=f"Timestamp out of order: seq_id {curr.seq_id} has earlier timestamp than seq_id {prev.seq_id} (can also indicate non-adjacent duplicate timestamps)",
            ))
        elif curr.timestamp == prev.timestamp:
            issues.append(IntegrityIssue(
                check_type="timestamp_duplicate",
                severity=ERROR,
                seq_ids=[prev.seq_id, curr.seq_id],
                details=f"Duplicate timestamp at seq_ids {prev.seq_id} and {curr.seq_id}",
            ))
    
    return issues


@dataclass
class PartAndMetadata:
    # TODO: double check all these fields requried
    part: ModelRequestPart | ModelResponsePart 
    idx: int # index of the msg where this part originated
    seq_id: int # seq_id of the msg where this part originated


def check_for_duplicate_content(records: Sequence[MessageRecord]) -> list[IntegrityIssue]:
    # mapping part content hash to the seq id which the part occured at
    CONTENT_LENGTH_THRESHOLD = 35
    SHORT_CONTENT_FREQ_THRESHOLD = 3
    integrity_issues = []

    try:
        messages = deserialize_messages(records)
    except ValueError:
        return [IntegrityIssue(
            check_type="deserialization",
            severity=ERROR,
            seq_ids=[],
            details="Deserialization error while attempting to perform duplication check. Manual intervention required!"
        )]

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

    for suspect_hash in part_hashes_suspected_of_duplication:
        suspect_part_and_meta_list = part_hash_table[suspect_hash]
        suspect_part_and_meta = suspect_part_and_meta_list[0]

        suspect_msgs_adjacent = False
        for i in range(1, len(suspect_part_and_meta_list)):
            # looking for ANY adjacent sus messages
            if (suspect_part_and_meta_list[i - 1].idx + 1) == suspect_part_and_meta_list[i].idx:
                suspect_msgs_adjacent = True
                break

        # we already know these parts have identical type and content, now determine how much we care
        if suspect_msgs_adjacent:
            severity = ERROR
            detail_preamble = "Duplicate content found in adjacent messages. Adjacent duplication is unlikely to naturally occur."
        elif len(suspect_part_and_meta.part.content) > CONTENT_LENGTH_THRESHOLD:
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
            integrity_issue = IntegrityIssue(
                check_type="content_duplicate",
                severity=severity,
                seq_ids=bad_seq_ids,
                details=detail_preamble + f" Duplication occured at seq_ids: {bad_seq_ids}",
                )
            integrity_issues.append(integrity_issue)

    return integrity_issues


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
    issues.extend(check_for_duplicate_content(records))
    
    
    return issues
