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
    
    return issues
