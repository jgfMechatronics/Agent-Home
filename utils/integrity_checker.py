"""Agent Data Integrity Checker.

Detects database corruption patterns in agent message history.
Top-level function: check_agent_integrity(session, agent_id) -> list[IntegrityIssue]
"""

from dataclasses import dataclass
from enum import Enum

from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from db.models import MessageRecord


class Severity(Enum):
    """Severity levels for integrity issues."""
    ERROR = "error"
    WARN = "warn"
    INFO = "info"


@dataclass
class IntegrityIssue:
    """A detected integrity problem in the message history."""
    check_type: str  # e.g., "seq_id_gap", "seq_id_duplicate", "adjacent_duplicate"
    severity: Severity
    seq_ids: list[int]  # seq_ids involved in the issue
    details: str  # human-readable description


def check_seq_id_consecutive(records: list[MessageRecord]) -> list[IntegrityIssue]:
    """Check that seq_ids are strictly consecutive starting at 0 (no gaps, no duplicates).
    
    Args:
        records: MessageRecords sorted by seq_id (caller's responsibility)
    
    Returns:
        List of IntegrityIssues for any gaps or duplicates found.
    """
    if not records:
        return []
    
    issues: list[IntegrityIssue] = []
    
    # Check first seq_id is 0
    if records[0].seq_id != 0:
        first_seq = records[0].seq_id
        issues.append(IntegrityIssue(
            check_type="seq_id_gap",
            severity=Severity.ERROR,
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
                severity=Severity.ERROR,
                seq_ids=[curr_seq],
                details=f"Duplicate seq_id {curr_seq} at positions {i-1} and {i}",
            ))
        elif curr_seq != prev_seq + 1:
            # Gap in seq_ids
            expected = prev_seq + 1
            issues.append(IntegrityIssue(
                check_type="seq_id_gap",
                severity=Severity.ERROR,
                seq_ids=[prev_seq, curr_seq],
                details=f"Gap in seq_ids: expected {expected}, got {curr_seq} (missing {expected} through {curr_seq - 1})",
            ))
    
    return issues


def check_seq_id_order_by_timestamp(records: list[MessageRecord]) -> list[IntegrityIssue]:
    """Check that seq_ids are monotonically increasing when sorted by timestamp.
    
    Detects cases where messages were persisted out of order (e.g., seq_id 2 before seq_id 1).
    
    Args:
        records: MessageRecords sorted by timestamp (caller's responsibility)
    
    Returns:
        List of IntegrityIssues for any out-of-order seq_ids.
    """
    if len(records) <= 1:
        return []
    
    issues: list[IntegrityIssue] = []
    
    for i in range(1, len(records)):
        prev_seq = records[i - 1].seq_id
        curr_seq = records[i].seq_id
        
        if curr_seq < prev_seq:
            issues.append(IntegrityIssue(
                check_type="seq_id_out_of_order",
                severity=Severity.ERROR,
                seq_ids=[prev_seq, curr_seq],
                details=f"seq_id out of order: {prev_seq} followed by {curr_seq} (by timestamp order)",
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
    issues: list[IntegrityIssue] = []
    
    # Fetch all messages ordered by seq_id (for consecutive check)
    stmt_by_seq = select(MessageRecord).where(MessageRecord.agent_id == agent_id).order_by(MessageRecord.seq_id)
    result = await session.execute(stmt_by_seq)
    records_by_seq = list(result.scalars().all())
    
    issues.extend(check_seq_id_consecutive(records_by_seq))
    
    # Fetch all messages ordered by timestamp (for order check)
    stmt_by_ts = select(MessageRecord).where(MessageRecord.agent_id == agent_id).order_by(MessageRecord.timestamp)
    result = await session.execute(stmt_by_ts)
    records_by_ts = list(result.scalars().all())
    
    issues.extend(check_seq_id_order_by_timestamp(records_by_ts))
    
    # TODO: Add more checks here
    
    return issues
