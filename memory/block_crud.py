"""
Block CRUD operations — Section 2.1

Read operations take (session, agent_id) — no lock required, allows concurrent reads.
Write operations take (deps: AgentDeps) — requires deps, proving caller holds per-agent lock.
"""
from collections.abc import Sequence

from sqlalchemy import func, select
from sqlalchemy.ext.asyncio import AsyncSession

from agent.types import AgentDeps
from db.models import MemoryBlockRecord


# --- Internal helpers ---

async def _persist(session: AsyncSession, commit: bool, record: MemoryBlockRecord | None = None) -> None:
    """Commit or flush the session, refreshing record if provided."""
    if commit:
        await session.commit()
        if record is not None:
            await session.refresh(record)
    else:
        await session.flush()


# --- Read operations (no lock) ---

async def get_blocks(session: AsyncSession, agent_id: str) -> Sequence[MemoryBlockRecord]:
    """Load all blocks for agent, ordered by position ascending."""
    stmt = (
        select(MemoryBlockRecord)
        .where(MemoryBlockRecord.agent_id == agent_id)
        .order_by(MemoryBlockRecord.position)
    )
    result = await session.execute(stmt)
    return result.scalars().all()


async def get_block(session: AsyncSession, agent_id: str, label: str) -> MemoryBlockRecord | None:
    """Load single block by label. Returns None if not found."""
    stmt = (
        select(MemoryBlockRecord)
        .where(MemoryBlockRecord.agent_id == agent_id)
        .where(MemoryBlockRecord.label == label)
    )
    result = await session.execute(stmt)
    return result.scalars().one_or_none()


# --- Write operations (require deps → lock held) ---

async def update_block(
    deps: AgentDeps,
    label: str,
    content: str,
    commit: bool = True,
    block: MemoryBlockRecord | None = None,
) -> MemoryBlockRecord:
    """
    Update block content.
    
    Raises if block doesn't exist or content exceeds char_limit.
    commit=False flushes instead of committing, for chaining ops atomically.
    block: Optional pre-fetched block to avoid redundant DB query if user 
    already has it
    """
    if block is None:
        block = await get_block(deps.session, deps.agent_id, label)
        if block is None:
            raise ValueError("block not found")

    if len(content) > block.char_limit:
        raise ValueError("new content exceeds char limit")

    block.content = content
    await _persist(deps.session, commit, block)
    return block


async def create_block(
    deps: AgentDeps,
    label: str,
    content: str = "",
    description: str = "",
    char_limit: int = 20000,
    position: int | None = None,
    commit: bool = True,
) -> MemoryBlockRecord:
    """
    Create new block.
    
    If position is None, appends to end (max existing position + 1).
    Raises if label already exists for this agent.
    """
    # Check for duplicate label
    existing = await get_block(deps.session, deps.agent_id, label)
    if existing is not None:
        raise ValueError(f"block with label '{label}' already exists")

    # Auto-assign position if not specified
    if position is None:
        stmt = select(func.max(MemoryBlockRecord.position)).where(
            MemoryBlockRecord.agent_id == deps.agent_id
        )
        result = await deps.session.execute(stmt)
        max_pos = result.scalar()
        position = 0 if max_pos is None else max_pos + 1

    block = MemoryBlockRecord(
        agent_id=deps.agent_id,
        label=label,
        content=content,
        description=description,
        char_limit=char_limit,
        position=position,
    )
    deps.session.add(block)
    await _persist(deps.session, commit, block)
    return block


async def delete_block(deps: AgentDeps, label: str, commit: bool = True) -> None:
    """Remove block. Raises if block doesn't exist (fail loudly)."""
    block = await get_block(deps.session, deps.agent_id, label)
    if block is None:
        raise ValueError("block not found")

    await deps.session.delete(block)
    await _persist(deps.session, commit)


async def reorder_blocks(deps: AgentDeps, labels_in_order: list[str], commit: bool = True) -> None:
    """
    Assign positions 0, 1, 2... based on list order.
    
    Raises if list doesn't include all blocks for agent or contains unknown labels.
    """
    blocks = await get_blocks(deps.session, deps.agent_id)
    existing_labels = {b.label for b in blocks}
    provided_labels = set(labels_in_order)

    # Validate: must be exact match
    if existing_labels != provided_labels:
        missing = existing_labels - provided_labels
        unknown = provided_labels - existing_labels
        errors = []
        if missing:
            errors.append(f"missing labels: {missing}")
        if unknown:
            errors.append(f"unknown labels: {unknown}")
        raise ValueError("; ".join(errors))

    # Build label -> block map for efficient lookup
    blocks_by_label = {b.label: b for b in blocks}

    # Clear positions first to avoid unique constraint collisions during reorder
    for i, block in enumerate(blocks):
        block.position = -(i + 1)  # Negative values won't collide with final 0,1,2...
    await deps.session.flush()

    # Assign final positions
    for position, label in enumerate(labels_in_order):
        blocks_by_label[label].position = position

    await _persist(deps.session, commit)
