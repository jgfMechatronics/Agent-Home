"""
Block CRUD operations — Section 2.1

Read operations take (session, agent_id) — no lock required, allows concurrent reads.
Write operations take (deps: AgentDeps) — requires deps, proving caller holds per-agent lock.

ORM objects should not outlive the session that loaded them — with one common
exception. Write operations return MemoryBlock DTOs (pydantic), safe to use after
the session closes. Read operations return ORM objects intended to be consumed
within the caller's read_scope.

Exception: reading simple column attributes of objects loaded from a read-only
session is safe after the scope closes. Read scopes never commit, so loaded
attributes are never expired — they stay populated on the detached object. The
hazard the rule guards against is any access that triggers a lazy load (expired
attributes, unloaded relationships), which needs a live session and fails outside
one (DetachedInstanceError, or MissingGreenlet in async context).
TODO: Consider making remaining read ops also return DTO
"""
from collections.abc import Sequence
from datetime import datetime

from pydantic import BaseModel, ConfigDict
from sqlalchemy import func, select
from sqlalchemy.exc import IntegrityError
from sqlalchemy.ext.asyncio import AsyncSession

from db.connection import ReadOnlySession

from agent.deps import AgentDeps
from agent.types import BlockSettings
from db.models import MemoryBlockRecord


class MemoryBlock(BaseModel):
    """Domain model for a memory block. Returned by write ops — safe to use after session closes.

    Built inside write_scope after flush (which populates Python-side defaults).
    """
    model_config = ConfigDict(from_attributes=True)

    id: str
    agent_id: str
    label: str
    description: str
    content: str
    char_limit: int
    position: int
    created_at: datetime
    updated_at: datetime


# --- Exceptions ---

class DuplicateBlockError(Exception):
    """Raised when attempting to create a block with a label that already exists for the agent."""


class BlockNotFoundError(Exception):
    """Raised when a block with the given label doesn't exist for the agent."""


class ContentExceedsLimitError(Exception):
    """Raised when new content exceeds the block's char_limit."""


class InvalidBlockOrderListError(Exception):
    """Raised when the label list for reorder doesn't match the agent's blocks."""


class DuplicatePositionError(Exception):
    """Raised when attempting to set a block's position to one already held by another block."""


# --- Read operations (no lock) ---

async def get_blocks(session: ReadOnlySession, agent_id: str) -> Sequence[MemoryBlockRecord]:
    """Load all blocks for agent, ordered by position ascending."""
    stmt = (
        select(MemoryBlockRecord)
        .where(MemoryBlockRecord.agent_id == agent_id)
        .order_by(MemoryBlockRecord.position)
    )
    result = await session.execute(stmt)
    return result.scalars().all()


async def get_block(session: ReadOnlySession, agent_id: str, label: str) -> MemoryBlockRecord | None:
    """Load single block by label. Returns None if not found."""
    stmt = (
        select(MemoryBlockRecord)
        .where(MemoryBlockRecord.agent_id == agent_id)
        .where(MemoryBlockRecord.label == label)
    )
    result = await session.execute(stmt)
    return result.scalars().one_or_none()


async def get_block_or_raise(session: ReadOnlySession, agent_id: str, label: str) -> MemoryBlockRecord:
    """Load single block by label. Raises BlockNotFoundError if not found."""
    block = await get_block(session, agent_id, label)
    if block is None:
        raise BlockNotFoundError(f"block with label '{label}' not found")
    return block


async def raise_if_label_exists(session: ReadOnlySession, agent_id: str, label: str) -> None:
    """Raise DuplicateBlockError if a block with this label already exists."""
    existing = await get_block(session, agent_id, label)
    if existing is not None:
        raise DuplicateBlockError(f"block with label '{label}' already exists")


# --- Write operations (require deps → lock held) ---

async def update_block(
    deps: AgentDeps,
    label: str,
    content: str,
) -> MemoryBlock:
    """
    Update block content. Returns MemoryBlock DTO (safe outside session scope).

    Raises if block doesn't exist or content exceeds char_limit.
    """
    async with deps.write_scope() as session:
        block = await get_block_or_raise(session, deps.agent_id, label)

        if len(content) > block.char_limit:
            raise ContentExceedsLimitError("new content exceeds char limit")

        block.content = content
        await session.flush()
        return MemoryBlock.model_validate(block)


async def create_block(
    deps: AgentDeps,
    settings: BlockSettings,
    content: str = "",
) -> MemoryBlock:
    """
    Create new block. Returns MemoryBlock DTO (safe outside session scope).

    If settings.position is None, appends to end (max existing position + 1).
    Raises if label already exists for this agent.
    """
    async with deps.write_scope() as session:
        await raise_if_label_exists(session, deps.agent_id, settings.label)

        # Auto-assign position if not specified
        position = settings.position
        if position is None:
            stmt = select(func.max(MemoryBlockRecord.position)).where(
                MemoryBlockRecord.agent_id == deps.agent_id
            )
            result = await session.execute(stmt)
            max_pos = result.scalar()
            position = 0 if max_pos is None else max_pos + 1

        block = MemoryBlockRecord(
            agent_id=deps.agent_id,
            label=settings.label,
            content=content,
            description=settings.description,
            char_limit=settings.char_limit,
            position=position,
        )
        session.add(block)
        await session.flush()
        return MemoryBlock.model_validate(block)


async def delete_block(deps: AgentDeps, label: str) -> None:
    """Remove block. Raises if block doesn't exist (fail loudly)."""
    async with deps.write_scope() as session:
        block = await get_block_or_raise(session, deps.agent_id, label)
        await session.delete(block)


async def reorder_blocks(deps: AgentDeps, labels_in_order: list[str]) -> None:
    """
    Assign positions 0, 1, 2... based on list order.
    
    Raises if list doesn't include all blocks for agent or contains unknown labels.
    """
    async with deps.write_scope() as session:
        blocks = await get_blocks(session, deps.agent_id)
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
            raise InvalidBlockOrderListError("; ".join(errors))

        # Build label -> block map for efficient lookup
        blocks_by_label = {b.label: b for b in blocks}

        # Clear positions first to avoid unique constraint collisions during reorder
        for i, block in enumerate(blocks):
            block.position = -(i + 1)  # Negative values won't collide with final 0,1,2...
        await session.flush()

        # Assign final positions
        for position, label in enumerate(labels_in_order):
            blocks_by_label[label].position = position


async def update_block_settings(
    deps: AgentDeps,
    label: str,
    settings: BlockSettings,
) -> MemoryBlock:
    """
    Update block settings (label, description, char_limit, position).
    Returns MemoryBlock DTO (safe outside session scope).

    Args:
        deps: Agent dependencies (proves caller holds lock)
        label: Current label of block to update (from URL path)
        settings: New settings to apply (settings.label may differ for rename)

    If settings.position is None, keeps the current position (no change).

    Raises BlockNotFoundError if block doesn't exist.
    Raises DuplicateBlockError if renaming to a label that already exists.
    Raises DuplicatePositionError if setting position to one already held by another block.
    """
    async with deps.write_scope() as session:
        block = await get_block_or_raise(session, deps.agent_id, label)

        # Check for label conflict on rename (before mutations)
        if settings.label != label:
            await raise_if_label_exists(session, deps.agent_id, settings.label)

        if settings.char_limit < len(block.content):
            raise ContentExceedsLimitError("new char_limit is less than current content length")

        # Apply mutations
        block.label = settings.label
        block.description = settings.description
        block.char_limit = settings.char_limit
        if settings.position is not None:
            block.position = settings.position

        try:
            await session.flush()
        except IntegrityError as e:
            # Label conflict already checked above; check for position conflict
            if "UNIQUE constraint failed: memory_block.agent_id, memory_block.position" in str(e):
                raise DuplicatePositionError(
                    f"position {settings.position} already held by another block"
                ) from e
            raise  # Re-raise unexpected IntegrityErrors

        return MemoryBlock.model_validate(block)
