"""
Block CRUD operations — Section 2.1

Read operations take (session, agent_id) — no lock required, allows concurrent reads.
Write operations take (deps: AgentDeps) — requires deps, proving caller holds per-agent lock.
"""
from sqlalchemy.ext.asyncio import AsyncSession

from agent.runner import AgentDeps
from db.models import MemoryBlockRecord


# --- Read operations (no lock) ---

async def get_blocks(session: AsyncSession, agent_id: str) -> list[MemoryBlockRecord]:
    """Load all blocks for agent, ordered by position ascending."""
    pass


async def get_block(session: AsyncSession, agent_id: str, label: str) -> MemoryBlockRecord | None:
    """Load single block by label. Returns None if not found."""
    pass


# --- Write operations (require deps → lock held) ---

async def update_block(deps: AgentDeps, label: str, content: str) -> MemoryBlockRecord:
    """Update block content. Raises if block doesn't exist or content exceeds char_limit."""
    pass


async def create_block(
    deps: AgentDeps,
    label: str,
    content: str = "",
    description: str = "",
    char_limit: int = 2000,
    position: int | None = None,
) -> MemoryBlockRecord:
    """
    Create new block. 
    
    If position is None, appends to end (max existing position + 1).
    Raises if label already exists for this agent.
    """
    pass


async def delete_block(deps: AgentDeps, label: str) -> None:
    """Remove block. Raises if block doesn't exist (fail loudly)."""
    pass


async def reorder_blocks(deps: AgentDeps, labels_in_order: list[str]) -> None:
    """
    Assign positions 0, 1, 2... based on list order.
    
    Raises if list doesn't include all blocks for agent or contains unknown labels.
    """
    pass
