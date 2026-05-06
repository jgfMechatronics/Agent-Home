"""
Agent CRUD operations
"""
from sqlalchemy import exists, select
from sqlalchemy.ext.asyncio import AsyncSession

from agent.types import AgentConfig
from db.models import AgentRecord


async def get_agent_record(session: AsyncSession, agent_id: str) -> AgentRecord | None:
    """
    Load agent by ID. Returns None if not found.
    # TODO: Review and maybe unit test as needed
    """
    stmt = select(AgentRecord).where(AgentRecord.id == agent_id)
    result = await session.execute(stmt)
    return result.scalars().one_or_none()


async def agent_exists(session: AsyncSession, agent_id: str) -> bool:
    """Return True if an agent with the given ID exists, without loading the full record."""
    stmt = select(exists().where(AgentRecord.id == agent_id))
    result = await session.execute(stmt)
    return result.scalar()


async def create_agent_record(
    session: AsyncSession,
    name: str,
    system_instructions: str,
    config: AgentConfig,
) -> AgentRecord:
    """Create a new agent and return the AgentRecord."""
    raise NotImplementedError
