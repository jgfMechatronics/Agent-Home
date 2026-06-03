"""
Agent CRUD operations
"""
from sqlalchemy import exists, select
from sqlalchemy.ext.asyncio import AsyncSession

from agent.types import AgentConfig, AgentDeps
from db.models import AgentRecord
from memory.system_prompt_compilation import compile_system_prompt


async def get_agent_record(session: AsyncSession, agent_id: str) -> AgentRecord | None:
    """Load agent by ID. Returns None if not found."""
    return await session.get(AgentRecord, agent_id)


async def agent_exists(session: AsyncSession, agent_id: str) -> bool:
    """Return True if an agent with the given ID exists, without loading the full record. TODO: Unit test"""
    stmt = select(exists().where(AgentRecord.id == agent_id))
    result = await session.execute(stmt)
    return result.scalar()


async def create_agent_record(
    session: AsyncSession,
    name: str,
    system_instructions: str,
    config: AgentConfig,
) -> AgentRecord:
    """Create a new agent, persist it, and return the AgentRecord."""
    record = AgentRecord(name=name, system_instructions=system_instructions, agent_config=config)
    session.add(record)
    await compile_system_prompt(AgentDeps(session, record))  # flushes session internally
    # TODO: Should commit here?
    return record
