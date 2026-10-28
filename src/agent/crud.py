"""
Agent CRUD operations
"""
from __future__ import annotations

from typing import TYPE_CHECKING

from sqlalchemy import exists, select
from sqlalchemy.ext.asyncio import AsyncSession

from db.connection import ReadOnlySession

if TYPE_CHECKING:
    from agent.deps import AgentDeps
from agent.types import AgentConfig
from db.models import AgentRecord
from memory.system_prompt_compilation import compile_system_prompt


async def get_agent_record(session: ReadOnlySession, agent_id: str) -> AgentRecord | None:
    """Load agent by ID. Returns None if not found."""
    return await session.get(AgentRecord, agent_id)


async def get_agent_record_by_name(session: ReadOnlySession, name: str) -> AgentRecord | None:
    """Load agent by name. Returns None if not found.

    Efficient single-row lookup — name has a unique constraint in the schema.
    """
    result = await session.execute(select(AgentRecord).where(AgentRecord.name == name))
    return result.scalar_one_or_none()


async def get_all_agents(session: ReadOnlySession) -> list[AgentRecord]:
    """Return all agents on the server."""
    result = await session.execute(select(AgentRecord))
    return list(result.scalars().all())


async def agent_exists(session: ReadOnlySession, agent_id: str) -> bool:
    """Return True if an agent with the given ID exists, without loading the full record. TODO: Unit test"""
    stmt = select(exists().where(AgentRecord.id == agent_id))
    result = await session.execute(stmt)
    return result.scalar()


async def replace_agent_config(deps: AgentDeps, config: AgentConfig) -> AgentConfig:
    """Replace agent config in DB. Agent is already verified to exist via deps.

    Resets compaction_warning_fired when soft_compaction_limit is raised — ensures
    the agent gets a fresh warning as the new, larger uncovered zone accumulates.
    Lowering the limit does not reset: compaction will fire sooner, evicting content
    the agent already consolidated when the original warning fired.
    """
    async with deps.write_scope():
        if config.soft_compaction_limit > deps.config.soft_compaction_limit:
            deps.compaction_warning_fired = False
        deps.config = config
    return deps.config


async def replace_system_instructions(deps: AgentDeps, instructions: str) -> str:
    """Replace system instructions in DB and recompile. Agent is already verified to exist via deps."""
    async with deps.write_scope() as session:
        deps.system_instructions = instructions
        await compile_system_prompt(deps, session)
    return deps.system_instructions


async def create_agent_record(
    session: AsyncSession,
    name: str,
    system_instructions: str,
    config: AgentConfig,
) -> AgentRecord:
    """Create a new agent record and flush. Does NOT commit — caller controls
    the transaction boundary (when the agent becomes visible to other requests).
    """
    record = AgentRecord(name=name, system_instructions=system_instructions, agent_config=config)
    session.add(record)
    await session.flush()  # populate record.id
    return record
