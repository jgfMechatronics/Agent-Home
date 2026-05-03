"""
Agent CRUD operations — Section 3.x
"""
from sqlalchemy.ext.asyncio import AsyncSession

from agent.types import AgentConfig


async def create_agent(
    session: AsyncSession,
    name: str,
    system_instructions: str,
    config: AgentConfig,
) -> str:
    """Create a new agent and return its ID."""
    raise NotImplementedError
