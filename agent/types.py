"""
Agent domain types — Section 3.0

Internal domain objects for the agent layer. API layer imports FROM here,
not the reverse.
"""
from dataclasses import dataclass

from pydantic import BaseModel
from sqlalchemy.ext.asyncio import AsyncSession


class AgentConfig(BaseModel):
    """
    Agent configuration stored as JSON in AgentRecord.agent_config.
    
    Stub for now — will be fleshed out as we implement agent runner.
    """
    model_name: str = "claude-sonnet-4-20250514"
    # TODO: Add tool settings, context limits, etc.


@dataclass
class AgentDeps:
    """
    Dependency bundle for agent operations. Outside tests, should only be constructed by get_deps.
    This enforces the connection between AgentDeps and the lock that get_deps holds, 
    making it so deps proves caller holds the per-agent lock.
    """
    session: AsyncSession
    agent_id: str
    config: AgentConfig
