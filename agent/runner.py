"""
Agent runner — Section 3.1

Provides AgentDeps (the capability token for write operations) and
agent construction utilities.
"""
from dataclasses import dataclass

from sqlalchemy.ext.asyncio import AsyncSession


@dataclass
class AgentDeps:
    """
    Dependency bundle for agent operations. Proves caller holds the per-agent lock.
    
    Write operations take this instead of raw session to enforce lock discipline.
    """
    session: AsyncSession
    agent_id: str
    # TODO: Add config when AgentConfig is implemented
