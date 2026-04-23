"""
Agent domain types — Section 3.0

Internal domain objects for the agent layer. API layer imports FROM here,
not the reverse.
"""
from dataclasses import dataclass

from pydantic import BaseModel, ConfigDict, field_validator
from sqlalchemy.ext.asyncio import AsyncSession


class AgentConfig(BaseModel):
    """
    Agent configuration stored as JSON in AgentRecord.agent_config.
    
    Required fields:
    - model_name: The LLM model to use (e.g., "claude-sonnet-4-20250514")
    - tool_names: List of tool names the agent can use
    - soft_compaction_limit: Token threshold for triggering compaction
    
    Optional fields:
    - compaction_target_percentage: Target context size after compaction as fraction of soft_compaction_limit
    - is_deletable: Whether agent can be deleted (default False)
    """
    model_config = ConfigDict(extra="forbid")
    
    model_name: str
    tool_names: list[str]
    soft_compaction_limit: int
    compaction_target_percentage: float = 0.25
    is_deletable: bool = False
    
    @field_validator("model_name")
    @classmethod
    def validate_model_name(cls, v: str) -> str:
        # TODO, once get_model implemented validate that str corresponds to a valid AnthropicModel.
        # Or, consider just storing model_name as an AnthropicModel and dealing with the DB integration.
        if not v.strip():
            raise ValueError("model_name cannot be empty")
        return v
    
    @field_validator("soft_compaction_limit")
    @classmethod
    def validate_soft_compaction_limit(cls, v: int) -> int:
        if v <= 0:
            raise ValueError("soft_compaction_limit must be positive")
        return v
    
    @field_validator("compaction_target_percentage")
    @classmethod
    def validate_compaction_target_percentage(cls, v: float) -> float:
        if not 0 < v < 1:
            raise ValueError("compaction_target_percentage must be between 0 and 1 (exclusive)")
        return v


@dataclass
class AgentDeps:
    """
    Dependency bundle for agent operations. Outside tests, should only be constructed by get_deps.
    This enforces the connection between AgentDeps and the lock that get_deps holds, 
    making it so deps proves caller holds the per-agent lock.
    TODO: consider having AgentDeps validate it came from get_deps, or associate the lock with AgentDeps
    """
    session: AsyncSession
    agent_id: str
    config: AgentConfig
    name: str
