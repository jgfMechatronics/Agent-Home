"""
Agent domain types — Section 3.0

Internal domain objects for the agent layer. API layer imports FROM here,
not the reverse.
"""
from dataclasses import dataclass, field
from datetime import datetime

from pydantic import BaseModel, ConfigDict, field_validator
from sqlalchemy.ext.asyncio import AsyncSession

from typing import TYPE_CHECKING
if TYPE_CHECKING:
    from db.models import AgentRecord


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


@dataclass(init=False)
class AgentDeps:
    """
    Dependency bundle for agent operations. Outside tests, should only be constructed by get_deps.
    This enforces the connection between AgentDeps and the lock that get_deps holds,
    making it so deps proves caller holds the per-agent lock.
    TODO: consider having AgentDeps validate it came from get_deps, or associate the lock with AgentDeps
    Best move here is probably have AgentDeps take the lock at construction or something and enforce locked. Unsure if it should live in deps (probably not to reduce unnecessary refs and access)

    _agent_record is private by convention — access via properties.

    All properties read through _agent_record. Callers that mutate the record must call
    await session.refresh(deps._agent_record) after any session.commit() to avoid
    MissingGreenlet on subsequent reads. Mutating tools always hold deps (proves lock),
    so the refresh site is always well-defined.
    """
    session: AsyncSession
    _agent_record: "AgentRecord" = field(repr=False)

    def __init__(self, session: AsyncSession, agent_record: "AgentRecord") -> None:
        self.session = session
        self._agent_record = agent_record

    @property
    def agent_id(self) -> str:
        return self._agent_record.id

    @property
    def name(self) -> str:
        return self._agent_record.name

    @property
    def config(self) -> AgentConfig:
        return self._agent_record.agent_config

    @property
    def system_instructions(self) -> str:
        return self._agent_record.system_instructions or ""

    @property
    def compiled_system_prompt(self) -> str | None:
        return self._agent_record.compiled_system_prompt

    @compiled_system_prompt.setter
    def compiled_system_prompt(self, value: str) -> None:
        # Setters take non-None values intentionally — callers always provide a compiled string
        self._agent_record.compiled_system_prompt = value

    @property
    def sys_prompt_compiled_at(self) -> datetime | None:
        return self._agent_record.sys_prompt_compiled_at

    @sys_prompt_compiled_at.setter
    def sys_prompt_compiled_at(self, value: datetime) -> None:
        # Setters take non-None values intentionally — callers always provide a concrete timestamp
        self._agent_record.sys_prompt_compiled_at = value

    @property
    def context_window_start(self) -> datetime | None:
        return self._agent_record.context_window_start

    @context_window_start.setter
    def context_window_start(self, value: datetime | None) -> None:
        self._agent_record.context_window_start = value
