"""
Agent domain types — Section 3.0

Internal domain objects for the agent layer. API layer imports FROM here,
not the reverse.
"""
import asyncio
from dataclasses import dataclass, field
from datetime import datetime
from typing import TYPE_CHECKING, Literal, get_args, get_origin

from pydantic import AliasChoices, BaseModel, ConfigDict, Field, field_validator
from pydantic_ai.models import parse_model_id
from pydantic_ai.providers import infer_provider_class
from pydantic_ai.settings import ThinkingLevel
from sqlalchemy.ext.asyncio import AsyncSession


if TYPE_CHECKING:
    from db.models import AgentRecord, MemoryBlockRecord


def validate_model_name(model_name: str) -> str:
    """Validate a model name string via pydantic-ai's provider registry.

    Uses parse_model_id + infer_provider_class to check the provider is known
    WITHOUT instantiating it — no API keys required. This makes validation safe
    for standalone DB readers (integrity checker, CLI tools, migration scripts).

    Accepts 'provider:model' format (e.g. 'anthropic:claude-haiku-4-5', 'zai:glm-5.3').
    Bare names are not supported (pydantic-ai 2.x dropped legacy resolution).

    Full instantiation (which requires API keys) happens naturally in
    factory._build_model_settings when the agent actually runs.

    Returns the name unchanged.
    """
    if not model_name.strip():
        raise ValueError("model_name cannot be empty")
    provider, model = parse_model_id(model_name)
    if provider is None:
        raise ValueError(f"Unknown model {model_name!r}: no provider prefix and no recognized model prefix")
    try:
        infer_provider_class(provider)
    except ValueError as e:
        raise ValueError(f"Invalid model_name {model_name!r}: {e}") from e
    if not model.strip():
        raise ValueError("model_name model part cannot be empty")
    return model_name


@dataclass
class AgentAppState:
    """Per-agent app-scoped state, held for the application lifetime.

    Created lazily on first run per agent. Both fields are permanent — the lock
    serializes concurrent requests, and the cancel_requested signals an in-flight
    run to stop.
    """
    lock: asyncio.Lock = field(default_factory=asyncio.Lock)
    cancel_requested: asyncio.Event = field(default_factory=asyncio.Event)


# --- Domain Exceptions ---
# Routes translate these to HTTP status codes (404, 409)

class AgentNotFoundError(Exception):
    """Raised when agent_id doesn't exist in DB."""
    pass


class AgentLockedError(Exception):
    """Raised when agent is already in use by another request."""
    pass


class MCPConnError(Exception):
    """Raised when an attached MCP server is unreachable at run start."""
    pass


def _thinking_effort_levels() -> tuple[str, ...]:
    """Extract the string effort levels from pydantic-ai's ThinkingLevel union.

    ThinkingLevel is `bool | Literal['minimal', ...]` — deriving the Literal's values
    keeps accepted levels in lockstep with pydantic-ai (no hand-maintained list).
    """
    for union_arg in get_args(ThinkingLevel):
        if get_origin(union_arg) is Literal:
            return get_args(union_arg)
    return ()  # unreachable while ThinkingLevel contains a Literal


THINKING_EFFORT_LEVELS: tuple[str, ...] = _thinking_effort_levels()


class AgentConfig(BaseModel):
    """
    Agent configuration stored as JSON in AgentRecord.agent_config.
    
    Required fields:
    - model_name: The LLM to use in 'provider:model' format (e.g., "anthropic:claude-haiku-4-5", "together:glm-4-flash")
      Straight model names will technically work in some cases as of Oct 1, 2026 but pydantic-ai plans to deprecate support for this
    - tool_names: List of tool names the agent can use
    - soft_compaction_limit: Token threshold for triggering compaction
    
    Optional fields:
    - compaction_target_fraction: Fraction of current message tokens to retain after compaction (not a fraction of soft_compaction_limit)
    - toolset_names: List of toolset names to attach (e.g., ["mcp_filesystem"]). Typically used for attaching MCP toolsets
    - is_deletable: Whether agent can be deleted (default False)
    - retries: how many times the agent can retry a failed tool call
    - thinking_mode: False (off) | True (on, provider default effort) | effort level
      ('minimal'/'low'/'medium'/'high'/'xhigh' — pydantic-ai ThinkingLevel, mapped per provider)
    """
    model_config = ConfigDict(extra="forbid") # prevent extra unexpected fields

    model_name: str
    tool_names: list[str]
    toolset_names: list[str] = []
    soft_compaction_limit: int
    compaction_target_fraction: float = 0.25
    is_deletable: bool = False
    retries: int = 4
    # TODO(alias-cleanup): Drop 'thinking_enabled' from AliasChoices once no stored configs
    # contain it. Records migrate lazily — any config write re-serializes with the new name.
    # Verify via: SELECT count(*) FROM agent WHERE agent_config LIKE '%thinking_enabled%';
    thinking_mode: ThinkingLevel = Field(
        default=False,
        validation_alias=AliasChoices("thinking_mode", "thinking_enabled"),
    )
    
    @field_validator("model_name")
    @classmethod
    def _validate_model_name(cls, v: str) -> str:
        return validate_model_name(v)
    
    @field_validator("soft_compaction_limit")
    @classmethod
    def validate_soft_compaction_limit(cls, v: int) -> int:
        if v <= 0:
            raise ValueError("soft_compaction_limit must be positive")
        return v
    
    @field_validator("compaction_target_fraction")
    @classmethod
    def validate_compaction_target_fraction(cls, v: float) -> float:
        if not 0 < v < 1:
            raise ValueError("compaction_target_fraction must be between 0 and 1 (exclusive)")
        return v

    @field_validator("retries")
    @classmethod
    def validate_retries(cls, v: int) -> int:
        if v < 0:
            raise ValueError("retries must be non-negative")
        return v


class BlockSettings(BaseModel):
    """Settings/metadata for a memory block (excludes content and timestamps).
    
    Used for both API requests/responses and as helper function parameter.
    Shared type so API layer and data layer speak the same language.
    
    Defaults align with create_block's defaults so BlockSettings() with just a label
    produces the same behavior as the old create_block(label=...) call.
    """
    label: str = Field(min_length=1)
    description: str = ""
    char_limit: int = Field(default=20000, gt=0)
    position: int | None = Field(default=None, ge=0)

    @classmethod
    def from_record(cls, block: "MemoryBlockRecord") -> "BlockSettings":
        return cls(
            label=block.label,
            description=block.description,
            char_limit=block.char_limit,
            position=block.position,
        )


@dataclass(init=False)
class AgentDeps:
    """
    Dependency bundle for agent operations. Outside tests, should only be constructed by build_deps.
    This enforces the connection between AgentDeps and the lock that build_deps holds,
    making it so deps proves caller holds the per-agent lock.
    TODO: consider having AgentDeps validate it came from build_deps, or associate the lock with AgentDeps
    Best move here is probably have AgentDeps take the lock at construction or something and enforce locked. Unsure if it should live in deps (probably not to reduce unnecessary refs and access)

    _agent_record is private by convention — access via properties.

    All properties read through _agent_record. Whenever possible call
    commit_changes_refresh_agent_record() rather than committing directly — it
    commits and refreshes _agent_record, preventing MissingGreenlet on
    subsequent reads. Mutating callers should always hold deps (proves lock), so the
    commit site is always well-defined.
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

    @config.setter
    def config(self, value: AgentConfig) -> None:
        self._agent_record.agent_config = value

    @property
    def system_instructions(self) -> str:
        return self._agent_record.system_instructions or ""

    @system_instructions.setter
    def system_instructions(self, value: str) -> None:
        self._agent_record.system_instructions = value

    @property
    def compiled_system_prompt(self) -> str:
        return self._agent_record.compiled_system_prompt

    @compiled_system_prompt.setter
    def compiled_system_prompt(self, value: str) -> None:
        self._agent_record.compiled_system_prompt = value

    @property
    def sys_prompt_compiled_at(self) -> datetime | None:
        return self._agent_record.sys_prompt_compiled_at

    @sys_prompt_compiled_at.setter
    def sys_prompt_compiled_at(self, value: datetime) -> None:
        # Setters take non-None values intentionally — callers always provide a concrete timestamp
        self._agent_record.sys_prompt_compiled_at = value

    @property
    def context_window_start(self) -> int:
        return self._agent_record.context_window_start

    @context_window_start.setter
    def context_window_start(self, value: int) -> None:
        self._agent_record.context_window_start = value

    @property
    def compaction_warning_fired(self) -> bool:
        return self._agent_record.compaction_warning_fired

    @compaction_warning_fired.setter
    def compaction_warning_fired(self, value: bool) -> None:
        self._agent_record.compaction_warning_fired = value

    async def commit_changes_refresh_agent_record(self) -> None:
        """Commit the session and immediately refresh _agent_record.

        These two operations should always be coupled: SQLAlchemy expires all ORM attributes
        after a commit, so any subsequent read of a mutable property (compiled_system_prompt,
        context_window_start, etc.) would trigger a lazy reload — which raises MissingGreenlet
        outside SQLAlchemy's own async machinery. Refreshing immediately after the commit
        reloads the record while the async context is still active, keeping the object live.

        Whenever possible, use this instead of calling session.commit() directly.
        """
        await self.session.commit()
        await self.session.refresh(self._agent_record)
