from datetime import datetime, timezone
import uuid

from sqlalchemy import ForeignKey, Index, JSON, UniqueConstraint
from sqlalchemy.orm import DeclarativeBase, Mapped, mapped_column, relationship
from sqlalchemy.types import TypeDecorator

from agent.types import AgentConfig


def utcnow() -> datetime:
    """Return current UTC time as a naive datetime.

    Naive (tzinfo-stripped) because SQLite has no timezone type — storing aware
    datetimes causes silent truncation or round-trip failures.

    TODO: Verify whether SQLite actually needs the strip. If the dialect treats
    UTC-aware datetimes (tzinfo=UTC, offset=0) as equivalent to naive UTC on
    round-trip, we could drop the .replace(tzinfo=None) and keep full awareness.
    """
    return datetime.now(timezone.utc).replace(tzinfo=None)

# TODO: Consider upgrading to UUID7 as a sortable UUID fallback if timestamps fail, but if timestamps fail we may be in trouble regardless

class AgentConfigType(TypeDecorator):
    """Stores AgentConfig as JSON in DB, exposes as AgentConfig instance in Python."""

    impl = JSON
    cache_ok = True

    def process_bind_param(self, value, dialect):
        """Python → Database (when writing/updating)."""
        if value is None:
            return None
        if not isinstance(value, AgentConfig):
            raise TypeError(f"Expected AgentConfig, got {type(value).__name__}")
        return value.model_dump()

    def process_result_value(self, value, dialect):
        """Database → Python (when reading/loading)."""
        if value is None:
            return None
        # Migration logic can be added here: value.setdefault("new_field", default)
        return AgentConfig.model_validate(value)


class Base(DeclarativeBase):
    pass


class AgentRecord(Base):
    __tablename__ = "agent"

    id: Mapped[str] = mapped_column(primary_key=True, default=lambda: str(uuid.uuid4()))
    name: Mapped[str] = mapped_column(unique=True)
    agent_config: Mapped[AgentConfig] = mapped_column(AgentConfigType())
    system_instructions: Mapped[str] = mapped_column(default='')
    compiled_system_prompt: Mapped[str] = mapped_column(default='')
    sys_prompt_compiled_at: Mapped[datetime | None]
    context_window_start: Mapped[int | None]
    # SQLite uses utc internally by default, matches our intent
    created_at: Mapped[datetime] = mapped_column(default=utcnow)
    updated_at: Mapped[datetime] = mapped_column(default=utcnow, onupdate=utcnow)

    memory_blocks: Mapped[list["MemoryBlockRecord"]] = relationship(cascade="all, delete-orphan")
    messages: Mapped[list["MessageRecord"]] = relationship(cascade="all, delete-orphan")

    def __repr__(self) -> str:
        return f"AgentRecord(id={self.id!r}, name={self.name!r})"


class MemoryBlockRecord(Base):
    __tablename__ = "memory_block"
    __table_args__ = (
        UniqueConstraint("agent_id", "label"),
        UniqueConstraint("agent_id", "position"),
        Index("ix_memory_block_agent_position", "agent_id", "position"),
        Index("ix_memory_block_agent_label", "agent_id", "label"),
    )

    id: Mapped[str] = mapped_column(primary_key=True, default=lambda: str(uuid.uuid4()))
    agent_id: Mapped[str] = mapped_column(ForeignKey("agent.id", ondelete="CASCADE"))
    label: Mapped[str]
    description: Mapped[str]
    content: Mapped[str]
    char_limit: Mapped[int]
    position: Mapped[int]
    created_at: Mapped[datetime] = mapped_column(default=utcnow)
    updated_at: Mapped[datetime] = mapped_column(default=utcnow, onupdate=utcnow)

    def __repr__(self) -> str:
        return f"MemoryBlockRecord(id={self.id!r}, agent_id={self.agent_id!r}, label={self.label!r})"


class MessageRecord(Base):
    __tablename__ = "message"
    __table_args__ = (
        # Primary access pattern: load history by agent in seq_id order
        Index("ix_message_agent_seq_id", "agent_id", "seq_id"),
    )

    id: Mapped[str] = mapped_column(primary_key=True, default=lambda: str(uuid.uuid4()))
    agent_id: Mapped[str] = mapped_column(ForeignKey("agent.id", ondelete="CASCADE"))
    # NOTE: Currently type is either "ModelResponse" or "ModelRequest" IE the union types of ModelMessage.
    # We may later want to make this more custom/useful, stuff like "Summary", "ToolCall", "ToolResponse"
    type: Mapped[str]
    content: Mapped[str]  # TEXT storing serialized ModelMessage JSON — not deserialized by SQLAlchemy
    total_tokens: Mapped[int | None]  # sum of input + output tokens for the LLM request associated with this message; None for ModelRequests and error rows
    seq_id: Mapped[int | None]  # per-agent monotonic ordinal; Used for ordering in requests and such; TODO: NULL until assigned
    timestamp: Mapped[datetime]

    def __repr__(self) -> str:
        return f"MessageRecord(id={self.id!r}, agent_id={self.agent_id!r}, type={self.type!r})"
