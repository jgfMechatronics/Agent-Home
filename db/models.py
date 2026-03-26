from datetime import datetime
import uuid

from sqlalchemy import ForeignKey, Index, JSON, UniqueConstraint, func
from sqlalchemy.orm import DeclarativeBase, Mapped, mapped_column, relationship


class Base(DeclarativeBase):
    pass


class AgentRecord(Base):
    __tablename__ = "agent"

    id: Mapped[str] = mapped_column(primary_key=True, default=lambda: str(uuid.uuid4()))
    name: Mapped[str]
    agent_config: Mapped[dict] = mapped_column(JSON)
    system_instructions: Mapped[str] = mapped_column(default='')
    compiled_system_prompt: Mapped[str] = mapped_column(default='')
    sys_prompt_compiled_at: Mapped[datetime | None]
    context_window_start: Mapped[datetime | None]
    # SQLite uses utc internally by default, matches our intent
    created_at: Mapped[datetime] = mapped_column(server_default=func.now())
    updated_at: Mapped[datetime] = mapped_column(server_default=func.now(), onupdate=func.now())

    memory_blocks: Mapped[list["MemoryBlockRecord"]] = relationship(cascade="all, delete-orphan")
    messages: Mapped[list["MessageRecord"]] = relationship(cascade="all, delete-orphan")

    def __repr__(self) -> str:
        return f"AgentRecord(id={self.id!r}, name={self.name!r})"


class MemoryBlockRecord(Base):
    __tablename__ = "memory_block"
    __table_args__ = (
        UniqueConstraint("agent_id", "label"),
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
    created_at: Mapped[datetime] = mapped_column(server_default=func.now())
    updated_at: Mapped[datetime] = mapped_column(server_default=func.now(), onupdate=func.now())

    def __repr__(self) -> str:
        return f"MemoryBlockRecord(id={self.id!r}, agent_id={self.agent_id!r}, label={self.label!r})"


class MessageRecord(Base):
    __tablename__ = "message"
    __table_args__ = (
        # Primary access pattern: load history by agent in timestamp order
        Index("ix_message_agent_timestamp", "agent_id", "timestamp"),
        # Type queries (e.g. find last ModelResponse with input_tokens) — timestamp DESC intent,
        # SQLite optimises both directions from a single index
        Index("ix_message_agent_type_timestamp", "agent_id", "type", "timestamp"),
    )

    id: Mapped[str] = mapped_column(primary_key=True, default=lambda: str(uuid.uuid4()))
    agent_id: Mapped[str] = mapped_column(ForeignKey("agent.id", ondelete="CASCADE"))
    type: Mapped[str]
    content: Mapped[str]  # TEXT storing serialized ModelMessage JSON — not deserialized by SQLAlchemy
    input_tokens: Mapped[int | None]
    timestamp: Mapped[datetime]

    def __repr__(self) -> str:
        return f"MessageRecord(id={self.id!r}, agent_id={self.agent_id!r}, type={self.type!r})"
