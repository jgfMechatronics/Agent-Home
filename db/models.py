from sqlalchemy import JSON, func
from sqlalchemy.orm import DeclarativeBase, Mapped, mapped_column
from datetime import datetime
import uuid

class Base(DeclarativeBase):
    pass

class AgentRecord(Base):
    __tablename__ = "agent"
    
    id: Mapped[str] = mapped_column(primary_key=True, default=lambda: str(uuid.uuid4()))
    name: Mapped[str]
    agent_config: Mapped[dict] = mapped_column(JSON)
    system_instructions: Mapped[str]
    compiled_system_prompt: Mapped[str] = mapped_column(default='')
    sys_prompt_compiled_at: Mapped[datetime | None]
    context_window_start: Mapped[datetime | None]
    # SQLite uses utc internally by default, matches our intent
    created_at: Mapped[datetime] = mapped_column(server_default=func.now())
    updated_at: Mapped[datetime] = mapped_column(server_default=func.now(), onupdate=func.now())

    def __repr__(self) -> str:
        return f"Agent ID: {self.id!r}, Name: {self.name!r}" # Sonnet please finish this repr