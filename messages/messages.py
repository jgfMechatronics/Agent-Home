"""
Message persistence and retrieval
"""
from pydantic_ai.messages import ModelMessage

from agent.types import AgentDeps

from typing import TYPE_CHECKING
if TYPE_CHECKING:
    from datetime import datetime
    from sqlalchemy.ext.asyncio import AsyncSession
    from db.models import MessageRecord


async def persist_messages(deps: AgentDeps, messages: list[ModelMessage], input_tokens: int) -> None:
    """Save each ModelMessage as its own row; set input_tokens on the final row only."""
    raise NotImplementedError


async def load_in_context_messages(session: "AsyncSession", agent_id: str) -> list[ModelMessage]:
    """Load messages from context_window_start, deserialized to list[ModelMessage]."""
    raise NotImplementedError


async def load_message_history(
    session: "AsyncSession",
    agent_id: str,
    start_timestamp: "datetime | None" = None,
) -> list["MessageRecord"]:
    """Load message history as raw records.
    
    If start_timestamp is provided, returns only messages equal to and after that point.
    Otherwise returns complete conversation history.
    """
    raise NotImplementedError
