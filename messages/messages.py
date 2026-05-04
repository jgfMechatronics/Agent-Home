"""
Message persistence and retrieval — Section 5
"""
from pydantic_ai.messages import ModelMessage

from agent.types import AgentDeps


async def persist_messages(deps: AgentDeps, messages: list[ModelMessage], input_tokens: int) -> None:
    """Save each ModelMessage as its own row; set input_tokens on the final row only."""
    raise NotImplementedError


async def load_in_context_messages(deps: AgentDeps) -> list[ModelMessage]:
    """Load messages from context_window_start, deserialized to list[ModelMessage]."""
    raise NotImplementedError
