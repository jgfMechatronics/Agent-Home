"""
Message persistence and retrieval — Section 5
"""
from pydantic_ai.messages import ModelMessage

from agent.types import AgentDeps


async def load_in_context_messages(deps: AgentDeps) -> list[ModelMessage]:
    """Load messages from context_window_start, deserialized to list[ModelMessage]."""
    raise NotImplementedError
