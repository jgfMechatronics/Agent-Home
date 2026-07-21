"""
Context reconstructor — reconstruct the exact context an LLM saw at any historical point.

This is a standalone module with direct read-only DB access. It does not require
the server to be running.
"""
from dataclasses import dataclass
from uuid import UUID

from sqlalchemy.ext.asyncio import AsyncSession

from db.models import MessageRecord


@dataclass
class ReconstructedContext:
    """The full context that existed when a specific message was processed/generated.
    
    Attributes:
        system_prompt: The compiled system prompt that was active
        tool_schemas: List of tool schema dicts that were available
        messages: MessageRecords from context_window_start up to (exclusive) target
        target_message: The message you asked about (the focal point)
        agent_id: The agent this context belongs to
    """
    system_prompt: str
    tool_schemas: list[dict]
    messages: list[MessageRecord]
    target_message: MessageRecord
    agent_id: str


async def reconstruct_context(session: AsyncSession, message_id: str) -> ReconstructedContext:
    """Reconstruct the context that existed when a specific message was processed.
    
    Args:
        session: SQLAlchemy async session
        message_id: UUID of the target message
        
    Returns:
        ReconstructedContext with system prompt, tools, message history, and target
        
    Raises:
        ValueError: If message_id not found
    """
    raise NotImplementedError("reconstruct_context not yet implemented")
