"""
Context reconstructor — reconstruct the exact context an LLM saw at any historical point.

This is a standalone module with direct read-only DB access. It does not require
the server to be running.

TODO CRITICAL: When setting this up to actually run standalone, MAKE SURE to use a READ ONLY ENGINE
"""
import json
from dataclasses import dataclass

from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from db.models import MessageRecord, SystemPromptSnapshot, ToolSchemaSnapshot
from messages.messages import load_messages


@dataclass
class ReconstructedContext:
    """The full context that existed when a specific message was processed/generated.
    
    Attributes:
        system_prompt: The compiled system prompt that was active
        tool_schemas: List of tool schema dicts that were available
        messages: MessageRecords from context_window_start up to (exclusive) target
        target_message: The message you asked about (the focal point)
        agent_id: The agent this context belongs to
    
    target_message is the ONLY message in the ReconstructedContext where the context is guarenteed
    to be as described. IE if you pick a different message from message list, that message may have been sent/generated
    with a different system prompt, different tool schema, or different messages in context.
    If you want the context associated with a different message from messages, then rerun reconstruct_context with said message ID
    
    target_message can be any message type, and the interpretation varies slightly based on type. 
    For example:
    When target_message is a ModelRequest, then the rest of the ReconstructedContext can be interpreted as the context which was sent
    along with the target when the target was sent

    When target_message is a ModelResponse, the rest of ReconstructedContext can be interpreted as the the context which the generation
    of target_message was conditioned on
    """
    system_prompt: str
    tool_schemas: list[dict]
    messages: list[MessageRecord]
    target_message: MessageRecord
    agent_id: str


async def reconstruct_context(session: AsyncSession, target_message_id: str) -> ReconstructedContext:
    """Reconstruct the context that existed when a specified target message was processed.
    
    Args:
        session: SQLAlchemy async session
        target_message_id: UUID of the target message
        
    Returns:
        ReconstructedContext with system prompt, tools, message history, and target
        
    Raises:
        ValueError: If target_message_id not found
    """
    target = await session.execute(
        select(MessageRecord).where(MessageRecord.id == target_message_id)
    )
    target = target.scalar_one_or_none()
    if target is None:
        raise ValueError(f"Message not found: {target_message_id}")
    
    sys_snapshot = await session.execute(
        select(SystemPromptSnapshot).where(
            SystemPromptSnapshot.id == target.system_prompt_hash
        )
    )
    sys_snapshot = sys_snapshot.scalar_one()
    
    tool_snapshot = await session.execute(
        select(ToolSchemaSnapshot).where(
            ToolSchemaSnapshot.id == target.tool_schema_hash
        )
    )
    tool_snapshot = tool_snapshot.scalar_one()
    
    # Fetch context_window_start message to get its seq_id
    context_start = await session.execute(
        select(MessageRecord).where(
            MessageRecord.id == target.context_window_start_msg_id
        )
    )
    context_start = context_start.scalar_one()
    
    messages = await load_messages(
        session,
        target.agent_id,
        start_seq_id=context_start.seq_id,
        end_seq_id=target.seq_id,
    )
    
    return ReconstructedContext(
        system_prompt=sys_snapshot.content,
        tool_schemas=json.loads(tool_snapshot.content),
        messages=messages,
        target_message=target,
        agent_id=target.agent_id,
    )
