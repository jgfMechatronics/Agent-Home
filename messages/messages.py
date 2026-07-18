"""
Message persistence and retrieval
"""
import logging
from datetime import datetime

from pydantic_ai.messages import (
    ModelMessage,
    ModelMessagesTypeAdapter,
    ModelRequest,
    ModelResponse,
    TextPart,
    ToolCallPart,
    ToolReturnPart,
    RetryPromptPart,
    UserPromptPart,
)
from sqlalchemy import func, select
from sqlalchemy.ext.asyncio import AsyncSession

from agent.types import AgentDeps
from db.models import MessageRecord, utcnow

log = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_orphan_replacement(
    msg: ModelMessage, part_type: type
) -> tuple[tuple[datetime, str], ModelResponse]:
    """Build the error entry and replacement ModelResponse for an orphaned tool message.

    part_type should be ToolCallPart, ToolReturnPart, or RetryPromptPart.
    Returns (error_entry, error_response) ready to append to errors and sanitized_msgs.
    """
    match part_type.__name__:
        case "ToolCallPart":
            label = "call"
        case "ToolReturnPart":
            label = "return"
        case "RetryPromptPart":
            label = "retry"
        case unexpected:
            label = f"part of unexpected type {unexpected}"

    # RetryPromptPart.tool_name may be None — filter to avoid join errors
    tool_names = [p.tool_name for p in msg.parts if isinstance(p, part_type) and p.tool_name is not None]
    error_text = f"[Orphaned tool {label}(s) dropped: {', '.join(tool_names)}]"
    return (_message_timestamp(msg), error_text), ModelResponse(parts=[TextPart(content=error_text)])


def _is_valid_tool_pair(call_msg: ModelMessage | None, return_msg: ModelMessage | None) -> bool:
    """True if call_msg/return_msg form a matched tool call/return pair.

    Requires call_msg to be a ModelResponse with ToolCallPart(s) and return_msg to be a
    ModelRequest with ToolReturnPart(s) or RetryPromptPart(s) sharing at least one tool_call_id.
    Both ToolReturnPart and RetryPromptPart are valid responses to a ToolCallPart.
    """
    if not isinstance(call_msg, ModelResponse) or not isinstance(return_msg, ModelRequest):
        return False
    call_ids = {p.tool_call_id for p in call_msg.parts if isinstance(p, ToolCallPart)}
    return_ids = {p.tool_call_id for p in return_msg.parts if isinstance(p, (ToolReturnPart, RetryPromptPart))}
    return call_ids == return_ids  # every call must have a return/retry and vice versa


def _replace_orphaned_tool_messages(
    messages: list[ModelMessage],
) -> tuple[list[ModelMessage], list[tuple[datetime, str]]]:
    """Replace orphaned tool calls and returns with error ModelResponses.

    - Orphaned call: a ModelResponse with ToolCallPart(s) not immediately followed by
      a ModelRequest with matching ToolReturnPart(s).
    - Orphaned return: a ModelRequest with ToolReturnPart(s) not immediately preceded by
      a ModelResponse with matching ToolCallPart(s).

    Returns (processed_messages, errors) where errors is a list of (original_ts, error_text)
    pairs suitable for summary warning appending at end of message chain in persist_messages.

    TODO: This algorithm is rather inefficient and a bit confusing
    The two cases could probably be commonized further but that doesn't solve touching every element on the list
    It also checks that every tool call has a return and every tool return has a call which is redundant. 
    It also cannot properly handle multiple tool parts in a single msg (parallel calls?)
    Def a better way to do this.
    """
    sanitized_msgs: list[ModelMessage] = []
    errors: list[tuple[datetime, str]] = []
    for i, msg in enumerate(messages):
        if isinstance(msg, ModelResponse) and any(isinstance(p, ToolCallPart) for p in msg.parts):
            next_msg = messages[i + 1] if i + 1 < len(messages) else None
            if not _is_valid_tool_pair(msg, next_msg):
                error_entry, error_response = _make_orphan_replacement(msg, ToolCallPart)
                errors.append(error_entry)
                sanitized_msgs.append(error_response)
                continue # Skips below append of original msg
        elif isinstance(msg, ModelRequest) and any(isinstance(p, (ToolReturnPart, RetryPromptPart)) for p in msg.parts):
            prev_msg = sanitized_msgs[-1] if sanitized_msgs else None
            if not _is_valid_tool_pair(prev_msg, msg):
                part_type = ToolReturnPart if any(isinstance(p, ToolReturnPart) for p in msg.parts) else RetryPromptPart
                error_entry, error_response = _make_orphan_replacement(msg, part_type)
                errors.append(error_entry)
                sanitized_msgs.append(error_response)
                continue
        sanitized_msgs.append(msg)
    return sanitized_msgs, errors


def _dump_msg_json(msg: ModelMessage) -> str:
    """Serialize a single ModelMessage to a JSON string (without outer array brackets)."""
    return ModelMessagesTypeAdapter.dump_json([msg]).decode()[1:-1]


def _message_timestamp(msg: ModelMessage) -> datetime:
    """Extract a naive UTC datetime from a ModelMessage for storage.

    ModelResponse always has .timestamp set (auto-assigned on construction).
    ModelRequest.timestamp is None by default; fall back to the first
    UserPromptPart.timestamp if available, otherwise use current time.

    TODO: revisit if ModelRequest ever carries its own timestamp in a future
    pydantic_ai version, or if non-UserPrompt parts need timestamp handling.
    """
    ts: datetime | None = msg.timestamp
    if ts is None and isinstance(msg, ModelRequest):
        ts = next(
            (p.timestamp for p in msg.parts if isinstance(p, UserPromptPart)),
            None,
        )
    if ts is None:
        return utcnow()
    # Strip tzinfo — SQLite stores naive datetimes
    return ts.replace(tzinfo=None) if ts.tzinfo is not None else ts


def _handle_serialization_error(
    msg: ModelMessage,
    e: Exception,
    agent_id: str,
) -> tuple[str, str, ModelMessage, tuple[datetime, str]]:
    """Build an error ModelResponse in place of a message that failed to serialize.

    Logs the failure and returns (content, msg_type, error_msg, error_entry) where
    error_entry is (original_ts, error_text) for the caller to append to its errors list.
    Must be called from within the except block so log.exception captures the active traceback.
    """
    original_ts = _message_timestamp(msg)
    log.exception(
        "persist_messages: unexpected serialization failure for agent %s (%s); injecting error record",
        agent_id, e,
    )
    error_text = f"[persist_messages serialization error]: {type(e).__name__}: {e}"
    error_msg = ModelResponse(parts=[TextPart(content=error_text)])
    content = _dump_msg_json(error_msg)
    error_to_append = (original_ts, error_text) # return this for caller to append to avoid sneakily mutating list
    return content, "ModelResponse", error_msg, error_to_append


# ---------------------------------------------------------------------------
# Functions
# ---------------------------------------------------------------------------

async def persist_messages(deps: AgentDeps, messages: list[ModelMessage]) -> int | None:
    """Save each ModelMessage as its own row; set total_tokens from ModelResponse.usage where available.

    total_tokens is extracted from each ModelResponse's usage (input + output tokens for that request).
    Set to None for ModelRequests and for ModelResponses with no token data (usage has all-zero fields).

    Returns the last non-None total_tokens seen across all messages, or None if no usage data was found.

    Pre-processing:
    - Orphaned tool calls/returns (unmatched ToolCallPart or ToolReturnPart) are replaced with an error ModelResponse.
    - Serialization failures inject an error ModelResponse in place of the bad message.
    - Both orphan replacements and serialization failures append a summary warning at the END of
      the chain (referencing the original message timestamp) so errors aren't buried in long histories.
    """
    if not messages:
        return None

    # Get next seq_id: MAX(seq_id) + 1, or 0 if no messages exist
    result = await deps.session.execute(
        select(func.max(MessageRecord.seq_id)).where(MessageRecord.agent_id == deps.agent_id)
    )
    max_seq_id = result.scalar()
    next_seq_id = max_seq_id + 1 if max_seq_id is not None else 0

    messages, errors = _replace_orphaned_tool_messages(messages)
    last_total_tokens: int | None = None

    for i, msg in enumerate(messages):
        try:
            # NOTE: The per msg serialization allows us to eliminate specific messages which have serialization failures,
            # but likely costs us some performance. This is an optimization opportunity: could have happy path try serializing the whole
            # list then on failure go message by message
            content = _dump_msg_json(msg)
            msg_type = type(msg).__name__
        except Exception as e:
            content, msg_type, msg, error_to_append = _handle_serialization_error(msg, e, deps.agent_id)
            errors.append(error_to_append)

        msg_total_tokens = msg.usage.total_tokens if isinstance(msg, ModelResponse) and msg.usage.has_values() else None
        if msg_total_tokens is not None:
            last_total_tokens = msg_total_tokens
        record = MessageRecord(
            agent_id=deps.agent_id,
            type=msg_type,
            content=content,
            total_tokens=msg_total_tokens,
            seq_id=next_seq_id + i,
            timestamp=_message_timestamp(msg),
        )
        deps.session.add(record)

    # Append notification of any errors that were encountered to the end of the message string
    # Continue seq_id sequence from where we left off
    error_seq_start = next_seq_id + len(messages)
    for j, (original_timestamp, error_text) in enumerate(errors):
        warning = (
            f"WARNING: A problem was encountered while persisting messages from the last turn: "
            f"'{error_text}'. A warning was injected in place of the problematic message, "
            f"problematic message timestamp was {original_timestamp}"
        )
        warning_msg = ModelResponse(parts=[TextPart(content=warning)])
        warning_content = _dump_msg_json(warning_msg)
        deps.session.add(MessageRecord(
            agent_id=deps.agent_id,
            type="ModelResponse",
            content=warning_content,
            total_tokens=None,
            seq_id=error_seq_start + j,
            timestamp=_message_timestamp(warning_msg),
        ))

    await deps.session.flush()
    return last_total_tokens


async def load_messages(
    session: AsyncSession,
    agent_id: str,
    start_seq_id: int = 0,
) -> list[MessageRecord]:
    """Load messages as ORM records in seq_id order.

    Returns messages where seq_id >= start_seq_id. Defaults to 0, which returns
    the full conversation history (all seq_ids are non-negative).
    """
    query = (
        select(MessageRecord)
        .where(MessageRecord.agent_id == agent_id)
        .where(MessageRecord.seq_id >= start_seq_id)
        .order_by(MessageRecord.seq_id)
    )

    result = await session.execute(query)
    return list(result.scalars().all())


def deserialize_messages(records: list[MessageRecord]) -> list[ModelMessage]:
    """Convert MessageRecords to Pydantic AI ModelMessages.

    Pure function — no database access. Handles all message types including summaries.
    Raises ValueError on invalid records — caller is responsible for error handling.
    """
    result: list[ModelMessage] = []
    for record in records:
        try:
            # Content is single-message JSON; wrap in array for TypeAdapter
            messages = ModelMessagesTypeAdapter.validate_json(f"[{record.content}]")
            if messages:
                result.append(messages[0])
        except Exception as e:
            raise ValueError(f"[Deserialization error for record {record.id}]: {e}") from e
    return result
