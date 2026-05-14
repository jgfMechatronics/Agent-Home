"""
Message persistence and retrieval
"""
import logging
from datetime import datetime, timedelta

from pydantic_ai.messages import (
    ModelMessage,
    ModelMessagesTypeAdapter,
    ModelRequest,
    ModelResponse,
    TextPart,
    ToolCallPart,
    ToolReturnPart,
    UserPromptPart,
)
from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from agent.types import AgentDeps
from db.models import MessageRecord, utcnow

log = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

async def _get_last_timestamp(session: AsyncSession, agent_id: str) -> datetime | None:
    """Return the timestamp of the most recent message for this agent, or None."""
    result = await session.execute(
        select(MessageRecord.timestamp)
        .where(MessageRecord.agent_id == agent_id)
        .order_by(MessageRecord.timestamp.desc())
        .limit(1)
    )
    return result.scalar_one_or_none()


def _replace_orphaned_tool_messages(
    messages: list[ModelMessage],
) -> tuple[list[ModelMessage], list[tuple[datetime, str]]]:
    """Replace orphaned tool calls and returns with error ModelResponses.

    - Orphaned call: a ModelResponse with ToolCallPart(s) not immediately followed by
      a ModelRequest with matching ToolReturnPart(s).
    - Orphaned return: a ModelRequest with ToolReturnPart(s) not immediately preceded by
      a ModelResponse with matching ToolCallPart(s).

    Returns (processed_messages, errors) where errors is a list of (original_ts, error_text)
    pairs suitable for summary warning appending in persist_messages.
    """
    result: list[ModelMessage] = []
    errors: list[tuple[datetime, str]] = []
    for i, msg in enumerate(messages):
        if isinstance(msg, ModelResponse) and any(isinstance(p, ToolCallPart) for p in msg.parts):
            next_msg = messages[i + 1] if i + 1 < len(messages) else None
            if not _is_matching_tool_return(next_msg, msg):
                tool_names = [p.tool_name for p in msg.parts if isinstance(p, ToolCallPart)]
                error_text = f"[Orphaned tool call(s) dropped: {', '.join(tool_names)}]"
                errors.append((_message_timestamp(msg), error_text))
                result.append(ModelResponse(parts=[TextPart(content=error_text)]))
                continue
        elif isinstance(msg, ModelRequest) and any(isinstance(p, ToolReturnPart) for p in msg.parts):
            prev_msg = result[-1] if result else None
            if not _has_matching_tool_call(prev_msg, msg):
                tool_names = [p.tool_name for p in msg.parts if isinstance(p, ToolReturnPart)]
                error_text = f"[Orphaned tool return(s) dropped: {', '.join(tool_names)}]"
                errors.append((_message_timestamp(msg), error_text))
                result.append(ModelResponse(parts=[TextPart(content=error_text)]))
                continue
        result.append(msg)
    return result, errors


def _has_matching_tool_call(prev_msg: ModelMessage | None, return_msg: ModelRequest) -> bool:
    """True if prev_msg is a ModelResponse with ToolCallPart(s) matching return_msg's returns."""
    if not isinstance(prev_msg, ModelResponse):
        return False
    call_ids = {p.tool_call_id for p in prev_msg.parts if isinstance(p, ToolCallPart)}
    return_ids = {p.tool_call_id for p in return_msg.parts if isinstance(p, ToolReturnPart)}
    return bool(call_ids & return_ids)


def _is_matching_tool_return(next_msg: ModelMessage | None, call_msg: ModelResponse) -> bool:
    """True if next_msg is a ModelRequest containing ToolReturnPart(s) matching call_msg's calls.

    TODO: Currently uses set intersection — if call_msg has calls A and B but next_msg only
    returns A, this returns True (partial match). Consider requiring ALL call_ids to be matched.
    """
    if not isinstance(next_msg, ModelRequest):
        return False
    call_ids = {p.tool_call_id for p in call_msg.parts if isinstance(p, ToolCallPart)}
    return_ids = {p.tool_call_id for p in next_msg.parts if isinstance(p, ToolReturnPart)}
    return bool(call_ids & return_ids)


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


# ---------------------------------------------------------------------------
# Functions
# ---------------------------------------------------------------------------

async def persist_messages(deps: AgentDeps, messages: list[ModelMessage], input_tokens: int) -> None:
    """Save each ModelMessage as its own row; set input_tokens on the final row only.

    Pre-processing:
    - Orphaned tool calls/returns (unmatched ToolCallPart or ToolReturnPart) are replaced with an error ModelResponse.
    - Serialization failures inject an error ModelResponse in place of the bad message.
    - Both orphan replacements and serialization failures append a summary warning at the END of
      the chain (referencing the original message timestamp) so errors aren't buried in long histories.
    - Timestamps are validated against the last DB record; out-of-order timestamps are bumped
      forward by 1 microsecond and a warning is logged.
    """
    if not messages:
        return

    messages, errors = _replace_orphaned_tool_messages(messages)
    last_ts = await _get_last_timestamp(deps.session, deps.agent_id)
    last_idx = len(messages) - 1

    for i, msg in enumerate(messages):
        try:
            content_bytes = ModelMessagesTypeAdapter.dump_json([msg])
            # Strip outer array brackets — store one message object per row
            content = content_bytes.decode()[1:-1]
            msg_type = type(msg).__name__
        except Exception as e:
            original_ts = _message_timestamp(msg)  # capture before replacing msg
            log.exception("persist_messages: unexpected serialization failure for agent %s (%s); injecting error record", deps.agent_id, e)
            error_text = f"[persist_messages serialization error]: {type(e).__name__}: {e}"
            error_msg = ModelResponse(parts=[TextPart(content=error_text)])
            content_bytes = ModelMessagesTypeAdapter.dump_json([error_msg])
            content = content_bytes.decode()[1:-1]
            msg_type = "ModelResponse"
            msg = error_msg
            errors.append((original_ts, error_text))

        ts = _message_timestamp(msg)
        if last_ts is not None and ts <= last_ts:
            log.warning(
                "persist_messages: timestamp ordering violation for agent %s "
                "(msg ts=%s <= last_ts=%s); bumping forward by 1µs",
                deps.agent_id, ts, last_ts,
            )
            ts = last_ts + timedelta(microseconds=1)
        last_ts = ts

        record = MessageRecord(
            agent_id=deps.agent_id,
            type=msg_type,
            content=content,
            input_tokens=input_tokens if i == last_idx else None,
            timestamp=ts,
        )
        deps.session.add(record)

    for original_ts, error_text in errors:
        warning = (
            f"WARNING: A problem was encountered while persisting messages from the last turn: "
            f"'{error_text}'. A warning was injected in place of the problematic message, "
            f"problematic message timestamp was {original_ts}"
        )
        warning_msg = ModelResponse(parts=[TextPart(content=warning)])
        warning_content = ModelMessagesTypeAdapter.dump_json([warning_msg]).decode()[1:-1]
        warn_ts = _message_timestamp(warning_msg)
        if last_ts is not None and warn_ts <= last_ts:
            warn_ts = last_ts + timedelta(microseconds=1)
        last_ts = warn_ts
        deps.session.add(MessageRecord(
            agent_id=deps.agent_id,
            type="ModelResponse",
            content=warning_content,
            input_tokens=None,
            timestamp=warn_ts,
        ))

    await deps.session.flush()


async def load_messages(
    session: AsyncSession,
    agent_id: str,
    start_timestamp: datetime | None = None,
) -> list[MessageRecord]:
    """Load messages as ORM records in chronological order.

    If start_timestamp is provided, returns only messages where timestamp >= start_timestamp.
    Otherwise returns the full conversation history.
    """
    query = (
        select(MessageRecord)
        .where(MessageRecord.agent_id == agent_id)
        .order_by(MessageRecord.timestamp)
    )
    if start_timestamp is not None:
        query = query.where(MessageRecord.timestamp >= start_timestamp)

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
