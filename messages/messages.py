"""
Message persistence and retrieval
"""
import dataclasses
import hashlib
import json
import logging
import uuid
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
from pydantic_ai.tools import ToolDefinition
from sqlalchemy import func, select
from sqlalchemy.dialects.sqlite import insert as sqlite_insert
from sqlalchemy.ext.asyncio import AsyncSession

from agent.types import AgentConfig, AgentDeps
from db.models import AgentConfigSnapshot, BaseSnapshot, MessageRecord, SystemPromptSnapshot, ToolDefinitionSnapshot, utcnow

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


def _compute_sha256(content: str) -> str:
    return hashlib.sha256(content.encode("utf-8")).hexdigest()


async def _ensure_content_snapshotted(
    session: AsyncSession,
    model_type: type[BaseSnapshot],
    content: str,
) -> str:
    """Hash content, insert a snapshot row of the given type if not already present, return the hash id."""
    hash_id = _compute_sha256(content)
    stmt = sqlite_insert(model_type).values(
        id=hash_id,
        content=content,
        created_at=utcnow(),
    ).on_conflict_do_nothing(index_elements=["id"])
    await session.execute(stmt)
    return hash_id


async def _ensure_system_prompt_snapshotted(session: AsyncSession, sys_prompt: str) -> str:
    """Insert a SystemPromptSnapshot for the given compiled system prompt (if not already present). Returns the hash id."""
    return await _ensure_content_snapshotted(session, SystemPromptSnapshot, sys_prompt)


async def _ensure_tool_definition_snapshotted(session: AsyncSession, schemas: list[ToolDefinition]) -> str:
    """Insert a ToolDefinitionSnapshot for the given tool definitions (if not already present). Returns the hash id."""
    content = json.dumps([dataclasses.asdict(s) for s in schemas], separators=(",", ":"))
    return await _ensure_content_snapshotted(session, ToolDefinitionSnapshot, content)


async def _ensure_agent_config_snapshotted(session: AsyncSession, config: AgentConfig) -> str:
    """Insert an AgentConfigSnapshot for the given agent config (if not already present). Returns the hash id."""
    return await _ensure_content_snapshotted(session, AgentConfigSnapshot, config.model_dump_json())


async def _get_context_window_start_msg_id(
    session: AsyncSession,
    agent_id: str,
    start_seq_id: int,
) -> str | None:
    """Return the UUID of the MessageRecord at start_seq_id, or None if it doesn't exist yet."""
    result = await session.execute(
        select(MessageRecord.id)
        .where(MessageRecord.agent_id == agent_id)
        .where(MessageRecord.seq_id == start_seq_id)
    )
    return result.scalar()


async def _persist_error_warnings(
    deps: AgentDeps,
    errors: list[tuple[datetime | None, str]],
    tool_schemas: list[ToolDefinition],
) -> None:
    """Build and persist error warning messages via recursive call to persist_messages."""
    warning_messages = [
        ModelResponse(parts=[TextPart(content=(
            f"WARNING: A problem was encountered while persisting messages from the last turn: "
            f"'{error_text}'. A warning was injected in place of the problematic message, "
            f"problematic message timestamp was {original_timestamp}"
        ))])
        for original_timestamp, error_text in errors
    ]
    await deps.session.flush()  # Ensure main messages visible to recursive call's MAX query
    await persist_messages(deps, warning_messages, tool_schemas, _is_error_pass=True)


# ---------------------------------------------------------------------------
# Functions
# ---------------------------------------------------------------------------

async def persist_messages(
    deps: AgentDeps,
    messages: list[ModelMessage],
    tool_schemas: list[ToolDefinition],
    *,
    _is_error_pass: bool = False,
) -> int | None:
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
    # NOTE: Concurrent calls would race on this query. Agent-level locking in the runner
    # (AgentAppState.lock) ensures only one persist_messages runs per agent at a time.
    result = await deps.session.execute(
        select(func.max(MessageRecord.seq_id)).where(MessageRecord.agent_id == deps.agent_id)
    )
    max_seq_id = result.scalar()
    next_seq_id = max_seq_id + 1 if max_seq_id is not None else 0

    # Snapshot hashes — computed once per call; **assumed** stable across the batch
    system_prompt_hash = await _ensure_system_prompt_snapshotted(deps.session, deps.compiled_system_prompt)
    tool_definition_hash = await _ensure_tool_definition_snapshotted(deps.session, tool_schemas)
    agent_config_hash = await _ensure_agent_config_snapshotted(deps.session, deps.config)

    # context_window_start_msg_id — look up existing start, or resolve self-referentially on first insert
    context_window_start_msg_id = await _get_context_window_start_msg_id(
        deps.session, deps.agent_id, deps.context_window_start
    )

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

        # Generate UUID explicitly so we can use it for self-referential context_window_start_msg_id
        # (SQLAlchemy column defaults are not applied until INSERT — record.id would be None otherwise)
        record_id = str(uuid.uuid4())
        if context_window_start_msg_id is None:
            # Self-referential for the very first message: no prior history exists in the DB
            context_window_start_msg_id = record_id
        
        record = MessageRecord(
            id=record_id,
            agent_id=deps.agent_id,
            type=msg_type,
            content=content,
            total_tokens=msg_total_tokens,
            seq_id=next_seq_id + i,
            timestamp=_message_timestamp(msg),
            system_prompt_hash=system_prompt_hash,
            tool_definition_hash=tool_definition_hash,
            agent_config_hash=agent_config_hash,
            context_window_start_msg_id=context_window_start_msg_id,
        )
        deps.session.add(record)

    # Persist error warnings via recursion (errors are simple TextPart messages, won't fail)
    if errors:
        if _is_error_pass:
            raise RuntimeError("Persistence errors during attempt to persist error notifications")
        await _persist_error_warnings(deps, errors, tool_schemas)
    else:
        await deps.session.flush()

    return last_total_tokens


async def load_messages(
    session: AsyncSession,
    agent_id: str,
    start_seq_id: int = 0,
    end_seq_id: int | None = None,
) -> list[MessageRecord]:
    """Load messages as ORM records in seq_id order.

    Returns messages where seq_id >= start_seq_id (and < end_seq_id if provided).
    Defaults to 0, which returns the full conversation history (all seq_ids are non-negative).
    """
    query = (
        select(MessageRecord)
        .where(MessageRecord.agent_id == agent_id)
        .where(MessageRecord.seq_id >= start_seq_id)
        .order_by(MessageRecord.seq_id)
    )
    if end_seq_id is not None:
        query = query.where(MessageRecord.seq_id < end_seq_id)

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
