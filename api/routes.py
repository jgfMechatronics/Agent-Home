"""API routes — Section 4.1.

TODO: Our current "read-only" access pattern isn't truly read-only. Read operations
take a full AsyncSession and may return mutable ORM objects still connected to the DB.
This works but violates principle of least privilege — callers that only need to read
have full write access. Worth revisiting when we have bandwidth.

TODO: We have some exception catching and mapping that doesn't use "raise ... from e", probably some places we want to add the chaining.
"""
import logging
from dataclasses import dataclass
from datetime import datetime
from typing import Any, AsyncGenerator, Awaitable, Protocol

from fastapi import APIRouter, Depends, HTTPException
from fastapi.sse import EventSourceResponse, ServerSentEvent
from pydantic_ai import Agent, AgentRunResultEvent
from sqlalchemy.ext.asyncio import AsyncSession

from agent.crud import agent_exists, create_agent_record, get_agent_record
from agent.types import AgentDeps
from api.fastapi_deps import get_session_dep, get_agent_and_deps, get_deps_dep
from api.schemas import (
    AgentMetadataResponse,
    CoreMemoryResponse,
    CreateAgentRequest,
    CreateMemoryBlockRequest,
    MemoryBlockResponse,
    MessageItem,
    MessageRequest,
    MessagesResponse,
)
from memory.block_crud import DuplicateBlockError, create_block, get_blocks
from memory.system_prompt_compilation import compile_system_prompt
from messages.messages import deserialize_messages, load_messages, persist_messages
from agent.compaction import compact, is_compaction_needed


logger = logging.getLogger(__name__)

router = APIRouter(prefix="/agents")


# --- Slash Commands ---

class SlashCommandHandler(Protocol):
    """Protocol for slash command handlers. All handlers receive deps and args."""
    async def __call__(self, deps: AgentDeps, args: str) -> ServerSentEvent: ...


@dataclass
class SlashCommandDef:
    """Definition for a slash command: handler + discovery metadata."""
    handler: SlashCommandHandler
    description: str
    hint: str | None = None


async def _handle_recompile(deps: AgentDeps, args: str) -> ServerSentEvent:
    """Handler for /recompile command. Recompiles the system prompt from current memory blocks."""
    await compile_system_prompt(deps)
    await deps.commit_changes_refresh_agent_record()
    return ServerSentEvent(
        data={"name": "user_recompile", "args": args, "result": "System prompt recompiled successfully", "status": "success"},
        event="SlashCommandResult",
    )


SLASH_COMMANDS: dict[str, SlashCommandDef] = {
    "recompile": SlashCommandDef(
        handler=_handle_recompile,
        description="Recompile memory blocks into system prompt",
    ),
}


def get_available_commands() -> list[dict[str, Any]]:
    """Build the availableCommands list for ACP discovery notification."""
    commands = []
    for name, cmd_def in SLASH_COMMANDS.items():
        cmd = {"name": name, "description": cmd_def.description}
        if cmd_def.hint:
            cmd["input"] = {"hint": cmd_def.hint}
        commands.append(cmd)
    return commands


def _parse_slash_cmd(msg: str) -> tuple[str, str] | None:
    """Parse a message as a slash command.
    
    Returns (command_name, args) if msg starts with '/' and command is in registry.
    Returns None otherwise (including unrecognized commands — those pass to model).
    """
    if not msg.startswith("/"):
        return None
    # Split into command and args: "/recompile foo bar" -> ("recompile", "foo bar")
    parts = msg[1:].split(maxsplit=1)
    if not parts:
        return None
    cmd = parts[0].lower()
    args = parts[1] if len(parts) > 1 else ""
    if cmd not in SLASH_COMMANDS:
        return None
    return (cmd, args)


def _is_slash_cmd(msg: str) -> bool:
    """Check if message is a recognized slash command."""
    return _parse_slash_cmd(msg) is not None


async def _handle_slash_cmd(deps: AgentDeps, msg: str) -> ServerSentEvent:
    """Dispatch a slash command to its handler and return the result SSE.
    
    Precondition: _is_slash_cmd(msg) is True.
    """
    parsed = _parse_slash_cmd(msg)
    if parsed is None:
        # Shouldn't happen if precondition met, but handle gracefully
        return ServerSentEvent(
            data={"name": "user_unknown", "args": msg, "result": "Unknown command", "status": "error"},
            event="SlashCommandResult"
        )
    cmd, args = parsed
    handler = SLASH_COMMANDS[cmd].handler
    try:
        return await handler(deps, args)
    except Exception as e:
        logger.exception("Slash command /%s failed", cmd)
        return ServerSentEvent(
            data={"name": f"user_{cmd}", "args": args, "result": f"Command failed: {e}", "status": "error"},
            event="SlashCommandResult"
        )


# --- Helpers ---

def map_to_sse(event: Any) -> ServerSentEvent:
    """Convert a Pydantic AI streaming event to a ServerSentEvent.

    The event type name goes in the SSE 'event' field, allowing clients to filter
    with addEventListener(). The event object is passed directly to 'data' and
    serialized by FastAPI's jsonable_encoder.

    TODO: Document the SSE event types in the API readme.
    """
    if isinstance(event, AgentRunResultEvent):
        # Stream-end signal only — don't expose the result object
        return ServerSentEvent(data={}, event="AgentRunResultEvent")
    return ServerSentEvent(data=event, event=type(event).__name__)


async def get_agent_record_or_404(session: AsyncSession, agent_id: str) -> Any:
    """Load agent record, raising 404 if not found."""
    record = await get_agent_record(session, agent_id)
    if record is None:
        raise HTTPException(status_code=404, detail=f"Agent {agent_id!r} not found")
    return record


async def _handle_message(agent: Agent, deps: AgentDeps, user_prompt: str) -> AsyncGenerator[ServerSentEvent, None]:
    records = await load_messages(deps.session, deps.agent_id, start_timestamp=deps.context_window_start)
    message_history = deserialize_messages(records)

    # TODO: Exceptions raised during run_stream_events result in entire turn being discarded.
    # this was discovered with "UnexpectedModelBehavior" due to excess retries
    async for event in agent.run_stream_events(user_prompt=user_prompt,
                                                message_history=message_history,
                                                deps=deps):
        yield map_to_sse(event)

        if isinstance(event, AgentRunResultEvent):
            final_result = event.result
            input_tokens = final_result.usage().input_tokens
            await persist_messages(deps=deps,
                                    messages=final_result.new_messages(),
                                    input_tokens=input_tokens)

            # commit before compaction — if compaction fails, the turn may still be valid
            await deps.commit_changes_refresh_agent_record()

            if is_compaction_needed(input_tokens, deps.config):
                await compact(deps, input_tokens)


# --- Routes ---

@router.get("/slash-commands")
async def get_slash_commands() -> list[dict[str, Any]]:
    """Return available slash commands for client discovery."""
    return get_available_commands()


@router.post("/{agent_id}/messages", response_class=EventSourceResponse)
async def send_message(
    agent_id: str,
    body: MessageRequest,
    agent_and_deps: tuple[Agent, AgentDeps] = Depends(get_agent_and_deps),
) -> AsyncGenerator[ServerSentEvent, None]:
    """TODO: Agent run should still be able to complete and persist in the event that client disconnects"""
    # AgentNotFoundError / AgentLockedError are translated to HTTP 404/503 by get_agent_and_deps
    agent, deps = agent_and_deps
    try:
        if _is_slash_cmd(body.message):
            yield await _handle_slash_cmd(deps, body.message)
        else:
            async for event in _handle_message(agent, deps, body.message):
                yield event
    except Exception as e:
        # TODO (low priority): put more thought into logging strategy (log levels, handler chain, structured logging)
        logger.exception("Unexpected error in send_message for agent %s", agent_id)
        await deps.session.rollback()
        yield ServerSentEvent(
            data={"message": f"Unexpected internal server error: '{type(e).__name__}: {str(e)}'"},
            event="Error",
        )


@router.post("/{agent_id}/recompile_system_prompt")
async def recompile_system_prompt_route_handler(
    agent_id: str,
    deps: AgentDeps = Depends(get_deps_dep)
) -> bool: # We may or may not want this to return a bool
    """
    TODO: This is temp just to be able to test out memory system functionality, we may not actually want this
    if we do, need to design proper and unit test
    """

    await compile_system_prompt(deps)
    return True


@router.post("/", status_code=201)
async def create_agent(
    body: CreateAgentRequest,
    session: AsyncSession = Depends(get_session_dep),
) -> AgentMetadataResponse:
    """Create a new agent and return its metadata.

    TODO: Compile the system prompt immediately after creation so the agent's memory blocks
    are visible from the first turn. Currently the agent starts without a compiled system
    prompt and won't see its blocks until an explicit recompile or first compaction.
    This should call compile_system_prompt (or equivalent) before returning, and the
    behaviour should be covered by tests in test_routes.py::TestCreateAgent.
    
    TODO: I was able to get an invalid model name through to the DB. Errored out when trying to do a model request but did persist
    """
    record = await create_agent_record(session, body.name, body.system_instructions, body.config)
    return AgentMetadataResponse.from_record(record)


@router.get("/{agent_id}")
async def get_agent_info(
    agent_id: str,
    session: AsyncSession = Depends(get_session_dep),
) -> AgentMetadataResponse:
    """Return metadata for an existing agent."""
    record = await get_agent_record_or_404(session, agent_id)
    return AgentMetadataResponse.from_record(record)


@router.get("/{agent_id}/memory/blocks")
async def get_memory_blocks(
    agent_id: str,
    session: AsyncSession = Depends(get_session_dep),
) -> CoreMemoryResponse:
    """Return core memory blocks for an agent."""
    blocks = await get_blocks(session, agent_id)
    # get_blocks returns empty list if agent DNE OR if agent has no blocks
    if not blocks and not (await agent_exists(session, agent_id)):
        raise HTTPException(status_code=404, detail=f"Agent {agent_id!r} not found")
    return CoreMemoryResponse(blocks=[MemoryBlockResponse.from_record(b) for b in blocks])


@router.post("/{agent_id}/memory/blocks", status_code=201)
async def create_memory_block(
    agent_id: str,
    body: CreateMemoryBlockRequest,
    deps: AgentDeps = Depends(get_deps_dep),
) -> MemoryBlockResponse:
    """Create a new memory block for an agent."""
    try:
        block = await create_block(deps, body.label, body.content, body.description, body.char_limit)
    except DuplicateBlockError as e:
        raise HTTPException(status_code=400, detail=f"Duplicate block: {e}") from e
    return MemoryBlockResponse.from_record(block)


@router.get("/{agent_id}/messages")
async def get_messages(
    agent_id: str,
    full: bool = False,
    after: datetime | None = None,
    session: AsyncSession = Depends(get_session_dep),
) -> MessagesResponse:
    """Return conversation history.

    - Default (no params): in-context messages only.
    - ?full=true: complete history.
    - ?after=<ISO datetime>: messages strictly after the given timestamp (exclusive).
      Intended for polling — pass the timestamp of the last received message to get only new ones.

    TODO: Another instance of bad read-only control
    """
    record = await get_agent_record_or_404(session, agent_id)
    if after is not None:
        start_timestamp, start_exclusive = after, True
    elif full:
        start_timestamp, start_exclusive = None, False
    else:
        # TODO: Don't need agent record if requesting full, but we're likely gonna rework this anyway
        start_timestamp, start_exclusive = record.context_window_start, False
    messages = await load_messages(session, agent_id, start_timestamp=start_timestamp, start_exclusive=start_exclusive)
    return MessagesResponse(messages=[
        MessageItem(id=m.id, type=m.type, content=m.content, timestamp=m.timestamp)
        for m in messages
    ])
