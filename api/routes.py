"""API routes — Section 4.1.

TODO: Our current "read-only" access pattern isn't truly read-only. Read operations
take a full AsyncSession and may return mutable ORM objects still connected to the DB.
This works but violates principle of least privilege — callers that only need to read
have full write access. Worth revisiting when we have bandwidth.

TODO: We have some exception catching and mapping that doesn't use "raise ... from e", probably some places we want to add the chaining.
"""
import logging
from typing import Any, AsyncGenerator

from fastapi import APIRouter, Depends, HTTPException
from fastapi.sse import EventSourceResponse, ServerSentEvent
from pydantic_ai import Agent, AgentRunResultEvent, capture_run_messages
from pydantic_ai.messages import ToolCallPart
from sqlalchemy.ext.asyncio import AsyncSession

from agent.crud import agent_exists, create_agent_record, get_agent_record
from agent.types import AgentAppState, AgentDeps
from api.fastapi_deps import get_session_dep, get_agent_and_deps, get_agent_app_states, get_deps_dep
from api.schemas import (
    AgentMetadataResponse,
    CoreMemoryResponse,
    CreateAgentRequest,
    CreateMemoryBlockRequest,
    MemoryBlockResponse,
    MessageRequest,
    MessagesResponse,
)
from memory.block_crud import DuplicateBlockError, create_block, get_blocks
from memory.system_prompt_compilation import compile_system_prompt
from messages.messages import deserialize_messages, load_messages, persist_messages
from agent.compaction import compact, is_compaction_needed


logger = logging.getLogger(__name__)

router = APIRouter(prefix="/agents")

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

    with capture_run_messages() as messages:
        async with agent.run_stream_events(user_prompt=user_prompt,
                                            message_history=message_history,
                                            deps=deps) as stream:
            last_persisted_idx = len(message_history) # track what we have persisted from messages
            last_total_tokens_value = None

            async for event in stream:
                yield map_to_sse(event)

                if len(messages) > last_persisted_idx:
                    # avoid persisting tool call before return comes in
                    if not isinstance(messages[-1].parts[-1], ToolCallPart):
                        total_tokens = await persist_messages(deps=deps, messages=messages[last_persisted_idx:])
                        await deps.commit_changes_refresh_agent_record()
                        last_persisted_idx = len(messages)
                        
                        if total_tokens is not None:
                            last_total_tokens_value = total_tokens

                if isinstance(event, AgentRunResultEvent):
                    # commit before compaction — if compaction fails, the turn may still be valid
                    await deps.commit_changes_refresh_agent_record()

                    if is_compaction_needed(last_total_tokens_value, deps.config):
                        await compact(deps, last_total_tokens_value)


# --- Routes ---

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


@router.post("/{agent_id}/cancel", status_code=202)
async def cancel_agent_run(
    agent_id: str,
    agent_app_states: dict[str, AgentAppState] = Depends(get_agent_app_states),
) -> None:
    """Cancel an active agent run.

    Sets the cancel_requested for the given agent if a run is currently active.
    Returns 200 if the cancel signal was sent, 409 if no run is active.

    Redundant cancels (event already set) succeed and return 200.
    """
    slot = agent_app_states.get(agent_id)
    if slot is None or not slot.lock.locked():
        raise HTTPException(status_code=409, detail=f"Agent {agent_id!r} has no active run")
    slot.cancel_requested.set() # If there was a previous unserviced cancellation request, no harm in setting again


@router.get("/{agent_id}/messages")
async def get_messages(
    agent_id: str,
    full: bool = False,
    session: AsyncSession = Depends(get_session_dep),
) -> MessagesResponse:
    """
    Return conversation history. Use ?full=true for complete history.
    TODO: Another instance of bad read-only control
    """
    record = await get_agent_record_or_404(session, agent_id)
    # TODO: Don't need agent record if requesting full, but we're likely gonna rework this anyway
    start_timestamp = None if full else record.context_window_start
    messages = await load_messages(session, agent_id, start_timestamp=start_timestamp)
    # Parse stored JSON and return — format TBD, this is throwaway (TODO)
    import json
    return MessagesResponse(messages=[json.loads(m.content) for m in messages])
