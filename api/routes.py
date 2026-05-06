"""API routes — Section 4.1."""
from typing import Any, AsyncGenerator

from fastapi import APIRouter, Depends, HTTPException
from fastapi.sse import EventSourceResponse, ServerSentEvent
from pydantic_ai import Agent, AgentRunResultEvent
from sqlalchemy.ext.asyncio import AsyncSession

from agent.crud import agent_exists, create_agent_record, get_agent_record
from agent.types import AgentDeps
from api.deps import get_session_dep, get_agent_and_deps
from api.schemas import (
    AgentMetadataResponse,
    CoreMemoryResponse,
    CreateAgentRequest,
    MemoryBlockResponse,
    MessageRequest,
    MessagesResponse,
)
from memory.block_crud import get_blocks
from messages.messages import load_in_context_messages, load_message_history, persist_messages
from agent.compaction import compact, is_compaction_needed

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


# --- Routes ---

@router.post("/{agent_id}/messages", response_class=EventSourceResponse)
async def send_message(
    agent_id: str,
    body: MessageRequest,
    agent_and_deps: tuple[Agent, AgentDeps] = Depends(get_agent_and_deps),
) -> AsyncGenerator[ServerSentEvent]:
    """TODO: Agent run should still be able to complete and persist in the event that client disconnects"""
    # AgentNotFoundError / AgentLockedError are translated to HTTP 404/503 by get_agent_and_deps
    agent, deps = agent_and_deps
    try:
        message_history = await load_in_context_messages(deps)

        async for event in agent.run_stream_events(user_prompt=body.message,
                                                    message_history=message_history,
                                                    deps=deps):
            yield map_to_sse(event)

            if isinstance(event, AgentRunResultEvent):
                final_result = event.result
                input_tokens = final_result.usage().input_tokens
                await persist_messages(deps=deps,
                                       messages=final_result.new_messages(),
                                       input_tokens=input_tokens)
                if is_compaction_needed(input_tokens, deps.config):
                    await compact(deps, input_tokens)
    except Exception as e:
        yield ServerSentEvent(
            data={"message": f"Unexpected internal server error: '{type(e).__name__}: {str(e)}'"},
            event="Error",
        )


@router.post("/", status_code=201)
async def create_agent(
    body: CreateAgentRequest,
    session: AsyncSession = Depends(get_session_dep),
) -> AgentMetadataResponse:
    """Create a new agent and return its metadata."""
    try:
        record = await create_agent_record(session, body.name, body.system_instructions, body.config)
        return AgentMetadataResponse.from_record(record)
    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=f"Exception during agent creation: {type(e).__name__}: {e}",
        )


@router.get("/{agent_id}")
async def get_agent_info(
    agent_id: str,
    session: AsyncSession = Depends(get_session_dep),
) -> AgentMetadataResponse:
    """Return metadata for an existing agent."""
    record = await get_agent_record_or_404(session, agent_id)
    return AgentMetadataResponse.from_record(record)


@router.get("/{agent_id}/core_memory")
async def get_core_memory(
    agent_id: str,
    session: AsyncSession = Depends(get_session_dep),
) -> CoreMemoryResponse:
    """Return core memory blocks for an agent."""
    blocks = await get_blocks(session, agent_id)
    # get_blocks returns empty list if agent DNE OR if agent has no blocks
    if not blocks and not (await agent_exists(session, agent_id)):
        raise HTTPException(status_code=404, detail=f"Agent {agent_id!r} not found")
    return CoreMemoryResponse(blocks=[
        MemoryBlockResponse(
            label=b.label,
            description=b.description,
            content=b.content,
            char_limit=b.char_limit,
            updated_at=b.updated_at,
        )
        for b in blocks
    ])


@router.get("/{agent_id}/messages")
async def get_messages(
    agent_id: str,
    full: bool = False,
    session: AsyncSession = Depends(get_session_dep),
) -> MessagesResponse:
    """Return conversation history. Use ?full=true for complete history."""
    await get_agent_record_or_404(session, agent_id)
    if full:
        messages = await load_message_history(session, agent_id)
    else:
        messages = await load_in_context_messages(session, agent_id)
    return MessagesResponse(messages=messages)
