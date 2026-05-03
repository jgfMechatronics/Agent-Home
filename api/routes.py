"""API routes — Section 4.1."""
import dataclasses
from typing import Any

from fastapi import APIRouter, Depends
from fastapi.responses import StreamingResponse
from fastapi.sse import EventSourceResponse
from pydantic_ai import AgentRunResultEvent, Agent
from pydantic_ai.messages import (
    FinalResultEvent,
    FunctionToolCallEvent,
    FunctionToolResultEvent,
    PartDeltaEvent,
    PartEndEvent,
    PartStartEvent,
)
from sqlalchemy.ext.asyncio import AsyncSession

from agent.factory import AgentFactory, get_agent_factory
from db.connection import get_session_dep
from api.schemas import (
    AgentMetadataResponse,
    CoreMemoryResponse,
    CreateAgentRequest,
    MessageRequest,
    MessagesResponse,
)
from messages.messages import load_in_context_messages 

router = APIRouter(prefix="/agents")


def map_to_sse(event: Any) -> dict:
    """Convert a Pydantic AI streaming event to an SSE-compatible dict.

    Each dict includes a 'type' key matching the event class name, plus
    event-specific fields. Sub-objects are serialized via dataclasses.asdict().
    TODO: The format defined here should definetly go in the readme for users of the stream content
    """
    type_name = type(event).__name__

    match event:
        case PartStartEvent() | PartEndEvent():
            return {"type": type_name, "index": event.index, "part": dataclasses.asdict(event.part)}
        case PartDeltaEvent():
            return {"type": type_name, "index": event.index, "delta": dataclasses.asdict(event.delta)}
        case FunctionToolCallEvent():
            return {"type": type_name, "part": dataclasses.asdict(event.part), "tool_call_id": event.part.tool_call_id}
        case FunctionToolResultEvent():
            return {"type": type_name, "result": dataclasses.asdict(event.result), "tool_call_id": event.result.tool_call_id}
        case FinalResultEvent():
            return {"type": type_name, "tool_name": event.tool_name}
        case AgentRunResultEvent():
            return {"type": type_name}
        case _:
            raise ValueError(f"Unhandled event type: {type_name}")


# --- Routes ---

@router.post("/{agent_id}/messages", response_class=EventSourceResponse)
async def send_message(
    agent_id: str,
    body: MessageRequest,
    factory: AgentFactory = Depends(get_agent_factory),
) -> StreamingResponse:
    """TODO: Agent run should still be able to complete and persist in the event that client disconnects"""
    
    async with factory.build_agent_and_deps(agent_id) as (agent, deps):
        message_history = await load_in_context_messages(deps)

        async for event in agent.run_stream_events(user_prompt=body.message,
                                                    message_history=message_history,
                                                    deps=deps):
            yield map_to_sse(event)
            
            # if event # In progress 


@router.post("/", status_code=201)
async def create_agent(
    body: CreateAgentRequest,
    session: AsyncSession = Depends(get_session_dep),
) -> AgentMetadataResponse:
    raise NotImplementedError


@router.get("/{agent_id}")
async def get_agent(
    agent_id: str,
    session: AsyncSession = Depends(get_session_dep),
) -> AgentMetadataResponse:
    raise NotImplementedError


@router.get("/{agent_id}/core_memory")
async def get_core_memory(
    agent_id: str,
    session: AsyncSession = Depends(get_session_dep),
) -> CoreMemoryResponse:
    raise NotImplementedError


@router.get("/{agent_id}/messages")
async def get_messages(
    agent_id: str,
    full: bool = False,
    session: AsyncSession = Depends(get_session_dep),
) -> MessagesResponse:
    raise NotImplementedError
