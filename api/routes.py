"""API routes — Section 4.1."""
import dataclasses
from typing import Any

from pydantic_ai import AgentRunResultEvent
from pydantic_ai.messages import (
    FinalResultEvent,
    FunctionToolCallEvent,
    FunctionToolResultEvent,
    PartDeltaEvent,
    PartEndEvent,
    PartStartEvent,
)


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
