"""Unit tests for map_to_sse — Section 4.1.

Verifies that each Pydantic AI streaming event type is mapped to the correct
ServerSentEvent — with the right event type in the 'event' field and correctly
serialized data. Pure unit tests: no DB, no HTTP, no async.

NOTE: BuiltinToolCallEvent and BuiltinToolResultEvent are intentionally not tested.
Agent Home uses custom function tools exclusively — we don't use provider-side
built-in tools (WebSearchTool, CodeExecutionTool, etc.).
"""
import json
import pytest
from unittest.mock import Mock

from fastapi.encoders import jsonable_encoder
from fastapi.sse import ServerSentEvent
from pydantic_ai import AgentRunResultEvent
from pydantic_ai.messages import (
    FinalResultEvent,
    FunctionToolCallEvent,
    FunctionToolResultEvent,
    PartDeltaEvent,
    PartEndEvent,
    PartStartEvent,
    TextPart,
    TextPartDelta,
    ThinkingPart,
    ThinkingPartDelta,
    ToolCallPart,
    ToolReturnPart,
)

from agent.broadcast_streaming import RunCompletedEvent, RunStartedEvent
from api.routes import map_to_sse


def serialize_sse_data(sse: ServerSentEvent) -> dict:
    """Serialize SSE data the same way FastAPI does."""
    if sse.data == {}:
        return {}
    return json.loads(json.dumps(jsonable_encoder(sse.data)))


# --- Module-level test data ---

TEXT_PART = TextPart(content="hello world")
TEXT_DELTA = TextPartDelta(content_delta="ello")
THINKING_PART = ThinkingPart(content="Let me think about this...")
THINKING_DELTA = ThinkingPartDelta(content_delta="thinking more...")
TOOL_CALL_PART = ToolCallPart(
    tool_name="memory_replace", args={"label": "notes"}, tool_call_id="call-1"
)
TOOL_RETURN_PART = ToolReturnPart(
    tool_name="memory_replace", content="Updated.", tool_call_id="call-1"
)

# All event types paired with their expected SSE event name.
ALL_EVENTS = [
    pytest.param(PartStartEvent(index=0, part=TEXT_PART), "PartStartEvent", id="PartStartEvent"),
    pytest.param(PartDeltaEvent(index=0, delta=TEXT_DELTA), "PartDeltaEvent", id="PartDeltaEvent"),
    pytest.param(PartEndEvent(index=0, part=TEXT_PART), "PartEndEvent", id="PartEndEvent"),
    pytest.param(FunctionToolCallEvent(part=TOOL_CALL_PART), "FunctionToolCallEvent", id="FunctionToolCallEvent"),
    pytest.param(FunctionToolResultEvent(part=TOOL_RETURN_PART), "FunctionToolResultEvent", id="FunctionToolResultEvent"),
    pytest.param(FinalResultEvent(tool_name=None, tool_call_id=None), "FinalResultEvent", id="FinalResultEvent"),
    pytest.param(AgentRunResultEvent(result=Mock()), "AgentRunResultEvent", id="AgentRunResultEvent"),
    pytest.param(RunStartedEvent(prompt="hello"), "RunStarted", id="RunStartedEvent"),
    pytest.param(RunCompletedEvent(status="success"), "RunCompleted", id="RunCompletedEvent"),
]

# Pydantic-ai events that map_to_sse passes through unchanged (data=event, event=type name).
# One entry per event type is sufficient — we're testing the passthrough contract, not
# the serialization of each field.
PASSTHROUGH_EVENTS = [
    pytest.param(PartStartEvent(index=0, part=TEXT_PART), id="PartStartEvent"),
    pytest.param(PartDeltaEvent(index=0, delta=TEXT_DELTA), id="PartDeltaEvent"),
    pytest.param(PartEndEvent(index=0, part=TEXT_PART), id="PartEndEvent"),
    pytest.param(FunctionToolCallEvent(part=TOOL_CALL_PART), id="FunctionToolCallEvent"),
    pytest.param(FunctionToolResultEvent(part=TOOL_RETURN_PART), id="FunctionToolResultEvent"),
    pytest.param(FinalResultEvent(tool_name=None, tool_call_id=None), id="FinalResultEvent"),
    pytest.param(PartStartEvent(index=0, part=THINKING_PART), id="PartStartEvent_thinking"),
    pytest.param(PartDeltaEvent(index=0, delta=THINKING_DELTA), id="PartDeltaEvent_thinking"),
]


class TestMapToSSEShared:
    """Behaviors shared across all event types."""

    @pytest.mark.parametrize("event,expected_type", ALL_EVENTS)
    def test_returns_server_sent_event_with_correct_type(self, event, expected_type):
        result = map_to_sse(event)
        assert isinstance(result, ServerSentEvent)
        assert result.event == expected_type


class TestDataPayload:
    """Verifies the serialized data payload for each event type."""

    @pytest.mark.parametrize("event", PASSTHROUGH_EVENTS)
    def test_passthrough_data_is_unchanged(self, event):
        """For pydantic-ai events, map_to_sse passes data through as-is.

        Asserts the result equals a directly constructed SSE with the same event and data —
        verifying map_to_sse doesn't modify or strip fields.
        """
        expected = ServerSentEvent(data=event, event=type(event).__name__)
        assert map_to_sse(event) == expected

    @pytest.mark.parametrize("event,expected_data", [
        pytest.param(RunStartedEvent(prompt="test prompt"), {"prompt": "test prompt"}, id="RunStartedEvent"),
        pytest.param(RunCompletedEvent(status="success"), {"status": "success"}, id="RunCompletedEvent_success"),
        pytest.param(RunCompletedEvent(status="cancelled"), {"status": "cancelled"}, id="RunCompletedEvent_cancelled"),
        pytest.param(RunCompletedEvent(status="error"), {"status": "error"}, id="RunCompletedEvent_error"),
    ])
    def test_custom_event_data(self, event, expected_data):
        """Custom events construct specific payloads — verify exact content."""
        assert serialize_sse_data(map_to_sse(event)) == expected_data

    def test_agent_run_result_exposes_no_data(self):
        """AgentRunResultEvent is a stream-end signal only — result content is not exposed over the wire.

        Clients accumulate the response via PartDeltaEvents; this event carries no payload.
        """
        assert serialize_sse_data(map_to_sse(AgentRunResultEvent(result=Mock()))) == {}
