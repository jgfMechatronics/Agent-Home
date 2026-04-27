"""Unit tests for map_to_sse — Section 4.1.

Verifies that each Pydantic AI streaming event type serializes to the correct
SSE-compatible dict. Pure unit tests: no DB, no HTTP, no async.

NOTE: BuiltinToolCallEvent and BuiltinToolResultEvent are intentionally not tested.
Agent Home uses custom function tools exclusively — we don't use provider-side
built-in tools (WebSearchTool, CodeExecutionTool, etc.).

NOTE: JF Skimmed this file but did not review in detail
"""
import pytest
from unittest.mock import Mock

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

from api.routes import map_to_sse


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

# One instance of each event type, paired with its expected "type" string.
ALL_EVENTS = [
    pytest.param(PartStartEvent(index=0, part=TEXT_PART), "PartStartEvent", id="PartStartEvent"),
    pytest.param(PartDeltaEvent(index=0, delta=TEXT_DELTA), "PartDeltaEvent", id="PartDeltaEvent"),
    pytest.param(PartEndEvent(index=0, part=TEXT_PART), "PartEndEvent", id="PartEndEvent"),
    pytest.param(FunctionToolCallEvent(part=TOOL_CALL_PART), "FunctionToolCallEvent", id="FunctionToolCallEvent"),
    pytest.param(FunctionToolResultEvent(result=TOOL_RETURN_PART), "FunctionToolResultEvent", id="FunctionToolResultEvent"),
    pytest.param(FinalResultEvent(tool_name=None, tool_call_id=None), "FinalResultEvent", id="FinalResultEvent"),
    pytest.param(AgentRunResultEvent(result=Mock()), "AgentRunResultEvent", id="AgentRunResultEvent"),
]

# PartStartEvent and PartEndEvent share structure (index + full part) — tested together.
# Use different index values to prevent false positives from a hardcoded return.
PART_BOUNDARY_EVENTS = [
    pytest.param(PartStartEvent(index=0, part=TEXT_PART), id="PartStartEvent"),
    pytest.param(PartEndEvent(index=3, part=TEXT_PART), id="PartEndEvent"),
]


class TestMapToSSEShared:
    """Behaviors shared across event types (7 tested, 2 builtin-tool events skipped)."""

    @pytest.mark.parametrize("event,expected_type", ALL_EVENTS)
    def test_returns_dict_with_correct_type(self, event, expected_type):
        result = map_to_sse(event)
        assert isinstance(result, dict)
        assert result["type"] == expected_type

    def test_raises_for_unknown_event_type(self):
        with pytest.raises(ValueError, match="Unhandled event type"):
            map_to_sse(object())


class TestPartBoundaryEvents:
    """PartStartEvent and PartEndEvent share structure: index + full part.

    Tested together to avoid duplicating identical assertions.
    """

    @pytest.mark.parametrize("event", PART_BOUNDARY_EVENTS)
    def test_includes_index(self, event):
        assert map_to_sse(event)["index"] == event.index

    @pytest.mark.parametrize("event", PART_BOUNDARY_EVENTS)
    def test_part_contains_content_and_kind(self, event):
        result = map_to_sse(event)
        assert result["part"]["content"] == TEXT_PART.content
        assert result["part"]["part_kind"] == TEXT_PART.part_kind


class TestPartDeltaEvent:
    def test_includes_index(self):
        event = PartDeltaEvent(index=1, delta=TEXT_DELTA)
        assert map_to_sse(event)["index"] == 1

    def test_delta_contains_content_delta(self):
        event = PartDeltaEvent(index=0, delta=TEXT_DELTA)
        assert map_to_sse(event)["delta"]["content_delta"] == TEXT_DELTA.content_delta


class TestFunctionToolCallEvent:
    def test_includes_tool_call_id(self):
        result = map_to_sse(FunctionToolCallEvent(part=TOOL_CALL_PART))
        assert result["tool_call_id"] == "call-1"

    def test_part_contains_tool_name_and_args(self):
        result = map_to_sse(FunctionToolCallEvent(part=TOOL_CALL_PART))
        assert result["part"]["tool_name"] == "memory_replace"
        assert result["part"]["args"] == {"label": "notes"}


class TestFunctionToolResultEvent:
    def test_includes_tool_call_id(self):
        result = map_to_sse(FunctionToolResultEvent(result=TOOL_RETURN_PART))
        assert result["tool_call_id"] == "call-1"

    def test_result_contains_tool_name_and_content(self):
        result = map_to_sse(FunctionToolResultEvent(result=TOOL_RETURN_PART))
        assert result["result"]["tool_name"] == "memory_replace"
        assert result["result"]["content"] == "Updated."


class TestFinalResultEvent:
    def test_tool_name_is_none_for_text_output(self):
        event = FinalResultEvent(tool_name=None, tool_call_id=None)
        assert map_to_sse(event)["tool_name"] is None

    def test_tool_name_is_set_for_tool_output(self):
        event = FinalResultEvent(tool_name="memory_replace", tool_call_id="call-1")
        assert map_to_sse(event)["tool_name"] == "memory_replace"


class TestAgentRunResultEvent:
    def test_is_minimal_signal_with_no_result_content(self):
        """Only "type" key — result content is not exposed over the wire.

        The client accumulates the response via PartDeltaEvents; AgentRunResultEvent
        is a stream-end signal only.
        """
        event = AgentRunResultEvent(result=Mock())
        result = map_to_sse(event)
        assert set(result.keys()) == {"type"}


class TestThinkingPart:
    """ThinkingPart flows through Part events — tests extended thinking/CoT streaming."""

    def test_part_start_with_thinking_part(self):
        event = PartStartEvent(index=0, part=THINKING_PART)
        result = map_to_sse(event)
        assert result["part"]["content"] == THINKING_PART.content
        assert result["part"]["part_kind"] == "thinking"

    def test_part_end_with_thinking_part(self):
        event = PartEndEvent(index=0, part=THINKING_PART)
        result = map_to_sse(event)
        assert result["part"]["content"] == THINKING_PART.content
        assert result["part"]["part_kind"] == "thinking"

    def test_part_delta_with_thinking_delta(self):
        event = PartDeltaEvent(index=0, delta=THINKING_DELTA)
        result = map_to_sse(event)
        assert result["delta"]["content_delta"] == THINKING_DELTA.content_delta
        assert result["delta"]["part_delta_kind"] == "thinking"
