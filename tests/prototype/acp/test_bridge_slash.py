"""
Tests for SlashCommandResult event handling in acp.bridge.dispatch_event.

SlashCommandResult is a synthetic SSE event type emitted by the server when
a slash command (e.g. /recompile) completes. The bridge translates it into a
single agent_message_chunk wrapped in system_alert XML tags so Nori renders it
visually distinct from a real agent response.
"""

import json
from unittest.mock import patch

import pytest

from prototype.acp.bridge import BridgeState, StreamState, process_sse_event


SESSION_ID = "test-session-id"


# --- Fixtures ---


@pytest.fixture
def state() -> BridgeState:
    return BridgeState()


@pytest.fixture
def stream_state() -> StreamState:
    return StreamState(tool_args={})


# --- Helpers ---


def slash_data(**overrides) -> str:
    """Build a SlashCommandResult data JSON string with sensible defaults."""
    payload = {
        "name": "user_recompile",
        "result": "System prompt recompiled successfully",
        "status": "success",
    }
    payload.update(overrides)
    return json.dumps(payload)


def get_update(mock_send, call_index: int) -> dict:
    """Extract the ACP update dict from a specific send() call."""
    return mock_send.call_args_list[call_index][0][0]["params"]["update"]


# --- Tests ---


class TestDispatchEventSlashCommandResult:
    """dispatch_event emits a single agent_message_chunk for SlashCommandResult events."""

    @pytest.mark.asyncio
    async def test_emits_single_agent_message_chunk(self, state, stream_state):
        """SlashCommandResult → exactly one agent_message_chunk."""
        with patch("prototype.acp.bridge.send") as mock_send:
            await process_sse_event(state, stream_state, SESSION_ID, "SlashCommandResult", slash_data(), {})

        assert mock_send.call_count == 1
        update = get_update(mock_send, 0)
        assert update["sessionUpdate"] == "agent_message_chunk"

    @pytest.mark.asyncio
    async def test_result_text_wrapped_in_system_alert_tags(self, state, stream_state):
        """Result text is wrapped in <system_alert> tags so Nori renders it distinctly."""
        result_text = "System prompt recompiled successfully"
        with patch("prototype.acp.bridge.send") as mock_send:
            await process_sse_event(state, stream_state, SESSION_ID, "SlashCommandResult", slash_data(result=result_text), {})

        update = get_update(mock_send, 0)
        assert update["content"]["text"] == f"<system_alert>{result_text}</system_alert>"

    @pytest.mark.asyncio
    async def test_error_result_text_still_surfaced(self, state, stream_state):
        """Failed slash commands still surface their result text in the chunk."""
        with patch("prototype.acp.bridge.send") as mock_send:
            await process_sse_event(
                state, stream_state, SESSION_ID, "SlashCommandResult",
                slash_data(status="error", result="Command failed: oops"), {}
            )

        update = get_update(mock_send, 0)
        assert "Command failed: oops" in update["content"]["text"]
