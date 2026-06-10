"""
Tests for SlashCommandResult event handling in acp.bridge.dispatch_event.

SlashCommandResult is a synthetic SSE event type emitted by the server when
a slash command (e.g. /recompile) completes. The bridge translates it into a
tool_call + tool_call_update ACP notification pair so toad can display it as
a tool call.
"""

import json
from unittest.mock import patch

import pytest

from acp.bridge import BridgeState, StreamState, process_sse_event


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
        "args": "",
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
    """dispatch_event emits a tool_call + tool_call_update pair for SlashCommandResult events."""

    @pytest.mark.asyncio
    async def test_happy_path_emits_correct_notification_pair(self, state, stream_state):
        """Success SlashCommandResult → tool_call (in_progress) + tool_call_update (completed)."""
        with patch("acp.bridge.send") as mock_send:
            await process_sse_event(state, stream_state, SESSION_ID, "SlashCommandResult", slash_data(), {})

        assert mock_send.call_count == 2

        first = get_update(mock_send, 0)
        assert first["sessionUpdate"] == "tool_call"
        assert first["status"] == "in_progress"
        assert first["toolCallId"].startswith("slash_user_recompile_")

        second = get_update(mock_send, 1)
        assert second["sessionUpdate"] == "tool_call_update"
        assert second["status"] == "completed"
        assert second["toolCallId"] == first["toolCallId"]  # IDs must match

    @pytest.mark.asyncio
    async def test_error_status_maps_to_failed(self, state, stream_state):
        """A SlashCommandResult with status='error' maps to 'failed' on the tool_call_update."""
        with patch("acp.bridge.send") as mock_send:
            await process_sse_event(
                state, stream_state, SESSION_ID, "SlashCommandResult",
                slash_data(status="error", result="Command failed: oops"), {}
            )

        second = get_update(mock_send, 1)
        assert second["status"] == "failed"
