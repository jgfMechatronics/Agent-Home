"""Tests for real-time SSE stream processing in acp.bridge.

Tests verify that _process_stream_event correctly handles:
- RunStarted: sends status=working to Nori
- RunCompleted: sends status=idle to Nori
- Content events: forwarded via process_sse_event
- state.stream_active: skips forwarding when user-initiated stream is active

No HITL review
"""

import json
from unittest.mock import AsyncMock, patch

import pytest

from acp.bridge import (
    BridgeState,
    StreamState,
    _process_stream_event,
)


SESSION_ID = "test-session-id"


class TestProcessStreamEvent:
    """Tests for _process_stream_event."""

    @pytest.fixture
    def state(self) -> BridgeState:
        """Fresh bridge state for each test."""
        return BridgeState()

    @pytest.fixture
    def stream_state(self) -> StreamState:
        """Fresh stream state for each test."""
        return StreamState()

    @pytest.mark.asyncio
    async def test_run_started_sends_working_status(self, state: BridgeState, stream_state: StreamState):
        """RunStarted event should send status=working to Nori."""
        with patch("acp.bridge.send") as mock_send:
            await _process_stream_event(
                state, stream_state, SESSION_ID, "RunStarted", "{}"
            )

        assert state.observer_turn_active is True
        mock_send.assert_called_once()
        call_arg = mock_send.call_args[0][0]
        assert call_arg["params"]["update"]["_meta"]["nori"]["status"] == "working"

    @pytest.mark.asyncio
    async def test_run_started_skipped_if_already_active(self, state: BridgeState, stream_state: StreamState):
        """RunStarted should not double-send if already in working state."""
        state.observer_turn_active = True
        with patch("acp.bridge.send") as mock_send:
            await _process_stream_event(
                state, stream_state, SESSION_ID, "RunStarted", "{}"
            )

        # Should not send anything since already active
        mock_send.assert_not_called()

    @pytest.mark.asyncio
    async def test_run_completed_sends_idle_status(self, state: BridgeState, stream_state: StreamState):
        """RunCompleted event should send status=idle to Nori."""
        state.observer_turn_active = True
        with patch("acp.bridge.send") as mock_send:
            await _process_stream_event(
                state, stream_state, SESSION_ID, "RunCompleted", '{"status": "success"}'
            )

        assert state.observer_turn_active is False
        mock_send.assert_called_once()
        call_arg = mock_send.call_args[0][0]
        assert call_arg["params"]["update"]["_meta"]["nori"]["status"] == "idle"

    @pytest.mark.asyncio
    async def test_run_completed_skipped_if_not_active(self, state: BridgeState, stream_state: StreamState):
        """RunCompleted should not send idle if not in working state."""
        state.observer_turn_active = False
        with patch("acp.bridge.send") as mock_send:
            await _process_stream_event(
                state, stream_state, SESSION_ID, "RunCompleted", '{"status": "success"}'
            )

        # Should not send anything since not active
        mock_send.assert_not_called()

    @pytest.mark.asyncio
    async def test_content_events_forwarded(self, state: BridgeState, stream_state: StreamState):
        """Content events should be forwarded via process_sse_event."""
        with patch("acp.bridge.process_sse_event", new_callable=AsyncMock) as mock_process:
            await _process_stream_event(
                state, stream_state, SESSION_ID, "PartDeltaEvent",
                '{"delta": {"content_delta": "hello"}}'
            )

        mock_process.assert_called_once()
        # Verify it was called with the right event type
        assert mock_process.call_args[0][3] == "PartDeltaEvent"

    @pytest.mark.asyncio
    async def test_skips_all_events_when_stream_active(self, state: BridgeState, stream_state: StreamState):
        """All events should be skipped when stream_active is True."""
        state.stream_active = True
        with patch("acp.bridge.send") as mock_send, \
             patch("acp.bridge.process_sse_event", new_callable=AsyncMock) as mock_process:
            # Test RunStarted
            await _process_stream_event(
                state, stream_state, SESSION_ID, "RunStarted", "{}"
            )
            # Test content event
            await _process_stream_event(
                state, stream_state, SESSION_ID, "PartDeltaEvent",
                '{"delta": {"content_delta": "hello"}}'
            )

        mock_send.assert_not_called()
        mock_process.assert_not_called()

    @pytest.mark.asyncio
    async def test_handles_invalid_json_gracefully(self, state: BridgeState, stream_state: StreamState):
        """Invalid JSON should be logged and skipped, not raise."""
        with patch("acp.bridge.send") as mock_send, \
             patch("acp.bridge.logger") as mock_logger:
            # Should not raise
            await _process_stream_event(
                state, stream_state, SESSION_ID, "RunStarted", "not valid json {"
            )

        mock_send.assert_not_called()
        mock_logger.warning.assert_called_once()
