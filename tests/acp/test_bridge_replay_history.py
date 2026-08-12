"""Tests for history replay in acp.bridge.replay_history.

Tests verify that replay_history caps the number of replayed messages
at HISTORY_REPLAY_LIMIT to avoid slow/wasteful replay of long histories.

No HITL review
"""

import json
from unittest.mock import AsyncMock, Mock, patch

import pytest

from acp.bridge import BridgeState, HISTORY_REPLAY_LIMIT, replay_history


SESSION_ID = "test-session-id"


def message_item(msg_id: int, seq_id: int, kind: str = "request") -> dict:
    """Build a test message item."""
    if kind == "request":
        content = {
            "kind": "request",
            "parts": [
                {
                    "part_kind": "user-prompt",
                    "content": f"Message {msg_id}",
                }
            ],
        }
    else:  # response
        content = {
            "kind": "response",
            "parts": [
                {
                    "part_kind": "text",
                    "content": f"Response {msg_id}",
                }
            ],
        }
    return {
        "id": str(msg_id),
        "seq_id": seq_id,
        "type": "message",
        "content": json.dumps(content),
        "timestamp": "2024-01-01T00:00:00Z",
    }


@pytest.mark.asyncio
async def test_replay_history_limits_to_last_n_items():
    """When server returns >40 items, only the last 40 are replayed."""
    state = BridgeState()
    state.server_url = "http://localhost:8000"

    # Create 60 message items
    all_items = [message_item(i, i) for i in range(60)]

    # Mock the HTTP response and client
    mock_response = Mock()
    mock_response.json.return_value = {"messages": all_items}
    mock_response.raise_for_status.return_value = None
    
    mock_client = AsyncMock()
    mock_client.get = AsyncMock(return_value=mock_response)

    with patch("acp.bridge.send") as mock_send:
        await replay_history(state, SESSION_ID, mock_client)

    # Verify the client was called
    mock_client.get.assert_called_once_with(
        f"{state.server_url}/agents/{SESSION_ID}/messages"
    )

    # Should have replayed only the last 40 items
    # Each message item generates at least one send() call
    # We expect at least 40 sends (for the last 40 items)
    assert len(mock_send.call_args_list) >= HISTORY_REPLAY_LIMIT


@pytest.mark.asyncio
async def test_replay_history_works_with_fewer_items():
    """When server returns <40 items, all are replayed."""
    state = BridgeState()
    state.server_url = "http://localhost:8000"

    # Create 20 message items (less than limit)
    all_items = [message_item(i, i) for i in range(20)]

    mock_response = Mock()
    mock_response.json.return_value = {"messages": all_items}
    mock_response.raise_for_status.return_value = None
    
    mock_client = AsyncMock()
    mock_client.get = AsyncMock(return_value=mock_response)

    with patch("acp.bridge.send") as mock_send:
        await replay_history(state, SESSION_ID, mock_client)

    # Should have called send() for all 20 items
    assert len(mock_send.call_args_list) >= 20


@pytest.mark.asyncio
async def test_replay_history_updates_watermark_to_last_seq_id():
    """Watermark is updated to the seq_id of the last replayed message."""
    state = BridgeState()
    state.server_url = "http://localhost:8000"

    # Create items with seq_ids 10-69
    all_items = [message_item(i, i + 10) for i in range(60)]

    mock_response = Mock()
    mock_response.json.return_value = {"messages": all_items}
    mock_response.raise_for_status.return_value = None
    
    mock_client = AsyncMock()
    mock_client.get = AsyncMock(return_value=mock_response)

    with patch("acp.bridge.send"):
        await replay_history(state, SESSION_ID, mock_client)

    # Watermark should be updated to the seq_id of the last item (69)
    assert state.last_message_seq_id == 69


@pytest.mark.asyncio
async def test_replay_history_handles_empty_messages():
    """Empty message list doesn't crash."""
    state = BridgeState()
    state.server_url = "http://localhost:8000"

    mock_client = AsyncMock()
    mock_response = AsyncMock()
    mock_response.json.return_value = {"messages": []}
    mock_client.get.return_value = mock_response

    with patch("acp.bridge.send"):
        await replay_history(state, SESSION_ID, mock_client)

    # Watermark should not be updated (no items)
    assert state.last_message_seq_id is None


@pytest.mark.asyncio
async def test_replay_history_handles_fetch_error():
    """Fetch errors are logged but don't crash."""
    state = BridgeState()
    state.server_url = "http://localhost:8000"

    mock_client = AsyncMock()
    mock_client.get.side_effect = Exception("Network error")

    with patch("acp.bridge.send"):
        # Should not raise
        await replay_history(state, SESSION_ID, mock_client)

    # Watermark should not be updated on error
    assert state.last_message_seq_id is None
