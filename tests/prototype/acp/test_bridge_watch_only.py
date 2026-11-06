"""Tests for the ACP bridge's watch-only mode.

In watch-only mode the bridge must never send a mutating request (prompt, cancel)
to the server, while history replay and the /stream subscription still run.

No HITL review
"""

import asyncio
import json
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from prototype.acp.bridge import (
    BridgeState,
    dispatch,
    env_flag,
    handle_session_cancel,
    handle_session_prompt,
    send_available_commands,
)


SESSION_ID = "agent-1"


def make_client() -> MagicMock:
    """Client double that fails the test on any request method that could mutate."""
    client = MagicMock()
    client.post = AsyncMock(side_effect=AssertionError("POST sent"))
    client.stream = MagicMock(side_effect=AssertionError("stream request sent"))
    client.get = AsyncMock(side_effect=AssertionError("GET sent"))
    return client


def prompt_msg(text: str) -> dict:
    return {"jsonrpc": "2.0", "id": 7, "method": "session/prompt",
            "params": {"sessionId": SESSION_ID, "prompt": [{"type": "text", "text": text}]}}


def updates(mock_send) -> list[dict]:
    return [c[0][0] for c in mock_send.call_args_list]


class TestWatchOnlyPrompt:
    @pytest.mark.asyncio
    @pytest.mark.parametrize("text", ["hello", "/clear", "/compact"])
    async def test_prompt_not_forwarded(self, text: str):
        state = BridgeState(agent_id=SESSION_ID, watch_only=True)
        client = make_client()
        with patch("prototype.acp.bridge.send") as mock_send:
            await handle_session_prompt(state, prompt_msg(text), client)

        client.post.assert_not_called()
        client.stream.assert_not_called()
        sent = updates(mock_send)
        assert sent[0]["params"]["update"]["sessionUpdate"] == "agent_message_chunk"
        assert "Watch-only" in sent[0]["params"]["update"]["content"]["text"]
        assert sent[-1]["id"] == 7
        assert sent[-1]["result"]["stopReason"] == "end_turn"
        assert state.stream_active is False

    @pytest.mark.asyncio
    async def test_normal_mode_still_posts(self):
        state = BridgeState(agent_id=SESSION_ID)
        client = make_client()
        client.stream = MagicMock(side_effect=RuntimeError("reached POST stream"))
        with patch("prototype.acp.bridge.send"), pytest.raises(RuntimeError, match="reached POST stream"):
            await handle_session_prompt(state, prompt_msg("hello"), client)
        assert client.stream.call_args[0][0] == "POST"


class TestWatchOnlyCancel:
    @pytest.mark.asyncio
    async def test_cancel_is_noop(self):
        state = BridgeState(agent_id=SESSION_ID, watch_only=True)
        client = make_client()
        await handle_session_cancel(state, {"method": "session/cancel"}, client)
        client.post.assert_not_called()

    @pytest.mark.asyncio
    async def test_normal_mode_cancel_posts(self):
        state = BridgeState(agent_id=SESSION_ID)
        client = make_client()
        client.post = AsyncMock(return_value=MagicMock(status_code=200))
        await handle_session_cancel(state, {"method": "session/cancel"}, client)
        client.post.assert_awaited_once()
        assert client.post.call_args[0][0].endswith(f"/agents/{SESSION_ID}/cancel")


class TestWatchOnlySlashCommands:
    @pytest.mark.asyncio
    async def test_no_fetch_and_nothing_advertised(self):
        state = BridgeState(agent_id=SESSION_ID, watch_only=True)
        client = make_client()
        with patch("prototype.acp.bridge.send") as mock_send:
            await send_available_commands(state, SESSION_ID, client)
        client.get.assert_not_called()
        mock_send.assert_not_called()

    @pytest.mark.asyncio
    async def test_normal_mode_fetches(self):
        state = BridgeState(agent_id=SESSION_ID)
        client = make_client()
        resp = MagicMock()
        resp.json.return_value = [{"name": "clear"}]
        client.get = AsyncMock(return_value=resp)
        with patch("prototype.acp.bridge.send") as mock_send:
            await send_available_commands(state, SESSION_ID, client)
        client.get.assert_awaited_once()
        mock_send.assert_called_once()


class TestWatchOnlyStillObserves:
    @pytest.mark.asyncio
    async def test_session_load_replays_and_starts_stream(self):
        """Replay and the /stream subscription start from session/load, independent of any prompt."""
        state = BridgeState(agent_id=SESSION_ID, watch_only=True)
        client = make_client()
        with patch("prototype.acp.bridge.send") as mock_send, \
             patch("prototype.acp.bridge.replay_history", new=AsyncMock()) as replay, \
             patch("prototype.acp.bridge.subscribe_to_agent_stream", new=AsyncMock()) as subscribe:
            await dispatch(state, {"id": 1, "method": "session/load",
                                   "params": {"sessionId": SESSION_ID}}, client)
            await asyncio.sleep(0)  # let the stream task start

        replay.assert_awaited_once()
        subscribe.assert_awaited_once()
        assert state.stream_task is not None
        assert updates(mock_send)[-1]["id"] == 1
        client.post.assert_not_called()

    @pytest.mark.asyncio
    async def test_stream_events_forwarded_in_watch_only(self):
        """The stream subscription does not depend on stream_active, which only prompts set."""
        from prototype.acp.bridge import StreamState, _process_stream_event
        state = BridgeState(agent_id=SESSION_ID, watch_only=True)
        with patch("prototype.acp.bridge.send") as mock_send:
            await _process_stream_event(state, StreamState(), SESSION_ID, "RunStartedEvent", "{}")
        assert state.stream_active is False
        mock_send.assert_called_once()

    @pytest.mark.asyncio
    async def test_session_new_only_gets(self):
        state = BridgeState(agent_id=SESSION_ID, watch_only=True)
        client = make_client()
        client.get = AsyncMock(return_value=MagicMock(status_code=200))
        with patch("prototype.acp.bridge.send"), \
             patch("prototype.acp.bridge.subscribe_to_agent_stream", new=AsyncMock()):
            await dispatch(state, {"id": 1, "method": "session/new", "params": {}}, client)
            await asyncio.sleep(0)
        assert client.get.call_count == 1  # agent existence check only; no slash-commands fetch
        client.post.assert_not_called()


class TestEnvFlag:
    @pytest.mark.parametrize("value,expected", [("1", True), ("true", True), ("ON", True),
                                                ("0", False), ("", False), ("no", False)])
    def test_values(self, monkeypatch, value: str, expected: bool):
        monkeypatch.setenv("X_FLAG", value)
        assert env_flag("X_FLAG") is expected

    def test_unset(self, monkeypatch):
        monkeypatch.delenv("X_FLAG", raising=False)
        assert env_flag("X_FLAG") is False
