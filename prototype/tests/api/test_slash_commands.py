"""
Tests for prototype slash command infrastructure (prototype/api/slash_commands.py).
"""
from unittest.mock import AsyncMock, Mock, patch

import pytest
from fastapi.sse import ServerSentEvent

from prototype.api.slash_commands import (
    _handle_recompile,
    _parse_slash_cmd,
    handle_slash_cmd,
    is_slash_cmd,
    SlashCommandDef,
)


class TestParseSlashCmd:
    """_parse_slash_cmd parsing — pure function, no I/O."""

    @pytest.mark.parametrize("msg,expected", [
        ("/recompile", ("recompile", "")),
        ("/recompile some args", ("recompile", "some args")),
        ("/RECOMPILE", ("recompile", "")),
        ("recompile", None),       # no leading slash
        ("/unknown_cmd", None),    # unrecognized command passes through to model
        ("/", None),               # slash with no command
        ("", None),                # empty string
    ])
    def test_parse_slash_cmd(self, msg, expected):
        assert _parse_slash_cmd(msg) == expected


class TestIsSlashCmd:
    """_is_slash_cmd — boolean wrapper around _parse_slash_cmd."""

    @pytest.mark.parametrize("msg,expected", [
        ("/recompile", True),
        ("/unknown_cmd", False),   # unrecognized → passes to model, not a slash cmd
        ("plain text", False),
    ])
    def test_is_slash_cmd(self, msg, expected):
        assert is_slash_cmd(msg) == expected


class TestHandleSlashCmd:
    """handle_slash_cmd dispatch logic."""

    @pytest.mark.parametrize("msg,expected_args", [
        ("/recompile", ""),
        ("/recompile some args", "some args"),
    ])
    async def test_dispatches_to_handler_with_correct_args(self, msg, expected_args):
        """handle_slash_cmd parses the message and calls the registered handler with the right args."""
        deps = Mock()
        expected_sse = ServerSentEvent(
            data={"name": "user_recompile", "args": expected_args, "result": "ok", "status": "success"},
            event="SlashCommandResult",
        )
        mock_handler = AsyncMock(return_value=expected_sse)

        mock_def = SlashCommandDef(handler=mock_handler, description="test")
        with patch.dict("prototype.api.slash_commands.SLASH_COMMANDS", {"recompile": mock_def}):
            result = await handle_slash_cmd(deps, msg)

        mock_handler.assert_awaited_once_with(deps, expected_args)
        assert result is expected_sse

    async def test_returns_error_sse_on_handler_exception(self):
        """Handler exceptions are caught and returned as a SlashCommandResult with status=error."""
        deps = Mock()
        mock_handler = AsyncMock(side_effect=RuntimeError("boom"))

        mock_def = SlashCommandDef(handler=mock_handler, description="test")
        with patch.dict("prototype.api.slash_commands.SLASH_COMMANDS", {"recompile": mock_def}):
            result = await handle_slash_cmd(deps, "/recompile")

        assert result.event == "SlashCommandResult"
        assert result.data["status"] == "error"
        assert "boom" in result.data["result"]

    async def test_returns_error_sse_when_precondition_violated(self):
        """Gracefully handles being called with a non-slash-command (precondition violated)."""
        deps = Mock()
        result = await handle_slash_cmd(deps, "not_a_slash_cmd")
        assert result.event == "SlashCommandResult"
        assert result.data["status"] == "error"


class TestHandleRecompile:
    """_handle_recompile handler — calls compile + commit, returns success SSE."""

    async def test_calls_compile_and_commit_then_returns_success_sse(self):
        deps = Mock()
        deps.commit_changes_refresh_agent_record = AsyncMock()

        with patch("prototype.api.slash_commands.compile_system_prompt", new_callable=AsyncMock) as mock_compile:
            result = await _handle_recompile(deps, "")

        mock_compile.assert_awaited_once_with(deps)
        deps.commit_changes_refresh_agent_record.assert_awaited_once()
        assert result.event == "SlashCommandResult"
        assert result.data["name"] == "user_recompile"
        assert result.data["status"] == "success"
        assert result.data["result"]  # non-empty result message
