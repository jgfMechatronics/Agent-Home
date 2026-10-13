"""
Tests for prototype slash command infrastructure (prototype/api/slash_commands.py).
"""
from unittest.mock import AsyncMock, Mock, patch

import pytest
from fastapi.sse import ServerSentEvent

from prototype.api.slash_commands import (
    _handle_clear,
    _handle_compact,
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
        ("/clear", ("clear", "")),
        ("/compact", ("compact", "")),
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


class TestHandleClear:
    """_handle_clear — calls clear_context, returns success SSE."""

    async def test_calls_clear_context_and_returns_success_sse(self):
        deps = Mock()

        with patch("prototype.api.slash_commands.clear_context", new_callable=AsyncMock) as mock_clear:
            result = await _handle_clear(deps, "")

        mock_clear.assert_awaited_once_with(deps)
        assert result.event == "SlashCommandResult"
        assert result.data["name"] == "user_clear"
        assert result.data["status"] == "success"
        assert result.data["result"]


class TestHandleCompact:
    """_handle_compact — calls compact_on_demand, returns success SSE with context-sensitive message."""

    async def test_reports_compaction_occurred_when_pointer_advanced(self):
        deps = Mock()

        with patch("prototype.api.slash_commands.compact", new_callable=AsyncMock, return_value=True):
            result = await _handle_compact(deps, "")

        assert result.event == "SlashCommandResult"
        assert result.data["status"] == "success"
        assert "complete" in result.data["result"].lower()

    async def test_reports_nothing_to_compact_when_context_too_small(self):
        deps = Mock()

        with patch("prototype.api.slash_commands.compact", new_callable=AsyncMock, return_value=False):
            result = await _handle_compact(deps, "")

        assert result.event == "SlashCommandResult"
        assert result.data["status"] == "success"
        assert "nothing" in result.data["result"].lower()
