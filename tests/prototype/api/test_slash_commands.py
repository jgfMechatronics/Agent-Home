"""
Tests for prototype slash command infrastructure (prototype/api/slash_commands.py).
"""
from unittest.mock import AsyncMock, Mock, patch

import pytest
from fastapi.sse import ServerSentEvent

from conftest import AsyncNoOpContextManager
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
        ("/recompile",            "recompile"),
        ("/recompile some args",  "recompile"),   # args present but ignored
        ("/RECOMPILE",            "recompile"),   # case-insensitive
        ("/clear",                "clear"),
        ("/compact",              "compact"),
        ("recompile",             None),          # no leading slash
        ("/unknown_cmd",          None),          # unrecognized → passes to model
        ("/",                     None),          # slash with no command
        ("",                      None),          # empty string
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

    @pytest.mark.parametrize("msg", ["/recompile", "/recompile some args"])
    async def test_dispatches_to_handler(self, msg):
        """handle_slash_cmd calls the registered handler with deps only (no args passed down)."""
        deps = Mock()
        expected_sse = ServerSentEvent(
            data={"name": "user_recompile", "result": "ok", "status": "success"},
            event="SlashCommandResult",
        )
        mock_handler = AsyncMock(return_value=expected_sse)

        mock_def = SlashCommandDef(handler=mock_handler, description="test")
        with patch.dict("prototype.api.slash_commands.SLASH_COMMANDS", {"recompile": mock_def}):
            result = await handle_slash_cmd(deps, msg)

        mock_handler.assert_awaited_once_with(deps)
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
    """_handle_recompile handler — compiles within write_scope, returns success SSE."""

    async def test_calls_compile_and_commit_then_returns_success_sse(self):
        deps = Mock()
        deps.write_scope = Mock(return_value=AsyncNoOpContextManager())

        with patch("prototype.api.slash_commands.compile_system_prompt", new_callable=AsyncMock) as mock_compile:
            result = await _handle_recompile(deps)

        mock_compile.assert_awaited_once_with(deps)
        assert result.event == "SlashCommandResult"
        assert result.data["name"] == "user_recompile"
        assert result.data["status"] == "success"
        assert result.data["result"]  # non-empty result message


@pytest.mark.parametrize("handler,patch_target,did_work_keyword", [
    (_handle_clear,   "prototype.api.slash_commands.clear_context", "cleared"),
    (_handle_compact, "prototype.api.slash_commands.compact",       "complete"),
])
@pytest.mark.parametrize("did_work", [True, False])
async def test_slash_handler_returns_success_sse_with_context_sensitive_message(
    handler, patch_target, did_work_keyword, did_work
):
    """Both /clear and /compact handlers return a success SSE whose message reflects whether
    work was done (True) or the operation was a no-op (False)."""
    deps = Mock()

    with patch(patch_target, new_callable=AsyncMock, return_value=did_work):
        result = await handler(deps)

    assert result.event == "SlashCommandResult"
    assert result.data["status"] == "success"
    if did_work:
        assert did_work_keyword in result.data["result"].lower()
    else:
        assert "nothing" in result.data["result"].lower()
