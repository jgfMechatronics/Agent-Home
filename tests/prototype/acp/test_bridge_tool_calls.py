"""Tests for ACP tool-call shaping in acp.bridge.

Nori renders tool calls from the structured ACP fields (kind, rawInput,
rawOutput, locations, content) rather than from display text. These tests pin
that contract so the bridge keeps giving Nori enough to use its native cells:

- kind: the tool name -> ACP ToolKind mapping (unmapped falls back to "other")
- rawOutput: execute results land in "stdout" verbatim, so newlines stay real
  newlines instead of turning into literal \\n inside a JSON string
- content: edit diffs, and bridge-rendered text for generic tools (Nori's
  fetch/think/other bucket), since Nori has no structured cell for those
- process_sse_event: the call/result pair carries args and tool name through

No HITL review
"""

import json
from unittest.mock import patch

import pytest

from prototype.acp.bridge import (
    BridgeState,
    StreamState,
    _replay_message_items,
    process_sse_event,
    tool_call,
    tool_call_complete,
    tool_call_update,
    tool_kind,
)

SESSION_ID = "test-session-id"
CALL_ID = "call-1"


def update_of(message: dict) -> dict:
    """Unwrap a session/update notification down to the update payload."""
    assert message["method"] == "session/update"
    assert message["params"]["sessionId"] == SESSION_ID
    return message["params"]["update"]


MULTILINE_RESULT = "line one\nline two\n\nline four"


class TestToolKind:
    @pytest.mark.parametrize(
        ("tool_name", "expected"),
        [
            ("Bash", "execute"),
            ("Read", "read"),
            ("Edit", "edit"),
            ("Write", "edit"),
            ("web_fetch", "fetch"),
            ("duckduckgo_search", "search"),
        ],
    )
    def test_maps_known_tools(self, tool_name: str, expected: str):
        assert tool_kind(tool_name) == expected

    def test_unmapped_tool_falls_back_to_other(self):
        assert tool_kind("memory_insert") == "other"

    def test_missing_name_falls_back_to_other(self):
        assert tool_kind(None) == "other"


class TestToolCall:
    def test_bash_call_uses_execute_kind_and_parses_json_args(self):
        args = json.dumps({"command": "ls -la", "description": "list files"})

        update = update_of(tool_call(SESSION_ID, CALL_ID, "Bash", args))

        assert update["sessionUpdate"] == "tool_call"
        assert update["toolCallId"] == CALL_ID
        assert update["title"] == "Bash"
        assert update["kind"] == "execute"
        assert update["status"] == "in_progress"
        assert update["rawInput"] == {"command": "ls -la", "description": "list files"}
        assert update["locations"] == []
        assert update["content"] == []

    def test_read_call_reports_its_path_as_location(self):
        update = update_of(
            tool_call(SESSION_ID, CALL_ID, "Read", {"file_path": "/repo/src/app.py"})
        )

        assert update["kind"] == "read"
        assert update["locations"] == [{"path": "/repo/src/app.py"}]

    def test_edit_call_carries_a_diff_content_block(self):
        update = update_of(
            tool_call(
                SESSION_ID,
                CALL_ID,
                "Edit",
                {
                    "file_path": "/repo/src/app.py",
                    "old_string": "old line",
                    "new_string": "new line",
                },
            )
        )

        assert update["kind"] == "edit"
        assert update["locations"] == [{"path": "/repo/src/app.py"}]
        assert update["content"] == [
            {
                "type": "diff",
                "path": "/repo/src/app.py",
                "oldText": "old line",
                "newText": "new line",
            }
        ]

    def test_edit_call_without_new_text_has_no_diff(self):
        update = update_of(
            tool_call(SESSION_ID, CALL_ID, "Edit", {"file_path": "/repo/src/app.py"})
        )

        assert update["content"] == []

    def test_malformed_json_args_are_preserved_under_raw(self):
        update = update_of(tool_call(SESSION_ID, CALL_ID, "Bash", "not json {"))

        assert update["rawInput"] == {"raw": "not json {"}

    def test_no_markdown_display_text_is_manufactured(self):
        update = update_of(
            tool_call(SESSION_ID, CALL_ID, "Bash", {"command": "echo hi"})
        )

        assert "**Arguments:**" not in json.dumps(update)


class TestToolCallUpdate:
    def test_bash_result_is_placed_in_stdout_verbatim(self):
        update = update_of(
            tool_call_update(
                SESSION_ID,
                CALL_ID,
                MULTILINE_RESULT,
                "completed",
                {"command": "cat file"},
                "Bash",
            )
        )

        assert update["sessionUpdate"] == "tool_call_update"
        assert update["kind"] == "execute"
        assert update["status"] == "completed"
        assert update["rawOutput"] == {"stdout": MULTILINE_RESULT}
        # Execute cells render from rawOutput; adding a text block would duplicate it.
        assert "content" not in update

    def test_bash_failure_still_reports_stdout_with_failed_status(self):
        update = update_of(
            tool_call_update(
                SESSION_ID, CALL_ID, "Exit code 2\nboom", "failed",
                {"command": "false"}, "Bash",
            )
        )

        assert update["status"] == "failed"
        assert update["rawOutput"] == {"stdout": "Exit code 2\nboom"}

    def test_read_result_has_no_text_content(self):
        update = update_of(
            tool_call_update(
                SESSION_ID, CALL_ID, "[file — lines 1-2 of 2]\n1\tone\n2\ttwo",
                "completed", {"file_path": "/repo/file.txt"}, "Read",
            )
        )

        assert update["kind"] == "read"
        assert "content" not in update

    def test_edit_update_repeats_the_diff(self):
        args = {
            "file_path": "/repo/src/app.py",
            "old_string": "old line",
            "new_string": "new line",
        }

        update = update_of(tool_call_update(SESSION_ID, CALL_ID, "ok", "completed", args, "Edit"))

        assert update["kind"] == "edit"
        assert update["content"] == [
            {
                "type": "diff",
                "path": "/repo/src/app.py",
                "oldText": "old line",
                "newText": "new line",
            }
        ]

    def test_generic_tool_result_is_plain_text_with_real_newlines(self):
        update = update_of(
            tool_call_update(
                SESSION_ID, CALL_ID, MULTILINE_RESULT, "completed", {}, "memory_insert"
            )
        )

        assert update["kind"] == "other"
        assert update["rawOutput"] == {"result": MULTILINE_RESULT}
        assert update["content"] == [
            {"type": "content", "content": {"type": "text", "text": MULTILINE_RESULT}}
        ]
        assert "\\n" not in update["content"][0]["content"]["text"]

    def test_generic_tool_failure_exposes_the_error_text(self):
        update = update_of(
            tool_call_update(
                SESSION_ID, CALL_ID, "block not found", "failed", {}, "memory_insert"
            )
        )

        assert update["status"] == "failed"
        assert update["rawOutput"] == {"error": "block not found"}

    def test_dict_result_passes_through_as_raw_output(self):
        result = {"content": "hello\nworld"}

        update = update_of(
            tool_call_update(SESSION_ID, CALL_ID, result, "completed", {}, "memory_read")
        )

        assert update["rawOutput"] == result

    def test_update_without_tool_name_omits_kind(self):
        update = update_of(tool_call_update(SESSION_ID, CALL_ID, "result"))

        assert "kind" not in update
        assert update["rawOutput"] == {"result": "result"}


class TestGenericDisplay:
    """Generic kinds (fetch/think/other): the bridge renders args and results."""

    def test_generic_call_renders_args_as_key_value_text(self):
        update = update_of(
            tool_call(
                SESSION_ID,
                CALL_ID,
                "memory_insert",
                {"label": "persona", "content": "line one\nline two"},
            )
        )

        assert update["kind"] == "other"
        assert update["content"] == [
            {
                "type": "content",
                "content": {
                    "type": "text",
                    "text": "label: persona\ncontent:\n  line one\n  line two",
                },
            }
        ]

    def test_generic_update_includes_args_then_result(self):
        update = update_of(
            tool_call_update(
                SESSION_ID,
                CALL_ID,
                "Message delivered.",
                "completed",
                {"target_name": "Haiku"},
                "send_message",
            )
        )

        text = update["content"][0]["content"]["text"]
        assert text == "target_name: Haiku\n\nMessage delivered."
        assert "\\n" not in text

    def test_generic_fetch_kind_uses_the_same_rendering(self):
        update = update_of(
            tool_call(SESSION_ID, CALL_ID, "web_fetch", {"url": "https://example.com"})
        )

        assert update["kind"] == "fetch"
        assert update["content"][0]["content"]["text"] == "url: https://example.com"

    def test_dict_result_renders_as_key_value(self):
        update = update_of(
            tool_call_update(
                SESSION_ID,
                CALL_ID,
                {"status": "ok", "detail": "line one\nline two"},
                "completed",
                {},
                "memory_read",
            )
        )

        text = update["content"][0]["content"]["text"]
        assert text == "status: ok\ndetail:\n  line one\n  line two"

    def test_long_display_is_middle_truncated(self):
        long_result = "\n".join(f"line {i}" for i in range(20))

        update = update_of(
            tool_call_update(
                SESSION_ID, CALL_ID, long_result, "completed", {}, "memory_read"
            )
        )

        lines = update["content"][0]["content"]["text"].split("\n")
        assert len(lines) == 15
        assert lines[0] == "line 0"
        assert lines[7] == "… +6 lines"
        assert lines[-1] == "line 19"

    def test_structured_kinds_still_send_no_content(self):
        for tool_name, args in (
            ("Bash", {"command": "ls"}),
            ("Read", {"file_path": "/repo/a.py"}),
            ("duckduckgo_search", {"query": "nori"}),
        ):
            update = update_of(tool_call(SESSION_ID, CALL_ID, tool_name, args))
            assert update["content"] == [], tool_name


class TestToolCallComplete:
    def test_replay_payload_uses_the_same_shaping(self):
        update = update_of(
            tool_call_complete(
                SESSION_ID,
                CALL_ID,
                "Bash",
                {"command": "ls"},
                MULTILINE_RESULT,
                "completed",
            )
        )

        assert update["sessionUpdate"] == "tool_call"
        assert update["kind"] == "execute"
        assert update["status"] == "completed"
        assert update["rawInput"] == {"command": "ls"}
        assert update["rawOutput"] == {"stdout": MULTILINE_RESULT}
        assert "**Arguments:**" not in json.dumps(update)


class TestProcessSseEventToolCalls:
    @pytest.fixture
    def state(self) -> BridgeState:
        return BridgeState()

    @pytest.fixture
    def stream_state(self) -> StreamState:
        return StreamState()

    @pytest.mark.asyncio
    async def test_tool_call_event_emits_structured_call(
        self, state: BridgeState, stream_state: StreamState
    ):
        data = json.dumps({
            "part": {
                "tool_call_id": CALL_ID,
                "tool_name": "Bash",
                "args": json.dumps({"command": "ls -la"}),
            }
        })

        with patch("prototype.acp.bridge.send") as mock_send:
            await process_sse_event(state, stream_state, SESSION_ID, "FunctionToolCallEvent", data, {})

        update = update_of(mock_send.call_args[0][0])
        assert update["kind"] == "execute"
        assert update["rawInput"] == {"command": "ls -la"}
        assert stream_state.tool_args[CALL_ID] == {"command": "ls -la"}
        assert stream_state.tool_names[CALL_ID] == "Bash"

    @pytest.mark.asyncio
    async def test_tool_result_event_reuses_kind_and_args(
        self, state: BridgeState, stream_state: StreamState
    ):
        stream_state.tool_args[CALL_ID] = {"command": "ls -la"}
        stream_state.tool_names[CALL_ID] = "Bash"
        data = json.dumps({
            "part": {
                "tool_call_id": CALL_ID,
                "content": MULTILINE_RESULT,
                "part_kind": "tool-return",
                "outcome": "success",
            }
        })

        with patch("prototype.acp.bridge.send") as mock_send:
            await process_sse_event(state, stream_state, SESSION_ID, "FunctionToolResultEvent", data, {})

        update = update_of(mock_send.call_args[0][0])
        assert update["kind"] == "execute"
        assert update["rawOutput"] == {"stdout": MULTILINE_RESULT}
        # State is consumed by the result event.
        assert CALL_ID not in stream_state.tool_args
        assert CALL_ID not in stream_state.tool_names

    @pytest.mark.asyncio
    async def test_retry_prompt_is_reported_as_failed(
        self, state: BridgeState, stream_state: StreamState
    ):
        stream_state.tool_names[CALL_ID] = "memory_insert"
        data = json.dumps({
            "part": {
                "tool_call_id": CALL_ID,
                "content": "block not found",
                "part_kind": "retry-prompt",
            }
        })

        with patch("prototype.acp.bridge.send") as mock_send:
            await process_sse_event(state, stream_state, SESSION_ID, "FunctionToolResultEvent", data, {})

        update = update_of(mock_send.call_args[0][0])
        assert update["status"] == "failed"
        assert update["rawOutput"] == {"error": "block not found"}

    @pytest.mark.asyncio
    async def test_thinking_and_text_chunks_are_forwarded(
        self, state: BridgeState, stream_state: StreamState
    ):
        with patch("prototype.acp.bridge.send") as mock_send:
            await process_sse_event(
                state, stream_state, SESSION_ID, "PartStartEvent",
                json.dumps({"part": {"part_kind": "thinking", "content": "pondering"}}), {},
            )
            await process_sse_event(
                state, stream_state, SESSION_ID, "PartEndEvent",
                json.dumps({"part": {"part_kind": "thinking"}}), {},
            )

        texts = [update_of(call[0][0])["content"]["text"] for call in mock_send.call_args_list]
        assert any("THINKING" in text for text in texts)
        assert "pondering" in texts


class TestReplayToolCalls:
    @pytest.fixture
    def state(self) -> BridgeState:
        return BridgeState()

    def test_replay_carries_tool_name_into_the_result_update(self):
        items = [
            {
                "id": "1",
                "seq_id": 1,
                "type": "message",
                "content": json.dumps({
                    "kind": "response",
                    "parts": [{
                        "part_kind": "tool-call",
                        "tool_call_id": CALL_ID,
                        "tool_name": "Bash",
                        "args": json.dumps({"command": "ls"}),
                    }],
                }),
            },
            {
                "id": "2",
                "seq_id": 2,
                "type": "message",
                "content": json.dumps({
                    "kind": "request",
                    "parts": [{
                        "part_kind": "tool-return",
                        "tool_call_id": CALL_ID,
                        "content": MULTILINE_RESULT,
                        "outcome": "success",
                    }],
                }),
            },
        ]

        with patch("prototype.acp.bridge.send") as mock_send:
            _replay_message_items(SESSION_ID, items)

        updates = [update_of(call[0][0]) for call in mock_send.call_args_list]
        call_update = next(u for u in updates if u["sessionUpdate"] == "tool_call")
        result_update = next(u for u in updates if u["sessionUpdate"] == "tool_call_update")
        assert call_update["kind"] == "execute"
        assert result_update["kind"] == "execute"
        assert result_update["rawOutput"] == {"stdout": MULTILINE_RESULT}
