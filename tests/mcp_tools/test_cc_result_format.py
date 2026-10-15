"""Tests for cc_result_format — serve JSON to CC-native text mapping.

Vectors marked [doc] are verbatim from the native-format capture
(E-LLM Docs/Misc/cc-native-tool-result-format.md, CC v2.1.300).
"""
import json

import pytest

from mcp_tools.cc_result_format import format_cc_tool_result


def _read_json(content: str, start_line: int = 1) -> str:
    return json.dumps({
        "type": "text",
        "file": {
            "filePath": "/x/file.txt",
            "content": content,
            "numLines": content.count("\n") + 1,
            "startLine": start_line,
            "totalLines": content.count("\n") + 1,
        },
    })


class TestRead:
    def test_numbered_lines_with_trailing_newline(self):
        """[doc] file `alpha\\nbeta\\ngamma\\n` renders as numbered lines incl. final empty."""
        assert (
            format_cc_tool_result("Read", _read_json("alpha\nbeta\ngamma\n"), False)
            == "1\talpha\n2\tbeta\n3\tgamma\n4\t"
        )

    def test_partial_read_keeps_absolute_numbering(self):
        """[doc] offset: 2, limit: 1 gives `2\\tbeta` — startLine carries the absolute line."""
        assert format_cc_tool_result("Read", _read_json("beta", start_line=2), False) == "2\tbeta"

    def test_file_without_trailing_newline(self):
        assert format_cc_tool_result("Read", _read_json("one\ntwo"), False) == "1\tone\n2\ttwo"

    def test_missing_file_error_passes_through(self):
        """[doc] Read errors arrive as isError text already in native form."""
        err = "File does not exist. Note: your current working directory is /x."
        assert format_cc_tool_result("Read", err, True) == err


class TestWrite:
    def test_create_message(self):
        payload = json.dumps({"type": "create", "filePath": "/x/new.md", "content": "",
                              "structuredPatch": [], "originalFile": None, "userModified": False})
        assert format_cc_tool_result("Write", payload, False) == (
            "File created successfully at: /x/new.md"
            " (file state is current in your context — no need to Read it back)"
        )

    def test_update_message(self):
        payload = json.dumps({"type": "update", "filePath": "/x/existing.md", "content": "",
                              "structuredPatch": [], "originalFile": "old", "userModified": False})
        assert format_cc_tool_result("Write", payload, False) == (
            "The file /x/existing.md has been updated successfully."
            " (file state is current in your context — no need to Read it back)"
        )


class TestEdit:
    def test_success_message(self):
        payload = json.dumps({"filePath": "/x/f.md", "oldString": "a", "newString": "b",
                              "originalFile": "a", "structuredPatch": [], "userModified": False,
                              "replaceAll": False})
        assert format_cc_tool_result("Edit", payload, False) == "The file /x/f.md has been updated successfully."

    def test_replace_all_message(self):
        """[doc] replace_all success carries its own wording."""
        payload = json.dumps({"filePath": "/x/f.md", "oldString": "a", "newString": "b",
                              "originalFile": "a", "structuredPatch": [], "userModified": False,
                              "replaceAll": True})
        assert format_cc_tool_result("Edit", payload, False) == (
            "The file /x/f.md has been updated. All occurrences were successfully replaced."
        )

    def test_not_found_error_rewrapped(self):
        """[doc] serve's `Tool Edit input is invalid:` prefix is stripped and wrapped in tool_use_error."""
        err = (
            "Tool Edit input is invalid: String to replace not found in file.\n"
            "String: needle"
        )
        assert format_cc_tool_result("Edit", err, True) == (
            "<tool_use_error>String to replace not found in file.\nString: needle</tool_use_error>"
        )

    def test_multi_match_error_rewrapped(self):
        err = (
            "Tool Edit input is invalid: Found 3 matches of the string to replace, but replace_all is "
            "false. To replace all occurrences, set replace_all to true. To replace only one occurrence, "
            "please provide more context to uniquely identify the instance.\nString: needle"
        )
        assert format_cc_tool_result("Edit", err, True) == (
            "<tool_use_error>Found 3 matches of the string to replace, but replace_all is false. "
            "To replace all occurrences, set replace_all to true. To replace only one occurrence, "
            "please provide more context to uniquely identify the instance.\nString: needle</tool_use_error>"
        )


class TestBash:
    def test_stdout_then_stderr_trailing_newline_trimmed(self):
        """[doc] `echo out-line; echo err-line 1>&2` gives `out-line\\nerr-line`."""
        payload = json.dumps({"stdout": "out-line\n", "stderr": "err-line\n",
                              "interrupted": False, "isImage": False, "noOutputExpected": False})
        assert format_cc_tool_result("Bash", payload, False) == "out-line\nerr-line"

    def test_stdout_only(self):
        payload = json.dumps({"stdout": "done\n", "stderr": "", "interrupted": False})
        assert format_cc_tool_result("Bash", payload, False) == "done"

    def test_no_output_marker(self):
        """[doc] empty output renders as the native marker."""
        payload = json.dumps({"stdout": "", "stderr": "", "interrupted": False})
        assert format_cc_tool_result("Bash", payload, False) == "(Bash completed with no output)"

    def test_nonzero_exit_passes_through(self):
        """[doc] serve already formats non-zero exits natively — passthrough, no rewrap."""
        err = "Exit code 2\nout2\nls: cannot access './nonexistent-dir-xyz': No such file or directory"
        assert format_cc_tool_result("Bash", err, True) == err

    def test_timeout_passes_through(self):
        """[doc] serve's timeout text (Exit code 143 + partial output) passes through."""
        err = "Exit code 143\nCommand timed out after 30000\npartial output"
        assert format_cc_tool_result("Bash", err, True) == err


class TestPassThrough:
    """The proxy must never turn a working result into an error — anything
    unexpected passes through unchanged."""

    def test_non_json_success_passes_through(self):
        text = "some future plain-text result"
        assert format_cc_tool_result("Read", text, False) == text

    def test_unknown_tool_json_passes_through(self):
        payload = json.dumps({"type": "text", "something": "else"})
        assert format_cc_tool_result("NotebookEdit", payload, False) == payload

    def test_json_array_passes_through(self):
        payload = json.dumps([{"content": "a"}])
        assert format_cc_tool_result("Read", payload, False) == payload

    def test_non_edit_error_passes_through_unchanged(self):
        """Only Edit's serve wrapper gets rewrapped; other errors pass through."""
        err = "Tool Read input is invalid: something"
        assert format_cc_tool_result("Read", err, True) == err

    def test_malformed_read_json_falls_back_gracefully(self):
        """Missing file envelope falls back to raw content, not a crash."""
        payload = json.dumps({"content": "just content, no envelope"})
        result = format_cc_tool_result("Read", payload, False)
        assert isinstance(result, str)  # any defensive string output, never an exception
