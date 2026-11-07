"""Tests for cc_result_format — serve JSON to CC-native text mapping.

Vectors marked [doc] are verbatim from the native-format capture
(E-LLM Docs/Misc/cc-native-tool-result-format.md, CC v2.1.300).
"""
import json

import pytest

from mcp_tools.cc_result_format import format_cc_tool_result, persisted_output_path


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
    # The header these vectors pin — "[path — lines X-Y of Z]" — is the one
    # thing the Agent Home ACP bridge reads out of a Read result, to populate
    # rawOutput["lines"] for Nori's exploring cell (bridge.py,
    # _CC_READ_HEADER on Cursed/TempPseudoMain / ImproveNoriToolDisplay).
    # Changing the header shape here means updating that parser and its tests.
    def test_numbered_lines_with_trailing_newline(self):
        """[doc] file `alpha\\nbeta\\ngamma\\n` renders as numbered lines incl. final empty,
        under a file header (structural stamp/content separation)."""
        assert (
            format_cc_tool_result("Read", _read_json("alpha\nbeta\ngamma\n"), False)
            == "[/x/file.txt — lines 1-4 of 4]\n1\talpha\n2\tbeta\n3\tgamma\n4\t"
        )

    def test_partial_read_keeps_absolute_numbering(self):
        """[doc] offset: 2, limit: 1 gives `2\\tbeta` — startLine carries the absolute line.
        The header reports the partial window against the file total."""
        payload = json.dumps({
            "type": "text",
            "file": {"filePath": "/x/file.txt", "content": "beta", "numLines": 1,
                     "startLine": 2, "totalLines": 40},
        })
        assert format_cc_tool_result("Read", payload, False) == (
            "[/x/file.txt — lines 2-2 of 40]\n2\tbeta"
        )

    def test_file_without_trailing_newline(self):
        assert format_cc_tool_result("Read", _read_json("one\ntwo"), False) == (
            "[/x/file.txt — lines 1-2 of 2]\n1\tone\n2\ttwo"
        )

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


def _cut_bash(size: int = 1288895, stdout: str = "x" * 30000) -> str:
    """Serve's result for output over its 30k cap (CC 2.1.300)."""
    return json.dumps({
        "stdout": stdout, "stderr": "", "interrupted": False, "isImage": False,
        "noOutputExpected": False, "persistedOutputPath": "/p/tool-results/a.txt",
        "persistedOutputSize": size,
    })


class TestBashCutOutput:
    def test_persisted_output_replaces_the_cut(self):
        full = "line\n" * 100000
        result = format_cc_tool_result("Bash", _cut_bash(len(full)), False, full)
        assert result == full.rstrip("\n")

    def test_spill_file_shorter_than_reported_size_gets_note(self):
        """Serve caps the spill file but reports the real size."""
        result = format_cc_tool_result("Bash", _cut_bash(), False, "ab\n" * 10)
        assert result.startswith("ab\n" * 9 + "ab\n[Output incomplete: only the first 30 of 1288895 bytes")

    def test_non_ascii_spill_file_counted_in_bytes(self):
        payload = json.dumps({"stdout": "", "persistedOutputPath": "/p", "persistedOutputSize": 4})
        assert format_cc_tool_result("Bash", payload, False, "éé\n"[:2]) == "éé"

    def test_unreadable_file_returns_cut_with_visible_note(self):
        result = format_cc_tool_result("Bash", _cut_bash(), False, None)
        assert result.startswith("x" * 30000 + "\n[Output incomplete")
        assert "1288895 bytes" in result
        assert "first 30000 characters" in result

    def test_path_found_only_for_cut_bash_results(self):
        assert persisted_output_path("Bash", _cut_bash(), False) == "/p/tool-results/a.txt"
        assert persisted_output_path("Read", _cut_bash(), False) is None
        assert persisted_output_path("Bash", _cut_bash(), True) is None
        assert persisted_output_path("Bash", "not json", False) is None
        assert persisted_output_path("Bash", "[1]", False) is None
        assert persisted_output_path("Bash", json.dumps({"stdout": "a"}), False) is None

    def test_cut_failed_command_gets_note(self):
        """Failed runs are cut at 30k with no file; the cut is flagged."""
        err = "Exit code 1\n" + "y" * 30000
        result = format_cc_tool_result("Bash", err, True)
        assert result.startswith(err + "\n[Output may be incomplete")
        assert result.endswith("Re-run with output redirected to a file to see all of it.]")

    def test_short_failed_command_unchanged(self):
        err = "Exit code 1\n" + "y" * 29999
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
