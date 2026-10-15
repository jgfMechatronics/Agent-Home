"""Format `claude mcp serve` JSON tool results into Claude Code's native model-facing strings.

Serve mode returns JSON-serialized objects; CC's own agents never see those —
they see plain text. This module maps serve JSON to the native formats,
captured from CC v2.1.300 print mode on 2026-10-14
(E-LLM Docs/Misc/cc-native-tool-result-format.md).

Pure functions only — no FastMCP imports — so the mapping is directly
unit-testable. Anything unexpected (unparseable text, unknown tool, missing
fields) passes through unchanged rather than crashing: the proxy must never
turn a working tool result into an error.
"""
import json

# Parenthetical appended to Write/Edit success messages: CC's native anti-
# redundant-Read nudge. The file state really is current in context — the
# model wrote it.
_NO_READ_BACK = " (file state is current in your context — no need to Read it back)"

# Serve wraps Edit failures as text with this prefix; native CC wraps the
# same failure in <tool_use_error> tags instead.
_EDIT_ERROR_PREFIX = "Tool Edit input is invalid: "


def format_cc_tool_result(tool_name: str, text: str, is_error: bool) -> str:
    """Dispatch a serve-mode tool result to its native-format string."""
    if is_error:
        return _format_error(tool_name, text)
    try:
        payload = json.loads(text)
    except json.JSONDecodeError:
        return text
    if not isinstance(payload, dict):
        return text
    formatter = _FORMATTERS.get(tool_name)
    if formatter is None:
        return text
    return formatter(payload)


def _format_read(payload: dict) -> str:
    """Numbered lines with a one-line file header: `<n>\\t<line>`, 1-based absolute.

    The header (`[/path — lines X-Y of Z]`) separates the (stamped) tool-return
    metadata from file content structurally — content lines could themselves be
    timestamps (logs, transcripts), so position alone can't disambiguate. It
    also carries partial-read context the numbered format loses (which file,
    what window of the total).

    A file ending with a newline produces a final empty line (`<n>\\t`),
    matching native. Partial reads keep absolute numbering via `startLine`.
    """
    file_info = payload.get("file", payload)
    if not isinstance(file_info, dict):
        return payload.get("content", "") if isinstance(payload, dict) else ""
    content = file_info.get("content", "")
    start_line = file_info.get("startLine", 1)
    num_lines = len(content.split("\n"))
    total_lines = file_info.get("totalLines", num_lines)
    file_path = file_info.get("filePath", "<unknown>")
    header = f"[{file_path} — lines {start_line}-{start_line + num_lines - 1} of {total_lines}]"
    numbered = "\n".join(
        f"{start_line + i}\t{line}" for i, line in enumerate(content.split("\n"))
    )
    return f"{header}\n{numbered}"


def _format_write(payload: dict) -> str:
    """Native Write messages; `type` reports create-vs-update of the write."""
    file_path = payload.get("filePath", "<unknown>")
    if payload.get("type") == "create":
        return f"File created successfully at: {file_path}{_NO_READ_BACK}"
    return f"The file {file_path} has been updated successfully.{_NO_READ_BACK}"


def _format_edit(payload: dict) -> str:
    """Native Edit messages; replace_all success carries its own wording."""
    file_path = payload.get("filePath", "<unknown>")
    if payload.get("replaceAll"):
        return f"The file {file_path} has been updated. All occurrences were successfully replaced."
    return f"The file {file_path} has been updated successfully."


def _format_bash(payload: dict) -> str:
    """stdout then stderr as plain text, trailing newline trimmed.

    Empty combined output renders as the native no-output marker.
    Non-zero-exit and timeout failures arrive as isError text and never
    reach this function.
    """
    combined = payload.get("stdout", "") + payload.get("stderr", "")
    combined = combined.rstrip("\n")
    return combined if combined else "(Bash completed with no output)"


def _format_error(tool_name: str, text: str) -> str:
    """Error results: rewrap Edit's serve wrapper natively, pass the rest through.

    Serve's non-zero-exit Bash text (`Exit code <N>\\n<output>`) is already
    native-formatted, as are Read's missing-file / EISDIR errors.
    """
    if tool_name == "Edit" and text.startswith(_EDIT_ERROR_PREFIX):
        stripped = text[len(_EDIT_ERROR_PREFIX):]
        return f"<tool_use_error>{stripped}</tool_use_error>"
    return text


_FORMATTERS = {
    "Read": _format_read,
    "Edit": _format_edit,
    "Write": _format_write,
    "Bash": _format_bash,
}
