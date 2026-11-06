"""Format `claude mcp serve` JSON tool results into Claude Code's native model-facing strings.

Serve mode returns JSON-serialized objects; CC's own agents never see those —
they see plain text. This module maps serve JSON to the native formats,
captured from CC v2.1.300 print mode on 2026-10-14
(E-LLM Docs/Misc/cc-native-tool-result-format.md).

Pure functions only — no FastMCP imports — so the mapping is directly
unit-testable. Reading a persisted-output file is I/O, so the proxy does it
and passes the text in. Anything unexpected (unparseable text, unknown tool, missing
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

# Serve cuts Bash output at this many characters (CC 2.1.300). Successful
# runs spill the full output to a file named in the result; failed runs
# (isError text) get the cut with no file and no marker.
_BASH_OUTPUT_CAP = 30000
_EXIT_CODE_PREFIX = "Exit code "


def persisted_output_path(tool_name: str, text: str, is_error: bool) -> str | None:
    """Path of the file holding the full output of a cut Bash result, if any."""
    if tool_name != "Bash" or is_error:
        return None
    try:
        payload = json.loads(text)
    except json.JSONDecodeError:
        return None
    if not isinstance(payload, dict):
        return None
    path = payload.get("persistedOutputPath")
    return path if isinstance(path, str) and path else None


def format_cc_tool_result(
    tool_name: str,
    text: str,
    is_error: bool,
    persisted_output: str | None = None,
) -> str:
    """Dispatch a serve-mode tool result to its native-format string.

    `persisted_output` is the contents of the file at `persisted_output_path`,
    or None if there was none or it couldn't be read.
    """
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
    if tool_name == "Bash":
        return _format_bash(payload, persisted_output)
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


def _format_bash(payload: dict, persisted_output: str | None = None) -> str:
    """stdout then stderr as plain text, trailing newline trimmed.

    When serve cut the output (`persistedOutputPath` set), the file holds the
    complete output (stdout and stderr interleaved in arrival order) and
    replaces the cut. Without it the cut is returned with an explicit note.
    Empty combined output renders as the native no-output marker.
    Non-zero-exit failures arrive as isError text and never reach this function.
    """
    combined = payload.get("stdout", "") + payload.get("stderr", "")
    if persisted_output is not None:
        combined = persisted_output.rstrip("\n")
        # Serve itself caps the spill file (observed: 64 MiB) while still
        # reporting the real size, so a short file means a second, silent cut.
        # Decoding can only lengthen the text, so this never misfires.
        size = payload.get("persistedOutputSize")
        have = len(persisted_output.encode("utf-8"))
        if isinstance(size, int) and have < size:
            combined += (
                f"\n[Output incomplete: only the first {have} of {size} bytes "
                "were kept by Claude Code.]"
            )
        return combined
    elif payload.get("persistedOutputPath"):
        size = payload.get("persistedOutputSize", "unknown")
        note = (
            f"[Output incomplete: the full output ({size} bytes) could not be "
            f"retrieved; only the first {len(combined)} characters are shown.]"
        )
        return combined.rstrip("\n") + "\n" + note
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
    if tool_name == "Bash" and text.startswith(_EXIT_CODE_PREFIX):
        # Failed runs are cut with no persisted file, so the rest is
        # unrecoverable; the cut is only detectable by its exact length.
        _, _, output = text.partition("\n")
        if len(output) >= _BASH_OUTPUT_CAP:
            return (
                f"{text}\n[Output may be incomplete: output of failed commands "
                f"is cut at {_BASH_OUTPUT_CAP} characters and the rest is not available.]"
            )
    return text


_FORMATTERS = {
    "Read": _format_read,
    "Edit": _format_edit,
    "Write": _format_write,
    "Bash": _format_bash,
}
