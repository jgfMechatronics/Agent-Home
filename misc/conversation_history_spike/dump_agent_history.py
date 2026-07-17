#!/usr/bin/env python3
"""Dump a Letta agent's conversation history to a file.

Supports range extraction via --start-text / --end-text markers, and
either human-readable text output (default) or raw JSON (--json).

Usage examples:
    # Full history, human-readable
    python3 dump_agent_history.py <agent_id> out.txt

    # Sliced range, raw JSON (for import testing)
    python3 dump_agent_history.py <agent_id> out.json --json \\
        --start-text "Hello, this is the beginning" \\
        --end-text "And this is the end"
"""

import argparse
import json
import sys
from datetime import datetime, timezone

import httpx

# Author: Sonnet


def _text_of(msg: dict) -> str:
    """Extract all searchable text from a message regardless of type."""
    parts = []
    mt = msg.get("message_type", "")

    content = msg.get("content")
    if content is not None:
        if isinstance(content, str):
            parts.append(content)
        elif isinstance(content, list):
            for block in content:
                if isinstance(block, dict):
                    parts.append(block.get("text", "") or "")

    if mt == "reasoning_message":
        parts.append(msg.get("reasoning", "") or "")
    elif mt == "tool_call_message":
        tc = msg.get("tool_call", {}) or {}
        parts.append(tc.get("name", ""))
        parts.append(tc.get("arguments", "") or "")
    elif mt == "tool_return_message":
        parts.append(msg.get("tool_return", "") or "")
    elif mt == "approval_request_message":
        for tc in msg.get("tool_calls") or []:
            parts.append(tc.get("name", ""))
            parts.append(tc.get("arguments", "") or "")
    elif mt == "approval_response_message":
        for a in msg.get("approvals") or []:
            parts.append(a.get("tool_return", "") or "")

    return " ".join(parts)


def _format_content(content) -> str:
    """Format a content field (str or list of text blocks) as plain text."""
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        blocks = [
            block.get("text", "")
            for block in content
            if isinstance(block, dict) and block.get("type") == "text"
        ]
        return "\n".join(blocks).strip()
    return str(content)


def _format_message(seq_i: int, msg: dict) -> list[str]:
    """Return lines representing one message in human-readable form."""
    mt = msg.get("message_type", "unknown")
    date = msg.get("date", "")
    lines = [f"[{seq_i}] {mt.upper()}  {date}", "-" * 40]

    if mt in ("user_message", "assistant_message"):
        lines.append(_format_content(msg.get("content", "")))

    elif mt == "reasoning_message":
        lines.append("  [THINKING]")
        for line in (msg.get("reasoning", "") or "").splitlines():
            lines.append(f"  {line}")
        lines.append("  [/THINKING]")

    elif mt == "tool_call_message":
        tc = msg.get("tool_call", {}) or {}
        lines.append(f"  [TOOL CALL: {tc.get('name', '?')} | id={tc.get('tool_call_id', '?')}]")
        args_str = tc.get("arguments", "") or ""
        if args_str:
            try:
                args_pretty = json.dumps(json.loads(args_str), indent=4)
            except (json.JSONDecodeError, ValueError):
                args_pretty = args_str
            for line in args_pretty.splitlines():
                lines.append(f"    {line}")

    elif mt == "tool_return_message":
        lines.append(
            f"  [TOOL RETURN: id={msg.get('tool_call_id', '?')} | status={msg.get('status', '?')}]"
        )
        tr = msg.get("tool_return", "") or ""
        preview = tr[:500] + ("..." if len(tr) > 500 else "")
        lines.append(f"  {preview}")

    elif mt == "approval_request_message":
        for tc in msg.get("tool_calls") or []:
            lines.append(
                f"  [APPROVAL REQUEST: {tc.get('name', '?')} | id={tc.get('tool_call_id', '?')}]"
            )
            args_str = tc.get("arguments", "") or ""
            if args_str:
                lines.append(f"  args: {args_str[:300]}")

    elif mt == "approval_response_message":
        for a in msg.get("approvals") or []:
            lines.append(
                f"  [APPROVAL RESPONSE: status={a.get('status', '?')} | id={a.get('tool_call_id', '?')}]"
            )
            tr = a.get("tool_return", "") or ""
            lines.append(f"  {tr[:300]}{'...' if len(tr) > 300 else ''}")

    else:
        lines.append(f"  (unhandled type: {mt})")
        safe = {k: v for k, v in msg.items() if k not in ("id", "otid", "signature")}
        lines.append(f"  {json.dumps(safe)[:400]}")

    lines.append("")
    return lines


def _find_range(
    messages: list[dict],
    start_text: str | None,
    end_text: str | None,
) -> tuple[int, int]:
    start_idx = 0
    end_idx = len(messages) - 1

    if start_text:
        found = False
        for i, msg in enumerate(messages):
            if start_text in _text_of(msg):
                start_idx = i
                found = True
                break
        if not found:
            print("WARNING: --start-text not found; using index 0", file=sys.stderr)

    if end_text:
        found_end = -1
        for i, msg in enumerate(messages):
            if end_text in _text_of(msg):
                found_end = i  # take the last match
        if found_end >= 0:
            end_idx = found_end
        else:
            print("WARNING: --end-text not found; using last message", file=sys.stderr)

    return start_idx, end_idx


def dump_history(
    agent_id: str,
    server_url: str,
    output_path: str,
    start_text: str | None,
    end_text: str | None,
    raw_json: bool,
) -> None:
    response = httpx.get(f"{server_url}/agents/{agent_id}/messages")
    response.raise_for_status()
    messages = response.json()  # Letta returns a bare list (not a wrapped object)

    if not isinstance(messages, list):
        print(f"ERROR: unexpected response shape: {type(messages)}", file=sys.stderr)
        sys.exit(1)

    start_idx, end_idx = _find_range(messages, start_text, end_text)
    sliced = messages[start_idx : end_idx + 1]

    print(
        f"Exporting [{start_idx}..{end_idx}] ({len(sliced)} of {len(messages)} messages)",
        file=sys.stderr,
    )

    if raw_json:
        with open(output_path, "w") as f:
            json.dump(sliced, f, indent=2)
        print(f"Wrote raw JSON → {output_path}", file=sys.stderr)
        return

    lines = [
        f"Agent: {agent_id}",
        f"Exported: {datetime.now(timezone.utc).isoformat()}",
        f"Range: [{start_idx}..{end_idx}]",
        f"Total messages in slice: {len(sliced)}",
        "=" * 80,
        "",
    ]

    for i, msg in enumerate(sliced, start=start_idx):
        lines.extend(_format_message(i, msg))

    lines.append("=" * 80)

    with open(output_path, "w") as f:
        f.write("\n".join(lines))

    print(f"Wrote text dump → {output_path}", file=sys.stderr)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Dump a Letta agent's conversation history to a file."
    )
    parser.add_argument("agent_id", help="Agent ID")
    parser.add_argument("output", help="Output file path")
    parser.add_argument(
        "--server",
        default="http://host.docker.internal:8283/v1",
        help="Letta server base URL (default: http://host.docker.internal:8283/v1)",
    )
    parser.add_argument("--start-text", help="Find first message containing this text (inclusive)")
    parser.add_argument(
        "--end-text",
        help="Find last message containing this text (inclusive; takes last match)",
    )
    parser.add_argument(
        "--json",
        dest="raw_json",
        action="store_true",
        help="Output raw JSON instead of human-readable text",
    )
    args = parser.parse_args()

    try:
        dump_history(
            args.agent_id,
            args.server,
            args.output,
            args.start_text,
            args.end_text,
            args.raw_json,
        )
    except httpx.HTTPStatusError as e:
        print(f"HTTP error: {e.response.status_code} {e.response.text}", file=sys.stderr)
        sys.exit(1)
    except httpx.RequestError as e:
        print(f"Request failed: {e}", file=sys.stderr)
        sys.exit(1)
