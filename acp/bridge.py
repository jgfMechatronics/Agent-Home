"""ACP bridge: translates stdio JSON-RPC ↔ Agent Home HTTP/SSE.

Phase 1: Happy path only. No cancellation, no permissions.
TODO: This bridge is too smart, its fine as a POC but really Agent-Home server should speak ACP and the bridge should just
map stdio->HTTP and back, with maybe some basic polling. Right now, Agent-Home isn't really ACP compatible.
"""

from __future__ import annotations

import asyncio
import json
import logging
import sys
from dataclasses import dataclass, field
from datetime import datetime
from typing import Any

import httpx

logger = logging.getLogger(__name__)


# =============================================================================
# JSON-RPC Helpers
# =============================================================================

def send(msg: dict[str, Any]) -> None:
    """Write a JSON-RPC message to stdout (newline-terminated, flushed)."""
    sys.stdout.buffer.write(json.dumps(msg).encode() + b"\n")
    sys.stdout.buffer.flush()


def response(id: int | str, result: dict[str, Any]) -> dict[str, Any]:
    """Build a JSON-RPC response."""
    return {"jsonrpc": "2.0", "id": id, "result": result}


def error_response(id: int | str | None, code: int, message: str) -> dict[str, Any]:
    """Build a JSON-RPC error response."""
    return {"jsonrpc": "2.0", "id": id, "error": {"code": code, "message": message}}


def notification(method: str, params: dict[str, Any]) -> dict[str, Any]:
    """Build a JSON-RPC notification (no id)."""
    return {"jsonrpc": "2.0", "method": method, "params": params}


# =============================================================================
# ACP Session/Update Builders
# =============================================================================

def session_update(session_id: str, update: dict[str, Any]) -> dict[str, Any]:
    """Build a session/update notification."""
    return notification("session/update", {"sessionId": session_id, "update": update})


def agent_message_chunk(session_id: str, text: str) -> dict[str, Any]:
    """Build an AgentMessageChunk update for text streaming."""
    return session_update(session_id, {
        "sessionUpdate": "agent_message_chunk",
        "content": {"type": "text", "text": text},
    })


def agent_thought_chunk(session_id: str, text: str) -> dict[str, Any]:
    """Build an AgentThoughtChunk update for thinking streaming."""
    return session_update(session_id, {
        "sessionUpdate": "agent_thought_chunk",
        "content": {"type": "text", "text": text},
    })


def user_message_chunk(session_id: str, text: str) -> dict[str, Any]:
    """Build a UserMessageChunk update for user messages (used in history replay)."""
    return session_update(session_id, {
        "sessionUpdate": "user_message_chunk",
        "content": {"type": "text", "text": text},
    })


def tool_call_complete(
    session_id: str, tool_call_id: str, tool_name: str,
    args: dict[str, Any], result: Any, status: str = "completed"
) -> dict[str, Any]:
    """Build a complete ToolCall for history replay (not streaming)."""
    # Format args + result as displayable content
    content_blocks = []
    if args:
        args_text = json.dumps(args, indent=2)
        content_blocks.append({
            "type": "content",
            "content": {"type": "text", "text": f"**Arguments:**\n```json\n{args_text}\n```"}
        })
    if result:
        result_text = json.dumps(result, indent=2) if not isinstance(result, str) else result
        status_label = "✓ Success" if status == "completed" else "✗ Failed"
        content_blocks.append({
            "type": "content",
            "content": {"type": "text", "text": f"**{status_label}:**\n```json\n{result_text}\n```"}
        })
    
    return session_update(session_id, {
        "sessionUpdate": "tool_call",
        "toolCallId": tool_call_id,
        "title": tool_name,
        "status": status,
        "rawInput": args,
        "rawOutput": {"result": result} if result else None,
        "content": content_blocks if content_blocks else None,
    })


def tool_call(
    session_id: str, tool_call_id: str, tool_name: str, args: dict[str, Any] | str | None
) -> dict[str, Any]:
    """Build a ToolCall update for tool invocation start."""
    # pydantic-ai sends args as JSON string, not dict
    if isinstance(args, dict):
        raw_input = args
    elif isinstance(args, str):
        try:
            raw_input = json.loads(args)
        except (json.JSONDecodeError, ValueError):
            raw_input = {"raw": args}
    else:
        raw_input = {}
    # Format args as displayable content for Toad
    args_text = json.dumps(raw_input, indent=2) if raw_input else ""
    content = [{"type": "content", "content": {"type": "text", "text": f"**Arguments:**\n```json\n{args_text}\n```"}}] if args_text else []
    
    return session_update(session_id, {
        "sessionUpdate": "tool_call",
        "toolCallId": tool_call_id,
        "title": tool_name,
        "status": "in_progress",
        "rawInput": raw_input,
        "content": content,
    })


def tool_call_update(
    session_id: str, tool_call_id: str, result: Any, status: str = "completed",
    args: dict[str, Any] | None = None
) -> dict[str, Any]:
    """Build a ToolCallUpdate for tool result."""
    raw_output = {"result": result} if not isinstance(result, dict) else result
    
    # Format combined args + result as displayable content for Toad
    content_blocks = []
    
    # Include args if present
    if args:
        args_text = json.dumps(args, indent=2)
        content_blocks.append({
            "type": "content", 
            "content": {"type": "text", "text": f"**Arguments:**\n```json\n{args_text}\n```"}
        })
    
    # Include result
    if raw_output:
        result_text = json.dumps(raw_output, indent=2)
        status_label = "✓ Success" if status == "completed" else "✗ Failed"
        content_blocks.append({
            "type": "content",
            "content": {"type": "text", "text": f"**{status_label}:**\n```json\n{result_text}\n```"}
        })
    
    return session_update(session_id, {
        "sessionUpdate": "tool_call_update",
        "toolCallId": tool_call_id,
        "status": status,
        "rawOutput": raw_output,
        "content": content_blocks if content_blocks else None,
    })


# =============================================================================
# Bridge State
# =============================================================================

@dataclass
class BridgeState:
    """Global bridge state."""
    server_url: str = "http://localhost:8000"
    agent_id: str | None = None  # Set via config or session/new

    # Accumulated usage for the current turn
    usage: dict[str, int] | None = None

    # Watermark for history polling — timestamp of the last message seen from any source.
    # Updated by replay_history, _update_watermark, and poll_for_new_messages.
    last_message_ts: datetime | None = None

    # True while a toad-initiated prompt stream is active — polling skips during this window
    # to avoid racing with the live stream and double-sending notifications.
    stream_active: bool = False

    # Handle for the background polling task — kept so it can be cancelled on session restart.
    polling_task: asyncio.Task | None = field(default=None, repr=False)


# =============================================================================
# SSE Event Processing
# =============================================================================

@dataclass
class StreamState:
    """Tracks state within a single SSE stream."""
    in_thinking: bool = False
    # Track args per tool_call_id so we can include them in the final update
    tool_args: dict[str, dict[str, Any]] = None
    
    def __post_init__(self):
        if self.tool_args is None:
            self.tool_args = {}


async def process_sse_stream(
    state: BridgeState, response: httpx.Response, session_id: str
) -> dict[str, Any]:
    """
    Process SSE events from Agent Home, forward as ACP notifications.
    
    Returns usage dict for the turn-complete response.
    """
    stream_state = StreamState()
    current_event_type: str | None = None
    current_data_lines: list[str] = []
    usage: dict[str, int] = {"input": 0, "output": 0, "cacheCreation": 0, "cacheRead": 0}
    
    async for line in response.aiter_lines():
        line = line.strip()
        
        if line.startswith("event:"):
            current_event_type = line[6:].strip()
        elif line.startswith("data:"):
            current_data_lines.append(line[5:].strip())
        elif line == "" and current_event_type:
            # End of event — process it
            data_str = "\n".join(current_data_lines)
            await process_sse_event(
                state, stream_state, session_id, current_event_type, data_str, usage
            )
            current_event_type = None
            current_data_lines = []
    
    return usage


async def process_sse_event(
    state: BridgeState,
    stream_state: StreamState,
    session_id: str,
    event_type: str,
    data_str: str,
    usage: dict[str, int],
) -> None:
    """Process a single SSE event and emit ACP notification if needed."""
    try:
        data = json.loads(data_str) if data_str else {}
    except json.JSONDecodeError:
        logger.warning(f"Failed to parse SSE data: {data_str[:100]}")
        return
    
    if event_type == "PartStartEvent":
        part = data.get("part", {})
        part_kind = part.get("part_kind")
        content = part.get("content", "")
        
        if part_kind == "thinking":
            stream_state.in_thinking = True
            if content:
                send(agent_thought_chunk(session_id, content))
        elif part_kind == "text":
            stream_state.in_thinking = False
            if content:
                send(agent_message_chunk(session_id, content))
    
    elif event_type == "PartDeltaEvent":
        delta = data.get("delta", {})
        content = delta.get("content_delta", "")
        if content:
            if stream_state.in_thinking:
                send(agent_thought_chunk(session_id, content))
            else:
                send(agent_message_chunk(session_id, content))
    
    elif event_type == "FunctionToolCallEvent":
        part = data.get("part", {})
        tool_call_id = part.get("tool_call_id", "")
        args = part.get("args")
        
        # Store args for later use in tool_call_update
        if args:
            if isinstance(args, str):
                try:
                    stream_state.tool_args[tool_call_id] = json.loads(args)
                except (json.JSONDecodeError, ValueError):
                    stream_state.tool_args[tool_call_id] = {"raw": args}
            else:
                stream_state.tool_args[tool_call_id] = args if isinstance(args, dict) else {}
        
        send(tool_call(session_id, tool_call_id, part.get("tool_name", "unknown"), args))
    
    elif event_type == "FunctionToolResultEvent":
        part = data.get("part", {})
        tool_call_id = part.get("tool_call_id", "")
        # Map outcome to status
        # Note: RetryPromptPart (ModelRetry failures) has no outcome field, only part_kind
        part_kind = part.get("part_kind", "tool-return")
        outcome = part.get("outcome", "success")
        status = "failed" if (part_kind == "retry-prompt" or outcome in ("failed", "denied")) else "completed"
        
        # Get saved args for this tool call
        saved_args = stream_state.tool_args.pop(tool_call_id, {})
        
        send(tool_call_update(
            session_id,
            tool_call_id,
            part.get("content", ""),
            status,
            saved_args,  # Pass args to include in combined content
        ))
    
    elif event_type == "AgentRunResultEvent":
        # Turn complete — extract usage if present
        if "usage" in data:
            u = data["usage"]
            usage["input"] = u.get("request_tokens", 0)
            usage["output"] = u.get("response_tokens", 0)
            # pydantic-ai doesn't expose cache stats directly
    
    elif event_type == "Error":
        # Emit as a message chunk so user sees it
        error_msg = data.get("message", "Unknown error")
        send(agent_message_chunk(session_id, f"\n[Error: {error_msg}]"))


# =============================================================================
# Request Handlers
# =============================================================================

async def handle_initialize(state: BridgeState, msg: dict[str, Any]) -> None:
    """Handle initialize request."""
    send(response(msg["id"], {
        "agentCapabilities": {
            "loadSession": False,
            "promptCapabilities": {
                "audio": False,
                "embeddedContent": False,
                "image": False,
            },
        },
        "authMethods": [],
        "protocolVersion": 1,
    }))


def _replay_message_items(session_id: str, items: list[dict[str, Any]]) -> datetime | None:
    """Convert a list of MessageItem dicts to session/update notifications.

    Each item has shape: {id, type, content, timestamp} where content is a
    serialized pydantic-ai ModelMessage JSON string.

    Returns the timestamp of the latest item, or None if items is empty.
    Used by both replay_history (initial load) and poll_for_new_messages (incremental).
    """
    tool_call_args: dict[str, dict[str, Any]] = {}
    latest_ts: datetime | None = None

    for item in items:
        ts_str = item.get("timestamp")
        if ts_str:
            latest_ts = datetime.fromisoformat(ts_str)

        # Unwrap the nested pydantic-ai message JSON
        try:
            msg = json.loads(item["content"])
        except (KeyError, json.JSONDecodeError):
            logger.warning("Skipping malformed message item: %r", item.get("id"))
            continue

        kind = msg.get("kind")
        parts = msg.get("parts", [])

        if kind == "request":
            for part in parts:
                part_kind = part.get("part_kind")
                if part_kind == "user-prompt":
                    content = part.get("content", "")
                    if content:
                        send(user_message_chunk(session_id, content))
                elif part_kind in ("tool-return", "retry-prompt"):
                    tool_call_id = part.get("tool_call_id", "")
                    content = part.get("content", "")
                    outcome = part.get("outcome", "success")
                    status = "failed" if (part_kind == "retry-prompt" or outcome in ("failed", "denied")) else "completed"
                    args = tool_call_args.get(tool_call_id)
                    send(tool_call_update(session_id, tool_call_id, content, status, args=args))

        elif kind == "response":
            for part in parts:
                part_kind = part.get("part_kind")
                if part_kind == "thinking":
                    content = part.get("content", "")
                    if content:
                        send(agent_thought_chunk(session_id, content))
                elif part_kind == "tool-call":
                    tool_call_id = part.get("tool_call_id", "")
                    tool_name = part.get("tool_name", "unknown")
                    args = part.get("args", {})
                    if isinstance(args, str):
                        try:
                            args = json.loads(args)
                        except (json.JSONDecodeError, ValueError):
                            args = {"raw": args}
                    tool_call_args[tool_call_id] = args
                    send(tool_call(session_id, tool_call_id, tool_name, args))
                elif part_kind == "text":
                    content = part.get("content", "")
                    if content:
                        send(agent_message_chunk(session_id, content))

    return latest_ts


async def replay_history(state: BridgeState, session_id: str, client: httpx.AsyncClient) -> None:
    """Fetch agent's in-context messages, replay as session/update notifications, update watermark."""
    try:
        resp = await client.get(f"{state.server_url}/agents/{session_id}/messages")
        resp.raise_for_status()
        items = resp.json().get("messages", [])
    except Exception as e:
        # History replay is best-effort — don't fail the session
        print(f"Warning: Failed to fetch history: {e}", file=sys.stderr)
        return

    latest_ts = _replay_message_items(session_id, items)
    if latest_ts is not None:
        state.last_message_ts = latest_ts


async def _update_watermark(state: BridgeState, session_id: str, client: httpx.AsyncClient) -> None:
    """Silently advance the watermark after a toad-initiated turn.

    Fetches messages after the current watermark and updates last_message_ts
    WITHOUT sending any notifications — the live stream already delivered those.
    Prevents the background poller from re-sending the same turn as notifications.
    """
    if state.last_message_ts is None:
        return
    try:
        resp = await client.get(
            f"{state.server_url}/agents/{session_id}/messages",
            params={"after": state.last_message_ts.isoformat()},
        )
        resp.raise_for_status()
        items = resp.json().get("messages", [])
        if items:
            latest_ts_str = items[-1].get("timestamp")
            if latest_ts_str:
                state.last_message_ts = datetime.fromisoformat(latest_ts_str)
    except Exception as e:
        logger.warning("Watermark update failed: %s", e)


async def poll_for_new_messages(
    state: BridgeState, session_id: str, client: httpx.AsyncClient
) -> None:
    """Background task: poll for new messages from non-toad sources every 2 seconds.

    Skips polling while a toad-initiated stream is active (stream_active=True) to
    avoid racing with the live SSE stream. Also serves as a heartbeat — if the
    server is unreachable, warnings are logged but the loop continues.
    """
    while True:
        await asyncio.sleep(2)
        if state.stream_active or state.last_message_ts is None:
            continue
        try:
            resp = await client.get(
                f"{state.server_url}/agents/{session_id}/messages",
                params={"after": state.last_message_ts.isoformat()},
            )
            resp.raise_for_status()
            items = resp.json().get("messages", [])
            if items:
                latest_ts = _replay_message_items(session_id, items)
                if latest_ts is not None:
                    state.last_message_ts = latest_ts
        except asyncio.CancelledError:
            raise  # Let cancellation propagate cleanly
        except Exception as e:
            logger.warning("Poll failed: %s", e)


async def handle_session_new(
    state: BridgeState, msg: dict[str, Any], client: httpx.AsyncClient
) -> None:
    """Handle session/new request."""
    # The agent_id should be configured — in Phase 1, we require it via env or config
    if not state.agent_id:
        send(error_response(msg["id"], -32000, "No agent_id configured"))
        return
    
    # Optionally verify agent exists
    try:
        resp = await client.get(f"{state.server_url}/agents/{state.agent_id}")
        if resp.status_code == 404:
            send(error_response(msg["id"], -32000, f"Agent not found: {state.agent_id}"))
            return
        resp.raise_for_status()
    except httpx.RequestError as e:
        send(error_response(msg["id"], -32000, f"Failed to verify agent: {e}"))
        return
    
    session_id = state.agent_id

    # Send response first so client knows session is ready
    send(response(msg["id"], {"sessionId": session_id}))

    # Replay history as session/update notifications and set initial watermark
    await replay_history(state, session_id, client)

    # Cancel any existing polling task (e.g. reconnect), then start fresh
    if state.polling_task is not None:
        state.polling_task.cancel()
    state.polling_task = asyncio.create_task(
        poll_for_new_messages(state, session_id, client)
    )


async def handle_session_prompt(
    state: BridgeState, msg: dict[str, Any], client: httpx.AsyncClient
) -> None:
    """Handle session/prompt request — the main turn handler."""
    params = msg.get("params", {})
    session_id = params.get("sessionId")
    prompt_blocks = params.get("prompt", [])
    
    if not session_id:
        send(error_response(msg["id"], -32602, "Missing sessionId"))
        return
    
    # Extract text from prompt blocks
    # Phase 1: only handle TextContent
    text_parts = []
    for block in prompt_blocks:
        if block.get("type") == "text":
            text_parts.append(block.get("text", ""))
    
    user_message = "\n".join(text_parts)
    
    if not user_message:
        send(error_response(msg["id"], -32602, "Empty prompt"))
        return
    
    # Pause background polling for the duration of this stream — we'll update the
    # watermark silently afterward to prevent the poller from re-sending these events.
    state.stream_active = True
    try:
        async with client.stream(
            "POST",
            f"{state.server_url}/agents/{session_id}/messages",
            json={"message": user_message},
            timeout=300.0,
        ) as http_response:
            if http_response.status_code != 200:
                body = await http_response.aread()
                send(error_response(
                    msg["id"], -32000, f"Agent Home error: {http_response.status_code} {body.decode()}"
                ))
                return

            # Stream events as ACP notifications
            usage = await process_sse_stream(state, http_response, session_id)

        # Advance watermark past this turn so the poller doesn't re-send it
        await _update_watermark(state, session_id, client)

    except httpx.RequestError as e:
        send(error_response(msg["id"], -32000, f"Request failed: {e}"))
        return
    finally:
        state.stream_active = False

    # Turn complete — send response
    send(response(msg["id"], {
        "sessionId": session_id,
        "stopReason": "end_turn",
        "usage": usage,
    }))


async def handle_session_cancel(state: BridgeState, msg: dict[str, Any]) -> None:
    """Handle session/cancel notification — Phase 1: ignore."""
    # This is a notification (no id), so we just silently ignore it
    logger.debug("Ignoring session/cancel (Phase 1)")


# =============================================================================
# Main Dispatch Loop
# =============================================================================

async def dispatch(state: BridgeState, msg: dict[str, Any], client: httpx.AsyncClient) -> None:
    """Dispatch a JSON-RPC message to the appropriate handler."""
    method = msg.get("method")
    
    if method == "initialize":
        await handle_initialize(state, msg)
    elif method == "session/new":
        await handle_session_new(state, msg, client)
    elif method == "session/prompt":
        await handle_session_prompt(state, msg, client)
    elif method == "session/cancel":
        await handle_session_cancel(state, msg)
    else:
        # Unknown method
        if "id" in msg:
            send(error_response(msg["id"], -32601, f"Method not found: {method}"))
        # If no id, it's a notification — silently ignore unknown notifications


async def main(agent_id: str | None = None, server_url: str = "http://localhost:8000") -> None:
    """Main entry point for the ACP bridge."""
    state = BridgeState(server_url=server_url, agent_id=agent_id)
    
    # Set up async stdin reader
    loop = asyncio.get_event_loop()
    reader = asyncio.StreamReader()
    protocol = asyncio.StreamReaderProtocol(reader)
    await loop.connect_read_pipe(lambda: protocol, sys.stdin)
    
    async with httpx.AsyncClient() as client:
        while True:
            line = await reader.readline()
            if not line:
                break  # EOF
            
            try:
                msg = json.loads(line.decode())
            except json.JSONDecodeError as e:
                logger.warning(f"Invalid JSON: {e}")
                continue
            
            await dispatch(state, msg, client)


if __name__ == "__main__":
    import os
    
    logging.basicConfig(
        level=logging.DEBUG if os.environ.get("ACP_DEBUG") else logging.WARNING,
        format="%(levelname)s: %(message)s",
        stream=sys.stderr,  # Logs to stderr, not stdout (which is for JSON-RPC)
    )
    
    agent_id = os.environ.get("AGENT_HOME_AGENT_ID")
    server_url = os.environ.get("AGENT_HOME_SERVER_URL", "http://localhost:8000")
    
    asyncio.run(main(agent_id=agent_id, server_url=server_url))
