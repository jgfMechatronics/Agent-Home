"""ACP bridge: translates stdio JSON-RPC ↔ Agent Home HTTP/SSE.

Phase 1: Happy path only. No cancellation, no permissions.
"""

from __future__ import annotations

import asyncio
import json
import logging
import sys
from dataclasses import dataclass
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


def tool_call(
    session_id: str, tool_call_id: str, tool_name: str, args: dict[str, Any] | str | None
) -> dict[str, Any]:
    """Build a ToolCall update for tool invocation start."""
    raw_input = args if isinstance(args, dict) else {}
    return session_update(session_id, {
        "sessionUpdate": "tool_call",
        "toolCallId": tool_call_id,
        "title": tool_name,
        "status": "in_progress",
        "rawInput": raw_input,
    })


def tool_call_update(
    session_id: str, tool_call_id: str, result: Any, status: str = "completed"
) -> dict[str, Any]:
    """Build a ToolCallUpdate for tool result."""
    return session_update(session_id, {
        "sessionUpdate": "tool_call_update",
        "toolCallId": tool_call_id,
        "status": status,
        "rawOutput": {"result": result} if not isinstance(result, dict) else result,
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


# =============================================================================
# SSE Event Processing
# =============================================================================

@dataclass
class StreamState:
    """Tracks state within a single SSE stream."""
    in_thinking: bool = False


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
        send(tool_call(
            session_id,
            part.get("tool_call_id", ""),
            part.get("tool_name", "unknown"),
            part.get("args"),
        ))
    
    elif event_type == "FunctionToolResultEvent":
        part = data.get("part", {})
        # Map outcome to status
        outcome = part.get("outcome", "success")
        status = "error" if outcome in ("failed", "denied") else "completed"
        send(tool_call_update(
            session_id,
            part.get("tool_call_id", ""),
            part.get("content", ""),
            status,
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
    
    # TODO: Replay history as session/update notifications
    # This would fetch GET /agents/{id}/messages?full=false
    # and emit each message as appropriate update notifications
    
    send(response(msg["id"], {"sessionId": session_id}))


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
    
    # POST to Agent Home
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
    
    except httpx.RequestError as e:
        send(error_response(msg["id"], -32000, f"Request failed: {e}"))
        return
    
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
