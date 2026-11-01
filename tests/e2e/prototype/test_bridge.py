"""E2E tests for the ACP bridge (prototype/acp/bridge.py).

Tests the full stack: Nori-style ACP client → bridge → Agent Home server → LLM.

Run with: pytest tests/e2e -m e2e

The bridge communicates via newline-delimited JSON-RPC on stdin/stdout.
Each test spawns a fresh bridge subprocess and drives it via the ACP protocol.
"""
import asyncio
import json
import sys
from contextlib import asynccontextmanager
from typing import AsyncIterator

import httpx
import pytest
import pytest_asyncio

from tests.e2e.conftest import TEST_AGENT_INSTRUCTIONS, PROJECT_ROOT, SERVER_URL, REQUEST_TIMEOUT

BRIDGE_TIMEOUT = REQUEST_TIMEOUT  # seconds to wait for bridge responses

# Deterministic tool-call probe: a memory block the tool_agent agent can read.
TOOL_PROBE_BLOCK = "e2e_probe"
TOOL_PROBE_CONTENT = "alpha\nbeta\ngamma\n"


# =============================================================================
# Bridge subprocess helpers
# =============================================================================

class BridgeProcess:
    """Wrapper around a bridge subprocess for sending/receiving ACP messages."""

    def __init__(self, proc: asyncio.subprocess.Process, agent_id: str):
        self._proc = proc
        self._agent_id = agent_id
        self._id_counter = 0

    def _next_id(self) -> int:
        self._id_counter += 1
        return self._id_counter

    async def send(self, method: str, params: dict) -> int:
        """Send a JSON-RPC request and return the request id."""
        msg_id = self._next_id()
        msg = {"jsonrpc": "2.0", "id": msg_id, "method": method, "params": params}
        line = json.dumps(msg).encode() + b"\n"
        self._proc.stdin.write(line)
        await self._proc.stdin.drain()
        return msg_id

    async def read_until(self, stop_condition, timeout: float = BRIDGE_TIMEOUT) -> list[dict]:
        """Read newline-delimited JSON messages until stop_condition(msg) returns True.

        Returns all messages read including the terminal one.
        Raises asyncio.TimeoutError if timeout is reached.
        """
        messages = []
        async def _read():
            while True:
                line = await self._proc.stdout.readline()
                if not line:
                    break
                msg = json.loads(line.decode())
                messages.append(msg)
                if stop_condition(msg):
                    break
        await asyncio.wait_for(_read(), timeout=timeout)
        return messages

    async def close(self):
        """Terminate the bridge process."""
        try:
            self._proc.stdin.close()
            self._proc.terminate()
            await asyncio.wait_for(self._proc.wait(), timeout=5.0)
        except Exception:
            pass


def _is_response(msg: dict, msg_id: int) -> bool:
    """True if msg is a JSON-RPC response to the given request id."""
    return msg.get("id") == msg_id and "result" in msg


def _is_prompt_response(msg: dict, prompt_id: int) -> bool:
    """True if msg is the JSON-RPC response to a session/prompt request.

    For user-initiated runs, the prompt response is the terminal signal —
    the bridge sends it after the full SSE stream from AH has been consumed.
    """
    return msg.get("id") == prompt_id and ("result" in msg or "error" in msg)


# =============================================================================
# Fixtures
# =============================================================================

@pytest_asyncio.fixture
async def e2e_agent(live_server: str) -> AsyncIterator[str]:
    """Get or create a test agent for bridge tests."""
    async with httpx.AsyncClient(timeout=30.0) as client:
        # Reuse existing agent if present
        resp = await client.get(f"{live_server}/agents")
        resp.raise_for_status()
        for agent in resp.json():
            if agent.get("name") == "e2e-bridge-test":
                yield agent["id"]
                return

        # Create new agent
        resp = await client.post(
            f"{live_server}/agents",
            json={
                "name": "e2e-bridge-test",
                "system_instructions": TEST_AGENT_INSTRUCTIONS,
                "config": {
                    "model_name": "anthropic:claude-haiku-4-5-20251001",
                    "tool_names": [],
                    "soft_compaction_limit": 100000,
                },
            },
        )
        resp.raise_for_status()
        yield resp.json()["id"]


@pytest_asyncio.fixture
async def tool_agent(live_server: str) -> AsyncIterator[str]:
    """Agent with one built-in tool, plus a memory block for that tool to edit.

    The bridge e2e agent has no tools, so this fixture provides the smallest
    deterministic setup that produces a real FunctionToolCall/Result pair:
    memory_insert against a known block (no network, no external state).
    """
    async with httpx.AsyncClient(timeout=30.0) as client:
        resp = await client.get(f"{live_server}/agents")
        resp.raise_for_status()
        agent_id = next(
            (agent["id"] for agent in resp.json() if agent.get("name") == "e2e-bridge-tool-test"),
            None,
        )
        if agent_id is None:
            resp = await client.post(
                f"{live_server}/agents",
                json={
                    "name": "e2e-bridge-tool-test",
                    "system_instructions": (
                        "You are a test assistant running as part of an automated e2e test. "
                        "When a message asks you to call a tool, call it; never answer from memory. "
                        "Your participation is not mandatory. If at any point you wish to not "
                        "participate, simply reply 'REFUSE' and nothing else."
                    ),
                    "config": {
                        "model_name": "anthropic:claude-haiku-4-5-20251001",
                        "tool_names": ["memory_insert"],
                        "soft_compaction_limit": 100000,
                    },
                },
            )
            resp.raise_for_status()
            agent_id = resp.json()["id"]

        # Seed the probe block (idempotent across repeated fixture use).
        resp = await client.get(f"{live_server}/agents/{agent_id}/memory/blocks")
        resp.raise_for_status()
        labels = {block.get("label") for block in resp.json()["blocks"]}
        if TOOL_PROBE_BLOCK not in labels:
            resp = await client.post(
                f"{live_server}/agents/{agent_id}/memory/blocks",
                json={"label": TOOL_PROBE_BLOCK, "content": TOOL_PROBE_CONTENT},
            )
            resp.raise_for_status()

        yield agent_id


@asynccontextmanager
async def _spawn_bridge(agent_id: str) -> AsyncIterator[BridgeProcess]:
    """Spawn a bridge subprocess connected to the given agent."""
    proc = await asyncio.create_subprocess_exec(
        sys.executable, "-m", "prototype.acp",
        agent_id, "--server-url", SERVER_URL,
        stdin=asyncio.subprocess.PIPE,
        stdout=asyncio.subprocess.PIPE,
        stderr=asyncio.subprocess.DEVNULL,
        cwd=PROJECT_ROOT,
    )
    bp = BridgeProcess(proc, agent_id)
    try:
        yield bp
    finally:
        await bp.close()


@pytest_asyncio.fixture
async def bridge(e2e_agent: str) -> AsyncIterator[BridgeProcess]:
    """Spawn a bridge subprocess connected to the e2e test agent."""
    async with _spawn_bridge(e2e_agent) as bp:
        yield bp


@pytest_asyncio.fixture
async def tool_bridge(tool_agent: str) -> AsyncIterator[BridgeProcess]:
    """Spawn a bridge subprocess connected to the tool-call test agent."""
    async with _spawn_bridge(tool_agent) as bp:
        yield bp


# =============================================================================
# Tests
# =============================================================================

async def test_initialize_returns_capabilities(bridge: BridgeProcess, e2e_agent: str):
    """Bridge responds to initialize with correct protocol version and capabilities."""
    msg_id = await bridge.send("initialize", {})
    msgs = await bridge.read_until(lambda m: _is_response(m, msg_id))

    response = msgs[-1]
    result = response["result"]

    assert result["protocolVersion"] == 1
    assert result["agentCapabilities"]["loadSession"] is True
    # Should advertise the agent_id for Nori auto-load
    assert result["_meta"]["nori"]["remoteControl"]["activeSessionId"] == e2e_agent


async def test_session_new_returns_session_id(bridge: BridgeProcess, e2e_agent: str):
    """Bridge responds to session/new with the agent's session id."""
    await bridge.send("initialize", {})
    await bridge.read_until(lambda m: "result" in m and m.get("id") == 1)

    msg_id = await bridge.send("session/new", {})
    msgs = await bridge.read_until(lambda m: _is_response(m, msg_id))

    response = msgs[-1]
    assert response["result"]["sessionId"] == e2e_agent


async def test_session_prompt_delivers_agent_response(bridge: BridgeProcess, e2e_agent: str):
    """session/prompt triggers agent_message_chunk notifications and ends with status=idle."""
    # Handshake
    init_id = await bridge.send("initialize", {})
    await bridge.read_until(lambda m: _is_response(m, init_id))

    new_id = await bridge.send("session/new", {})
    await bridge.read_until(lambda m: _is_response(m, new_id))

    # Send prompt
    prompt_id = await bridge.send("session/prompt", {
        "sessionId": e2e_agent,
        "prompt": [{"type": "text", "text": "Say 'hello' and nothing else."}],
    })

    # Collect until bridge sends the prompt response (signals turn complete)
    msgs = await bridge.read_until(lambda m: _is_prompt_response(m, prompt_id))

    # Should have received at least one agent_message_chunk
    chunk_updates = [
        m for m in msgs
        if m.get("method") == "session/update"
        and m.get("params", {}).get("update", {}).get("sessionUpdate") == "agent_message_chunk"
    ]
    assert len(chunk_updates) > 0, "Expected at least one agent_message_chunk notification"

    # Assert the exact reply, not just that chunks arrived: the bridge forwards
    # server-side failures as "[Error: ...]" chunks, so a non-empty list alone
    # can hide an auth or model failure.
    chunk_texts = [
        m["params"]["update"]["content"]["text"] for m in chunk_updates
    ]
    assert "".join(chunk_texts) == "hello"

    # prompt response should have been received
    prompt_responses = [m for m in msgs if _is_response(m, prompt_id)]
    assert len(prompt_responses) == 1


def _update_payloads(msgs: list[dict], session_update: str) -> list[dict]:
    """Filter protocol messages down to one session/update kind."""
    return [
        m["params"]["update"] for m in msgs
        if m.get("method") == "session/update"
        and m.get("params", {}).get("update", {}).get("sessionUpdate") == session_update
    ]


async def test_tool_call_flows_through_with_structured_fields(
    tool_bridge: BridgeProcess, tool_agent: str
):
    """A real tool call arrives as tool_call + tool_call_update with usable fields.

    Pins the Nori-facing contract end to end: kind is set, the result rides in
    rawOutput as plain text with real newlines (not JSON-escaped), and generic
    tools keep a text content block so their output is still visible.
    """
    init_id = await tool_bridge.send("initialize", {})
    await tool_bridge.read_until(lambda m: _is_response(m, init_id))

    new_id = await tool_bridge.send("session/new", {})
    await tool_bridge.read_until(lambda m: _is_response(m, new_id))

    prompt_id = await tool_bridge.send("session/prompt", {
        "sessionId": tool_agent,
        "prompt": [{
            "type": "text",
            "text": (
                f"Call the memory_insert tool on block '{TOOL_PROBE_BLOCK}' with "
                "content 'delta' and after '<end>'. Invoke the tool rather than "
                "answering directly."
            ),
        }],
    })
    msgs = await tool_bridge.read_until(lambda m: _is_prompt_response(m, prompt_id))

    calls = _update_payloads(msgs, "tool_call")
    assert calls, "Expected a tool_call notification for the memory_insert call"
    call = calls[0]
    assert call["title"] == "memory_insert"
    assert call["kind"] == "other"
    assert call["status"] == "in_progress"
    assert call["rawInput"]["label"] == TOOL_PROBE_BLOCK

    results = _update_payloads(msgs, "tool_call_update")
    assert results, "Expected a tool_call_update notification for the tool result"
    result = results[0]
    assert result["status"] == "completed"
    assert result["kind"] == "other"

    # Real newlines, not literal "\n", end to end through JSON-RPC.
    text = result["content"][0]["content"]["text"]
    assert "alpha" in text
    assert "\n" in text
    assert "\\n" not in text
