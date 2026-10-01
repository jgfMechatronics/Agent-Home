"""E2E tests for the ACP bridge (prototype/acp/bridge.py).

Tests the full stack: Nori-style ACP client → bridge → Agent Home server → LLM.

Run with: pytest tests/e2e -m e2e

The bridge communicates via newline-delimited JSON-RPC on stdin/stdout.
Each test spawns a fresh bridge subprocess and drives it via the ACP protocol.
"""
import asyncio
import json
import sys
from typing import AsyncIterator

import httpx
import pytest
import pytest_asyncio

from tests.e2e.conftest import TEST_AGENT_INSTRUCTIONS, PROJECT_ROOT, SERVER_URL, REQUEST_TIMEOUT

BRIDGE_TIMEOUT = REQUEST_TIMEOUT  # seconds to wait for bridge responses


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
async def bridge(e2e_agent: str) -> AsyncIterator[BridgeProcess]:
    """Spawn a bridge subprocess connected to the e2e test agent."""
    proc = await asyncio.create_subprocess_exec(
        sys.executable, "-m", "prototype.acp",
        e2e_agent, "--server-url", SERVER_URL,
        stdin=asyncio.subprocess.PIPE,
        stdout=asyncio.subprocess.PIPE,
        stderr=asyncio.subprocess.DEVNULL,
        cwd=PROJECT_ROOT,
    )
    bp = BridgeProcess(proc, e2e_agent)
    yield bp
    await bp.close()


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

    # prompt response should have been received
    prompt_responses = [m for m in msgs if _is_response(m, prompt_id)]
    assert len(prompt_responses) == 1
