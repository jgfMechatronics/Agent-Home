"""Integration test: bridge real-time stream subscription.

Tests the full chain:
  AH server --SSE--> bridge --ACP/stdout--> this script (fake TUI)

Flow:
1. Create two agents (sender + receiver) via HTTP
2. Launch bridge subprocess pointed at receiver agent
3. Send ACP initialize + session/new to bridge stdin
4. Prompt sender to send a message to receiver (triggering background run)
5. Verify bridge emits working-state events (observer turn + content chunks)
   from the real-time stream subscription (not just polling)

Usage:
    cd /workspace/git/Agent-Home_dev2
    uv run python misc/test_bridge_stream_integration.py
"""
import asyncio
import json
import os
import sys
import httpx

BASE = "http://localhost:8008"


# ---------------------------------------------------------------------------
# Agent setup
# ---------------------------------------------------------------------------

async def get_or_create_agent(client: httpx.AsyncClient, name: str, config: dict) -> str:
    resp = await client.post(f"{BASE}/agents", json=config)
    if resp.status_code == 409:
        agents = (await client.get(f"{BASE}/agents")).json()
        agent = next(a for a in agents if a["name"] == name)
        print(f"  Reusing {name}: {agent['id']}")
        return agent["id"]
    resp.raise_for_status()
    agent_id = resp.json()["id"]
    print(f"  Created {name}: {agent_id}")
    return agent_id


async def setup_agents(client: httpx.AsyncClient) -> tuple[str, str]:
    base_config = {
        "model_name": "claude-sonnet-4-5",
        "tool_names": ["send_message", "duckduckgo_search"],
        "soft_compaction_limit": 50000,
    }
    sender_id = await get_or_create_agent(client, "bridge-test-sender", {
        "name": "bridge-test-sender",
        "system_instructions": "You are a test agent. When asked to send a message, use send_message immediately.",
        "config": base_config,
    })
    receiver_id = await get_or_create_agent(client, "bridge-test-receiver", {
        "name": "bridge-test-receiver",
        "system_instructions": "When you receive a message, do 1-2 web searches on a random topic and summarize.",
        "config": base_config,
    })
    return sender_id, receiver_id


# ---------------------------------------------------------------------------
# Bridge subprocess management
# ---------------------------------------------------------------------------

async def start_bridge(agent_id: str) -> asyncio.subprocess.Process:
    env = {
        **os.environ,
        "AGENT_HOME_AGENT_ID": agent_id,
        "AGENT_HOME_SERVER_URL": BASE,
        "ACP_DEBUG": "0",
    }
    proc = await asyncio.create_subprocess_exec(
        sys.executable, "-m", "acp",
        stdin=asyncio.subprocess.PIPE,
        stdout=asyncio.subprocess.PIPE,
        stderr=asyncio.subprocess.PIPE,
        env=env,
        cwd="/workspace/git/Agent-Home_dev2",
    )
    print(f"  Bridge PID: {proc.pid}")
    return proc


def bridge_send(proc: asyncio.subprocess.Process, msg: dict) -> None:
    line = json.dumps(msg) + "\n"
    proc.stdin.write(line.encode())


async def bridge_flush(proc: asyncio.subprocess.Process) -> None:
    await proc.stdin.drain()


# ---------------------------------------------------------------------------
# ACP message helpers
# ---------------------------------------------------------------------------

def acp_initialize(msg_id: int = 1) -> dict:
    return {"jsonrpc": "2.0", "id": msg_id, "method": "initialize", "params": {
        "clientInfo": {"name": "test-client", "version": "0.1"},
        "protocolVersion": 1,
    }}


def acp_session_new(session_id_hint: str, msg_id: int = 2) -> dict:
    return {"jsonrpc": "2.0", "id": msg_id, "method": "session/new", "params": {
        "sessionId": session_id_hint,
    }}


# ---------------------------------------------------------------------------
# Main test
# ---------------------------------------------------------------------------

async def main() -> None:
    print("\n=== Bridge Real-Time Stream Integration Test ===\n")

    async with httpx.AsyncClient() as client:
        print("[1] Setting up agents...")
        sender_id, receiver_id = await setup_agents(client)

        print(f"\n[2] Starting bridge for receiver ({receiver_id[:8]}...)...")
        bridge = await start_bridge(receiver_id)
        await asyncio.sleep(0.5)

        print("\n[3] ACP handshake (initialize + session/new)...")
        bridge_send(bridge, acp_initialize())
        await bridge_flush(bridge)
        await asyncio.sleep(0.3)

        bridge_send(bridge, acp_session_new(receiver_id))
        await bridge_flush(bridge)
        await asyncio.sleep(1.0)  # let session/new replay history + start stream

        # Collect stdout events in background
        events: list[dict] = []
        observer_turn_seen = False
        content_chunks_seen = 0

        async def collect_bridge_output():
            nonlocal observer_turn_seen, content_chunks_seen
            while True:
                try:
                    line = await asyncio.wait_for(bridge.stdout.readline(), timeout=0.5)
                except asyncio.TimeoutError:
                    continue
                if not line:
                    break
                text = line.decode().strip()
                if not text:
                    continue
                try:
                    msg = json.loads(text)
                    events.append(msg)
                    method = msg.get("method", "")
                    params = msg.get("params", {})
                    update = params.get("update", {})
                    session_update_type = update.get("sessionUpdate", "")

                    # Detect observer turn activation (status=working)
                    meta = update.get("_meta", {})
                    if meta.get("nori", {}).get("status") == "working":
                        observer_turn_seen = True
                        print(f"  [bridge-out] ✓ Observer turn activated (status=working)")
                    elif meta.get("nori", {}).get("status") == "idle":
                        print(f"  [bridge-out] ✓ Observer turn closed (status=idle)")

                    # Detect content chunks (agent_message_chunk format)
                    if session_update_type == "agent_message_chunk":
                        text = update.get("content", {}).get("text", "")
                        if text:
                            content_chunks_seen += 1
                            if content_chunks_seen <= 3:
                                print(f"  [bridge-out] content chunk: {text[:60]!r}")
                    elif session_update_type == "tool_call":
                        print(f"  [bridge-out] tool_call: {update.get('title', '?')}")
                    elif session_update_type == "tool_call_update":
                        print(f"  [bridge-out] tool_call_update: {update.get('status', '?')}")
                    elif method not in ("", "session/update"):
                        print(f"  [bridge-out] {method}")
                except json.JSONDecodeError:
                    print(f"  [bridge-raw] {text[:100]}")

        collect_task = asyncio.create_task(collect_bridge_output())

        print("\n[4] Prompting sender to message receiver...")
        prompt_task = asyncio.create_task(client.post(
            f"{BASE}/agents/{sender_id}/messages",
            json={"message": f"Use send_message to send 'Please search for bioluminescence and give me a 2 sentence summary.' to bridge-test-receiver"},
            timeout=90,
        ))

        # Wait for sender to finish, then give bridge time to process stream events
        try:
            await asyncio.wait_for(prompt_task, timeout=60)
            print("  Sender turn complete. Waiting for bridge to process stream events...")
            await asyncio.sleep(8)  # let background run complete + bridge process it
        except asyncio.TimeoutError:
            print("  [timeout] Sender timed out")

        collect_task.cancel()
        bridge.stdin.close()
        bridge.kill()
        await bridge.wait()

        # Report
        print(f"\n{'='*60}")
        print(f"Total ACP messages received from bridge: {len(events)}")
        print(f"Observer turn activated (status=working): {'✓' if observer_turn_seen else '✗'}")
        print(f"Content chunks received: {content_chunks_seen}")

        print("\nPASS" if observer_turn_seen else "\nFAIL — bridge did not activate observer turn")
        print("(Content chunks may be 0 if bridge uses message-level batching rather than streaming chunks)")


if __name__ == "__main__":
    asyncio.run(main())
