"""End-to-end test for the real-time agent stream endpoint.

Creates two agents, subscribes to the background agent's stream, then has
the foreground agent send it a message. Confirms RunStarted/content/RunCompleted
events arrive on the stream in real time.

Usage:
    cd /workspace/git/Agent-Home_dev2
    uv run python misc/test_realtime_stream.py
"""
import asyncio
import json
import httpx

BASE = "http://localhost:8008"

AGENT_A_CONFIG = {
    "name": "stream-test-sender",
    "system_instructions": (
        "You are a test agent. When asked to send a message to another agent, "
        "use the send_message tool to do so immediately."
    ),
    "config": {
        "model_name": "claude-sonnet-4-5",
        "tool_names": ["send_message", "duckduckgo_search"],
        "soft_compaction_limit": 50000,
    },
}

AGENT_B_CONFIG = {
    "name": "stream-test-receiver",
    "system_instructions": (
        "You are a test agent. When you receive a message, do 2-3 web searches "
        "on an interesting topic, then respond with a brief summary."
    ),
    "config": {
        "model_name": "claude-sonnet-4-5",
        "tool_names": ["duckduckgo_search"],
        "soft_compaction_limit": 50000,
    },
}


async def create_agent(client: httpx.AsyncClient, config: dict) -> str:
    resp = await client.post(f"{BASE}/agents", json=config)
    if resp.status_code == 409:
        # Already exists — find by name
        all_agents = (await client.get(f"{BASE}/agents")).json()
        agent = next(a for a in all_agents if a["name"] == config["name"])
        print(f"Reusing {config['name']}: {agent['id']}")
        return agent["id"]
    resp.raise_for_status()
    agent_id = resp.json()["id"]
    print(f"Created {config['name']}: {agent_id}")
    return agent_id


async def subscribe_stream(client: httpx.AsyncClient, agent_id: str, events: list, done: asyncio.Event):
    """Collect SSE events from the stream until done is set."""
    print(f"\n[stream] Subscribing to agent {agent_id[:8]}...")
    async with client.stream("GET", f"{BASE}/agents/{agent_id}/stream", timeout=60) as resp:
        print(f"[stream] Connected (status {resp.status_code})")
        async for line in resp.aiter_lines():
            if done.is_set():
                break
            if line.startswith("event:"):
                event_type = line[len("event:"):].strip()
                events.append({"type": event_type, "data": None})
                print(f"[stream] event: {event_type}")
            elif line.startswith("data:"):
                data_str = line[len("data:"):].strip()
                if events:
                    events[-1]["data"] = json.loads(data_str) if data_str else {}
                    print(f"[stream] data:  {data_str[:120]}")
            if events and events[-1].get("type") == "RunCompleted":
                print("[stream] RunCompleted received — signalling done")
                done.set()


async def send_message_prompt(client: httpx.AsyncClient, agent_id_a: str, agent_b_name: str):
    """Tell agent A to send a message to agent B."""
    prompt = f"Please use the send_message tool to send a message to {agent_b_name!r}. Tell them: 'Hello! Please do some research on bioluminescence and summarize what you find.'"
    print(f"\n[prompt] Sending prompt to agent A...")
    async with client.stream(
        "POST",
        f"{BASE}/agents/{agent_id_a}/messages",
        json={"message": prompt},
        timeout=120,
    ) as resp:
        async for line in resp.aiter_lines():
            if line.startswith("event:"):
                print(f"[agent-a] event: {line[6:].strip()}")
            elif line.startswith("data:") and "AgentRunResultEvent" not in line:
                data = line[5:].strip()[:100]
                if data and data != "{}":
                    print(f"[agent-a] data:  {data}")


async def main():
    async with httpx.AsyncClient() as client:
        # Create agents
        agent_a = await create_agent(client, AGENT_A_CONFIG)
        agent_b = await create_agent(client, AGENT_B_CONFIG)

        events: list = []
        done = asyncio.Event()

        # Subscribe to B's stream BEFORE prompting A
        stream_task = asyncio.create_task(
            subscribe_stream(client, agent_b, events, done)
        )
        await asyncio.sleep(0.5)  # let connection establish

        # Prompt A to send a message to B (triggers background run on B)
        prompt_task = asyncio.create_task(
            send_message_prompt(client, agent_a, "stream-test-receiver")
        )

        # Wait for both, with timeout
        try:
            await asyncio.wait_for(
                asyncio.gather(prompt_task, done.wait()),
                timeout=90,
            )
        except asyncio.TimeoutError:
            print("\n[timeout] Test timed out waiting for RunCompleted")
        finally:
            done.set()
            stream_task.cancel()
            try:
                await stream_task
            except (asyncio.CancelledError, Exception):
                pass

        # Report
        print(f"\n{'='*60}")
        print(f"Events received on B's stream: {len(events)}")
        for e in events:
            print(f"  {e['type']}: {e['data']}")

        has_started = any(e["type"] == "RunStarted" for e in events)
        has_completed = any(e["type"] == "RunCompleted" for e in events)
        has_content = any(e["type"] in ("PartStartEvent", "PartDeltaEvent") for e in events)

        print(f"\nRunStarted:  {'✓' if has_started else '✗'}")
        print(f"Content:     {'✓' if has_content else '✗'}")
        print(f"RunCompleted:{'✓' if has_completed else '✗'}")
        print("\nPASS" if (has_started and has_completed) else "\nFAIL")


if __name__ == "__main__":
    asyncio.run(main())
