"""E2E tests for broadcast streaming.

These tests require a live server running on localhost:8008.
Start with: ./start_server.sh

Run with: pytest -m e2e
"""

import asyncio
import json

import httpx
import pytest

SERVER_URL = "http://localhost:8008"


async def subscribe_to_stream(agent_id: str, events: list, ready_event: asyncio.Event):
    """Subscribe to agent's broadcast stream and collect events."""
    async with httpx.AsyncClient(timeout=30.0) as client:
        async with client.stream("GET", f"{SERVER_URL}/agents/{agent_id}/stream") as response:
            ready_event.set()  # Signal that subscription is active
            async for line in response.aiter_lines():
                if line.startswith("data:"):
                    data = json.loads(line[5:].strip())
                    events.append(data)
                elif line.startswith("event:"):
                    event_type = line[6:].strip()
                    events.append({"_event_type": event_type})


async def send_message(agent_id: str, message: str) -> list[dict]:
    """Send message to agent and collect SSE events from response."""
    events = []
    async with httpx.AsyncClient(timeout=30.0) as client:
        async with client.stream(
            "POST",
            f"{SERVER_URL}/agents/{agent_id}/messages",
            json={"message": message},
        ) as response:
            async for line in response.aiter_lines():
                if line.startswith("data:"):
                    data = json.loads(line[5:].strip())
                    events.append(data)
                elif line.startswith("event:"):
                    event_type = line[6:].strip()
                    events.append({"_event_type": event_type})
    return events


async def get_or_create_test_agent() -> str:
    """Get existing test agent or create one. Returns agent_id."""
    async with httpx.AsyncClient(timeout=10.0) as client:
        # List agents
        response = await client.get(f"{SERVER_URL}/agents")
        response.raise_for_status()
        agents = response.json()
        
        # Look for existing e2e test agent
        for agent in agents:
            if agent.get("name") == "e2e-broadcast-test":
                return agent["id"]
        
        # Create new agent
        response = await client.post(
            f"{SERVER_URL}/agents",
            json={
                "name": "e2e-broadcast-test",
                "system_instructions": "You are a helpful test assistant. Keep responses brief.",
                "config": {
                    "model_name": "claude-sonnet-4-20250514",
                    "tool_names": [],
                    "soft_compaction_limit": 100000,
                },
            },
        )
        response.raise_for_status()
        return response.json()["id"]


async def test_broadcast_stream_receives_events():
    """
    Verify that /stream receives broadcast events when /messages is called.
    
    The broadcast stream should receive:
    - RunStartedEvent (synthetic, before agent runs)
    - All pydantic-ai streaming events (same as /messages response)
    - RunCompletedEvent (synthetic, after agent completes)
    """
    agent_id = await get_or_create_test_agent()
    
    broadcast_events: list[dict] = []
    subscription_ready = asyncio.Event()
    
    # Start subscription task
    subscription_task = asyncio.create_task(
        subscribe_to_stream(agent_id, broadcast_events, subscription_ready)
    )
    
    try:
        # Wait for subscription to be active
        await asyncio.wait_for(subscription_ready.wait(), timeout=5.0)
        
        # Small delay to ensure subscription is fully established
        await asyncio.sleep(0.1)
        
        # Send message and collect response events
        message_events = await send_message(agent_id, "Say 'hello' and nothing else.")
        
        # Give broadcast events time to arrive
        await asyncio.sleep(0.5)
        
    finally:
        subscription_task.cancel()
        try:
            await subscription_task
        except asyncio.CancelledError:
            pass
    
    # Verify we got events from both streams
    assert len(message_events) > 0, "Should receive events from /messages"
    assert len(broadcast_events) > 0, "Should receive events from /stream broadcast"
    
    # Check for synthetic bookend events in broadcast
    event_types = [e.get("_event_type") for e in broadcast_events if "_event_type" in e]
    assert "RunStartedEvent" in event_types, f"Broadcast should include RunStartedEvent. Got: {event_types}"
    assert "RunCompletedEvent" in event_types, f"Broadcast should include RunCompletedEvent. Got: {event_types}"
    
    print(f"\n✓ Message stream events: {len(message_events)}")
    print(f"✓ Broadcast stream events: {len(broadcast_events)}")
    print(f"✓ Broadcast event types: {event_types}")
