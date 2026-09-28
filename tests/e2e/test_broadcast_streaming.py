"""E2E tests for broadcast streaming.

Run with: pytest -m e2e
Server is auto-started/stopped by the live_server fixture.
"""

import asyncio

import httpx
from httpx_sse import aconnect_sse

from tests.e2e.conftest import sse_to_dict, send_message, TEST_AGENT_INSTRUCTIONS


async def subscribe_to_stream(
    client: httpx.AsyncClient, server_url: str, agent_id: str, events: list, ready_event: asyncio.Event
):
    """Subscribe to agent's broadcast stream and collect events."""
    async with aconnect_sse(client, "GET", f"{server_url}/agents/{agent_id}/stream") as event_source:
        ready_event.set()  # Signal that subscription is active
        async for sse in event_source.aiter_sse():
            events.append(sse_to_dict(sse))


async def get_or_create_test_agent(client: httpx.AsyncClient, server_url: str) -> str:
    """Get existing test agent or create one. Returns agent_id."""
    # List agents
    response = await client.get(f"{server_url}/agents")
    response.raise_for_status()
    agents = response.json()
    
    # Look for existing e2e test agent
    for agent in agents:
        if agent.get("name") == "e2e-broadcast-test":
            return agent["id"]
    
    # Create new agent
    response = await client.post(
        f"{server_url}/agents",
        json={
            "name": "e2e-broadcast-test",
            "system_instructions": TEST_AGENT_INSTRUCTIONS,
            "config": {
                "model_name": "claude-haiku-4-5-20251001",
                "tool_names": [],
                "soft_compaction_limit": 100000,
            },
        },
    )
    response.raise_for_status()
    return response.json()["id"]


async def test_broadcast_stream_receives_events(live_server: str, client: httpx.AsyncClient):
    """
    Verify that /stream receives broadcast events when /messages is called.
    
    The broadcast stream should receive:
    - RunStartedEvent (synthetic, before agent runs)
    - All pydantic-ai streaming events (same as /messages response)
    - RunCompletedEvent (synthetic, after agent completes)
    """
    agent_id = await get_or_create_test_agent(client, live_server)
    
    broadcast_events: list[dict] = []
    subscription_ready = asyncio.Event()
    
    # Start subscription task
    subscription_task = asyncio.create_task(
        subscribe_to_stream(client, live_server, agent_id, broadcast_events, subscription_ready)
    )
    
    try:
        # Wait for subscription to be active
        await asyncio.wait_for(subscription_ready.wait(), timeout=5.0)
        
        # Small delay to ensure subscription is fully established
        await asyncio.sleep(0.1)
        
        # Send message and collect response events
        message_events = await send_message(client, live_server, agent_id, "Say 'hello' and nothing else.")
        
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
    event_types = [e["event"] for e in broadcast_events]
    assert "RunStartedEvent" in event_types, f"Broadcast should include RunStartedEvent. Got: {event_types}"
    assert "RunCompletedEvent" in event_types, f"Broadcast should include RunCompletedEvent. Got: {event_types}"
    
    print(f"\n✓ Message stream events: {len(message_events)}")
    print(f"✓ Broadcast stream events: {len(broadcast_events)}")
    print(f"✓ Broadcast event types: {event_types}")
