"""
Tests for BroadcastHub and the /agents/{agent_id}/stream endpoint.

Top-down TDD: tests drive the interface design, implementation follows.
"""
import asyncio

import pytest
from fastapi import FastAPI
from httpx import AsyncClient

from agent.broadcast_streaming import BroadcastHub, RunStartedEvent
from db.models import AgentRecord


class TestStreamEndpoint:
    """GET /agents/{agent_id}/stream — SSE stream of broadcast events."""

    @pytest.fixture(autouse=True)
    def setup_broadcast_hub(self, app: FastAPI):
        """Install BroadcastHub on app state (lifespan doesn't run in tests)."""
        app.state.broadcast_hub = BroadcastHub()

    @pytest.fixture
    def hub(self, app: FastAPI) -> BroadcastHub:
        """Provide BroadcastHub from app state."""
        return app.state.broadcast_hub

    async def test_receives_broadcast_event(self, client: AsyncClient, agent_record: AgentRecord, hub):
        """Client subscribed to /stream receives events broadcast to that agent."""
        agent_id = str(agent_record.id)
        received_lines = []

        async def subscribe_and_collect():
            async with client.stream("GET", f"/agents/{agent_id}/stream") as response:
                assert response.status_code == 200
                async for line in response.aiter_lines():
                    received_lines.append(line)

        async def broadcast_then_shutdown():
            await asyncio.sleep(0.05)  # Let subscriber connect
            await hub.broadcast(agent_id, RunStartedEvent(prompt="hello"))
            await asyncio.sleep(0.05)  # Let event propagate
            await hub.shutdown()  # Causes iterator to exit via ShutdownEvent

        # Run concurrently - shutdown() causes subscriber's iterator to exit cleanly
        await asyncio.wait_for(
            asyncio.gather(subscribe_and_collect(), broadcast_then_shutdown()),
            timeout=2.0
        )

        # Verify SSE format: event type and data (dataclass serializes to dict with field names)
        expected = [
            "event: RunStartedEvent",
            'data: {"prompt": "hello"}',
            "",  # Empty line marks end of SSE event
        ]
        assert received_lines == expected
