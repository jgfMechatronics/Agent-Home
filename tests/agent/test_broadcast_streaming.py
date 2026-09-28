"""Unit tests for BroadcastHub."""
import asyncio
import pytest
from unittest.mock import AsyncMock, Mock

from agent.broadcast_streaming import BroadcastHub, RunStartedEvent, RunCompletedEvent


class TestBroadcastHub:
    """Tests for BroadcastHub public interface: subscribe, broadcast, shutdown."""

    @pytest.fixture
    def hub(self):
        return BroadcastHub()

    def make_mock_request(self):
        """Factory for creating multiple independent mock requests."""
        request = Mock()
        request._disconnected = False
        request.is_disconnected = AsyncMock(side_effect=lambda: request._disconnected)
        return request

    async def collect_events(self, hub: BroadcastHub, agent_id: str, request: Mock, collector: list):
        """Subscribe and collect events until disconnected or shutdown."""
        async with hub.subscribe(agent_id, request) as events:
            async for event in events:
                collector.append(event)

    async def test_broadcast_delivers_to_correct_subscribers(self, hub):
        """Events broadcast to agent_id reach only that agent's subscribers."""
        # Events to broadcast for each agent
        events_a = [RunStartedEvent(prompt="a1"), RunCompletedEvent(status="success")]
        events_b = [RunStartedEvent(prompt="b1"), RunStartedEvent(prompt="b2")]

        # Create mock requests (one for a, two for b)
        request_a = self.make_mock_request()
        request_b1 = self.make_mock_request()
        request_b2 = self.make_mock_request()

        # Collectors for received events
        received_a: list = []
        received_b1: list = []
        received_b2: list = []

        async def broadcast_then_disconnect():
            """Broadcast all events, then signal disconnect."""
            await asyncio.sleep(0.01)  # Let subscribers start

            for event in events_a:
                await hub.broadcast("a", event)
            for event in events_b:
                await hub.broadcast("b", event)

            await asyncio.sleep(0.01)  # Let events propagate

            # Signal all subscribers to disconnect
            request_a._disconnected = True
            request_b1._disconnected = True
            request_b2._disconnected = True

        await asyncio.wait_for(
            asyncio.gather(
                self.collect_events(hub, "a", request_a, received_a),
                self.collect_events(hub, "b", request_b1, received_b1),
                self.collect_events(hub, "b", request_b2, received_b2),
                broadcast_then_disconnect(),
            ),
            timeout=7.0,  # Must exceed hub's 5s disconnect check interval
        )

        # Each subscriber received exactly what was broadcast to their agent
        assert received_a == events_a
        assert received_b1 == events_b
        assert received_b2 == events_b

    async def test_shutdown_terminates_all_subscribers(self, hub):
        """shutdown() causes all active subscribe iterators to exit."""
        request_a = self.make_mock_request()
        request_b = self.make_mock_request()

        received_a: list = []
        received_b: list = []

        async def shutdown_after_delay():
            await asyncio.sleep(0.01)  # Let subscribers start
            await hub.shutdown()

        # Should complete without timeout — shutdown terminates iterators
        await asyncio.wait_for(
            asyncio.gather(
                self.collect_events(hub, "a", request_a, received_a),
                self.collect_events(hub, "b", request_b, received_b),
                shutdown_after_delay(),
            ),
            timeout=2.0,
        )

        # Subscribers exited cleanly (no events were broadcast)
        assert received_a == []
        assert received_b == []
