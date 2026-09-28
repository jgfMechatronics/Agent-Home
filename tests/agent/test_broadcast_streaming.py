"""Unit tests for BroadcastHub and /stream endpoint integration."""
import asyncio
import pytest
from unittest.mock import AsyncMock, Mock, patch

from fastapi import FastAPI
from httpx import AsyncClient

from agent.broadcast_streaming import (
    BroadcastHub,
    RunStartedEvent,
    RunCompletedEvent,
    run_agent_with_broadcast,
)
from db.models import AgentRecord


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

    @staticmethod
    async def broadcast_events(hub: BroadcastHub, events_a: list, events_b: list):
        """Broadcast events to agents a and b."""
        await asyncio.sleep(0.01)  # Let subscribers start
        for event in events_a:
            hub.broadcast("a", event)
        for event in events_b:
            hub.broadcast("b", event)
        await asyncio.sleep(0.01)  # Let events propagate

    @staticmethod
    async def terminate_via_disconnect(hub, events_a, events_b, requests):
        """Broadcast events, then signal client disconnect."""
        await TestBroadcastHub.broadcast_events(hub, events_a, events_b)
        for req in requests:
            req._disconnected = True

    @staticmethod
    async def terminate_via_shutdown(hub, events_a, events_b, requests):
        """Broadcast events, then trigger hub shutdown."""
        await TestBroadcastHub.broadcast_events(hub, events_a, events_b)
        await hub.shutdown()

    @pytest.mark.parametrize("termination_method,timeout", [
        pytest.param(terminate_via_disconnect, 7.0, id="disconnect"),
        pytest.param(terminate_via_shutdown, 2.0, id="shutdown"),
    ])
    async def test_broadcast_delivers_to_correct_subscribers(self, hub, termination_method, timeout):
        """Events broadcast to agent_id reach only that agent's subscribers.
        
        Tests both termination mechanisms: client disconnect and server shutdown.
        """
        # Events to broadcast for each agent
        events_a = [RunStartedEvent(prompt="a1"), RunCompletedEvent(status="success")]
        events_b = [RunStartedEvent(prompt="b1"), RunStartedEvent(prompt="b2")]

        # Create mock requests (one for a, two for b)
        request_a = self.make_mock_request()
        request_b1 = self.make_mock_request()
        request_b2 = self.make_mock_request()
        all_requests = [request_a, request_b1, request_b2]

        # Collectors for received events
        received_a: list = []
        received_b1: list = []
        received_b2: list = []

        await asyncio.wait_for(
            asyncio.gather(
                self.collect_events(hub, "a", request_a, received_a),
                self.collect_events(hub, "b", request_b1, received_b1),
                self.collect_events(hub, "b", request_b2, received_b2),
                termination_method(hub, events_a, events_b, all_requests),
            ),
            timeout=timeout,
        )

        # Each subscriber received exactly what was broadcast to their agent
        assert received_a == events_a
        assert received_b1 == events_b
        assert received_b2 == events_b


class TestRunAgentWithBroadcast:
    """Tests for run_agent_with_broadcast wrapper function."""

    @pytest.fixture
    def mock_deps(self):
        deps = Mock()
        deps.agent_id = "test-agent"
        return deps

    @pytest.fixture
    def mock_state(self):
        state = Mock()
        state.cancel_requested = asyncio.Event()
        return state

    @pytest.fixture
    def hub(self):
        return Mock()

    async def test_success_status(self, mock_deps, mock_state, hub):
        """Normal run broadcasts started, events, and completed(success)."""
        mock_events = [Mock(name="event1"), Mock(name="event2")]

        async def mock_runner(agent, deps, state, prompt):
            for e in mock_events:
                yield e

        with patch("agent.broadcast_streaming.run_stateful_agent", mock_runner):
            yielded = [e async for e in run_agent_with_broadcast(
                Mock(), mock_deps, mock_state, "test prompt", hub
            )]

        assert yielded == mock_events
        assert hub.broadcast.call_args_list == [
            (("test-agent", RunStartedEvent(prompt="test prompt")),),
            (("test-agent", mock_events[0]),),
            (("test-agent", mock_events[1]),),
            (("test-agent", RunCompletedEvent(status="success")),),
        ]

    async def test_error_status(self, mock_deps, mock_state, hub):
        """Exception broadcasts completed(error) and re-raises."""
        mock_event = Mock()

        async def mock_runner(agent, deps, state, prompt):
            yield mock_event
            raise ValueError("test error")

        with patch("agent.broadcast_streaming.run_stateful_agent", mock_runner):
            with pytest.raises(ValueError, match="test error"):
                async for _ in run_agent_with_broadcast(
                    Mock(), mock_deps, mock_state, "test prompt", hub
                ):
                    pass

        assert hub.broadcast.call_args_list == [
            (("test-agent", RunStartedEvent(prompt="test prompt")),),
            (("test-agent", mock_event),),
            (("test-agent", RunCompletedEvent(status="error")),),
        ]

    async def test_cancelled_status(self, mock_deps, mock_state, hub):
        """If cancel_requested is set, status is 'cancelled'."""
        mock_event = Mock()

        async def mock_runner(agent, deps, state, prompt):
            yield mock_event
            mock_state.cancel_requested.set()

        with patch("agent.broadcast_streaming.run_stateful_agent", mock_runner):
            async for _ in run_agent_with_broadcast(
                Mock(), mock_deps, mock_state, "test prompt", hub
            ):
                pass

        assert hub.broadcast.call_args_list == [
            (("test-agent", RunStartedEvent(prompt="test prompt")),),
            (("test-agent", mock_event),),
            (("test-agent", RunCompletedEvent(status="cancelled")),),
        ]


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
            hub.broadcast(agent_id, RunStartedEvent(prompt="hello"))
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
