"""
BroadcastHub — real-time event broadcasting to SSE subscribers.

Manages subscriber queues and event distribution for agent activity streaming.
Enables TUIs and other clients to observe agent runs without polling.
"""
import asyncio
from contextlib import asynccontextmanager
from dataclasses import dataclass, field
from typing import AsyncIterator

from starlette.requests import Request


# ---------------------------------------------------------------------------
# Synthetic event types (not emitted by pydantic-ai)
# ---------------------------------------------------------------------------

@dataclass
class RunStartedEvent:
    """Emitted before the first pydantic-ai event in a run."""
    prompt: str
    event_kind: str = field(default="run_started", init=False)


@dataclass
class RunCompletedEvent:
    """Emitted after all pydantic-ai events in a run."""
    status: str  # "success" | "cancelled" | "error"
    event_kind: str = field(default="run_completed", init=False)


@dataclass
class ShutdownEvent:
    """Internal sentinel signaling subscribers to exit. Should never be yielded to consumers."""
    event_kind: str = field(default="shutdown", init=False)


# ---------------------------------------------------------------------------
# BroadcastHub
# ---------------------------------------------------------------------------

class BroadcastHub:
    """Central hub for broadcasting agent events to SSE subscribers.
    
    Lives on app.state.broadcast_hub. Routes and tools use broadcast() to send
    events; the /stream endpoint uses subscribe() to receive them.
    """

    def __init__(self):
        self._subscribers: dict[str, list[asyncio.Queue]] = {}

    async def broadcast(self, agent_id: str, event: object) -> None:
        """Push an event to all registered subscribers for agent_id."""
        for queue in self._subscribers.get(agent_id, []):
            await queue.put(event)

    @asynccontextmanager
    async def subscribe(self, agent_id: str, request: Request) -> AsyncIterator[AsyncIterator[object]]:
        """Subscribe to events for agent_id. Yields async iterator of events.
        
        Handles registration/cleanup automatically. Checks for client disconnect
        and shutdown sentinel internally.
        """
        queue: asyncio.Queue = asyncio.Queue()
        self._subscribers.setdefault(agent_id, []).append(queue)
        try:
            yield self._event_iterator(agent_id, queue, request)
        finally:
            self._unregister(agent_id, queue)

    async def _event_iterator(
        self, agent_id: str, queue: asyncio.Queue, request: Request
    ) -> AsyncIterator[object]:
        """Internal iterator that yields events until disconnect or shutdown."""
        while not await request.is_disconnected():
            try:
                event = await asyncio.wait_for(queue.get(), timeout=1.0)
            except asyncio.TimeoutError:
                continue
            if isinstance(event, ShutdownEvent):
                return
            yield event

    def _unregister(self, agent_id: str, queue: asyncio.Queue) -> None:
        """Remove a queue from the registry."""
        subs = self._subscribers.get(agent_id)
        if subs and queue in subs:
            subs.remove(queue)
            if not subs:
                del self._subscribers[agent_id]

    async def shutdown(self) -> None:
        """Broadcast ShutdownEvent to all subscribers. Called on server shutdown."""
        for agent_id in list(self._subscribers.keys()):
            for queue in self._subscribers[agent_id]:
                await queue.put(ShutdownEvent())
