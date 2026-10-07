"""
BroadcastHub — real-time event broadcasting to SSE subscribers.

Manages subscriber queues and event distribution for agent activity streaming.
Enables TUIs and other clients to observe agent runs without polling.
"""
import asyncio
from contextlib import asynccontextmanager
from dataclasses import dataclass
from typing import TYPE_CHECKING, AsyncGenerator, AsyncIterator

from starlette.requests import Request

from agent.runner import run_stateful_agent

if TYPE_CHECKING:
    from pydantic_ai import Agent

    from agent.runner import AgentAppState
    from agent.types import AgentDeps

# ---------------------------------------------------------------------------
# Synthetic event types (not emitted by pydantic-ai)
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class RunStartedEvent:
    """Emitted before the first pydantic-ai event in a run."""
    prompt: str


@dataclass(frozen=True)
class RunCompletedEvent:
    """Emitted after all pydantic-ai events in a run."""
    status: str  # "success" | "cancelled" | "error"


@dataclass(frozen=True)
class RunErrorEvent:
    """Emitted when an agent run fails with an unhandled exception.

    Broadcast before RunCompletedEvent(status='error') so subscribers see the
    error details before the terminal signal. Mirrors the Error SSE emitted by
    the handle_message route for direct (non-background) runs.
    """
    message: str


@dataclass(frozen=True)
class ShutdownEvent:
    """Internal sentinel signaling subscribers to exit. Should never be yielded to consumers."""
    pass


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

    def broadcast(self, agent_id: str, event: object) -> None:
        """Push an event to all registered subscribers for agent_id.
        
        Uses put_nowait to avoid possibly blocking the caller.
        """
        for queue in self._subscribers.get(agent_id, []):
            # If the queue fills we have a bug which should be addressed 
            queue.put_nowait(event)

    @asynccontextmanager
    async def subscribe(self, agent_id: str, request: Request) -> AsyncIterator[AsyncIterator[object]]:
        """Subscribe to events for agent_id. Yields async iterator of events.
        
        Handles registration/cleanup automatically. Checks for client disconnect
        and shutdown sentinel internally.
        """
        queue: asyncio.Queue = asyncio.Queue()
        self._register(agent_id, queue)
        try:
            yield self._event_iterator(agent_id, queue, request)
        finally:
            self._unregister(agent_id, queue)

    async def _event_iterator(
        self, agent_id: str, queue: asyncio.Queue, request: Request
    ) -> AsyncGenerator[object, None]:
        """Internal iterator that yields events until disconnect or shutdown."""
        while not await request.is_disconnected():
            try:
                event = await asyncio.wait_for(queue.get(), timeout=5.0)
            except asyncio.TimeoutError:
                continue
            if isinstance(event, ShutdownEvent):
                return
            yield event

    def _register(self, agent_id: str, queue: asyncio.Queue) -> None:
        """Add a queue to the registry."""
        self._subscribers.setdefault(agent_id, []).append(queue)

    def _unregister(self, agent_id: str, queue: asyncio.Queue) -> None:
        """Remove a queue from the registry. Removes agent_id entry if this was the last queue."""
        queue_list = self._subscribers.get(agent_id)
        if queue_list and queue in queue_list:
            queue_list.remove(queue)
            if not queue_list:
                del self._subscribers[agent_id]

    async def shutdown(self) -> None:
        """Broadcast ShutdownEvent to all subscribers. Called on server shutdown."""
        for agent_id in list(self._subscribers.keys()):
            for queue in self._subscribers[agent_id]:
                await queue.put(ShutdownEvent())


# ---------------------------------------------------------------------------
# Agent runner wrapper function
# ---------------------------------------------------------------------------

async def run_agent_with_broadcast(
    agent: "Agent",
    deps: "AgentDeps",
    agent_app_state: "AgentAppState",
    user_prompt: str,
    hub: BroadcastHub,
) -> AsyncGenerator[object, None]:
    """Wrap run_stateful_agent with broadcast logic.
    
    Broadcasts RunStartedEvent before the run, all pydantic-ai events during,
    and RunCompletedEvent after. Yields events through so caller can also
    consume them (e.g., for SSE response).
    
    Status in RunCompletedEvent:
    - "success" if run completes normally
    - "cancelled" if cancel_requested was set
    - "error" if an exception occurred (re-raised after broadcasting)
    """

    agent_id = deps.agent_id
    hub.broadcast(agent_id, RunStartedEvent(prompt=user_prompt))
    status = "success"
    try:
        async for event in run_stateful_agent(agent, deps, agent_app_state, user_prompt):
            hub.broadcast(agent_id, event)
            yield event
    except Exception as e:
        status = "error"
        hub.broadcast(agent_id, RunErrorEvent(
            message=f"\n\nUnexpected internal server error: '{type(e).__name__}: {str(e)}'"
        ))
        raise
    finally:
        if agent_app_state.cancel_requested.is_set():
            status = "cancelled"
        hub.broadcast(agent_id, RunCompletedEvent(status=status))
