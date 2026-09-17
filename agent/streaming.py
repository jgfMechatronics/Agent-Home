"""Agent activity stream — subscriber registry for real-time run event broadcasting.

Agents running in the background (e.g. via send_message) emit events here so
that connected clients (e.g. the ACP bridge) can stream them in real time without
polling history.

Architecture note: This is a module-level singleton for prototype simplicity.
In production this should move to app.state to avoid state leaking between test
instances. The subscriber registry is intentionally decoupled from SSE formatting
— callers queue raw pydantic-ai events plus our synthetic event types, and the
route layer handles SSE serialisation.
"""
import asyncio
from dataclasses import dataclass, field


# ---------------------------------------------------------------------------
# Synthetic event types (not emitted by pydantic-ai)
# ---------------------------------------------------------------------------

@dataclass
class UserPromptEvent:
    """Emitted before RunStarted — carries the message that initiated the background run.

    Allows the TUI to display what triggered the agent's turn (e.g. an inter-agent
    message) before the agent starts responding, mirroring how user-initiated turns
    show the prompt.
    """
    content: str
    event_kind: str = field(default="user_prompt", init=False)


@dataclass
class RunStartedEvent:
    """Emitted before the first pydantic-ai event in a background run."""
    event_kind: str = field(default="run_started", init=False)


@dataclass
class RunCompletedEvent:
    """Emitted after all pydantic-ai events in a background run."""
    status: str  # "success" | "cancelled" | "error"
    event_kind: str = field(default="run_completed", init=False)


# ---------------------------------------------------------------------------
# Subscriber registry
# ---------------------------------------------------------------------------

# agent_id → list of queues, one per connected subscriber
_subscribers: dict[str, list[asyncio.Queue]] = {}


def register_subscriber(agent_id: str) -> asyncio.Queue:
    """Create and register a new event queue for agent_id. Caller must unregister."""
    queue: asyncio.Queue = asyncio.Queue()
    _subscribers.setdefault(agent_id, []).append(queue)
    return queue


def unregister_subscriber(agent_id: str, queue: asyncio.Queue) -> None:
    """Remove a queue from the registry. Safe to call even if already removed."""
    subs = _subscribers.get(agent_id)
    if subs and queue in subs:
        subs.remove(queue)
        if not subs:
            del _subscribers[agent_id]


async def broadcast(agent_id: str, event: object) -> None:
    """Push an event to all registered subscribers for agent_id. Fire-and-forget per subscriber."""
    for queue in _subscribers.get(agent_id, []):
        await queue.put(event)
