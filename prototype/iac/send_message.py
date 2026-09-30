"""send_message tool implementation for inter-agent communication.

NOTE: This architecture will likely be discarded in favor of a centralized
fire-and-forget queue and server-level background task for handling such agent runs.
We're doing too much and passing too much down into the tool level here, way too much
responsibility for this tool. This prototype is useful for learning about overall
behavior though.
"""
import asyncio
import contextvars
import logging

from pydantic_ai import RunContext
from pydantic_ai.exceptions import ModelRetry

from agent.crud import get_all_agents
from agent.types import AgentDeps

logger = logging.getLogger(__name__)

# Timeout for acquiring the target agent's lock in the background task
SEND_MESSAGE_LOCK_TIMEOUT_SECONDS: float = 20.0

# GC-safe set: keeps tasks alive until done (event loop holds only weak refs to tasks)
background_tasks: set[asyncio.Task] = set()


def _format_inter_agent_message(sender_name: str, content: str) -> str:
    """Prepend the standard inter-agent origin marker to a message."""
    return f"[INTER AGENT MESSAGE. If you want to reply, use the 'send_message' tool. From: {sender_name}]\n{content}"


async def send_message(
    ctx: RunContext[AgentDeps],
    target_name: str,
    content: str,
) -> str:
    """Send a message to another agent by name.

    Resolves the target agent by name, then spawns a background task to deliver
    the message. Returns a delivery confirmation once the target's lock is acquired,
    or an error string if the target is not found or is unavailable.

    Args:
        ctx: Pydantic AI run context with AgentDeps
        target_name: Name of the agent to message
        content: The message content to send

    Returns:
        A confirmation string on success.

    Raises:
        ModelRetry: If target agent not found, registry not configured, or target is busy.
    """
    deps = ctx.deps

    # Resolve target name → agent_id
    agents = await get_all_agents(deps.session)
    target_record = next((a for a in agents if a.name == target_name), None)
    if target_record is None:
        raise ModelRetry(f"Agent {target_name!r} not found.")

    # Reject self-messages
    if target_record.id == deps.agent_id:
        raise ModelRetry("You cannot send a message to yourself.")

    # Format message with origin marker
    formatted_content = _format_inter_agent_message(sender_name=deps.name, content=content)

    # Ensure registry is available (only present when agent has send_message configured)
    if deps.agent_app_state_reg is None:
        raise ModelRetry("send_message is not configured for this agent (missing registry).")

    # Extract engine from session — always available via session.bind
    # This is intended to avoid passing engine down on deps. Eventually we hope to not need the engine at this level at all.
    engine = deps.session.bind

    # Spawn background task with delivery confirmation via Future
    loop = asyncio.get_running_loop()
    future: asyncio.Future[bool] = loop.create_future()

    task = asyncio.create_task(
        _deliver_message(
            agent_id=target_record.id,
            user_prompt=formatted_content,
            engine=engine,
            agent_app_state_reg=deps.agent_app_state_reg,
            delivery_future=future,
        ),
        context=contextvars.Context(),
    )
    background_tasks.add(task)
    task.add_done_callback(background_tasks.discard)

    # Await delivery confirmation (lock acquired or failed)
    success = await future

    if success:
        return f"Message delivered to {target_name!r}."
    raise ModelRetry(f"Agent {target_name!r} is busy and could not be reached.")


async def _deliver_message(
    agent_id: str,
    user_prompt: str,
    engine: object,
    agent_app_state_reg: dict,
    delivery_future: "asyncio.Future[bool]",
    timeout: float = SEND_MESSAGE_LOCK_TIMEOUT_SECONDS,
) -> None:
    """Background task: acquire target lock, signal delivery, run agent to completion.

    Signals delivery_future with True once the lock is acquired (delivery confirmed),
    or False if the lock cannot be acquired within the timeout.
    """
    # Deferred imports to avoid circular imports at module load
    from agent.factory import AgentFactory, AgentLockedError
    from agent.runner import run_stateful_agent
    from db.connection import get_session
    try:
        async with get_session(engine) as session:
            factory = AgentFactory(agent_id, agent_app_state_reg, session)
            try:
                async with factory.build_agent_and_deps(timeout=timeout) as (agent, deps):
                    # Lock is held — signal delivery confirmation
                    delivery_future.set_result(True)
                    async for _ in run_stateful_agent(agent, deps, agent_app_state_reg[agent_id], user_prompt):
                        pass
            except AgentLockedError:
                if not delivery_future.done():
                    delivery_future.set_result(False)
    except Exception as e:
        logger.error("send_message background task failed for agent %r: %s", agent_id, e)
        if not delivery_future.done():
            delivery_future.set_exception(e)
        raise
