"""
Agent tool registry — Section 3.2

Maps tool name strings to callable tool functions for agent construction.
Memory tools raise ModelRetry on failure (for model self-correction).
TODO: Add a memory_delete which just wraps memory replace with an empty new content, for convenience
and associated tests
"""
import asyncio
import logging
from typing import Callable

from pydantic_ai import RunContext
from pydantic_ai.common_tools.duckduckgo import duckduckgo_search_tool
from pydantic_ai.common_tools.web_fetch import web_fetch_tool
from pydantic_ai.tools import Tool
from pydantic_ai.exceptions import ModelRetry

from agent.crud import get_all_agents
from agent.types import AgentDeps
from memory.block_crud import get_block, update_block

logger = logging.getLogger(__name__)

# Timeout for acquiring the target agent's lock in the background task
SEND_MESSAGE_LOCK_TIMEOUT_SECONDS: float = 60.0

# GC-safe set: keeps tasks alive until done (event loop holds only weak refs to tasks)
_BACKGROUND_TASKS: set[asyncio.Task] = set()


def _get_edit_line_info(content: str, edit_start_idx: int, new_text: str) -> tuple[int, int]:
    """Get line number and line count for snippet computation.
    
    Args:
        content: The content AFTER the edit (temporally after, not positionally)
        edit_start_idx: Character index where edit started in original content
        new_text: The text that was inserted/replaced
    
    Returns:
        (edit_start_line, edit_line_count) for _compute_snippet
    """
    # Count newlines before the edit position to get 0-indexed line number
    edit_start_line = content[:edit_start_idx].count("\n")
    # Count lines in the new text
    edit_line_count = new_text.count("\n") + 1
    return edit_start_line, edit_line_count


def _compute_snippet(
    content: str, edit_start_idx: int, new_text: str, context_lines: int = 3
) -> str:
    """Extract a snippet of content around an edited region.

    Args:
        content: The full content (after edit) to extract from
        edit_start_idx: Character index where the edit begins
        new_text: The text that was inserted/replaced (used to determine line span)
        context_lines: Number of lines of context before + after

    Returns:
        A string containing the snippet with context around the edit
    """
    edit_start_line, edit_line_count = _get_edit_line_info(content, edit_start_idx, new_text)
    lines = content.split("\n")
    start = max(0, edit_start_line - context_lines)
    end = min(len(lines), edit_start_line + edit_line_count + context_lines)
    return "\n".join(lines[start:end])


def _find_occurrences(content: str, target: str) -> list[int]:
    """Find all non-overlapping start indices of target in content."""
    indices = []
    start = 0
    while True:
        idx = content.find(target, start)
        if idx == -1:
            break
        indices.append(idx)
        start = idx + len(target)  # Non-overlapping: skip past this match
    return indices


def _resolve_occurrence(
    content: str, target: str, occurrence: int | None, label: str
) -> int:
    """Resolve target string to a character index.
    
    Args:
        content: The block content to search
        target: The string to find
        occurrence: Which occurrence (1-indexed), or None for unique match
        label: Block label for error messages
    
    Returns:
        Start character index on success.
    
    Raises:
        ModelRetry: If target not found, ambiguous, or occurrence invalid.
    """
    indices = _find_occurrences(content, target)
    
    if not indices:
        raise ModelRetry(f"'{target}' not found in block '{label}'")
    
    if len(indices) > 1 and occurrence is None:
        raise ModelRetry(f"'{target}' appears {len(indices)} times. Specify occurrence (1-{len(indices)}).")
    
    # Convert to 0-indexed
    target_idx = 0 if occurrence is None else occurrence - 1
    
    if target_idx < 0:
        raise ModelRetry("occurrence must be >= 1 (1-indexed)")
    
    if target_idx >= len(indices):
        raise ModelRetry(f"occurrence {occurrence} not found (only {len(indices)} occurrences exist)")
    
    return indices[target_idx]


async def memory_replace(
    ctx: RunContext[AgentDeps],
    label: str,
    old_string: str,
    new_string: str,
    occurrence: int | None = None,
) -> str:
    """Replace text in a memory block.
    
    Args:
        ctx: Pydantic AI run context with AgentDeps
        label: The label of the memory block to edit
        old_string: The text to find and replace
        new_string: The replacement text
        occurrence: Which occurrence to replace (1-indexed). Required if multiple matches.
    
    Returns:
        Snippet of updated content on success.
    
    Raises:
        ModelRetry: On validation failure (block not found, target not found, etc.)
    """
    deps = ctx.deps
    
    if not old_string:
        raise ModelRetry("old_string cannot be empty")
    
    block = await get_block(deps.session, deps.agent_id, label)
    if block is None:
        raise ModelRetry(f"block '{label}' not found")
    
    # Resolve occurrence to character position (raises ModelRetry on failure)
    start_pos = _resolve_occurrence(block.content, old_string, occurrence, label)
    
    # Perform replacement
    end_pos = start_pos + len(old_string)
    new_content = block.content[:start_pos] + new_string + block.content[end_pos:]
    
    # handles char limit check and persistence
    try:
        await update_block(deps, label, new_content, commit=False, block=block)
    except ValueError as e:
        raise ModelRetry(str(e))
    
    # Compute and return snippet
    return _compute_snippet(new_content, start_pos, new_string)


async def memory_insert(
    ctx: RunContext[AgentDeps],
    label: str,
    content: str,
    after: str | None = None,
    occurrence: int | None = None,
) -> str:
    """Insert text into a memory block.
    
    Args:
        ctx: Pydantic AI run context with AgentDeps
        label: The label of the memory block to edit
        content: The text to insert
        after: Where to insert. Use '<start>' for beginning, '<end>' for end,
               or any string to insert after that anchor.
        occurrence: Which occurrence of anchor to insert after (1-indexed).
                   Required if anchor appears multiple times.
    
    Returns:
        Snippet of updated content on success.
    
    Raises:
        ModelRetry: On validation failure (block not found, anchor not found, etc.)
    """
    deps = ctx.deps
    
    if not after:
        raise ModelRetry("'after' cannot be empty. Use '<start>' or '<end>' for boundary insertions.")
    
    # Get block
    block = await get_block(deps.session, deps.agent_id, label)
    if block is None:
        raise ModelRetry(f"block '{label}' not found")
    
    # Handle special markers
    if after in ("<start>", "<end>"):
        if occurrence is not None:
            raise ModelRetry("occurrence cannot be used with '<start>' or '<end>'")
        insert_pos = 0 if after == "<start>" else len(block.content)
    else:
        # Resolve anchor occurrence to character position (raises ModelRetry on failure)
        anchor_pos = _resolve_occurrence(block.content, after, occurrence, label)
        # Insert AFTER the anchor
        insert_pos = anchor_pos + len(after)
    
    # Perform insertion
    new_content = block.content[:insert_pos] + content + block.content[insert_pos:]
    
    # handles char limit check and persistence
    try:
        await update_block(deps, label, new_content, commit=False, block=block)
    except ValueError as e:
        raise ModelRetry(str(e))
    
    return _compute_snippet(new_content, insert_pos, content)


# =============================================================================
# Inter-agent communication
# =============================================================================

def _format_inter_agent_message(sender_name: str, content: str) -> str:
    """Prepend the standard inter-agent origin marker to a message.

    Format: '[INTER AGENT MESSAGE. From: <sender>]\\n<content>'
    """
    return f"[INTER AGENT MESSAGE. From: {sender_name}]\n{content}"


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
        A string describing the outcome — success or reason for failure.
    """
    deps = ctx.deps

    # Resolve target name → agent_id
    agents = await get_all_agents(deps.session)
    target_record = next((a for a in agents if a.name == target_name), None)
    if target_record is None:
        return f"Error: Agent {target_name!r} not found."

    # Format message with origin marker
    formatted_content = _format_inter_agent_message(sender_name=deps.name, content=content)

    # Ensure engine + registry are available (only present when built via AgentFactory with engine)
    if deps.engine is None or deps.agent_app_state_reg is None:
        return "Error: send_message is not configured for this agent (missing engine or registry)."

    # Spawn background task with delivery confirmation via Future
    loop = asyncio.get_running_loop()
    future: asyncio.Future[bool] = loop.create_future()

    task = asyncio.create_task(
        _deliver_message(
            agent_id=target_record.id,
            user_prompt=formatted_content,
            engine=deps.engine,
            agent_app_state_reg=deps.agent_app_state_reg,
            delivery_future=future,
        )
    )
    _BACKGROUND_TASKS.add(task)
    task.add_done_callback(_BACKGROUND_TASKS.discard)

    # Await delivery confirmation (lock acquired or failed)
    success = await future

    if success:
        return f"Message delivered to {target_name!r}."
    else:
        return f"Error: Agent {target_name!r} is busy and could not be reached."


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
            factory = AgentFactory(agent_id, agent_app_state_reg, session, engine)
            try:
                async with factory.build_agent_and_deps() as (agent, deps):
                    # Lock is held — signal delivery confirmation
                    delivery_future.set_result(True)
                    # Run agent to completion; discard yielded events (caller doesn't need them)
                    async for _ in run_stateful_agent(agent, deps, agent_app_state_reg[agent_id], user_prompt):
                        pass
            except AgentLockedError:
                if not delivery_future.done():
                    delivery_future.set_result(False)
    except Exception as e:
        logger.error("send_message background task failed for agent %r: %s", agent_id, e)
        if not delivery_future.done():
            delivery_future.set_exception(e)


# =============================================================================
# Tool Registry
# =============================================================================

TOOL_REGISTRY: dict[str, Callable | Tool[AgentDeps]] = {
    "memory_replace": memory_replace,
    "memory_insert": memory_insert,
    "duckduckgo_search": duckduckgo_search_tool(max_results=5),
    "web_fetch": web_fetch_tool(),
    "send_message": send_message,
}


def get_tools_for_agent(tool_names: list[str]) -> list[Callable | Tool[AgentDeps]]:
    """Return the list of tool callables for the given tool names.
    
    Args:
        tool_names: List of tool name strings to look up
        
    Returns:
        List of tool callables or Tool instances
        
    Raises:
        KeyError: If any tool name is not found in the registry
    """
    tools = []
    for name in tool_names:
        if name not in TOOL_REGISTRY:
            raise KeyError(f"Unknown tool: {name}")
        tools.append(TOOL_REGISTRY[name])
    return tools
