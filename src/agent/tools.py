"""
Agent tool registry — Section 3.2

Maps tool name strings to callable tool functions for agent construction.
Memory tools raise ModelRetry on failure (for model self-correction).
TODO: Add a memory_delete which just wraps memory replace with an empty new content, for convenience
and associated tests
"""
from typing import Callable

from pydantic_ai import RunContext
from pydantic_ai.common_tools.duckduckgo import duckduckgo_search_tool
from pydantic_ai.common_tools.web_fetch import web_fetch_tool
from pydantic_ai.tools import Tool
from pydantic_ai.exceptions import ModelRetry

from agent.types import AgentDeps
from memory.block_crud import get_block, update_block, ContentExceedsLimitError
from prototype.iac.send_message import send_message


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


def _snap_to_next_line_start(content: str, anchor_end: int) -> int:
    """Return the insertion point: start of the line following the line containing the anchor's last character.

    The anchor may end mid-line or exactly at a line boundary; either way the
    insertion follows the line the anchor ends on.
    """
    nl = content.find("\n", max(anchor_end - 1, 0))
    return len(content) if nl == -1 else nl + 1


def _insert_at_line_boundary(content: str, insert_pos: int, new_text: str) -> tuple[str, int]:
    """Insert new_text at a line boundary and return (new_content, content_start_idx).

    Guarantees line separation on both sides: if the position doesn't follow a
    newline (end of an unterminated block), a junction newline is added first.
    new_text is inserted verbatim, then exactly one trailing newline is appended.
    """
    prefix = content[:insert_pos]
    if prefix and not prefix.endswith("\n"):
        prefix += "\n"
    new_content = prefix + new_text + "\n" + content[insert_pos:]
    return new_content, len(prefix)


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
        await update_block(deps, label, new_content, block=block)
    except ContentExceedsLimitError as e:
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
    """Insert text into a memory block at a line boundary.

    Insertion always lands on a fresh line — mid-line splicing is impossible by
    construction. The tool guarantees line separation on both sides of the insert;
    the agent controls blank-line separation by including leading/trailing
    newlines in content (e.g. content="\\nNew entry\\n" inserts a cleanly
    blank-separated entry).

    Args:
        ctx: Pydantic AI run context with AgentDeps
        label: The label of the memory block to edit
        content: The text to insert, verbatim (its newlines are preserved exactly)
        after: Where to insert. Use '<start>' for beginning, '<end>' for end,
               or any string to insert after the line containing that anchor
               (the anchor's line stays intact — nothing is ever split mid-line).
        occurrence: Which occurrence of anchor to insert after (1-indexed).
                   Required if anchor appears multiple times.

    Returns:
        Snippet of updated content on success.

    Raises:
        ModelRetry: On validation failure (block not found, anchor not found, etc.)
    """
    deps = ctx.deps

    if not content:
        raise ModelRetry("content cannot be empty")

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
        # Snap to the start of the line following the anchor's line
        insert_pos = _snap_to_next_line_start(block.content, anchor_pos + len(after))

    # Line-boundary insertion: verbatim content + exactly one trailing newline
    new_content, content_start = _insert_at_line_boundary(block.content, insert_pos, content)

    # handles char limit check and persistence
    try:
        await update_block(deps, label, new_content, block=block)
    except ContentExceedsLimitError as e:
        raise ModelRetry(str(e))

    return _compute_snippet(new_content, content_start, content)


async def memory_read(
    ctx: RunContext[AgentDeps],
    label: str,
    offset: int = 0,
    limit: int = 100,
) -> str:
    """Read a window of a memory block's CURRENT content, with computed line numbers.

    Use this to get exact current strings — for example, to retry a memory edit
    that failed because your view of the block (from the system prompt) is stale.
    This tool is NOT for normal recall: block content is already visible in your
    system prompt, and changes from your own edits are in your context. Only use
    it when you suspect significant drift AND that's causing a problem (a failed
    edit, many accumulated edits, etc.).

    Workflow: eyeball the approximate line number of your target from your
    system-prompt view (the block's lines_current metadata gives the total line
    count), read a window, page around by adjusting offset until you have the
    target region, then re-attempt your edit using the exact current strings.

    Returns lines verbatim (directly copyable) under a header giving the window
    range and total line count.

    Args:
        ctx: Pydantic AI run context with AgentDeps
        label: The label of the memory block to read
        offset: 0-indexed line to start reading from
        limit: Maximum number of lines to return

    Returns:
        A header line ([memory_read: block '<label>', lines X-Y of Z]) followed
        by the requested lines verbatim.

    Raises:
        ModelRetry: If the block is not found, offset/limit are invalid, or the
        requested window is out of range (message includes the total line count
        and requested range for paging).
    """
    deps = ctx.deps

    if offset < 0:
        raise ModelRetry("offset must be >= 0")
    if limit <= 0:
        raise ModelRetry("limit must be positive")

    block = await get_block(deps.session, deps.agent_id, label)
    if block is None:
        raise ModelRetry(f"block '{label}' not found")

    lines = block.content.splitlines()
    if not lines:
        return f"[memory_read: block '{label}' is empty]"

    window = lines[offset : offset + limit]
    if not window:
        raise ModelRetry(
            f"[memory_read: block '{label}' has {len(lines)} lines — "
            f"lines {offset + 1}-{offset + limit} are out of range. Adjust offset.]"
        )

    header = f"[memory_read: block '{label}', lines {offset + 1}-{offset + len(window)} of {len(lines)}]"
    return header + "\n" + "\n".join(window)


# =============================================================================
# Tool Registry
# =============================================================================

TOOL_REGISTRY: dict[str, Callable | Tool[AgentDeps]] = {
    "memory_replace": memory_replace,
    "memory_insert": memory_insert,
    "memory_read": memory_read,
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
