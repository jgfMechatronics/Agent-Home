"""
System prompt compilation — assembles memory blocks into the full system prompt.

compile_system_prompt(deps, session) — builds prompt from blocks, stores in agent.compiled_system_prompt
get_system_prompt(ctx) — returns cached prompt for Pydantic AI instructions param
TODO: More instances of accessing db directly rather than going through the agent_record on deps.
"""
from __future__ import annotations

from typing import TYPE_CHECKING

from sqlalchemy.ext.asyncio import AsyncSession

if TYPE_CHECKING:
    from agent.deps import AgentDeps
from db.models import MemoryBlockRecord, utcnow
from db.connection import ReadOnlySession
from memory.block_crud import get_blocks
from pydantic_ai import RunContext


def _format_block(block: MemoryBlockRecord) -> str:
    """Format a single memory block as XML."""
    return (
        f"<{block.label}>\n"
        f"<description>\n{block.description}\n</description>\n"
        f"<metadata>\n"
        f"chars_current: {len(block.content)}\n"
        f"chars_limit: {block.char_limit}\n"
        f"lines_current: {len(block.content.splitlines())}\n"
        f"</metadata>\n"
        f"<content>\n{block.content}\n</content>\n"
        f"</{block.label}>"
    )


async def compile_system_prompt(deps: AgentDeps, session: AsyncSession) -> None:
    """
    Assemble memory blocks into compiled_system_prompt.

    - Fetches all blocks for deps.agent_id via the provided session
    - Formats each block as XML with label, description, metadata, content
    - Prepends system_instructions wrapped in XML
    - Stores result in agent.compiled_system_prompt
    - Updates agent.sys_prompt_compiled_at

    Caller is responsible for providing a session (typically from write_scope) —
    the signature makes this explicit and ensures reads and writes share a transaction
    and that any uncommitted changes in caller's session are reflected here
    """
    # Cast to ReadOnlySession even though its not, only way to use caller's session is violate RO rule here.
    blocks = await get_blocks(ReadOnlySession(session), deps.agent_id)

    # Build prompt: system_instructions first, then blocks
    parts = [f"<system_instructions>\n{deps.system_instructions}\n</system_instructions>"]

    for block in blocks:
        parts.append(_format_block(block))

    # Store result
    deps.compiled_system_prompt = "\n".join(parts)
    deps.sys_prompt_compiled_at = utcnow()


async def get_system_prompt(ctx: RunContext[AgentDeps]) -> str:
    """
    Return cached compiled_system_prompt for Pydantic AI instructions param.

    Registered as the pydantic-ai instructions callback — always called with RunContext.
    For direct access, use deps.compiled_system_prompt instead.

    - Returns agent.compiled_system_prompt (empty string if NULL/empty)
    - Does NOT recompile — that's deferred compilation
    """
    return ctx.deps.compiled_system_prompt or ""
