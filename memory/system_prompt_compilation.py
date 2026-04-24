"""
System prompt compilation — assembles memory blocks into the full system prompt.

compile_system_prompt(deps) — builds prompt from blocks, stores in agent.compiled_system_prompt
get_system_prompt(ctx) — returns cached prompt for Pydantic AI instructions param
"""
from datetime import datetime, UTC

from agent.types import AgentDeps
from db.models import AgentRecord, MemoryBlockRecord
from memory.block_crud import get_blocks
from pydantic_ai import RunContext


def _format_block(block: MemoryBlockRecord) -> str:
    """Format a single memory block as XML."""
    return (
        f"<{block.label}>\n"
        f"<description>\n{block.description}\n</description>\n"
        f"<metadata>\n"
        f"- chars_current={len(block.content)}\n"
        f"- chars_limit={block.char_limit}\n"
        f"</metadata>\n"
        f"<content>\n{block.content}\n</content>\n"
        f"</{block.label}>"
    )


async def compile_system_prompt(deps: AgentDeps) -> None:
    """
    Assemble memory blocks into compiled_system_prompt.
    
    - Fetches all blocks for deps.agent_id, ordered by position
    - Formats each block as XML with label, description, metadata, content
    - Prepends system_instructions wrapped in XML
    - Stores result in agent.compiled_system_prompt
    - Updates agent.sys_prompt_compiled_at
    """
    session = deps.session
    
    # Load agent
    agent = await session.get(AgentRecord, deps.agent_id)

    if agent is None:
        raise ValueError(f"Agent{deps.agent_id} not found in DB during system prompt compilation attempt")
    
    # Load blocks in position order
    blocks = await get_blocks(session, deps.agent_id)
    
    # Build prompt: system_instructions first, then blocks
    parts = [f"<system_instructions>\n{agent.system_instructions}\n</system_instructions>"]
    
    for block in blocks:
        parts.append(_format_block(block))
    
    # Store result
    agent.compiled_system_prompt = "".join(parts)
    agent.sys_prompt_compiled_at = datetime.now(UTC)
    
    await session.flush()


async def get_system_prompt(ctx_or_deps : RunContext[AgentDeps] | AgentDeps) -> str:
    """
    Return cached compiled_system_prompt for Pydantic AI instructions param.
    
    - Extracts agent from ctx.deps OR from deps depending on what is provided.
      Needs to be able to accept ctx so pydantic AI can call it
    - Returns agent.compiled_system_prompt (empty string if NULL/empty)
    - Does NOT recompile — that's deferred compilation
    """
    if isinstance(ctx_or_deps, RunContext):
        deps = ctx_or_deps.deps
    elif isinstance(ctx_or_deps, AgentDeps):
        deps = ctx_or_deps
    else:
        raise TypeError("ctx_or_deps must be of type RunContext[AgentDeps] or AgentDeps")

    session = deps.session
    agent = await session.get(AgentRecord, deps.agent_id)
    
    return agent.compiled_system_prompt or ""
