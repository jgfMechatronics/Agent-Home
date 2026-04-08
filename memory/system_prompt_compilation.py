"""
System prompt compilation — assembles memory blocks into the full system prompt.

compile_system_prompt(deps) — builds prompt from blocks, stores in agent.compiled_system_prompt
get_system_prompt(ctx) — returns cached prompt for Pydantic AI instructions param
"""
from agent.types import AgentDeps


async def compile_system_prompt(deps: AgentDeps) -> None:
    """
    Assemble memory blocks into compiled_system_prompt.
    
    - Fetches all blocks for deps.agent_id, ordered by position
    - Formats each block as XML with label, description, content
    - Prepends system_instructions
    - Stores result in agent.compiled_system_prompt
    - Updates agent.sys_prompt_compiled_at
    """
    raise NotImplementedError("Section 2.2 implementation pending")


async def get_system_prompt(ctx) -> str:
    """
    Return cached compiled_system_prompt for Pydantic AI instructions param.
    
    - Extracts agent_id from ctx.deps
    - Returns agent.compiled_system_prompt (empty string if NULL)
    - Does NOT recompile — that's deferred compilation
    """
    raise NotImplementedError("Section 2.2 implementation pending")
