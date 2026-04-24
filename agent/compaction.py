"""Compaction functions — Section 3.3

Handles context window management by advancing the message history pointer
when token limits are exceeded.
"""
from agent.types import AgentConfig, AgentDeps
from memory.system_prompt_compilation import compile_system_prompt


def is_compaction_needed(input_tokens: int, config: AgentConfig) -> bool:
    """Check if compaction should be triggered based on token count.
    
    Returns True when input_tokens exceeds the soft_compaction_limit.
    """
    raise NotImplementedError


async def compact(deps: AgentDeps, input_tokens: int) -> None:
    """Advance context_window_start to reduce context size.
    
    Estimates system prompt tokens from character count, calculates average
    tokens per message, and advances the pointer to hit the target percentage
    of soft_compaction_limit.
    
    Guarantees:
    - Never evicts the most recent 4 messages
    - No-op if 4 or fewer messages in context
    - Does NOT delete messages (pointer-only)
    - Calls compile_system_prompt after advancing pointer
    """
    raise NotImplementedError
