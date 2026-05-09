"""Compaction functions — Section 3.3

Handles context window management by advancing the message history pointer
when token limits are exceeded.
"""
from agent.types import AgentConfig, AgentDeps
from memory.system_prompt_compilation import compile_system_prompt
from messages.messages import load_message_history


def is_compaction_needed(input_tokens: int, config: AgentConfig) -> bool:
    """Check if compaction should be triggered based on token count.
    
    Returns True when input_tokens exceeds the soft_compaction_limit.
    """
    return input_tokens > config.soft_compaction_limit


async def compact(deps: AgentDeps, input_tokens: int) -> None:
    """Advance context_window_start to reduce context size.
    
    Estimates system prompt tokens from character count, calculates average
    tokens per message, and advances the pointer to hit the target percentage
    of soft_compaction_limit.
    
    Guarantees:
    - Never evicts the most recent 4 messages
    - No-op if 4 or fewer messages in context
    - Does NOT delete messages (pointer manipulation only)
    - Calls compile_system_prompt after advancing pointer
    """
    messages = await load_message_history(
        deps.session, deps.agent_id, start_timestamp=deps.context_window_start
    )

    # small context guard/avoid div by 0
    if len(messages) <= 4:
        return

    # TODO (low priority): we may eventually want a more sophisticated way to estiamte tokens, and some sort of 
    # check and loop on resulting in-context message token count to be more accurate if we find it necessary
    sys_tokens = len(deps.compiled_system_prompt or "") / 4
    msg_tokens = input_tokens - sys_tokens
    avg_tokens_per_msg = msg_tokens / len(messages)
    if avg_tokens_per_msg <= 0:
        return  # System prompt dominates token budget — can't estimate, skip this turn
    target_tokens = deps.config.compaction_target_percentage * deps.config.soft_compaction_limit
    n_msg_to_keep = max(4, int((target_tokens - sys_tokens) / avg_tokens_per_msg))

    if n_msg_to_keep >= len(messages):
        return

    deps.context_window_start = messages[-n_msg_to_keep].timestamp
    await deps.session.flush()
    await compile_system_prompt(deps)