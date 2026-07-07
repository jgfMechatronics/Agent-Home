"""Compaction functions — Section 3.3

Handles context window management by advancing the message history pointer
when token limits are exceeded.
"""
import logging

from agent.types import AgentConfig, AgentDeps
from memory.system_prompt_compilation import compile_system_prompt
from messages.messages import deserialize_messages, load_messages
from pydantic_ai.messages import RetryPromptPart, ToolReturnPart

logger = logging.getLogger(__name__)


def is_compaction_needed(total_tokens: int | None, config: AgentConfig) -> bool:
    """Check if compaction should be triggered based on token count.

    Returns True when total_tokens exceeds the soft_compaction_limit.
    Returns False and logs a warning when total_tokens is None, which indicates
    no usage data was available for the turn (unexpected in normal operation).
    """
    if total_tokens is None:
        logger.warning("is_compaction_needed called with total_tokens=None; skipping compaction")
        return False
    return total_tokens > config.soft_compaction_limit


async def compact(deps: AgentDeps, total_tokens: int) -> None:
    """Advance context_window_start to reduce context size.
    
    Estimates system prompt tokens from character count, calculates average
    tokens per message, and advances the pointer to hit the target percentage
    of soft_compaction_limit.
    
    Guarantees:
    - Never evicts the most recent 4 messages
    - No-op if 4 or fewer messages in context
    - Does NOT delete messages (pointer manipulation only)
    - Tool call/return pairs are kept atomic (never split across the boundary)
    - Calls compile_system_prompt after advancing pointer
    """
    messages = await load_messages(
        deps.session, deps.agent_id, start_timestamp=deps.context_window_start
    )

    # small context guard/avoid div by 0
    if len(messages) <= 4:
        return

    # TODO (low priority): we may eventually want a more sophisticated way to estiamte tokens, and some sort of 
    # check and loop on resulting in-context message token count to be more accurate if we find it necessary
    sys_tokens = len(deps.compiled_system_prompt or "") / 4
    msg_tokens = total_tokens - sys_tokens
    avg_tokens_per_msg = msg_tokens / len(messages)
    if avg_tokens_per_msg <= 0:
        return  # System prompt dominates token budget — can't estimate, skip this turn
    target_tokens = deps.config.compaction_target_percentage * deps.config.soft_compaction_limit
    n_msg_to_keep = max(4, int((target_tokens - sys_tokens) / avg_tokens_per_msg))

    if n_msg_to_keep >= len(messages):
        return

    # Ensure tool call/return pairs are never split across the compaction boundary.
    # If the candidate start message is a ToolReturnPart or RetryPromptPart, include
    # the preceding ToolCallPart message too.
    candidate = messages[-n_msg_to_keep]
    if candidate.type == "ModelRequest":
        [deserialized] = deserialize_messages([candidate])
        if any(isinstance(p, (ToolReturnPart, RetryPromptPart)) for p in deserialized.parts):
            n_msg_to_keep += 1

    # A ToolReturnPart/RetryPromptPart can never be the first message in a context window,
    # so in practice this is at most an efficiency thing (turns compact into a no op if n_msg_to_keep == len(messages))
    # rather than a proper safety guard
    if n_msg_to_keep >= len(messages):
        return

    deps.context_window_start = messages[-n_msg_to_keep].timestamp
    await compile_system_prompt(deps)
    await deps.commit_changes_refresh_agent_record()
