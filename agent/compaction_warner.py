"""CompactionWarner capability — warns agent when approaching compaction threshold.

Injects a warning message when context tokens exceed 75% of soft_compaction_limit.
Fire-once behavior: only warns once per compaction cycle (flag resets when compaction runs).
"""
from dataclasses import dataclass

from typing import TYPE_CHECKING

from pydantic_ai import RunContext
from pydantic_ai.capabilities import AbstractCapability
from pydantic_ai.messages import UserPromptPart
from pydantic_ai.models import ModelRequestContext

if TYPE_CHECKING:
    from pydantic_ai.messages import ModelResponse

from agent.types import AgentDeps


COMPACTION_WARNING_THRESHOLD_FRACTION = 0.75

COMPACTION_WARNING_TEXT = """\
You are nearing compaction. Oldest messages at the beginning of your context will soon be evicted.
Please scan your context and look for anything that should be moved to an appropriate core memory block.
Start your scan from the OLDEST messages. Compaction starts with the oldest messages and works forward until the token ct target is reached.
You do NOT need to re-save anything you have already saved in this context, or anything already captured in core memory.
Remember, anything compacted away and not archived or saved to core-memory will be GONE, as if it never happened from your perspective!
Take your time with the memory consolidation! Your active task will still be waiting for you when you're done. Preserving memories is more important than resuming the task quickly.
Once you are finished with consolidation, you can resume your previous task. The compaction will be seamless from your perspective
If your memory is already up to date with your active context, you can ignore this message."""


@dataclass
class CompactionWarner(AbstractCapability[AgentDeps]):
    """Capability that warns the agent when approaching compaction threshold.
    
    Fires once when context tokens cross the warning threshold (75% of soft limit).
    Resets when compaction runs (via compaction_warning_fired flag on AgentRecord).
    """

    async def after_model_request(
        self,
        ctx: RunContext[AgentDeps],
        *,
        request_context: ModelRequestContext,
        response: "ModelResponse",
    ) -> "ModelResponse":
        """Check token usage after response and enqueue warning if threshold crossed.
        
        Uses ctx.enqueue() so the warning is delivered on the NEXT model request,
        also ensures it gets captured and persisted in message history.
        """
        # Already warned this compaction cycle
        if ctx.deps.compaction_warning_fired:
            return response
        
        # Calculate threshold
        threshold = int(ctx.deps.config.soft_compaction_limit * COMPACTION_WARNING_THRESHOLD_FRACTION)
        
        # Check if we've crossed the threshold
        total_tokens = ctx.usage.total_tokens if ctx.usage else 0
        if total_tokens >= threshold:
            # Set flag (will be persisted by runner's commit)
            ctx.deps.compaction_warning_fired = True
            
            # Enqueue warning — delivered on next request, available for persistence in message history
            ctx.enqueue(UserPromptPart(content=COMPACTION_WARNING_TEXT))
        
        return response
