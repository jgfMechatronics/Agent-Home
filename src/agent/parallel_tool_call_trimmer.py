"""ParallelToolCallTrimmer capability — client-side enforcement of one tool call per model response.

BANDAID (Oct 2026). Persistence in run_stateful_agent can't handle multiple tool calls in a single
ModelResponse (partial persistence, then message sanitation eats the orphaned calls). The model
setting parallel_tool_calls=False is only a *request* to the provider: Anthropic enforces it
(tool_choice.disable_parallel_tool_use), but on OpenRouter it's forwarded as an OpenAI-style hint
that many upstream providers/serving stacks ignore.

This capability makes the guarantee locally: after each model response, every function ToolCallPart
after the first is dropped before the response enters message history or tool execution. The model
then sees that it made exactly one call, gets its return, and re-issues anything it still needs.

Caveat: streamed PartStart/PartDelta events for the dropped calls have already gone out to clients
by the time this hook runs (it fires after the stream completes). They are never executed or persisted.

REMOVE when the runner refactor handles parallel tool calls in persistence.
"""
import logging
from dataclasses import dataclass, replace

from typing import TYPE_CHECKING

from pydantic_ai import RunContext
from pydantic_ai.capabilities import AbstractCapability
from pydantic_ai.messages import ToolCallPart
from pydantic_ai.models import ModelRequestContext

if TYPE_CHECKING:
    from pydantic_ai.messages import ModelResponse

from agent.types import AgentDeps


logger = logging.getLogger(__name__)


@dataclass
class ParallelToolCallTrimmer(AbstractCapability[AgentDeps]):
    """Keep only the first function tool call in each model response.

    Non-tool-call parts (thinking, text) and native/builtin tool call parts are left untouched;
    native tool calls execute provider-side and never go through our tool execution or the
    tool-pair persistence path.
    """

    async def after_model_request(
        self,
        ctx: RunContext[AgentDeps],
        *,
        request_context: ModelRequestContext,
        response: "ModelResponse",
    ) -> "ModelResponse":
        kept_parts = []
        dropped: list[ToolCallPart] = []
        seen_tool_call = False
        for part in response.parts:
            if isinstance(part, ToolCallPart):
                if seen_tool_call:
                    dropped.append(part)
                    continue
                seen_tool_call = True
            kept_parts.append(part)

        if not dropped:
            return response

        logger.warning(
            "Model %r emitted %d parallel tool calls despite parallel_tool_calls=False; "
            "kept %r, dropped %r",
            response.model_name,
            len(dropped) + 1,
            next(p.tool_name for p in kept_parts if isinstance(p, ToolCallPart)),
            [p.tool_name for p in dropped],
        )
        return replace(response, parts=kept_parts)
