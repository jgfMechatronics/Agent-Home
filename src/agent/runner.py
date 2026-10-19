# ASSUMPTION: we always start fresh runs via run_stream_events(user_prompt=...) — we never use
# pydantic-ai's native run resumption (suspended runs / durable execution). That mechanism reads
# the system prompt from ModelRequest.instructions in stored history rather than the agent callback
# (_agent_graph.py: _get_history_instructions). Since persist_messages strips instructions before
# storing, a resumed run would receive instructions=None and lose its system prompt.
# If we ever adopt pydantic-ai's resume mechanism, stripping must be revisited first.

import httpx
import logging
from collections.abc import Sequence
from typing import AsyncGenerator, TYPE_CHECKING

from pydantic_ai import Agent, AgentRunResultEvent, RunCancelled
from pydantic_ai.usage import UsageLimits
from pydantic_ai.messages import (
    AgentStreamEvent,
    ToolResultEvent,
    ModelRequest,
    ModelResponse,
    ToolCallPart,
    ToolReturnPart,
    UserPromptPart,
)
from agent.compaction import compact, is_compaction_needed
from agent.timestamping import add_timestamp_prefix
from agent.types import AgentAppState, AgentDeps, AgentOutput, MCPConnError
from messages.messages import deserialize_messages, format_system_alert, load_messages, persist_messages

if TYPE_CHECKING:
    from pydantic_ai.toolsets import AbstractToolset

logger = logging.getLogger(__name__)

# pydantic-ai's default request_limit is 50, which proved too low for tool-heavy autonomous
# sessions. 100 gives headroom while still catching genuinely runaway agents.
# Per-agent configurability (like AgentConfig.retries) is the natural extension if 100 ever proves tight.
_REQUEST_LIMIT = 100

COMPACTION_RESUME_NOTICE = format_system_alert("Compaction complete, resuming run. Oldest context was trimmed to stay within limits. This notice does not mark the trim boundary, there are messages prior to this alert which were retained.\n" \
                                               "Your system prompt has been recompiled. Any memory edits up to this notice are now reflected in your core memory blocks.\n"
                                               "No action is required on your part, you can continue where you left off.", tags_on_newline=True)


async def _handle_run_cancellation(
    canceled_e: RunCancelled,
    deps: AgentDeps,
    toolsets: "Sequence[AbstractToolset]",
):
    cancel_notice = ModelRequest(parts=[UserPromptPart(
        content=format_system_alert("Turn cancelled by user.")
    )])
    await persist_messages(deps=deps, messages=[cancel_notice], toolsets=toolsets)
    await deps.commit_changes_refresh_agent_record()
    return True


async def run_stateful_agent(agent: Agent,
                             deps: AgentDeps,
                             agent_app_state: AgentAppState,
                             user_prompt: str) -> AsyncGenerator[AgentStreamEvent | AgentRunResultEvent[AgentOutput], None]:
    """
    The core loop that drives the Pydantic AI agent, persists messages, handles cancellation, handles compaction.
    
    Yields raw AgentEvent objects from pydantic_ai.Agent.run_stream_events(). The caller is responsible for
    converting these to ServerSentEvent format if needed (typically via map_to_sse in the API layer).
    
    Mid-turn compaction: If context exceeds the compaction threshold during a turn, we compact and automatically
    resume with a fresh run. The original user_prompt is replaced with a resume notice on subsequent iterations.
    NOTE: TEMPORARY STAND IN PRIOR TO AGENTIC COMPACTION
    
    TODO: This function is currently tested through the handle_message route. We should consider moving the bulk of that
    testing into unit testing of this function

    TODO: Convert to agent.iter instead of agent.run_stream_events. Many benefits including getting rid of the funny capture_run_messages
    Also would allows us to consider switching to RunContext.tool_manager for capturing tool scheams which may be cleaner.
    agent.iter exposes a RunContext at this level I believe.
    """

    # Agents have no clock — anchor every inbound user message with a wall-clock
    # timestamp (agents otherwise fabricate times in memory entries). Compaction
    # resume notices below are system alerts and intentionally unstamped.
    user_prompt = add_timestamp_prefix(user_prompt)
        
    records = await load_messages(deps.session, deps.agent_id, start_seq_id=deps.context_window_start)
    message_history = deserialize_messages(records)        

    try:
        async with agent.iter(user_prompt=user_prompt,
                              deps=deps,
                              message_history=message_history,
                              cancellation_token=agent_app_state.cancellation_token,
                              usage_limits=UsageLimits(request_limit=_REQUEST_LIMIT)) as run:
            try:            
                persisted_msgs_cursor = 0
                
                async for node in run:
                    # persist messages if required. TODO: break into fcn?
                    msgs_to_persist = run.new_messages()[persisted_msgs_cursor:]
                    # Avoid passing a tool call part into persit without return. Only need to check last part of last msg.
                    # If there's a lone call somewhere else, the history is degenerate and repair_messages() call in persist_messages() will clean up
                    if msgs_to_persist and (not isinstance(msgs_to_persist[-1].parts[-1], ToolCallPart)):
                        tool_defs = run.ctx.deps.tool_manager.tool_defs
                        await persist_messages(deps, msgs_to_persist, tool_defs)
                        await deps.commit_changes_refresh_agent_record() # TODO: Why don't we just commit inside persist
                        persisted_msgs_cursor += len(msgs_to_persist)

                    if Agent.is_model_request_node(node):
                        async with node.stream(run.ctx) as event_stream:
                            async for event in event_stream:
                                yield event
            except RunCancelled as canceled_e:
                tool_defs = run.ctx.deps.tool_manager.tool_defs
                _handle_run_cancellation(canceled_e, tool_defs)
                # No need to propagate, expected exc we handled internally
                return
    except RuntimeError as e:
        if isinstance(e.__cause__, httpx.ConnectError):
            raise MCPConnError(
                "At least 1 attached MCP server is unreachable — check status of attached MCP servers."
                ) from e
        raise
