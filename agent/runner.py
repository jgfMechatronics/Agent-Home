import logging
from typing import AsyncGenerator, TYPE_CHECKING

from pydantic_ai import Agent, AgentRunResultEvent, capture_run_messages
from pydantic_ai.messages import (
    AgentStreamEvent,
    ToolResultEvent,
    ModelRequest,
    ToolCallPart,
    ToolReturnPart,
    UserPromptPart,
)
from pydantic_ai.toolsets.function import FunctionToolset

from agent.compaction import compact, is_compaction_needed
from agent.types import AgentAppState, AgentDeps
from messages.messages import deserialize_messages, format_system_alert, load_messages, persist_messages

if TYPE_CHECKING:
    from collections.abc import Sequence

    from pydantic_ai.tools import ToolDefinition
    from pydantic_ai.toolsets import AbstractToolset

logger = logging.getLogger(__name__)

COMPACTION_RESUME_NOTICE = format_system_alert("Resuming after compaction. Context was trimmed to stay within limits.")


def _extract_tool_definitions(toolsets: "Sequence[AbstractToolset]", agent_id: str) -> "list[ToolDefinition]":
    tool_schemas = []
    for ts in toolsets:
        if isinstance(ts, FunctionToolset):
            for tool in ts.tools.values():
                tool_schemas.append(tool.tool_def)
        else:
            logger.error(
                "Agent %s has a non-FunctionToolset toolset (%s); "
                "tool definitions for context reconstruction will be incomplete.",
                agent_id, ts.label,
            )
    return tool_schemas


def _count_adjacent_model_request_merges(messages: list) -> int:
    """Count adjacent ModelRequest pairs that pydantic-ai will merge.
    
    pydantic-ai merges consecutive ModelRequests in message_history, which affects
    indexing when tracking new messages to persist. Returns the count of merges
    (i.e., how many messages will "disappear" due to merging).
    """
    return sum(1 for i in range(len(messages) - 1)
               if isinstance(messages[i], ModelRequest) 
               and isinstance(messages[i+1], ModelRequest))


async def _check_and_handle_cancel(
    agent_app_state: AgentAppState,
    deps: AgentDeps,
    tool_schemas: "list[ToolDefinition]",
) -> bool:
    if not agent_app_state.cancel_requested.is_set():
        return False
    cancel_notice = ModelRequest(parts=[UserPromptPart(
        content=format_system_alert("Turn cancelled by user.")
    )])
    await persist_messages(deps=deps, messages=[cancel_notice], tool_schemas=tool_schemas)
    await deps.commit_changes_refresh_agent_record()
    return True


async def run_stateful_agent(agent: Agent,
                             deps: AgentDeps,
                             agent_app_state: AgentAppState,
                             user_prompt: str) -> AsyncGenerator[AgentStreamEvent | AgentRunResultEvent[str], None]:
    """
    The core loop that drives the Pydantic AI agent, persists messages, handles cancellation, handles compaction.
    
    Yields raw AgentEvent objects from pydantic_ai.Agent.run_stream_events(). The caller is responsible for
    converting these to ServerSentEvent format if needed (typically via map_to_sse in the API layer).
    
    Mid-turn compaction: If context exceeds the compaction threshold during a turn, we compact and automatically
    resume with a fresh run. The original user_prompt is replaced with a resume notice on subsequent iterations.
    NOTE: TEMPORARY STAND IN PRIOR TO AGENTIC COMPACTION
    
    TODO: This function is currently tested through the handle_message route. We should consider moving the bulk of that
    testing into unit testing of this function
    """
    tool_schemas = _extract_tool_definitions(agent.toolsets, deps.agent_id)
    
    interrupted_by_compaction = True
    while interrupted_by_compaction:
        interrupted_by_compaction = False
        
        records = await load_messages(deps.session, deps.agent_id, start_seq_id=deps.context_window_start)
        message_history = deserialize_messages(records)
        
        # pydantic-ai merges adjacent ModelRequests, which shifts indices in the captured messages list
        merge_adjustment = _count_adjacent_model_request_merges(message_history)
        # Track where new messages start for persistence; adjust for merges pydantic-ai will perform
        new_message_idx = len(message_history) - merge_adjustment

        with capture_run_messages() as messages:
            async with agent.run_stream_events(user_prompt=user_prompt,
                                                message_history=message_history,
                                                deps=deps) as stream:
                last_total_tokens_value = None

                async for event in stream:
                    yield event

                    messages_to_persist = []
                    last_part_of_last_msg = messages[-1].parts[-1] if messages else None

                    if (isinstance(event, ToolResultEvent)
                        and isinstance(event.part, ToolReturnPart)
                        and not isinstance(last_part_of_last_msg, ToolReturnPart)
                        and isinstance(last_part_of_last_msg, ToolCallPart)):
                        # As of 1.97.0, pydantic-ai adds the ToolReturn to the captured messages list only
                        # when the next step starts, not when ToolResultEvent is yielded. Persist the tool pair atomically
                        # from the event data directly, so we don't lose it on cancel
                        # The last two gating conditions are a sanity check: Ensure the tool return is NOT available but the tool call IS
                        tool_return_msg = ModelRequest(parts=[event.part])
                        messages_to_persist = messages[new_message_idx:] + [tool_return_msg]
                    elif (len(messages) > new_message_idx) and not isinstance(last_part_of_last_msg, ToolCallPart):
                        messages_to_persist = messages[new_message_idx:]

                    if messages_to_persist:
                        total_tokens = await persist_messages(deps=deps, messages=messages_to_persist, tool_schemas=tool_schemas)
                        await deps.commit_changes_refresh_agent_record()
                        new_message_idx += len(messages_to_persist)
                        if total_tokens is not None:
                            last_total_tokens_value = total_tokens

                    if await _check_and_handle_cancel(agent_app_state, deps, tool_schemas):
                        return

                    # Mid-turn compaction check (after cancel — cancel supersedes compaction).
                    # Only fire at clean message boundaries: after a tool returns or at natural turn end.
                    if isinstance(event, (ToolResultEvent, AgentRunResultEvent)) and is_compaction_needed(last_total_tokens_value, deps.config):
                        await compact(deps, last_total_tokens_value)

                        if not isinstance(event, AgentRunResultEvent):
                            # agent wasn't finished, restart them with a fresh run.
                            user_prompt = COMPACTION_RESUME_NOTICE
                            interrupted_by_compaction = True
                            break
