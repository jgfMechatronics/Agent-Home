import logging
from typing import AsyncGenerator, TYPE_CHECKING

from pydantic_ai import Agent, AgentRunResultEvent, capture_run_messages
from pydantic_ai.messages import (
    AgentStreamEvent,
    ToolResultEvent,
    ModelRequest,
    ModelResponse,
    ToolCallPart,
    ToolReturnPart,
    UserPromptPart,
)
from pydantic_ai.mcp import MCPToolset
from pydantic_ai.toolsets.function import FunctionToolset
from pydantic_ai.tools import ToolDefinition

from agent.compaction import compact, is_compaction_needed
from agent.types import AgentAppState, AgentDeps
from messages.messages import deserialize_messages, format_system_alert, load_messages, persist_messages

if TYPE_CHECKING:
    from collections.abc import Sequence

    from pydantic_ai.toolsets import AbstractToolset

logger = logging.getLogger(__name__)

COMPACTION_RESUME_NOTICE = format_system_alert("Resuming after compaction. Context was trimmed to stay within limits.")


async def _extract_tool_definitions(toolsets: "Sequence[AbstractToolset]", agent_id: str) -> list[ToolDefinition]:
    tool_schemas: list[ToolDefinition] = []
    for ts in toolsets:
        if isinstance(ts, FunctionToolset):
            for tool in ts.tools.values():
                tool_schemas.append(tool.tool_def)
        elif isinstance(ts, MCPToolset):
            for mcp_tool in await ts.list_tools():
                tool_schemas.append(ToolDefinition(
                    name=mcp_tool.name,
                    description=mcp_tool.description,
                    parameters_json_schema=mcp_tool.inputSchema,
                ))
        else:
            logger.error(
                "Agent %s has an unsupported toolset type (%s); "
                "tool definitions for context reconstruction will be incomplete."
                "Supported toolset types are FunctionToolset and MCPToolset.",
                agent_id, ts.label,
            )
    return tool_schemas


def _count_adjacent_message_merges(messages: list) -> int:
    """Count adjacent message pairs that pydantic-ai will merge into one.

    pydantic-ai merges consecutive same-type messages in message_history, which
    reduces the effective list length and shifts indices in the captured messages
    list. Returns the total count of merges (i.e., how many messages will
    "disappear" due to merging).

    ModelRequest merge condition: instructions are compatible (neither has
    instructions, or they match). Adjacent requests with differing instructions
    are left separate.

    ModelResponse merge condition: both messages have no provider metadata
    (provider_response_id, provider_name, model_name all None). Responses from
    real API calls carry this metadata and are never merged.
    """
    def _request_will_merge(a: ModelRequest, b: ModelRequest) -> bool:
        return (not a.instructions or not b.instructions or a.instructions == b.instructions)

    def _response_will_merge(a: ModelResponse, b: ModelResponse) -> bool:
        return (a.provider_response_id is None and a.provider_name is None and a.model_name is None
                and b.provider_response_id is None and b.provider_name is None and b.model_name is None)

    count = 0
    for i in range(len(messages) - 1):
        a, b = messages[i], messages[i + 1]
        if isinstance(a, ModelRequest) and isinstance(b, ModelRequest) and _request_will_merge(a, b):
            count += 1
        elif isinstance(a, ModelResponse) and isinstance(b, ModelResponse) and _response_will_merge(a, b):
            count += 1
    return count


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

    TODO: Convert to agent.iter instead of agent.run_stream_events. Many benefits including getting rid of the funny capture_run_messages
    Also would allows us to consider switching to RunContext.tool_manager for capturing tool scheams which may be cleaner.
    agent.iter exposes a RunContext at this level I believe.
    """
    
    interrupted_by_compaction = True
    while interrupted_by_compaction:
        interrupted_by_compaction = False
        
        records = await load_messages(deps.session, deps.agent_id, start_seq_id=deps.context_window_start)
        message_history = deserialize_messages(records)
        
        # pydantic-ai merges adjacent same-type messages, which shifts indices in the captured messages list
        merge_adjustment = _count_adjacent_message_merges(message_history)
        # Track where new messages start for persistence; adjust for merges pydantic-ai will perform
        new_message_idx = len(message_history) - merge_adjustment

        with capture_run_messages() as messages:
            async with agent.run_stream_events(user_prompt=user_prompt,
                                                message_history=message_history,
                                                deps=deps) as stream:
                last_total_tokens_value = None

                async for event in stream:
                    yield event

                    # putting tool_schemas capture here should support agent self modifying attached tools.
                    # not that we have any means or tests for that yet
                    tool_schemas = await _extract_tool_definitions(agent.toolsets, deps.agent_id)
                    messages_to_persist = []
                    last_part_of_last_msg = messages[-1].parts[-1] if (messages and messages[-1].parts) else None

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
