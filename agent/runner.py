from typing import AsyncGenerator

from pydantic_ai import Agent, AgentRunResultEvent, capture_run_messages
from pydantic_ai.messages import (
    AgentStreamEvent,
    FunctionToolResultEvent,
    ModelRequest,
    ModelResponse,
    TextPart,
    ToolCallPart,
    ToolReturnPart,
)

from agent.compaction import compact, is_compaction_needed
from agent.types import AgentAppState, AgentDeps
from messages.messages import deserialize_messages, load_messages, persist_messages


async def run_stateful_agent(agent: Agent,
                             deps: AgentDeps,
                             agent_app_state: AgentAppState,
                             user_prompt: str) -> AsyncGenerator[AgentStreamEvent | AgentRunResultEvent[str], None]:
    """
    The core loop that drives the Pydantic AI agent, persists messages, handles cancellation, handles compaction.
    
    Yields raw AgentEvent objects from pydantic_ai.Agent.run_stream_events(). The caller is responsible for
    converting these to ServerSentEvent format if needed (typically via map_to_sse in the API layer).
    
    TODO: This function is currently tested through the handle_message route. We should consider moving the bulk of that
    testing into unit testing of this function
    """
    records = await load_messages(deps.session, deps.agent_id, start_timestamp=deps.context_window_start)
    message_history = deserialize_messages(records)

    with capture_run_messages() as messages:
        async with agent.run_stream_events(user_prompt=user_prompt,
                                            message_history=message_history,
                                            deps=deps) as stream:
            new_message_idx = len(message_history)  # track what we have persisted already from messages
            last_total_tokens_value = None

            async for event in stream:
                yield event

                messages_to_persist = []
                last_part_of_last_msg = messages[-1].parts[-1] if messages else None

                if (isinstance(event, FunctionToolResultEvent)
                    and isinstance(event.part, ToolReturnPart)
                    and not isinstance(last_part_of_last_msg, ToolReturnPart)
                    and isinstance(last_part_of_last_msg, ToolCallPart)):
                    # As of 1.97.0, pydantic-ai adds the ToolReturn to the captured messages list only
                    # when the next step starts, not when FunctionToolResultEvent is yielded. Persist the tool pair atomically
                    # from the event data directly, so we don't lose it on cancel
                    # The last two gating conditions are a sanity check: Ensure the tool return is NOT available but the tool call IS
                    tool_return_msg = ModelRequest(parts=[event.part])
                    messages_to_persist = messages[new_message_idx:] + [tool_return_msg]
                elif (len(messages) > new_message_idx) and not isinstance(last_part_of_last_msg, ToolCallPart):
                    messages_to_persist = messages[new_message_idx:]

                if messages_to_persist:
                    total_tokens = await persist_messages(deps=deps, messages=messages_to_persist)
                    await deps.commit_changes_refresh_agent_record()
                    new_message_idx += len(messages_to_persist)
                    if total_tokens is not None:
                        last_total_tokens_value = total_tokens

                if agent_app_state.cancel_requested.is_set():
                    # NOTE: Ideally this would be a ModelRequest (user message), but pydantic-ai merges
                    # consecutive ModelRequests, breaking cursor-based persistence. Using ModelResponse
                    # avoids the merge. Consider switching back after migrating to agent.iter().
                    cancel_notice = ModelResponse(parts=[TextPart(
                        content="<system_message>Turn cancelled by user.</system_message>"
                    )])
                    await persist_messages(deps=deps, messages=[cancel_notice])
                    await deps.commit_changes_refresh_agent_record()
                    return

                if isinstance(event, AgentRunResultEvent):
                    if is_compaction_needed(last_total_tokens_value, deps.config):
                        await compact(deps, last_total_tokens_value)
