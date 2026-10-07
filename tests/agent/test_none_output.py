"""
Behavioral tests for the None output union member (pydantic-ai allows_none path).

Agents' output types include None so empty and thinking-only model responses
complete the run with a None result instead of triggering pydantic-ai's output
retry ("Please return text or call a tool."). Models often finish their work
via tool calls and have nothing left to say — in upstream's words, "forcing a
retry just makes them produce unnecessary follow-up text."

Historical context: our pydantic-ai 1.97 → 2.54 upgrade crossed upstream #6403
(v2.8.0), which removed v1.97's text-recovery path (reusing discarded
commentary text as the final output). Since then, empty responses retry. The
None union member is upstream's designed replacement for exactly this case.

Follows the tool_output_limits test pattern: a bare Agent on the production
output types (AGENT_OUTPUT_TYPES from the factory) driven by a scripted
FunctionModel. Capabilities are not included — output completion behavior is
purely output_type-driven. Factory wiring is pinned separately in
test_factory.py (TestBuildAgentAndDeps.test_output_type_matches_agent_output_types).
"""
import pytest
from pydantic_ai import Agent, RunContext
from pydantic_ai.messages import (
    ModelMessage,
    ModelResponse,
    RetryPromptPart,
    TextPart,
    ThinkingPart,
    ToolCallPart,
)
from pydantic_ai.models.function import AgentInfo, FunctionModel

from agent.factory import AGENT_OUTPUT_TYPES


# --- Fixtures and helpers ---

DUMMY_TOOL_NAME = "dummy_tool"
DUMMY_TOOL_CALL = ToolCallPart(tool_name=DUMMY_TOOL_NAME, args="{}", tool_call_id="tc-1")
COMPLETION_TEXT = "Turn complete."


def _make_agent(steps: list[ModelResponse]) -> tuple[Agent, list[int]]:
    """Build a bare agent on AGENT_OUTPUT_TYPES whose scripted model consumes
    steps in order, one per model call. Returns (agent, call_log) where
    call_log grows by one element per model invocation."""
    call_log: list[int] = []

    async def scripted_model(messages: list[ModelMessage], info: AgentInfo) -> ModelResponse:
        call_log.append(1)
        return steps[len(call_log) - 1]

    agent = Agent(FunctionModel(scripted_model), output_type=AGENT_OUTPUT_TYPES)

    @agent.tool
    async def dummy_tool(ctx: RunContext) -> str:
        return "ok"

    return agent, call_log


def _retry_parts(messages: list[ModelMessage]) -> list[RetryPromptPart]:
    return [
        part
        for message in messages
        for part in getattr(message, "parts", [])
        if isinstance(part, RetryPromptPart)
    ]


def _shape(messages: list[ModelMessage]) -> list[tuple[str, tuple[str, ...]]]:
    """Structural fingerprint of a message chain: (message type, part kinds) per message.

    Avoids equality pitfalls with run-generated timestamps/usage/run ids while
    still asserting chain length, order, and part composition exactly."""
    return [(type(m).__name__, tuple(type(p).__name__ for p in m.parts)) for m in messages]


def _kinds(parts: list) -> tuple[str, ...]:
    return tuple(type(p).__name__ for p in parts)


# --- Tests ---

@pytest.mark.asyncio
@pytest.mark.parametrize(
    "final_response,expected_output",
    [
        (ModelResponse(parts=[]), None),  # truly empty — the Sonnet incident shape
        (ModelResponse(parts=[ThinkingPart(content="all done")]), None),  # thinking-only
        (ModelResponse(parts=[TextPart(content=COMPLETION_TEXT)]), COMPLETION_TEXT),  # control: text unaffected
    ],
    ids=["empty", "thinking-only", "text"],
)
async def test_final_response_shapes(final_response: ModelResponse, expected_output):
    """After a tool call, empty and thinking-only responses must complete the run
    with a None result (no retry); a text response still completes with its text.

    The model must be called exactly twice (tool call + final) — a third call
    would mean a retry fired. No RetryPromptPart may appear in history, and the
    history must be exactly the scripted messages, nothing extra."""
    agent, call_log = _make_agent([ModelResponse(parts=[DUMMY_TOOL_CALL]), final_response])

    result = await agent.run("go")

    assert result.output == expected_output
    assert len(call_log) == 2, f"Model called {len(call_log)}x — retry fired?"

    messages = result.all_messages()
    assert not _retry_parts(messages), f"Unexpected retry prompts in history: {_retry_parts(messages)}"
    # History must be exactly: user prompt, tool call, tool return, final response —
    # nothing extra (no retry prompts, no injected follow-ups).
    assert _shape(messages) == [
        ("ModelRequest", ("UserPromptPart",)),
        ("ModelResponse", ("ToolCallPart",)),
        ("ModelRequest", ("ToolReturnPart",)),
        ("ModelResponse", _kinds(final_response.parts)),
    ]


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "response,expected_output",
    [
        (ModelResponse(parts=[]), None),
        (ModelResponse(parts=[ThinkingPart(content="nothing to say")]), None),
    ],
    ids=["empty", "thinking-only"],
)
async def test_empty_first_response_completes_without_retry(response: ModelResponse, expected_output):
    """An empty/thinking-only response directly to the user prompt (no tool work
    at all) also completes with None — allows_none applies to any empty response,
    not just post-tool-call ones. Exactly one model call."""
    agent, call_log = _make_agent([response])

    result = await agent.run("go")

    assert result.output == expected_output
    assert len(call_log) == 1, f"Model called {len(call_log)}x — retry fired?"

    messages = result.all_messages()
    assert not _retry_parts(messages)
    assert _shape(messages) == [
        ("ModelRequest", ("UserPromptPart",)),
        ("ModelResponse", _kinds(response.parts)),
    ]
