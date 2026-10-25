"""Unit tests for ParallelToolCallTrimmer capability (bandaid — remove with the runner refactor).

Integration coverage (trimmer through run_stateful_agent with a model that emits parallel calls)
lives in test_runner.py: TestMemoryToolsThroughRunner.test_parallel_calls_trimmed_to_first.
"""
from unittest.mock import Mock

import pytest
from pydantic_ai.messages import (
    ModelResponse,
    NativeToolCallPart,
    TextPart,
    ThinkingPart,
    ToolCallPart,
)

from agent.parallel_tool_call_trimmer import ParallelToolCallTrimmer


async def _trim(response: ModelResponse) -> ModelResponse:
    # ctx and request_context are unused by the trimmer
    return await ParallelToolCallTrimmer().after_model_request(Mock(), request_context=Mock(), response=response)


@pytest.mark.asyncio
class TestParallelToolCallTrimmer:

    async def test_keeps_first_tool_call_and_non_tool_parts(self):
        """Thinking/text survive in order; only the first function tool call is kept."""
        response = ModelResponse(
            parts=[
                ThinkingPart(content="plan"),
                TextPart(content="reading files"),
                ToolCallPart("read", {"path": "a"}, tool_call_id="c1"),
                ToolCallPart("read", {"path": "b"}, tool_call_id="c2"),
                ToolCallPart("grep", {"pattern": "x"}, tool_call_id="c3"),
            ],
            model_name="some-openrouter-model",
        )

        trimmed = await _trim(response)

        assert trimmed.parts == [
            ThinkingPart(content="plan"),
            TextPart(content="reading files"),
            ToolCallPart("read", {"path": "a"}, tool_call_id="c1"),
        ]
        # Response metadata is preserved — only parts change
        assert trimmed.model_name == "some-openrouter-model"

    async def test_text_after_dropped_call_is_kept(self):
        """Non-tool parts interleaved after a dropped call are retained."""
        response = ModelResponse(parts=[
            ToolCallPart("read", {"path": "a"}, tool_call_id="c1"),
            ToolCallPart("read", {"path": "b"}, tool_call_id="c2"),
            TextPart(content="trailing"),
        ])

        trimmed = await _trim(response)

        assert trimmed.parts == [
            ToolCallPart("read", {"path": "a"}, tool_call_id="c1"),
            TextPart(content="trailing"),
        ]

    @pytest.mark.parametrize("parts", [
        [TextPart(content="no tools")],
        [ThinkingPart(content="t"), ToolCallPart("read", {"path": "a"}, tool_call_id="c1")],
    ])
    async def test_zero_or_one_call_passes_through_unchanged(self, parts):
        """Responses that already comply are returned as the same object (no copy)."""
        response = ModelResponse(parts=parts)
        assert await _trim(response) is response

    async def test_native_tool_calls_untouched(self):
        """Provider-executed native tool calls aren't function tool calls and are never dropped."""
        native = NativeToolCallPart(tool_name="web_search", args={"q": "x"}, tool_call_id="n1")
        response = ModelResponse(parts=[
            native,
            ToolCallPart("read", {"path": "a"}, tool_call_id="c1"),
            NativeToolCallPart(tool_name="web_search", args={"q": "y"}, tool_call_id="n2"),
        ])

        assert await _trim(response) is response
