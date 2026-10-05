"""
Behavioral tests for the ToolOutputLimits spill capability as wired by the factory.

Constructs a bare Agent on the production capability pipeline (_build_capabilities)
and drives it with a scripted FunctionModel. An oversized tool return must NOT land
in message history: the model sees a handle + preview instead, the payload lands in
the LocalFileStore, and read_tool_result pages the original content back. Small
returns pass through untouched.

The store is deliberately NOT redirected to a tmp_path: default LocalFileStore
construction IS the production path, and we assert on the real default root
(per-user dir under system temp). Spill files accumulate across runs in the test
container; that is accepted (bounded by the store's cleanup_after TTL).
"""
import json
import os
import re
import tempfile
from pathlib import Path

import pytest
from pydantic_ai import Agent, RunContext
from pydantic_ai.messages import ModelMessage, ModelResponse, TextPart, ToolCallPart, ToolReturnPart
from pydantic_ai.models.function import FunctionModel
from pydantic_ai_harness.tool_output_limits import READ_TOOL_NAME

from agent.factory import TOOL_OUTPUT_SPILL_THRESHOLD_CHARS, _build_capabilities
from agent.types import AgentDeps


# --- Payload fixtures ---

PAYLOAD_MARKER = "the needle in the haystack"


def _payload() -> str:
    """A payload comfortably over the spill threshold, with a needle buried mid-file.

    The needle sits in the middle so it falls in the omitted section of the
    head/tail preview — if it ever shows up in history, the spill didn't happen.
    """
    lines = [f"line {i:05d}: {'x' * 60}" for i in range(400)]
    lines[200] = f"line 00200: {PAYLOAD_MARKER}"
    payload = "\n".join(lines)
    assert len(payload) > TOOL_OUTPUT_SPILL_THRESHOLD_CHARS
    return payload


SMALL_RETURN = "small payload, passes through"


# --- FunctionModel scripting ---

BIG_TOOL_NAME = "big_tool"
BIG_TOOL_CALL_ID = "tc-big-1"
READBACK_TOOL_CALL_ID = "tc-read-1"
COMPLETION = "done."

TOOL_CALL_STEP = ModelResponse(parts=[ToolCallPart(
    tool_name=BIG_TOOL_NAME, args='{"n": 1}', tool_call_id=BIG_TOOL_CALL_ID,
)])
COMPLETION_STEP = ModelResponse(parts=[TextPart(content=COMPLETION)])


class _ScriptedFunction:
    """FunctionModel non-streamed function consuming one step per model invocation.

    Each step is either a ModelResponse to return, or a callable receiving the
    live message history (list of ModelMessage) and returning a ModelResponse —
    used to script read_tool_result calls that depend on a handle spilled
    earlier in the same run. Running out of steps raises IndexError: fail loudly.
    """

    def __init__(self, steps: list):
        self._steps = steps
        self.invocation = 0

    def __call__(self, messages, info) -> ModelResponse:
        step = self._steps[self.invocation]
        self.invocation += 1
        if callable(step):
            step = step(list(messages))
        return step


def _build_agent(steps: list, deps: AgentDeps, *, payload: str | None = None, toolsets: list | None = None) -> Agent:
    """Agent on the production capability pipeline.

    Takes real AgentDeps (fixture-built): the production pipeline always runs
    with deps — CompactionWarner reads them on every response.

    Tool source is exactly one of: `payload` (registers a function tool
    returning it) or `toolsets` (MCP variant — callers script calls against
    the MCP tools).
    """
    assert (payload is None) != (toolsets is None), \
        "Provide exactly one of payload (function tool) or toolsets (MCP tools)"
    function = _ScriptedFunction(steps)
    agent = Agent(
        FunctionModel(function=function),
        deps_type=AgentDeps,
        capabilities=_build_capabilities(),
        toolsets=toolsets or [],
    )

    if payload is not None:
        @agent.tool
        async def big_tool(ctx: RunContext, n: int) -> str:
            return payload

    return agent


# --- Helpers ---

def _default_store_root() -> Path:
    """Mirror of LocalFileStore's default root (per-user dir under system temp)."""
    return Path(tempfile.gettempdir()) / f"pyai_harness_overflow-{os.geteuid()}"


def _extract_handle(content: str) -> str:
    match = re.search(r"stored to handle '([^']+)'", content)
    assert match, f"no spill handle found in content: {content[:200]!r}"
    return match.group(1)


def _find_tool_return(messages: list[ModelMessage], tool_name: str) -> ToolReturnPart:
    """Return the first ToolReturnPart for `tool_name` in a message history."""
    returns = [
        part
        for msg in messages
        for part in getattr(msg, "parts", [])
        if isinstance(part, ToolReturnPart) and part.tool_name == tool_name
    ]
    assert returns, f"no ToolReturnPart for {tool_name!r} in history"
    return returns[0]


# --- Spill behavior ---

class TestSpill:
    """Oversized tool returns are spilled, not persisted; small ones pass through."""

    async def test_oversized_return_spilled_not_persisted(self, agent_deps: AgentDeps):
        """An oversized return is replaced in history by a handle + preview.

        The payload must be absent from history, present in the store file the
        handle points at.
        """
        payload = _payload()
        agent = _build_agent([TOOL_CALL_STEP, COMPLETION_STEP], agent_deps, payload=payload)

        result = await agent.run("go", deps=agent_deps)

        ret = _find_tool_return(result.all_messages(), BIG_TOOL_NAME)
        content = str(ret.content)
        assert PAYLOAD_MARKER not in content, "payload leaked into message history"
        assert "Tool output too large" in content
        assert len(content) < TOOL_OUTPUT_SPILL_THRESHOLD_CHARS, \
            "spill stand-in should be small, not a near-copy of the payload"

        handle = _extract_handle(content)
        spill_file = _default_store_root() / handle
        assert spill_file.is_file(), f"spill file missing: {spill_file}"
        assert PAYLOAD_MARKER.encode() in spill_file.read_bytes()

    async def test_small_return_passes_through_untouched(self, agent_deps: AgentDeps):
        """A return under the threshold is exactly the tool's return value."""
        agent = _build_agent([TOOL_CALL_STEP, COMPLETION_STEP], agent_deps, payload=SMALL_RETURN)

        result = await agent.run("go", deps=agent_deps)

        ret = _find_tool_return(result.all_messages(), BIG_TOOL_NAME)
        assert ret.content == SMALL_RETURN


class TestReadback:
    """The model can page the spilled payload back via read_tool_result."""

    async def test_read_tool_result_pages_back_the_payload(self, agent_deps: AgentDeps):
        """A read_tool_result call with the spilled handle returns original content."""
        payload = _payload()

        def readback_step(messages: list[ModelMessage]) -> ModelResponse:
            """Second invocation: page back the spill via the capability's read tool."""
            spill = _find_tool_return(messages, BIG_TOOL_NAME)
            handle = _extract_handle(str(spill.content))
            args = json.dumps({
                "handle": handle, "offset": 0, "limit": 600,
                "from_end": False, "pattern": None,
            })
            return ModelResponse(parts=[ToolCallPart(
                tool_name=READ_TOOL_NAME, args=args, tool_call_id=READBACK_TOOL_CALL_ID,
            )])

        agent = _build_agent([TOOL_CALL_STEP, readback_step, COMPLETION_STEP], agent_deps, payload=payload)

        result = await agent.run("go", deps=agent_deps)

        readback = _find_tool_return(result.all_messages(), READ_TOOL_NAME)
        assert PAYLOAD_MARKER in str(readback.content)


# --- MCP variant ---

@pytest.fixture
def big_mcp_toolset():
    """Real in-process FastMCP server exposing a big-return tool — no HTTP, no mocking.

    Mirrors the in_process_mcp_toolset fixture pattern (tests/conftest.py), with a
    payload large enough to trigger the spill. Our production agents' tools are
    MCP tools, so the hook must fire for MCP results, not just function tools.
    """
    from fastmcp import FastMCP
    from pydantic_ai.mcp import MCPToolset

    mcp = FastMCP("big-tool-mcp-server")

    @mcp.tool()
    def mcp_big_tool(n: int) -> str:
        """Return a big payload."""
        return _payload()

    return MCPToolset(mcp)


class TestMCPSpill:
    """Oversized returns from MCP tools are spilled too."""

    async def test_oversized_mcp_return_spilled(self, agent_deps: AgentDeps, big_mcp_toolset):
        """An oversized MCP tool return is spilled exactly like a function tool's."""
        steps = [
            ModelResponse(parts=[ToolCallPart(
                tool_name="mcp_big_tool", args='{"n": 1}', tool_call_id="tc-mcp-1",
            )]),
            COMPLETION_STEP,
        ]
        agent = _build_agent(steps, agent_deps, toolsets=[big_mcp_toolset])

        result = await agent.run("go", deps=agent_deps)

        ret = _find_tool_return(result.all_messages(), "mcp_big_tool")
        content = str(ret.content)
        assert PAYLOAD_MARKER not in content, "MCP payload leaked into message history"
        assert "Tool output too large" in content

        handle = _extract_handle(content)
        spill_file = _default_store_root() / handle
        assert spill_file.is_file(), f"spill file missing: {spill_file}"
        assert PAYLOAD_MARKER.encode() in spill_file.read_bytes()
