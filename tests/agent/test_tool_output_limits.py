"""
Behavioral tests for the ToolOutputLimits spill capability as wired by the factory.

Constructs a bare Agent on the production capability pipeline (_build_capabilities)
and drives it with a scripted FunctionModel. An oversized tool return must NOT land
in message history: the model sees a handle + preview instead, the payload lands in
the LocalFileStore, and read_tool_result pages the original content back. Small
returns pass through untouched.

The tool source is parametrized (function toolset vs in-process MCP toolset) —
the hook must treat both identically since our agents' tools are MCP tools.

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
from fastmcp import FastMCP
from pydantic_ai import Agent, RunContext
from pydantic_ai.mcp import MCPToolset
from pydantic_ai.messages import ModelMessage, ModelResponse, TextPart, ToolCallPart, ToolReturnPart
from pydantic_ai.models.function import FunctionModel
from pydantic_ai.toolsets import FunctionToolset
from pydantic_ai_harness.tool_output_limits import READ_TOOL_NAME

from agent.factory import TOOL_OUTPUT_SPILL_THRESHOLD_CHARS, _build_capabilities
from agent.types import AgentDeps


# --- Payload fixtures ---

PAYLOAD_MARKER = "the needle in the haystack"


def _payload() -> str:
    """A payload ~3x the spill threshold, with a needle buried mid-file.

    The needle sits at the midpoint so it falls in the omitted section of the
    head/tail preview — if it ever shows up in history, the spill didn't happen.
    """
    filler = "x" * 60
    line = f"line {{i:05d}}: {filler}"
    line_count = 3 * TOOL_OUTPUT_SPILL_THRESHOLD_CHARS // len(line.format(i=0)) + 1
    lines = [line.format(i=i) for i in range(line_count)]
    needle = line_count // 2
    lines[needle] = f"line {needle:05d}: {PAYLOAD_MARKER}"
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
    
    TODO: We should consider if this _ScriptedFunction and the corresponding FunctionModel build from it could replace the FunctionModelTestAgent
    or at least inspire it.
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


# --- Toolset builders (parametrized tool sources) ---

def _function_toolset(payload: str) -> FunctionToolset:
    """FunctionToolset whose big_tool returns `payload`."""
    toolset = FunctionToolset()

    @toolset.tool
    async def big_tool(ctx: RunContext, n: int) -> str:
        return payload

    return toolset


def _mcp_toolset(payload: str) -> MCPToolset:
    """In-process FastMCP server whose big_tool returns `payload` — no HTTP, no mocking.

    Mirrors the in_process_mcp_toolset fixture pattern (tests/conftest.py). Our
    production agents' tools are MCP tools, so the spill hook must fire for MCP
    results, not just function tools.
    """
    mcp = FastMCP("big-tool-mcp-server")

    @mcp.tool()
    def big_tool(n: int) -> str:
        """Return a payload."""
        return payload

    return MCPToolset(mcp)


TOOLSET_BUILDERS = [
    pytest.param(_function_toolset, id="function"),
    pytest.param(_mcp_toolset, id="mcp"),
]


def _build_agent(steps: list, deps: AgentDeps, toolsets: list) -> Agent:
    """Agent on the production capability pipeline with tools from `toolsets`.

    Takes real AgentDeps (fixture-built): the production pipeline always runs
    with deps — CompactionWarner reads them on every response.
    """
    function = _ScriptedFunction(steps)
    return Agent(
        FunctionModel(function=function),
        deps_type=AgentDeps,
        capabilities=_build_capabilities(),
        toolsets=toolsets,
    )


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
    """Oversized tool returns are spilled, not persisted; small ones pass through.

    Parametrized over tool source: the hook must treat function tools and MCP
    tools identically (our agents' tools are all MCP).
    """

    @pytest.mark.parametrize("toolset_builder", TOOLSET_BUILDERS)
    async def test_oversized_return_spilled_not_persisted(self, agent_deps: AgentDeps, toolset_builder):
        """An oversized return is replaced in history by a handle + preview.

        The payload must be absent from history, present in the store file the
        handle points at.
        """
        payload = _payload()
        agent = _build_agent([TOOL_CALL_STEP, COMPLETION_STEP], agent_deps, [toolset_builder(payload)])

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

    @pytest.mark.parametrize("toolset_builder", TOOLSET_BUILDERS)
    async def test_small_return_passes_through_untouched(self, agent_deps: AgentDeps, toolset_builder):
        """A return under the threshold is exactly the tool's return value."""
        agent = _build_agent(
            [TOOL_CALL_STEP, COMPLETION_STEP], agent_deps, [toolset_builder(SMALL_RETURN)],
        )

        result = await agent.run("go", deps=agent_deps)

        ret = _find_tool_return(result.all_messages(), BIG_TOOL_NAME)
        assert ret.content == SMALL_RETURN


class TestReadback:
    """The model can page the spilled payload back via read_tool_result."""

    async def test_read_tool_result_pages_back_the_payload(self, agent_deps: AgentDeps):
        """A read_tool_result call with the spilled handle returns original content.

        Function toolset only: once spilled, the read path is capability-level
        and agnostic to where the payload came from (source parity is covered
        by the parametrized spill tests).
        """
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

        agent = _build_agent([TOOL_CALL_STEP, readback_step, COMPLETION_STEP], agent_deps,
                             [_function_toolset(payload)])

        result = await agent.run("go", deps=agent_deps)

        readback = _find_tool_return(result.all_messages(), READ_TOOL_NAME)
        assert PAYLOAD_MARKER in str(readback.content)
