"""
Spike: does an immediate raise in step 1's stream generator propagate through
run_stream_events to the caller's error handling?

CONTEXT
-------
FunctionModelTestAgent._stream is called once per model invocation.
Typical run: 2 invocations.
  - Step 0 (call 0): yields DeltaToolCalls → tool runs → tool returns
  - Step 1 (call 1): currently yields "start..." THEN raises (tuple convention)

James/Opus question: can we simplify step 1 to raise IMMEDIATELY (no prior yield)?
The original "yield then raise" existed for peek() — but by step 1 the tool has
already completed, so peek() concern may not apply.

QUESTION
--------
If step 1's generator raises RuntimeError immediately (before yielding anything),
does that exception propagate through run_stream_events to the caller's try/except?

EXPECTED
--------
YES — by step 1, pydantic-ai has already processed a complete tool call cycle.
The second model call's exception should propagate normally regardless of peek().

RUN
---
    cd /workspace/git/Agent-Home
    uv run python misc/immediate_raise_spike.py
"""

import asyncio
from pydantic_ai import Agent
from pydantic_ai.models.function import FunctionModel, AgentInfo, DeltaToolCall, DeltaToolCalls


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _has_tool_return(messages) -> bool:
    from pydantic_ai.messages import ModelRequest, ToolReturnPart
    return any(
        isinstance(part, ToolReturnPart)
        for msg in messages if isinstance(msg, ModelRequest)
        for part in msg.parts
    )


TOOL_CALL = DeltaToolCalls({0: DeltaToolCall(name="dummy_tool", json_args='{"arg": "dummy"}', tool_call_id="tc-a1")})
COMPLETION = "Turn complete."
CRASH = RuntimeError("Simulated crash mid-stream")

DEFAULT_STEPS = [TOOL_CALL, COMPLETION]
CRASH_STEPS   = [TOOL_CALL, CRASH]


def make_stream_fn(steps: list):
    calls = []

    async def _stream(messages, info: AgentInfo):
        calls.append(list(messages))
        step = steps[len(calls) - 1]
        if isinstance(step, Exception):
            raise step
        yield step

    return _stream, calls


def make_agent(steps: list):
    stream_fn, calls = make_stream_fn(steps)
    agent: Agent = Agent(FunctionModel(stream_function=stream_fn))

    async def dummy_tool(arg: str) -> str:
        return "dummy_tool_output"

    agent.tool_plain(dummy_tool)
    return agent, calls


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------

async def test_happy_path():
    """Sanity check: normal run completes without exception."""
    print("\n--- test_happy_path ---")
    agent, calls = make_agent(DEFAULT_STEPS)
    events = []
    try:
        async with agent.run_stream_events("go", message_history=[]) as stream:
            async for event in stream:
                events.append(type(event).__name__)
    except Exception as e:
        print(f"  UNEXPECTED EXCEPTION: {e}")
        return

    print(f"  calls: {len(calls)}")
    print(f"  events: {events}")
    print(f"  PASS: completed without exception")


async def test_immediate_raise_step1():
    """Key question: does immediate raise in step 1 propagate to caller's try/except?"""
    print("\n--- test_immediate_raise_step1 ---")
    agent, calls = make_agent(CRASH_STEPS)
    caught_exception = None
    events = []

    try:
        async with agent.run_stream_events("go", message_history=[]) as stream:
            async for event in stream:
                events.append(type(event).__name__)
    except RuntimeError as e:
        caught_exception = e

    print(f"  calls: {len(calls)}")
    print(f"  events before exception: {events}")
    if caught_exception:
        print(f"  PASS: exception propagated to caller: {caught_exception}")
    else:
        print(f"  FAIL: exception was NOT propagated to caller's try/except")


async def test_yield_then_raise_step1():
    """Baseline: current tuple behavior — yield 'start...' THEN raise."""
    print("\n--- test_yield_then_raise_step1 ---")
    YIELD_THEN_CRASH_STEPS = [TOOL_CALL, ("start...", RuntimeError("crash"))]

    calls = []
    async def _stream(messages, info: AgentInfo):
        calls.append(list(messages))
        step = YIELD_THEN_CRASH_STEPS[len(calls) - 1]
        if isinstance(step, tuple):
            yield step[0]
            raise step[1]
        else:
            yield step

    agent: Agent = Agent(FunctionModel(stream_function=_stream))
    async def dummy_tool(arg: str) -> str:
        return "dummy_tool_output"
    agent.tool_plain(dummy_tool)

    caught_exception = None
    events = []
    try:
        async with agent.run_stream_events("go", message_history=[]) as stream:
            async for event in stream:
                events.append(type(event).__name__)
    except RuntimeError as e:
        caught_exception = e

    print(f"  calls: {len(calls)}")
    print(f"  events before exception: {events}")
    if caught_exception:
        print(f"  PASS (baseline): exception propagated: {caught_exception}")
    else:
        print(f"  FAIL (baseline): exception was NOT propagated")


async def main():
    print("=" * 60)
    print("Spike: immediate raise in step 1 vs yield-then-raise")
    print("=" * 60)
    await test_happy_path()
    await test_immediate_raise_step1()
    await test_yield_then_raise_step1()
    print("\nDone.")


if __name__ == "__main__":
    asyncio.run(main())
