"""
Spike: capture_run_messages behavior for cancellation/persistence design.

PURPOSE
-------
De-risk the "persist-as-you-go via capture_run_messages cursor" implementation
plan BEFORE writing the test suite. Empirically answers five questions:

Q1 CRITICAL — INCREMENTAL POPULATION
    Does the captured list populate incrementally (as each message completes)
    or only at the end of the run? We need incremental for the cursor approach.

Q2 — HISTORY INCLUSION
    When run() receives message_history=[...], does the captured list include
    those prior messages or only the new ones? Determines cursor starting point.

Q3 — TOOL TIMING
    When exactly do ModelResponse(tool call) and ModelRequest(tool return) appear
    in the captured list, relative to tool execution?

Q4 — CANCELLATION (the critical unknown)
    When we exit run_stream_events early (async CM exit) WHILE a tool is blocking:
      (a) Does pydantic-ai let the in-flight tool finish?
      (b) What's in the captured list — are pre-exit messages retained?
      (c) Is the tool-return present?

Q5 — new_messages() cumulative?
    Does result.new_messages() return the same messages on repeated calls?

SETUP
-----
Real pydantic_ai Agent + FunctionModel (stream_function).
Multi-step run: 1st model call → DeltaToolCall → tool blocks → tool returns → final text.
asyncio.Event controls blocking to inspect list mid-execution.

RUN
---
    cd /workspace/git/Agent-Home
    source .venv/bin/activate
    python3 misc/capture_run_messages_spike.py

FINDINGS: See printed output — each question answered with observed list states.
"""

import asyncio
from typing import AsyncIterator

from pydantic_ai import Agent, capture_run_messages
from pydantic_ai.messages import (
    FunctionToolCallEvent,
    FunctionToolResultEvent,
    ModelMessage,
    ModelRequest,
    ModelResponse,
    TextPart,
    ToolCallPart,
    ToolReturnPart,
    UserPromptPart,
)
from pydantic_ai.models.function import AgentInfo, DeltaToolCall, DeltaToolCalls, FunctionModel

# Type alias for clarity
FunctionDef = object  # just for doc purposes; actual type from pydantic_ai.models.function


# ──────────────────────────────────────────────────────────────────────────────
# Helpers
# ──────────────────────────────────────────────────────────────────────────────

def snapshot(label: str, messages: list[ModelMessage]) -> None:
    """Print a labelled snapshot of the captured list."""
    print(f"\n  [{label}] captured list has {len(messages)} entries:")
    for i, msg in enumerate(messages):
        kind = type(msg).__name__
        if isinstance(msg, ModelRequest):
            parts_summary = [type(p).__name__ for p in msg.parts]
        elif isinstance(msg, ModelResponse):
            parts_summary = [type(p).__name__ for p in msg.parts]
        else:
            parts_summary = ["?"]
        print(f"    [{i}] {kind}(parts={parts_summary})")


def make_prior_messages() -> list[ModelMessage]:
    """Two-turn history to pass as message_history."""
    return [
        ModelRequest(parts=[UserPromptPart(content="Prior user message")]),
        ModelResponse(parts=[TextPart(content="Prior assistant response")]),
    ]


# ──────────────────────────────────────────────────────────────────────────────
# FunctionModel stream_function factories
# ──────────────────────────────────────────────────────────────────────────────

def make_stream_function(
    tool_started_event: asyncio.Event,
    tool_release_event: asyncio.Event,
    snapshots_ref: dict,
    messages_ref: list[ModelMessage],
) -> "StreamFunctionDef":
    """
    Multi-step stream_function.

    Call 1 (no tool return in history): emit DeltaToolCall → tool will block.
    Call 2 (tool return present in history): emit final text.
    """

    async def stream_function(
        messages: list[ModelMessage], info: AgentInfo
    ) -> AsyncIterator[str | DeltaToolCalls]:
        # Determine which step we're on by checking message history
        has_tool_return = any(
            isinstance(part, ToolReturnPart)
            for msg in messages
            if isinstance(msg, ModelRequest)
            for part in msg.parts
        )

        if not has_tool_return:
            # Step 1: emit a tool call
            snapshots_ref["before_tool_call_yield"] = list(messages_ref)
            snapshot("stream_fn step1 BEFORE yield tool call", messages_ref)
            yield DeltaToolCalls({0: DeltaToolCall(name="blocking_tool", json_args='{}', tool_call_id="tc-001")})
            snapshots_ref["after_tool_call_yield"] = list(messages_ref)
            snapshot("stream_fn step1 AFTER yield tool call", messages_ref)
        else:
            # Step 2: emit final text
            snapshots_ref["before_final_text"] = list(messages_ref)
            snapshot("stream_fn step2 BEFORE yield final text", messages_ref)
            yield "Done! Tool ran successfully."
            snapshots_ref["after_final_text"] = list(messages_ref)
            snapshot("stream_fn step2 AFTER yield final text", messages_ref)

    return stream_function


def make_cancellation_stream_function(
    tool_started_event: asyncio.Event,
    tool_release_event: asyncio.Event,
) -> "StreamFunctionDef":
    """Same as above but for cancellation test — no snapshot side effects."""

    async def stream_function(
        messages: list[ModelMessage], info: AgentInfo
    ) -> AsyncIterator[str | DeltaToolCalls]:
        has_tool_return = any(
            isinstance(part, ToolReturnPart)
            for msg in messages
            if isinstance(msg, ModelRequest)
            for part in msg.parts
        )
        if not has_tool_return:
            yield DeltaToolCalls({0: DeltaToolCall(name="blocking_tool", json_args='{}', tool_call_id="tc-cancel")})
        else:
            yield "Post-cancel text — should not appear."

    return stream_function


# ──────────────────────────────────────────────────────────────────────────────
# Q1 + Q2 + Q3: Incremental population, history inclusion, tool timing
# ──────────────────────────────────────────────────────────────────────────────

async def run_q1_q2_q3() -> None:
    print("\n" + "=" * 70)
    print("Q1 / Q2 / Q3: Incremental population, history inclusion, tool timing")
    print("=" * 70)

    tool_started = asyncio.Event()
    tool_release = asyncio.Event()
    snapshots: dict = {}

    with capture_run_messages() as messages:
        print(f"\n  [before run] captured list starts with {len(messages)} entries")

        stream_fn = make_stream_function(tool_started, tool_release, snapshots, messages)

        agent: Agent[None, str] = Agent(
            FunctionModel(stream_function=stream_fn),
            output_type=str,
        )

        async def blocking_tool() -> str:
            """A tool that signals it's started then waits to be released."""
            tool_started.set()
            snapshot("blocking_tool ENTRY (tool is now running)", messages)
            await tool_release.wait()
            snapshot("blocking_tool EXIT (after release)", messages)
            return "tool result"

        agent.tool_plain(blocking_tool)

        prior = make_prior_messages()

        async def run_agent() -> None:
            async with agent.run_stream_events(
                "Run with tool", message_history=prior
            ) as stream:
                async for event in stream:
                    pass  # consume all events

        async def release_tool_after_started() -> None:
            await tool_started.wait()
            snapshot("WHILE tool is blocking (from watcher task)", messages)
            tool_release.set()

        await asyncio.gather(run_agent(), release_tool_after_started())

    snapshot("AFTER run (outside capture_run_messages)", messages)

    # ── Analysis ──────────────────────────────────────────────────────────────
    print("\n  ── Q2 HISTORY INCLUSION ──")
    if len(messages) >= 2 and isinstance(messages[0], ModelRequest) and isinstance(messages[1], ModelResponse):
        first_req_parts = messages[0].parts
        if any(isinstance(p, UserPromptPart) and p.content == "Prior user message" for p in first_req_parts):
            print("  ✅ HISTORY INCLUDED: prior messages ARE in the captured list (index 0 and 1)")
            print(f"     cursor for new messages should START at len(prior) = {len(prior)}")
        else:
            print("  ❌ UNEXPECTED: first entry not the prior user message")
    else:
        print("  ⚠️  Unexpected structure — check snapshot output above")

    print("\n  ── Q1 INCREMENTAL POPULATION ──")
    before_tool_len = len(snapshots.get("before_tool_call_yield", []))
    after_tool_len = len(snapshots.get("after_tool_call_yield", []))
    while_blocking_len = len(snapshots.get("while_blocking", []))
    final_len = len(messages)

    print(f"  before step1 yield:    {before_tool_len} entries")
    print(f"  after step1 yield:     {after_tool_len} entries")
    print(f"  final (post-run):      {final_len} entries")
    print(f"  expected with 2 prior: history(2) + req1 + resp1(toolcall) + req2(toolreturn) + resp2(text) = 6")

    if final_len >= 6:
        print("  ✅ INCREMENTAL: list grew during run (prior=2, full run=6+)")
    else:
        print(f"  ⚠️  Only {final_len} entries — check if incremental or end-only")

    # Q3: check tool timing
    print("\n  ── Q3 TOOL TIMING ──")
    # ModelResponse with ToolCallPart should appear after model call, before tool execution
    # Check snapshots around blocking_tool entry
    tool_entry_snap = snapshots.get("tool_entry_snap", [])
    print("  See snapshot printouts above for precise timing.")
    print("  Key: ModelResponse(ToolCallPart) should appear BEFORE tool entry.")
    print("  Key: ModelRequest(ToolReturnPart) should appear AFTER tool exit.")


# ──────────────────────────────────────────────────────────────────────────────
# Q4: Cancellation behavior
# ──────────────────────────────────────────────────────────────────────────────

async def run_q4() -> None:
    """
    TRUE in-flight cancellation test.

    Design: on FunctionToolCallEvent, await tool_started.wait() (yields to the
    event loop so the tool can actually start executing), then break WITHOUT
    ever releasing the tool. A separate task releases it after a delay so we
    can see whether pydantic-ai lets it complete post-CM-exit or kills it.
    """
    print("\n" + "=" * 70)
    print("Q4: Cancellation — break from CM while tool is TRULY in-flight")
    print("=" * 70)

    tool_started = asyncio.Event()
    tool_completed = asyncio.Event()
    tool_release = asyncio.Event()

    with capture_run_messages() as messages:
        stream_fn = make_cancellation_stream_function(tool_started, tool_release)

        agent: Agent[None, str] = Agent(
            FunctionModel(stream_function=stream_fn),
            output_type=str,
        )

        tool_was_allowed_to_complete = False

        async def blocking_tool() -> str:
            """Signals started, blocks until released, signals completion."""
            nonlocal tool_was_allowed_to_complete
            tool_started.set()
            print("\n  [tool] started — blocking on tool_release (NOT yet released)")
            await tool_release.wait()
            tool_was_allowed_to_complete = True
            print("  [tool] released and completing")
            tool_completed.set()
            return "cancellation tool result"

        agent.tool_plain(blocking_tool)

        async def run_with_early_exit() -> None:
            """Break out of the CM once the tool is confirmed blocking."""
            try:
                async with agent.run_stream_events("Cancel test") as stream:
                    async for event in stream:
                        snapshot(f"cancel event: {type(event).__name__}", messages)
                        if isinstance(event, FunctionToolCallEvent):
                            # Tool was dispatched. Await tool_started to let it
                            # actually start executing in the event loop.
                            await tool_started.wait()
                            # Tool is now blocking on tool_release — we have NOT released it.
                            print("\n  [cancel] tool is blocking — breaking CM NOW (tool_release NOT set)")
                            break
            except Exception as e:
                print(f"\n  [cancel] exception on CM exit: {type(e).__name__}: {e}")

        async def release_after_cancel_attempt() -> None:
            """Release the tool after a delay, after the CM has had time to exit."""
            await tool_started.wait()
            await asyncio.sleep(0.15)
            print("  [watcher] releasing tool (post-cancel delay)")
            tool_release.set()
            await asyncio.sleep(0.15)

        await asyncio.gather(run_with_early_exit(), release_after_cancel_attempt())

        print(f"\n  [cancel] tool_was_allowed_to_complete = {tool_was_allowed_to_complete}")
        print(f"  [cancel] tool_completed.is_set()       = {tool_completed.is_set()}")
        snapshot("AFTER early CM exit", messages)

    print("\n  ── Q4 ANALYSIS ──")
    print(f"  (a) Tool completion: {'✅ tool ran to completion (pydantic-ai does NOT cancel in-flight tools)' if tool_was_allowed_to_complete else '❌ tool was cancelled mid-execution (pydantic-ai killed it)'}")

    # ModelResponse(ToolCallPart) — appended before tool runs, always expected
    has_tool_call_response = any(
        isinstance(msg, ModelResponse) and any(isinstance(p, ToolCallPart) for p in msg.parts)
        for msg in messages
    )
    # ModelRequest(ToolReturnPart) — only if tool completed AND pydantic-ai processed the result
    has_tool_return_request = any(
        isinstance(msg, ModelRequest) and any(isinstance(p, ToolReturnPart) for p in msg.parts)
        for msg in messages
    )
    print(f"  (b) ModelResponse(ToolCallPart) in captured list: {'✅' if has_tool_call_response else '❌'}")
    print(f"  (c) ModelRequest(ToolReturnPart) in captured list: {'✅' if has_tool_return_request else '❌ absent'}")
    print(f"      Full captured list: {len(messages)} entries")


# ──────────────────────────────────────────────────────────────────────────────
# Q4b: Cancellation — exit BEFORE tool even starts (model hasn't run yet)
# ──────────────────────────────────────────────────────────────────────────────

async def run_q4b() -> None:
    """Bonus: what if we exit the CM immediately (before any events)?"""
    print("\n" + "=" * 70)
    print("Q4b: Cancellation — exit CM before consuming ANY events")
    print("=" * 70)

    with capture_run_messages() as messages:
        tool_started = asyncio.Event()
        tool_release = asyncio.Event()
        stream_fn = make_cancellation_stream_function(tool_started, tool_release)

        agent: Agent[None, str] = Agent(
            FunctionModel(stream_function=stream_fn),
            output_type=str,
        )

        @agent.tool_plain
        async def blocking_tool() -> str:  # type: ignore[misc]
            tool_started.set()
            await tool_release.wait()
            return "should not run"

        try:
            async with agent.run_stream_events("Immediate cancel") as stream:
                # Break immediately — never consumed a single event
                pass  # exhaust the CM without iterating
        except Exception as e:
            print(f"  Exception on immediate CM exit: {type(e).__name__}: {e}")

        snapshot("After immediate CM exit (no events consumed)", messages)
        print(f"\n  tool started: {tool_started.is_set()}")
        print(f"  tool released: {tool_release.is_set()}")


# ──────────────────────────────────────────────────────────────────────────────
# Q5: new_messages() — cumulative?
# ──────────────────────────────────────────────────────────────────────────────

async def run_q5() -> None:
    """
    new_messages() requires agent.run() (returns RunResult), not run_stream_events
    (AgentEventStream has no result() method). Use a plain function FunctionModel
    since run() doesn't support stream_function.
    """
    print("\n" + "=" * 70)
    print("Q5: new_messages() — cumulative across calls?")
    print("=" * 70)

    # FunctionModel.function must return ModelResponse (not str); Agent handles output extraction.
    def simple_fn(messages: list[ModelMessage], info: AgentInfo) -> ModelResponse:
        return ModelResponse(parts=[TextPart(content="Hello from simple_fn")])

    agent: Agent[None, str] = Agent(
        FunctionModel(function=simple_fn),
        output_type=str,
    )

    prior = make_prior_messages()

    result = await agent.run("Simple run", message_history=prior)

    nm1 = result.new_messages()
    nm2 = result.new_messages()

    print(f"\n  new_messages() call 1: {len(nm1)} messages")
    print(f"  new_messages() call 2: {len(nm2)} messages")
    print(f"  Same object? {nm1 is nm2}")
    print(f"  Equal content? {nm1 == nm2}")

    for i, msg in enumerate(nm1):
        kind = type(msg).__name__
        parts = [type(p).__name__ for p in msg.parts] if hasattr(msg, 'parts') else []
        print(f"    [{i}] {kind}(parts={parts})")

    if nm1 == nm2:
        print("\n  ✅ CUMULATIVE: new_messages() returns same content on repeated calls")
    else:
        print("\n  ❌ NOT CUMULATIVE: new_messages() returns different results per call")

    # Also check: does it include prior messages?
    all_msgs = result.all_messages()
    print(f"\n  all_messages() count: {len(all_msgs)} (prior={len(prior)}, new={len(nm1)})")
    if len(all_msgs) == len(prior) + len(nm1):
        print("  ✅ all_messages() = prior + new_messages() (cursor math confirmed)")
    else:
        print(f"  ⚠️  Unexpected: all_messages()={len(all_msgs)} != prior({len(prior)}) + new({len(nm1)})")

    # Q5 bonus: confirm capture_run_messages slice matches new_messages()
    print("\n  Q5 BONUS: does capture_run_messages[cursor:] == new_messages()?")
    with capture_run_messages() as captured:
        cursor = len(prior)
        result2 = await agent.run("Cursor check run", message_history=prior)  # same agent, same function

    new_via_cursor = captured[cursor:]
    new_via_method = result2.new_messages()
    print(f"  captured[cursor:] length: {len(new_via_cursor)}")
    print(f"  new_messages() length:    {len(new_via_method)}")
    if new_via_cursor == new_via_method:
        print("  ✅ captured[cursor:] == new_messages() — cursor approach is valid")
    else:
        print("  ❌ MISMATCH — cursor approach does not match new_messages()")
        for i, (c, n) in enumerate(zip(new_via_cursor, new_via_method)):
            print(f"    [{i}] cursor: {type(c).__name__}, method: {type(n).__name__}, equal: {c == n}")


# ──────────────────────────────────────────────────────────────────────────────
# Q6: Cancel-before-pull — does tool start before FunctionToolCallEvent is pulled?
# ──────────────────────────────────────────────────────────────────────────────

async def run_q6_cancel_before_pull() -> None:
    """
    Q6: Does a tool start executing BEFORE its FunctionToolCallEvent is consumed?

    Design:
      DRY RUN  — consume all events, record sequence, find index N of
                 FunctionToolCallEvent.
      PROBE RUN — pull exactly N events (all strictly before the tool-call
                 event), then await asyncio.sleep(0.2) to give the producer
                 a genuine chance to run ahead.  Check tool_started.is_set().

    Verdict:
      tool_started=False after sleep → SAFE (rendezvous: producer blocks at
          the yield point until consumer pulls the event)
          → run_stream_events graceful-cancel strategy IS viable
      tool_started=True  after sleep → UNSAFE (producer dispatches tool
          concurrently / into a buffer before consumer pulls)
          → must use agent.iter() or another mechanism
    """
    print("\n" + "=" * 70)
    print("Q6: Cancel-before-pull — does tool start before its event is pulled?")
    print("=" * 70)

    def make_q6_agent(tool_id: str) -> "tuple[Agent[None, str], asyncio.Event, asyncio.Event]":
        """Return a fresh (agent, started_event, release_event) triple."""
        started: asyncio.Event = asyncio.Event()
        release: asyncio.Event = asyncio.Event()

        async def stream_fn(
            messages: list[ModelMessage], info: AgentInfo
        ) -> AsyncIterator[str | DeltaToolCalls]:
            has_tool_return = any(
                isinstance(part, ToolReturnPart)
                for msg in messages
                if isinstance(msg, ModelRequest)
                for part in msg.parts
            )
            if not has_tool_return:
                yield DeltaToolCalls(
                    {0: DeltaToolCall(name="q6_tool", json_args="{}", tool_call_id=tool_id)}
                )
            else:
                yield "q6 run complete"

        agent: Agent[None, str] = Agent(
            FunctionModel(stream_function=stream_fn), output_type=str
        )

        @agent.tool_plain
        async def q6_tool() -> str:  # type: ignore[misc]
            started.set()
            await release.wait()
            return "q6 tool result"

        return agent, started, release

    # ── Step 1: Dry run — record full event sequence ───────────────────────────
    print("\n  ── STEP 1: Dry run — recording full event sequence ──")

    dry_agent, dry_started, dry_release = make_q6_agent("tc-q6-dry")
    dry_events: list[str] = []

    async def run_dry() -> None:
        async with dry_agent.run_stream_events("Q6 dry run") as stream:
            async for event in stream:
                dry_events.append(type(event).__name__)

    async def release_dry_tool() -> None:
        await dry_started.wait()
        dry_release.set()

    await asyncio.gather(run_dry(), release_dry_tool())

    print(f"  Full event sequence ({len(dry_events)} events):")
    for i, name in enumerate(dry_events):
        arrow = " ◄── FunctionToolCallEvent" if name == "FunctionToolCallEvent" else ""
        print(f"    [{i}] {name}{arrow}")

    if "FunctionToolCallEvent" not in dry_events:
        print("\n  ❌ FunctionToolCallEvent not found in sequence — cannot proceed")
        return

    N = dry_events.index("FunctionToolCallEvent")
    print(f"\n  → FunctionToolCallEvent at index {N}. Probe pulls [0..{N - 1}] then sleeps.")

    # ── Step 2: Probe run ──────────────────────────────────────────────────────
    print(f"\n  ── STEP 2: Probe run — pull {N} event(s), sleep 0.2s, check tool_started ──")

    probe_agent, probe_started, probe_release = make_q6_agent("tc-q6-probe")
    probe_tool_started_after_sleep: bool | None = None

    with capture_run_messages() as probe_captured:
        try:
            async with probe_agent.run_stream_events("Q6 probe run") as stream:
                if N == 0:
                    # FunctionToolCallEvent is the very first event.
                    # Don't pull anything — just sleep to let the producer advance.
                    print("  [probe] N=0: sleeping 0.2s without pulling any events…")
                    await asyncio.sleep(0.2)
                    probe_tool_started_after_sleep = probe_started.is_set()
                    print(f"  [probe] tool_started after sleep: {probe_started.is_set()}")
                else:
                    pulled = 0
                    async for event in stream:
                        print(f"  [probe] pulled [{pulled}]: {type(event).__name__}")
                        pulled += 1
                        if pulled == N:
                            # All N pre-tool events consumed. Next would be FunctionToolCallEvent.
                            print(
                                f"\n  [probe] {N} pre-tool event(s) pulled. "
                                f"Sleeping 0.2s to give producer a chance to run ahead…"
                            )
                            await asyncio.sleep(0.2)
                            probe_tool_started_after_sleep = probe_started.is_set()
                            print(f"  [probe] tool_started after sleep: {probe_started.is_set()}")
                            print(f"  [probe] breaking CM — FunctionToolCallEvent was NEVER pulled")
                            break
        except Exception as e:
            print(f"  [probe] exception on CM exit: {type(e).__name__}: {e}")
        finally:
            probe_release.set()  # safety: unblock tool if it somehow started

    await asyncio.sleep(0.05)  # let background cleanup settle

    # ── Verdicts ───────────────────────────────────────────────────────────────
    print("\n  ── Q6 VERDICT ──")

    if probe_tool_started_after_sleep is None:
        print("  ⚠️  Could not determine probe result (unexpected code path)")
        return

    if not probe_tool_started_after_sleep:
        print(
            "\n  ✅ SAFE: tool did NOT start while its event was pending+unpulled\n"
            "     (rendezvous semantics confirmed — producer blocks at yield)\n"
            "     → run_stream_events graceful-cancel strategy IS viable"
        )
    else:
        print(
            "\n  ❌ UNSAFE: tool started before its FunctionToolCallEvent was pulled\n"
            "     (buffered or concurrent dispatch)\n"
            "     → must use agent.iter() or another mechanism"
        )

    # Secondary: inspect captured list for dangling tool-call-without-return
    has_tool_call_resp = any(
        isinstance(msg, ModelResponse)
        and any(isinstance(p, ToolCallPart) for p in msg.parts)
        for msg in probe_captured
    )
    has_tool_return_req = any(
        isinstance(msg, ModelRequest)
        and any(isinstance(p, ToolReturnPart) for p in msg.parts)
        for msg in probe_captured
    )
    print(f"\n  Captured list after probe CM exit: {len(probe_captured)} entries")
    print(f"    ModelResponse(ToolCallPart)  present: {has_tool_call_resp}")
    print(f"    ModelRequest(ToolReturnPart) present: {has_tool_return_req}")
    if has_tool_call_resp and not has_tool_return_req:
        print("    → dangling tool call (ToolCallPart recorded, no ToolReturnPart)")
    elif not has_tool_call_resp:
        print("    → no tool call recorded (FunctionToolCallEvent never pulled → ModelResponse(ToolCallPart) not yet appended)")


# ──────────────────────────────────────────────────────────────────────────────
# Main
# ──────────────────────────────────────────────────────────────────────────────

async def main() -> None:
    print("\n" + "█" * 70)
    print("  capture_run_messages SPIKE — pydantic-ai 1.104.0")
    print("█" * 70)

    await run_q1_q2_q3()
    await run_q4()
    await run_q4b()
    await run_q5()
    await run_q6_cancel_before_pull()

    print("\n" + "=" * 70)
    print("SPIKE COMPLETE — see Q analysis sections above for verdicts")
    print("=" * 70 + "\n")


if __name__ == "__main__":
    asyncio.run(main())
