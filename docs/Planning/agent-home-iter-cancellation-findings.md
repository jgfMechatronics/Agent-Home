# Agent Home: iter() migration + turn cancellation — findings & tentative plan

**Date:** Jun 6, 2026 | **pydantic-ai:** 1.104.0 | **Status:** iter() migration DEFERRED (see plan)

## Why this exists
Investigated how to add (a) safe turn **cancellation** and (b) **partial persistence**
to our FastAPI SSE route (`api/routes.py` `send_message`) for the Agent Home / Toad TUI work.
The route currently streams a turn via `agent.run_stream_events(...)` and persists ONLY at the
terminal event. Two TODOs motivate change: line-73 (persist on client disconnect) and line-80
(exceptions discard the whole turn).

Detailed source crawls live in:
- `/workspace/git/misc/run_stream_events_findings.md`
- `/workspace/git/misc/iter_investigation_findings.md`
- `/workspace/git/misc/event_parity_test.py` (empirical event-parity test, runnable)

---

## ESTABLISHED FACTS (source-verified + empirically tested)

### 1. DECISION: use `agent.iter()` for the real cancellation/persist implementation
- `run_stream_events` wraps `self.run()` in a **background asyncio.Task**; its ONLY cancellation
  hook is a hard `task.cancel()` that can land mid-tool-dispatch → violates our dispatch-boundary
  invariant (never abandon an in-flight tool → orphan/desync).
- `iter()` gives node-by-node control: cancel checkpoint BETWEEN nodes is clean; in-flight tools
  finish naturally; `run.new_messages()` is LIVE between nodes → assembled messages for partial
  persist WITHOUT reimplementing pydantic's delta accumulation.
- Pydantic docs explicitly endorse iter() for streaming "both events and output at every step."
- `run_stream_events` exposes NO assembled-messages handle short of the terminal `AgentRunResultEvent`
  — confirmed in docs ("piece together ... yourself from PartStartEvent/PartDeltaEvents").

### 2. EVENT PARITY: iter()+node.stream() reproduces run_stream_events EXACTLY
Empirically verified (event_parity_test.py, tool-call scenario, identical type sequences):
```
PartStartEvent, PartEndEvent, FunctionToolCallEvent, FunctionToolResultEvent,
PartStartEvent, FinalResultEvent, PartDeltaEvent×3, PartEndEvent, AgentRunResultEvent
```
- Per-node events are the SAME objects: `ModelResponseStreamEvent` (PartStart/PartDelta/PartEnd/
  FinalResult) from `ModelRequestNode.stream()`; `HandleResponseEvent` (FunctionToolCall/
  FunctionToolResult) from `CallToolsNode.stream()`.
- The ONLY gap: `AgentRunResultEvent` is NOT emitted by nodes — synthesize it from `run.result`
  after the loop (one line). With that, the external SSE contract is IDENTICAL.
- **Consequence:** the external contract does NOT change between a run_stream_events impl and an
  iter() impl. This is what makes the Phase 1 / 1.5 split safe (see plan).

### 3. Cancellation mechanics (the gating trick ALSO works, but we're not using it)
- anyio channel `max_buffer_size=0` (rendezvous) + emit-then-execute (_agent_graph.py:1670-1701)
  means a blocked `send(FunctionToolCallEvent)` PROVES the tool hasn't started. So even on
  run_stream_events you could gate cancellation at the tool boundary — but it gives no clean
  partial-persist, so iter() wins on persistence, not cancellation soundness.

### 4. Incidental gotchas (for whoever implements)
- Import nodes from PRIVATE `pydantic_ai._agent_graph` (`ModelRequestNode`, `CallToolsNode`);
  `End` from `pydantic_graph.basenode`. `AgentRunResultEvent` lives in `pydantic_ai.run` (NOT messages).
- `UserPromptNode` and `End` are yielded by `async for node in run` but produce no events — skip them.
- On early `break`, `run.result is None` — guard before synthesizing AgentRunResultEvent.
- Bare `async for node in run` skips `wrap_node_run` capability hooks (run.py:193-200) — use
  `run.next(node)` if we ever rely on capabilities.
- `FunctionToolResultEvent.result` is DEPRECATED → use `.part`. Check `map_to_sse` doesn't use `.result`.
- Route grows ~15 → ~40+ lines; two asymmetric streaming APIs (ModelRequestNode.stream yields an
  AgentStream you iterate; CallToolsNode.stream yields a HandleResponseEvent iterator directly).

---

## TENTATIVE PLAN (subject to change — details not yet committed)

### Phasing (Jun 6 decision with James)
- **Phase 1 — basic TUI, happy path, LIVE agent, NO cancellation.** Uses the EXISTING
  run_stream_events route unchanged. Zero dependency on iter(). Pure ACP/Toad integration risk —
  the thing most likely to "kill us anyway" and make cancellation design moot. Dogfood it.
- **Phase 1.5 — cancellation.** Two separable sub-risks:
  - (a) ACP cancel TRANSPORT (does Toad send session/cancel as expected; can the bridge catch it
    mid-stream — the listen-while-streaming concurrency). Provable with a STUB cancel handler.
    Caveat: with a live agent under it, the stub WILL desync (acks "cancelled" while the real turn
    finishes + persists). Accepted as a known throwaway spike artifact — NOT real semantics.
  - (b) real agent-loop cancellation (iter(), dispatch-boundary halt, partial persist) — the
    DEFERRED keeper work, done properly via full TDD process later.
- iter() migration itself = run through our FULL process (spec + reviewed TDD), NOT spiked. It's
  well-scoped, low-risk, high-value, but slow (mostly James's time to learn the pydantic-node model).
  Gated behind Phase 1 proving the approach viable at all.

### Tentative route sketch (iter() impl) — see iter_investigation_findings.md Q2 for full version
```python
async with agent.iter(user_prompt=..., message_history=..., deps=deps) as run:
    async for node in run:
        if isinstance(node, (ModelRequestNode, CallToolsNode)):
            async with node.stream(run.ctx) as s:
                async for event in s:
                    yield map_to_sse(event)
        if cancellation_requested:               # between-node checkpoint
            await persist(run.new_messages()); break
if run.result is not None:                       # synthesize terminal event
    await persist(run.result.new_messages())
    yield map_to_sse(AgentRunResultEvent(run.result))
```

### Governing invariant (unchanged)
Never create a state where reality ≠ the agent's recorded history. In-flight tool → let it finish,
persist real result, THEN halt. Cancelled agent must KNOW it was cancelled (inject a message part so
the next LLM call sees "interrupted by user here").
