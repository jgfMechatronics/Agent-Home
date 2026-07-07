# capture_run_messages Spike — Findings
**Date:** June 11, 2026  
**pydantic-ai version:** 1.104.0  
**Script:** `misc/capture_run_messages_spike.py`  
**Author:** Sonnet

---

## Q1 — Incremental population: ✅ YES

The captured list grows during the run, not just at the end. Same object as `ctx.state.message_history`.

Timing (with 2 prior messages):
- Before stream_fn step1 yield: **3 entries** (2 prior + req1)
- After stream_fn step1 completes: **4 entries** (+ resp1 with ToolCallPart)
- At tool ENTRY: **4 entries**  
- After tool returns: **5 entries** (+ req2 with ToolReturnPart)
- After final text: **6 entries** (+ resp2 with TextPart)

**Cursor approach is valid.** Reading `captured[cursor:]` during the run gives you exactly the messages appended so far.

---

## Q2 — History inclusion: ✅ YES — HISTORY IS INCLUDED

Prior messages passed as `message_history` ARE in the captured list at indices `0..len(prior)-1`.  
Cursor for new messages must start at `len(prior)`.

---

## Q3 — Tool timing: confirmed

Exact sequence:
1. `ModelRequest(UserPromptPart)` appended **before** model call
2. `stream_fn` yields `DeltaToolCall` → `ModelResponse(ToolCallPart)` appended **after** `stream_fn` completes
3. Tool runs (captured list has 4 entries at tool ENTRY)
4. Tool returns → `ModelRequest(ToolReturnPart)` appended **after** tool EXIT
5. `stream_fn` step2 yields text → `ModelResponse(TextPart)` appended after step2 completes

Key for persistence: `ModelResponse(ToolCallPart)` is safe to persist as soon as it appears — tool hasn't run yet.

---

## Q4 — Cancellation (break from CM while tool in-flight): ❌ TOOL IS CANCELLED

When we `break` out of `run_stream_events` CM while a tool is blocking:

**(a) Tool completion:** ❌ — pydantic-ai injects `CancelledError` into the in-flight tool coroutine. The tool does **not** complete.

**(b) `ModelResponse(ToolCallPart)` in captured list:** ✅ — retained (was appended before tool ran)

**(c) `ModelRequest(ToolReturnPart)` in captured list:** ❌ — absent (tool never returned)

**Full captured list after cancel:** `[req(UserPromptPart), resp(ToolCallPart)]` — 2 entries only.

### Q4b — Exit CM before consuming any events

If we exit the CM before iterating any events: **0 captured entries**, tool never started.

---

## Q5 — new_messages() cumulative: ✅ YES

`new_messages()` returns the same content on repeated calls (returns a new list object each time, but equal content).  
`all_messages() == prior + new_messages()` — confirmed exactly.

**Bonus:** `captured[cursor:]` == `new_messages()` — the cursor approach produces identical results.

**Note on `FunctionModel` usage for Q5:** `FunctionModel(function=...)` requires the function to return `ModelResponse`, **not** a plain `str`. To use `agent.run()` and access `result.new_messages()`, the function must return `ModelResponse(parts=[TextPart(content="...")])`.

---

---

## Q6 — Cancel-before-pull (tool starts before FunctionToolCallEvent is consumed?): ✅ SAFE

**Strategy:** Dry run recorded full event sequence. Probe run pulled the N=2 events before `FunctionToolCallEvent`, slept 0.2s, then checked `tool_started`.

**Dry run event sequence (8 events):**
```
[0] PartStartEvent
[1] PartEndEvent
[2] FunctionToolCallEvent  ← N=2
[3] FunctionToolResultEvent
[4] PartStartEvent
[5] FinalResultEvent
[6] PartEndEvent
[7] AgentRunResultEvent
```

**Result:** `tool_started.is_set()` = **False** after 0.2s sleep.

**Verdict:**
```
✅ SAFE: tool did NOT start while its event was pending+unpulled
   (rendezvous semantics confirmed — producer blocks at yield)
   → run_stream_events graceful-cancel strategy IS viable
```

**Secondary finding (captured list after probe CM exit):** 2 entries.
- `ModelResponse(ToolCallPart)` present: **True**  ← surprising — see note below
- `ModelRequest(ToolReturnPart)` present: **False**
- → dangling tool call in captured list

**Important note on the dangling tool call:** Even though `FunctionToolCallEvent` was never pulled by the consumer, `ModelResponse(ToolCallPart)` was already appended to the captured list. This means pydantic-ai appends the `ModelResponse` to captured list *as part of producing the `FunctionToolCallEvent`* — before the consumer pulls it. The producer therefore holds the dangling `ModelResponse(ToolCallPart)` in the buffer, waiting for the consumer to pull. Our persistence logic must handle the case where we break before pulling `FunctionToolCallEvent`: `captured[cursor:]` will contain a `ModelResponse(ToolCallPart)` with no matching `ToolReturnPart`.

---

## Design implications

### Normal run
`capture_run_messages + cursor` is fully viable:
- Start cursor at `len(prior)` 
- Read `captured[cursor:]` at each persistence checkpoint to get new messages since last save
- `new_messages()` on the final `RunResult` gives the same slice

### Cancellation run
`capture_run_messages + cursor` is also viable, with caveats:

**Q4 (cancel while tool in-flight):**
- pydantic-ai **cancels** the in-flight tool (injects `CancelledError`)
- `captured` list has everything up to the cancel point, **excluding** the cancelled tool's return
- After CM exit: `[req(UserPromptPart), resp(ToolCallPart)]` — no ToolReturnPart

**Q6 (cancel before FunctionToolCallEvent is pulled):**
- Tool does NOT start (rendezvous confirmed)
- But `ModelResponse(ToolCallPart)` IS in the captured list (appended by producer before consumer pulls the event)
- After CM exit: same dangling pattern — ToolCallPart present, ToolReturnPart absent

**In both cancellation cases**, `captured[cursor:]` will contain a `ModelResponse(ToolCallPart)` without a matching `ModelRequest(ToolReturnPart)`. Persistence logic must handle this gracefully — either skip the dangling entry or persist it as-is (it accurately reflects partial execution).

### Graceful cancellation strategy (per Requirement 4 — let active tool complete)
Q6 confirms the strategy is **viable**:
- Set a cancel flag when cancellation is requested
- In the stream-consuming loop: after each `FunctionToolCallEvent`, check the cancel flag *before pulling the next event* — but only exit the CM at a safe boundary (after the matching `FunctionToolResultEvent` is consumed)
- This lets the in-flight tool complete naturally; the CM is only exited between tool executions, not mid-tool

The rendezvous guarantee means: if cancel arrives while `FunctionToolCallEvent` is unpulled, we can simply pull it (starting the tool), wait for `FunctionToolResultEvent`, then exit — no risk of orphaning a mid-execution tool.
