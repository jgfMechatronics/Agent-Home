# Notes from accidental exploration of an unreleased dev branch
---

## File Trace: prompt_async → LLM Request

Traced June 4, 2026. Repo: `/workspace/git/misc/Agent Home Benchmarking Repos/opencode/`

```
1. packages/server/src/handlers/v2/session.ts
   → "prompt" handler
   → calls session.prompt({ sessionID, id, prompt, delivery, resume })
   → delivery param controls sync vs async (not separate routes)

2. packages/core/src/session.ts  — V2Session.prompt()
   → SessionInput.admit(db, ...)     ← durable write to DB first
   → enqueueWake(sessionID)          ← triggers execution
   → calls execution.wake(sessionID)

3. packages/core/src/session/execution/local.ts  — SessionExecution.wake()
   → routes to correct coordinator via LocationServiceMap
   → calls coordinator.wake(sessionID)
   (comment: "Future remote placement belongs here" — designed for multi-node)

4. packages/core/src/session/run-coordinator.ts  — concurrency manager
   → at most one drain per session key, with coalescing
   → wake() coalesces multiple wakes into one execution
   → run() vs wake() mode: explicit run joins/upgrades, wake coalesces
   → calls runner.run({ sessionID, force: mode === "run" })

5. packages/core/src/session/runner/llm.ts  — main agent loop
   → run(): checks pending work (steers/queues), loops up to MAX_STEPS=25
   → runTurn():
       - resolves model for session
       - loads context: store.context(sessionID)    ← FRESH EACH TURN ✓
       - builds: LLM.request({ model, messages: toLLMMessages(context), tools })
       - streams: llm.stream(request)
       - for each event: publishes to event bus
       - for tool-call events: executes via tools.settle(), publishes result
       - loops if tool calls need continuation

6. @opencode-ai/llm (packages/llm/)
   → actual LLM API call — NOT YET TRACED
```

---

## Key Observations (so far)

### The good
- **Context loaded fresh each turn** (line 141 in llm.ts) — not frozen like OpenHands. Architecture could support our per-turn memory injection.
- **Concurrency handled cleanly** — RunCoordinator is well-designed, coalesces wakes, prevents concurrent runs per session.
- **Durable-first design** — message written to DB before execution kicks off. Recovery-oriented.
- **Location abstraction** — designed for future multi-node/remote execution.
- **Tool execution pluggable** — via ToolRegistry.Service, settled per call.
- **Event bus throughout** — all activity published as events, which is what feeds the SSE stream.

### The concerning
- **V2 is significantly incomplete.** Major TODOs in `runner/llm.ts` (lines 25-74):
  - `[ ]` Load agent and effective permissions
  - `[ ]` Build provider/model-specific base instructions
  - `[ ]` Load project instructions (AGENTS.md, etc.)
  - `[ ]` Compact or summarize history when context pressure requires it
  - `[ ]` Apply steering reminders, plugin transforms
  - `[ ]` Bound provider retries
  - `[ ]` Durable status tracking
- **Several operations throw OperationUnavailableError:** `compact`, `wait`, `switchAgent`, `switchModel`, `shell`, `skill` — all unimplemented in V2.
- **System prompt assembly is the unbuilt part** — exactly the part we need most.

### Still unknown
- How does `store.context()` work? What does it currently put in the LLM context?
- How is the system prompt currently constructed (what little IS built)?
- `toLLMMessages()` — how does history get assembled into LLM messages?
- The `@opencode-ai/llm` package — how does it talk to Anthropic/etc.?
- How does persistence/storage work (SQLite? what schema?)
- What does agentpool (phil65) actually implement — full protocol or partial?

---

## Next Steps (paused here)

1. Continue trace: `store.context()` — what's in the context right now?
2. Read `toLLMMessages()` — how is history assembled?
3. Look at `packages/llm/` — LLM client abstraction
4. Look at agentpool (phil65) — what does a pydantic-ai OpenCode protocol impl look like?
5. Synthesize: framework adoption viable? What would we be building into vs. building ourselves?
