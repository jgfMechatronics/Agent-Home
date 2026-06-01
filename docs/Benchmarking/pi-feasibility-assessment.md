# Pi Feasibility: Client-Side Tools + Agent Home State

**Date:** May 24, 2026  
**Author:** Sonnet  
**Question from Opus:** Can Pi be adapted to do client-side tool execution while Agent Home handles agent state/memory/history/context?

---

## Short Answer

**Tractable, but requires choosing one of three paths.** Pi and Agent Home have a fundamental architectural mismatch that can't be bridged without work. The right question is which bridge is worth building for Phase 3.

---

## The Mismatch

**Pi's architecture** (`pi-agent-core/src/agent-loop.ts`):
- `runLoop()` calls `streamSimple()` **directly to the LLM** 
- Then `executeToolCalls()` **locally** in Pi's process
- `Agent` class owns transcript, queues, tool list — all state lives client-side
- `AgentLoopConfig` has hooks: `convertToLlm`, `transformContext`, `beforeToolCall`, `afterToolCall`, `prepareNextTurn`

**Agent Home's architecture** (current):
- `POST /agents/{id}/messages` → server runs **full agent loop** (LLM + tools) via pydantic-ai
- SSE delivers text/thinking events + `AgentRunResultEvent` (done signal)
- Tools run **server-side**. No "pending tool call" concept — no way for client to pause and return results.

**The gap:** Pi expects to own the agent loop and run tools locally. Agent Home runs the entire loop server-side. Meeting in the middle (client tools + server state) requires a new protocol that doesn't exist in either codebase today.

---

## Three Paths

### Path A: Pi as Thin Client (Recommended for Phase 3)

Pi becomes a **display layer** — TUI/UX preserved, but agent loop removed entirely.

- Pi's `runLoop` / `executeToolCalls` stripped out
- Replaced with: HTTP call to `POST /agents/{id}/messages`, stream SSE into Pi's TUI
- Pi's bash/read/write tools become **server-side tools** registered in Agent Home
- No new protocol needed — Path A from the Letta Code analysis applies directly

**Tradeoffs:**
- ✅ Tractable: 1-2 weeks
- ✅ Tests API portability cleanly — second client after Letta Code
- ✅ Pi's TUI value preserved (streaming display, keyboard handling)
- ❌ Pi's local tool execution removed — tools now run in server process
- ❌ For us: tools are co-located (same machine), so this is acceptable
- ❌ For remote use cases: tools running server-side means server needs filesystem access

### Path B: Client-Side Tool Execution Protocol (Major Effort)

Server pauses agent run on tool calls, Pi executes locally, POSTs results back to resume.

Requires new Agent Home protocol:
- Run IDs (identify a paused run)
- Pause/resume mechanism in agent loop
- `POST /agents/{id}/runs/{run_id}/tool_results` endpoint (or similar)
- Pi polling or webhook to receive tool call requests

**Tradeoffs:**
- ✅ True client-side tool execution — Pi bash runs locally
- ✅ Aligns with Letta's `ClientToolSchema` pattern (proven design)
- ❌ 4-6 weeks minimum, significant design risk
- ❌ Not Phase 3 scope — this is Phase 4+ territory

**Note:** Letta already solved this with `ClientToolSchema` + `requires_approval`. If we implement it, the blueprint exists. But it's a different question from "what do we build in Phase 3."

### Path C: Pi Loop + Agent Home Persistence Only

Pi keeps its own agent loop + LLM calls. Agent Home used **only for persistence** (message history, memory blocks).

- Pi calls Agent Home's storage API after each turn: `POST /agents/{id}/messages` (no streaming, just persist)
- Pi retrieves history: `GET /agents/{id}/messages`
- Memory blocks: new CRUD endpoints on Agent Home

**Tradeoffs:**
- ✅ Medium effort: 2-3 weeks
- ✅ Pi's local tool execution fully preserved
- ❌ Agent Home becomes a dumb database, not an agent server
- ❌ Doesn't test API portability (Letta Code compatibility irrelevant)
- ❌ Two separate agent loops (Pi's and Agent Home's) — which one is authoritative?

---

## Recommendation

**Phase 3: Path A (thin client).** It answers the portability question: "does our API surface work for a second client?" If the answer is yes, we know the abstraction is right. Pi's TUI is legitimately good UX and worth preserving.

**Path B** is the right long-term answer if we ever want agents running tools on user machines (privacy-preserving agentic work, no server filesystem access needed). File it as a future protocol design — not now.

**Path C** is interesting only if we need Pi's specific local tool execution for something we can't replicate server-side. Not obvious we do.

---

## What Path A Requires

From Agent Home:
1. `POST /v1/conversations/{id}/messages` — send message, receive SSE (this is our Phase 3 core anyway)
2. `GET /v1/agents` — list agents for Pi's session selector
3. `POST /v1/agents` / `GET /v1/agents/{id}` — create/retrieve agent
4. `POST /v1/conversations` — create conversation
5. `GET /v1/conversations/{id}/messages` — history retrieval

From Pi:
1. Strip `runLoop` / `executeToolCalls` — replace with HTTP + SSE consumer
2. Register Pi's tools (bash, read, write, etc.) as Agent Home server-side tools
3. Wire Pi's TUI to render `LettaStreamingResponse` events

The tool registration question is the interesting one: do Pi's tools live in Agent Home's DB, or are they injected per-request? Letta Code uses server-managed tools. For Pi's bash tool, "server-side" means our agent process runs bash commands — fine for co-located dev use, needs thought for remote.

---

## Files Referenced

- `pi-agent-core/src/agent-loop.ts` — `runLoop()`, `executeToolCalls()`
- `pi-agent-core/src/agent.ts` — `Agent` class, state ownership
- `pi-coding-agent/src/core/agent-session.ts` — `AgentSession`, bash exec, compaction
- `Agent-Home/src/routes.py` — current `POST /agents/{id}/messages` implementation
- `letta-tool-architecture.md` — Letta's `ClientToolSchema` pause/resume pattern (Path B blueprint)
