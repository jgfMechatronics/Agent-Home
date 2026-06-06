# Toad TUI Integration Plan for Agent Home

## Executive Summary

**Recommendation: Adopt toad as our TUI client.**

Toad natively supports 11 of 12 requirements, with the remaining one (image paste) on a reasonable path. Required work on our side: an ACP stdio-to-HTTP bridge CLI (the immediate Phase 1 spike), plus — deferred to later phases — mid-turn-interrupt control flow (cancellation and a server-side permission pause/resume) that we'd need regardless of TUI choice. The interrupt work shares machinery and rides on a planned `agent.iter()` migration; Phase 1 proves the happy path first on the existing route, with no interrupts.

---

## Conceptual Model: Sessions vs Agents

### The Mismatch

ACP assumes a separation between agents and sessions:
- **Agent** = identity/capabilities (e.g., "Claude Sonnet")
- **Session** = a conversation instance (create many, load/resume them)

Agent Home fuses these concepts:
- **Agent** = identity + conversation + memory + continuity (what ACP would call agent+session combined)

There is no "fresh conversation" with an Agent Home agent — the history is constitutive of who they are.

### Our Mapping

| ACP Concept | Agent Home Mapping |
|-------------|-------------------|
| Agent identity | Agent Home agent ID |
| Session | The agent itself — one per agent, continuous |
| `session/new` | "Connect to this agent" (returns the one persistent session) |
| `session/load` | Not used — we declare `loadSession: false` |

**Key decision: `loadSession: false`**

- Toad won't save session IDs locally or offer a resume UI
- Every launch calls `session/new`, we return the agent's continuous session
- Toad becomes a stateless thin client — we own persistence entirely
- Sidesteps the mismatch: we're saying "we handle continuity, you just connect"

### History Replay

ACP has no history fields in responses — history is entirely server-push via `session/update` notifications. Toad renders whatever `session/update` events arrive, with no gate on whether it was `session/new` or `session/load`.

**Implementation:**
```
Client: session/new
Server: {"sessionId": "agent-abc123"}           ← response
Server: session/update {user: "Hello"}          ← history replay
Server: session/update {assistant: "Hi!"}       ← history replay
...                                             ← last N turns
<agent now live, ready for input>
```

User opens toad → history appears → ready to type. Indistinguishable from a "resumed" session.

**What we replay: the agent's in-context messages.** Rather than an arbitrary "last N messages" (Letta's approach, which is decoupled from what the agent actually sees), we replay exactly the messages currently in the agent's context window. This is trivial for the spike — it's already our default message-history behavior — and it has a real UX virtue: it shows the user precisely what the agent can see. Worth considering as a permanent default, not just a spike shortcut.

**Display history ≠ agent memory.** The replayed messages are visual scrollback only. The agent's true continuity comes from core memory blocks + recall, which extend far beyond the in-context window. No scrollback pagination in the prototype — scrolling further back than the in-context messages is a non-goal.

---

## Requirements Assessment

### Native Support (No Work Required)

| Req | Description | Toad Implementation |
|-----|-------------|---------------------|
| 1 | Streaming typewriter | `AgentMessageChunk` streaming via `session/update` |
| 2 | External activity display | Server-push via JSON-RPC notifications — bridge forwards our SSE |
| 3 | Tool call display | `ToolCall` widget with status, expand/collapse, content types |
| 4 | Git-style diffs | `textual_diff_view.DiffView` — split/unified, syntax highlighted |
| 5 | Approve/Deny tools | `Question` widget with a/A/r/R keybindings |
| 7 | Esc to halt | Esc twice within 3s → `session/cancel` RPC |
| 8 | Landing page | Agent picker + project directory selector |
| 9 | Two display modes | Sidebar toggle (ctrl+B) is native; custom panels require Phase 3 fork |
| 11 | Resize-friendly | Textual framework handles natively |
| 12 | Non-blocking streaming | Prompt stays active; server handles concurrent sends |

### Requires Our Work

| Req | Description | Approach |
|-----|-------------|----------|
| 6 | Manage tool permissions | **Workaround:** Add custom sidebar panel exposing MCP server status via extended session state. OR wait for toad's "UI for MCP servers" roadmap item. |

### Currently Unsupported

| Req | Description | Approach |
|-----|-------------|----------|
| 10 | Image paste | **Fork required.** Protocol supports `ImageContent`, toad explicitly disables it. Plan: implement in Phase 3 fork, contribute upstream if quality is good. |

---

## Architecture

### The Bridge

Toad speaks stdio JSON-RPC. Agent Home speaks HTTP/SSE. We need a bridge CLI:

```
┌─────────────┐      stdio       ┌─────────────────┐      HTTP/SSE      ┌──────────────┐
│    Toad     │ ←──────────────→ │  agent-home acp │ ←────────────────→ │  Agent Home  │
│   (TUI)     │   JSON-RPC       │    (bridge)     │                    │   (server)   │
└─────────────┘                  └─────────────────┘                    └──────────────┘
```

### Bridge Responsibilities

1. **Stdin reader:** Parse JSON-RPC requests from toad
2. **HTTP client:** Translate to Agent Home API calls
3. **SSE subscriber:** Connect to Agent Home event stream
4. **Stdout writer:** Forward server events as JSON-RPC notifications
5. **History replay:** After `session/new`, push the agent's in-context messages as `session/update` events

### RPC Methods to Implement

| Method | Direction | Bridge Behavior |
|--------|-----------|-----------------|
| `initialize` | toad→bridge | Return capabilities with `loadSession: false` |
| `session/new` | toad→bridge | Connect to agent, replay history via `session/update`, return `sessionId` = agent ID |
| `session/prompt` | toad→bridge | `POST /agents/{id}/messages` (SSE response), forward as `session/update` |
| `session/cancel` | toad→bridge | `POST /agents/{id}/cancel` (Phase 1.5 — transport via stub, real mechanism rides deferred iter()) |
| `session/update` | bridge→toad | Forward from SSE stream |
| `session/request_permission` | bridge→toad | Forward from SSE, wait for response, POST approval back (Phase 1.6 — mechanism only) |

**Not implemented:**
- `session/load` — we declare `loadSession: false`, toad won't call this
- `session/set_mode` — defer unless we implement modes

### fs/terminal Callbacks

ACP lets the *client* expose `fs/read_text_file`, `fs/write_text_file`, and `terminal/*` for the agent to call — i.e., the client is the execution environment (e.g., Zed running an agent's edits against the user's open project).

**This is the opposite of our architecture.** Agent Home executes all tools server-side via MCP — that's how sandboxing works. The agent never reaches back to the client to touch files or run commands.

**Decision: toad is display-only.** Tool execution happens server-side via MCP; toad renders the resulting tool-call events from the SSE stream. The bridge never forwards an `fs/*` or `terminal/*` request to toad, because the server never makes one — so toad's local implementations are simply never invoked.

Toad may still advertise these client capabilities during `initialize`; that's harmless since they're never exercised.

### Permissions: Mechanism vs Policy

These have very different risk profiles and should be separated.

**The mechanism** — the mid-turn pause/resume round-trip:
1. Agent hits a tool needing approval → Pydantic AI defers the call → agent loop persists state and suspends
2. Server emits a "permission needed" event over SSE
3. Approval comes back via a response endpoint (SSE is one-way, so this needs a separate POST-back channel)
4. Agent resumes from suspended state

This is the architecturally novel control flow. Our stack currently assumes a turn runs start-to-finish (message in → stream out → done). Pausing *mid-turn*, holding state, and resuming is a different shape — and it's exactly what "auto-approve everything" papers over. It shares the mid-turn-suspend machinery with cancellation and depends on the same iter()-based loop ownership, so (see Implementation Phases) it lands in **Phase 1.6**, after the happy-path spike proves the approach and alongside the deferred iter() work — not in the Phase 1 happy path.

**The policy** — which tools need approval, and headless behavior:
- Memory tools (truly server-side) → always auto-approve
- Sandbox MCP (designated agent workspace) → YOLO auto-approve
- External MCP (host system, outside sandbox) → approval required
- Autonomous activation (self-wake / inter-agent, no client attached) → deny? defer? queue? **(tricky — deferred)**

This is configuration *on top of* the mechanism. Genuinely deferrable.

**Decision (Phase 1.6):** Build the mechanism, defer the policy. Designate *one* tool as approval-required (a real external MCP tool or a deliberate test tool) purely to exercise the full round-trip end to end (Pydantic AI → Agent Home API → bridge → toad → back). Everything else stays auto-approved. This validates the pause/resume architecture across all layers without designing the policy engine or solving the autonomous question yet. (Phase 1 ships with no permission round-trip at all — every tool auto-approves.)

### Agent Home Server Work

| Item | Description | Phase |
|------|-------------|-------|
| **History endpoint** | `GET /agents/{id}/history` — return the agent's in-context messages for replay | Phase 1 |
| **Cancel endpoint** | `POST /agents/{id}/cancel` — interrupt running turn. Transport provable via stub; real impl rides deferred iter() migration. | Phase 1.5 |
| **Standing event stream** | Persistent SSE for unsolicited events (self-wake, inter-agent) | Phase 2 |
| **Permission mechanism** | Mid-turn pause/resume round-trip + approval response endpoint. One tool designated approval-required to exercise it. | Phase 1.6 |
| **Permission policy** | Tool differentiation (sandbox vs external MCP), headless/autonomous behavior | Deferred |

---

## Agent Registration

Create `/workspace/git/toad/src/toad/data/agents/agenthome.dev.toml`:

```toml
identity = "agenthome.dev"
name = "Agent Home"
short_name = "agenthome"
url = "https://github.com/your-org/agent-home"
protocol = "acp"
author_name = "Agent Home Team"
type = "coding"
description = "Ethical AI agent framework with persistent memory and development infrastructure."
run_command."*" = "agent-home acp"

[actions.\"*\".install]
command = "uv tool install agent-home"
bootstrap_uv = true
description = "Install Agent Home CLI"
```

### Agent Discovery

How does toad's picker learn which agents exist?

**Spike:** Skip the picker entirely. Pass the agent ID (or name) to the bridge as a launch argument/config. The bridge sends the connection request; the server resolves it. Discovery UI is pure polish, deferred.

**Later:** Trivial to add a server endpoint that enumerates agent names/IDs, and to modify toad's picker to populate from it. Not a concern — bounded work whenever we want it.

---

## Implementation Phases

> **Working mode (Jun 6): exploratory spike.** This is the high-uncertainty "does the approach work at all" phase. We deliberately *skip* the heavyweight spec → reviewed-TDD → iterate loop. Sonnet and Opus dogfood a working prototype fast to retire integration risk; TDD is the implementers' discretion. Rigor returns for the keeper work (e.g. the iter() migration), not the spike.

### Phase 1: Happy path, live agent (prove the approach)
Basic TUI round-trip against a **live agent**, **no mid-turn interrupts** (no cancellation, no permission pause/resume). Uses the **existing `run_stream_events` route unchanged** — zero dependency on the deferred iter() work. This is pure ACP/Toad integration risk: the thing most likely to kill the approach and moot all the interrupt design. Retire it first.
- [ ] `agent-home acp` CLI skeleton
- [ ] `initialize` → return capabilities with `loadSession: false`
- [ ] `session/new` → connect to agent, replay history, return sessionId
- [ ] `session/prompt` → POST message, subscribe SSE, forward chunks
- [ ] `session/update` forwarding from SSE
- [ ] Basic error handling
- [ ] **Separate console for Agent Home status** (no fork yet)

### Phase 1.5: Cancellation
- [ ] **ACP cancel transport** — `session/cancel` wired toad → bridge, bridge catches it mid-stream (listen-while-streaming concurrency), acks. Provable with a **stub** server-side handler. *Caveat:* with a live agent under it, the stub will desync (acks "cancelled" while the real turn finishes + persists) — accepted as a known throwaway spike artifact, NOT real semantics.
- [ ] **Real agent-loop cancellation** — dispatch-boundary halt + partial persist. Rides on the **deferred iter() migration** (see Open Design Questions). Done properly via full TDD, not in the spike.

### Phase 1.6 — Permission mechanism  **[Jun 6 — flag for final read-through]**
The mid-turn pause/resume round-trip (one approval-required tool, full round-trip + POST-back endpoint). **Moved out of Phase 1** by the same logic as cancellation: it's novel mid-turn-*suspend* control flow that depends on the same iter()-based loop ownership we're deferring — and arguably harder than cancellation (suspend + resume vs. just halt). You can't "auto-approve everything papers over it" in the happy-path phase *and* prove the suspend/resume mechanism there. Confirm this re-scoping.

### Phase 2: Full Protocol Support
- [ ] **Permission policy** — tool differentiation (sandbox vs external MCP), headless/autonomous behavior
- [ ] Standing SSE for unsolicited events
- [ ] `session/set_mode` if we implement modes

### Phase 3: Toad Integration (Fork)
- [ ] Fork toad
- [ ] Add image paste support (`ImageContent` handling in UI)
- [ ] Add custom sidebar panels (MCP status, context stats)
- [ ] Contribute both upstream if quality is good

---

## Goals Assessment

| Goal | Status |
|------|--------|
| 1. Claude Code styling | ✅ Toad already matches this aesthetic |
| 2. Custom theming | ✅ Extensive settings system ("Almost everything may be tweaked") |
| 3. Reusable widgets | ✅ Already componentized — not our concern as users |

---

## Risks and Mitigations

| Risk | Mitigation |
|------|------------|
| **Python ≥3.14 required** | Toad requires bleeding-edge Python — verify deployment environment compatibility |
| Toad development direction diverges from our needs | AGPL allows forking; toad is well-structured |
| Bridge complexity underestimated | Start with spike, iterate. Core is ~4 RPC methods for Phase 1. |
| Image support becomes blocking | Fork toad to add (Phase 3), contribute upstream |
| ACP protocol changes | Protocol is versioned; toad maintains compatibility |
| Custom panels need fork | Phase 1 uses separate console; Phase 3 fork adds panels properly |
| History replay after `session/new` is unconventional | ACP convention, not protocol enforcement. Toad handles it fine (verified in source). |
| **Multi-client concurrency** | `loadSession: false` + sessionId = agent ID means every launch connects to the same persistent session. Two toad windows on one agent, or a window open while self-wake fires → potential interleaved/duplicate streams. **Acknowledged, punted for now.** We're already async; likely manageable with single-writer enforcement or fan-out broadcast. May surface complications around push notifications. Revisit when it bites. |

---

## Open Design Questions

### Cancellation Mechanism (active discussion)

Cancellation is **essential** and in scope for the spike — a coding TUI without a working stop button is unusable, and an agent burning tokens in the wrong direction with no recourse is unacceptable. It also shares "interrupt the running turn" machinery with the permission mechanism (permission = *interrupt and wait*; cancel = *interrupt and abort*), so getting the interrupt architecture right once serves both.

**The core challenge:** Our agent server runs on cooperative multitasking (asyncio). Unlike the permission flow — where the agent *voluntarily* reaches a yield point and waits for approval — cancellation must inject a halt *unexpectedly* into a running turn, gracefully, with no true preemption available. We can only act at await points.

**Governing invariant:** Never create a state where reality and the agent's recorded history disagree. A confused agent acting on a false world-model causes *compounding* damage — worse than any delay from waiting.

**RESOLVED — *what* cancel means (the dispatch-boundary model):**

Cancellation difficulty is not uniform; it depends on where in the turn the agent is. The clean boundary is **tool dispatch = the commit point:**
- *Mid-generation*, including while the agent is still *forming* a tool call (nothing dispatched yet) → discard the partial, clean cancel.
- *Right before dispatch* → last clean exit; halt without dispatching.
- *Tool dispatched / in-flight* → **let it finish, persist its real result, then halt.** We never abandon an in-flight tool, because abandoning desyncs reality from history (tool ran, agent thinks it didn't → orphaned call + side effects).
- *After tool result / between steps* → trivial clean checkpoint.

This maps directly onto a **flag-at-checkpoints** mechanism — check the cancel flag (a) between stream chunks, (b) immediately before each tool dispatch, (c) after each tool result. **Never** inside an in-flight tool await. "Let the tool finish" isn't a compromise — it's the direct consequence of placing checkpoints only at clean boundaries.

Consequences:
- **We do NOT need MCP cancellation support** (`notifications/cancelled`) — we designed around ever needing it. It only returns if we someday want true mid-tool interruption, which we likely never do.
- **No destructive-action emergency kill.** If you'd need a dirty-state mid-tool kill to prevent catastrophe, the failure already happened upstream (permissions / sandboxing / access). Designing for it optimizes for an already-lost scenario and tempts reliance on it over fixing the real layer. Excluded.
- **Agent must know it was cancelled:** after persisting the completed tool result and halting, inject a message part (e.g. a user/system message) into history so the next LLM call sees an explicit "you were interrupted by the user here" — legible to the agent, not just the UI. Easy.

**RESOLVED — *how* (the mechanism): use `agent.iter()`, check a cancel flag between nodes.**

We did the Pydantic AI control-flow pass (Jun 6, pydantic-ai 1.104.0). Findings:
- `agent.run_stream_events()` runs the agent in a **background `asyncio.Task`**, and its only cancellation hook is a hard `task.cancel()` — which can land *mid-tool-dispatch*, violating the dispatch-boundary invariant. So run_stream_events cannot give us clean cancellation.
- `agent.iter()` hands us the agent graph **node-by-node**. We own the loop, so we check the cancel flag at clean boundaries (between nodes / after each node's event stream), in-flight tools finish naturally, and `run.new_messages()` yields assembled messages for **partial persistence** for free. This is the flag-at-checkpoints model, realized concretely.
- The gating trick on run_stream_events (rendezvous channel + emit-then-execute proves "tool not started") technically works too, but gives no clean partial-persist — so iter() wins on persistence, not on cancellation soundness.

**The real cancellation mechanism is therefore DEFERRED to a separate `iter()` migration** — run through our full process (well-scoped, low-risk, but slow; mostly the time for James to learn the pydantic-node model). It is gated behind Phase 1 proving the whole approach viable.

**Critically — deferring is SAFE because the external event contract is preserved.** Empirically verified (`/workspace/git/misc/event_parity_test.py`): `iter()` + per-node `node.stream()` (plus a synthesized terminal `AgentRunResultEvent`) reproduces the *exact* event sequence `run_stream_events` emits — identical types and order, tool-call scenario included. So a Phase 1 bridge built on the existing `run_stream_events` route exposes the same SSE contract the eventual `iter()` route will. Switching later changes server internals, not the wire. Full findings: `/workspace/git/AgentMemory/Opus/agent-home-iter-cancellation-findings.md`.

**Status: *what* resolved (dispatch-boundary model); *how* resolved (iter() + between-node flag); real mechanism deferred behind Phase 1, safely, thanks to confirmed event-contract parity.**

---

## Decision

**Proceed with toad integration.**

The bridge is bounded (~4 RPC methods for Phase 1). Our `loadSession: false` + history replay approach sidesteps the session/agent conceptual mismatch cleanly — toad becomes a stateless view, we own persistence entirely.

Toad handles all the hard TUI problems (streaming, diffs, permissions, resize, styling). We focus on our differentiator: the memory system and agent loop.

Next step: Implement Phase 1 bridge as spike to validate the approach.
