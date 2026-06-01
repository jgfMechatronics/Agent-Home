# Agent Home Strategic Plan

*High-level roadmap to first dogfooding milestone.*

---

## Goal: Dogfooding with Agentic Coding CLI

The first major milestone is having a Claude Code-like CLI connected to the Agent Home server. This unlocks:
- Hands-on testing and iteration of the server USING The server
- Real display format decisions (informed by actual usage)
- Memory system validation
- Motivation — seeing the work come to life

---

## Path to Dogfooding

### Phase 1: Minimal Stack Complete

Continue top-down from API layer. Complete the first-pass implementation plan, punting decisions that benefit from CLI context (tracked in `implementation_plan.md` → "Near Term Deferred Work").

**"Minimal stack" = everything in implementation plan Section 1-5, except explicitly deferred items.**

Key remaining work:
- Section 3.3: Compaction (James implementing)
- Section 3.4: Agent CRUD
- Section 4.1: Routes (in progress — gap filling)
- Section 4.3: App & Lifespan
- Section 5: Message persistence

### Phase 2: Simple Validation CLI

Spin up a temporary CLI — minimum viable interface to hit the API. **Not intended to carry forward.**

Scope:
- Streaming display of SSE events
- Raw display of ModelMessage content initially
- Basic parsing for sensible chat interface
- Enough to validate memory tools and server behavior

Ownership: Opus + Sonnet, minimal James input, no human code review (throwaway code).

### Phase 3: Agentic CLI Research & Selection

Once we can see the server working, research coding CLI options:

| Option | Notes |
|--------|-------|
| **Letta Code** | Same architecture (shell handles tool execution, connects to server for LLM). TypeScript. Has some cruft. |
| **Plandex** | TBD — needs evaluation |
| **OpenCode** | TBD — needs evaluation |
| **Pi** | TBD |
research other options as well

Decision criteria will emerge from Phase 2 hands-on experience.

UPDATE: This research lead to a new high level architecture design as specified in /docs/Architecture/architecture-evaluation.md
We didn't like how things were shaping up with integrating with any existing agentic coding CLIs.
We searched the space of existing solutions pretty thoroughly, and while there are some similar existing architecutres (most notably Openhands agent server), none met our needs as an existing platform to build on.


### Phase 4: Agent Harness (Current)

Build the agent harness greenfield on pydantic-ai + FastAPI. Key work items:

**TUI Client:**
- Display-only thin client (streaming + tool calls + approvals)
- Likely Python Textual — needs prototyping
- "Eager refresh" — shows agent activity regardless of who initiated (self-wake, inter-agent, etc.)
- HTTP+SSE connection to Core

**Tool Execution (MCP):**
- Evaluate existing MCP servers for filesystem/shell (Anthropic reference implementation, others)
- Possibly Integrate via pydantic-ai's `MCPToolset` (merged May 7, 2026)
- Gap: content search (grep) — Anthropic's server lacks it, may need supplemental solution

**Letta parity (our custom version) Features:**
- Archival memory (semantic search, tagging)
- Conversation search
- Inter-agent messaging
- Self-wake scheduling (asyncio background task)

**Agentic Compaction**


See `/docs/Architecture/architecture-evaluation.md` for connection decisions and `/docs/Architecture/architecture-vision.md` for requirements.

---

## Tool Execution Model

**Committed:** Memory tools run server-side (they're about agent state).

**Committed:** Coding tools run via MCP servers (isolated from Core process — "can't rm -rf your own hippocampus").

**Result:** Hybrid execution — "one stateful agent, many mech suits." The agent is the persistent reasoning entity; different CLIs/interfaces are different operational contexts it can inhabit.

Architecture decisions documented in `/docs/Architecture/architecture-evaluation.md`.

---

## Principles

1. **Top-down discovery** — Start from API, let needs surface, punt what needs more context.
2. **Punt wisely** — Track deferred decisions; don't let them accumulate silently.
3. **See it work** — Motivation matters. Get to something tangible before polishing.
4. **Throwaway is okay** — The simple CLI doesn't need to be good, just functional enough to validate.

---

*Last updated: May 29, 2026*
