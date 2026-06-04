# Sonnet TUI Benchmark Notes
*Started: June 4, 2026 — Wide-net survey, first pass*

## Core Architectural Constraint

Our TUI must be a **thin HTTP client** to our Agent Home server (separate process, communicates via REST + SSE). This rules out any "embedded" TUI where the agent loop runs inside the TUI process.

Most existing tools are embedded — the TUI IS the agent. The client/server split we need is less common but exists.

---

## Candidates Surveyed

### 1. Solenoid (Google ADK) ⭐ MOST RELEVANT
**What it is:** FastAPI backend + Textual TUI client connected via AG-UI protocol (SSE-based).
**Stack:** Python throughout. FastAPI server, Textual TUI, SSE for streaming.
**Pattern:** TUI subscribes to SSE events from FastAPI backend for real-time updates. Exactly our intended architecture.
**Issues:**
- Coupled to Google ADK agent design — backend is AG-UI specific
- AG-UI protocol (from CopilotKit) is real but may be overkill; we can adopt SSE event naming
- Small project, may be sparse
**Verdict:** Best architectural match. Worth a real deep dive — can we strip the backend coupling from the TUI client layer? If yes, the Textual TUI side might be adaptable.
**Lookup:** Search "Solenoid Google ADK Textual TUI FastAPI SSE AG-UI"

---

### 2. tcode (hifar / Python version) ⭐ WORTH EXAMINING
**What it is:** Python project with FastAPI server (server.py) + TUI client + event bus bridge.
**Stack:** Python, FastAPI, runtui or basic TUI, SQLite persistence.
**Pattern:** Agent loop in separate process, TUI connects to server. Event bus bridges core to UI.
**Issues:**
- TUI designed for its own agent backend — not a generic HTTP client
- Multiple repos with confusing naming (hifar/tcode vs erickrus/tcode, different projects)
- runtui dependency (small framework, less mature than Textual)
**Verdict:** Architecture pattern worth stealing. Read server.py + bridge.py for event patterns. Don't adopt wholesale.

---

### 3. Textual v4 (TUI Framework) ⭐ LIKELY FOUNDATION
**What it is:** Python TUI framework by Will McGugan. Not an agent tool — just the framework.
**Key capability:** Added streaming Markdown support (MarkdownStream widget) in July 2025 (PR #5966). Async-native with Workers API for background tasks.
**Why it matters:**
- `MarkdownStream` widget handles partial text updates → typewriter effect for streaming
- Workers API: non-blocking background tasks that can consume SSE stream without blocking UI
- Rich integration: syntax highlighting, panels, tables all built-in
- Reactive/event-driven: clean separation of data model and display
- Resize-friendly by design
- Active development, large community
**httpx + Textual pattern:** `async with httpx.AsyncClient().stream("GET", url) as r: async for chunk in r.aiter_text()` inside a Worker → post events to app.
**Verdict:** This is probably our foundation if we build custom. The question is whether anything on top of it is worth adapting vs. building from scratch.

---

### 4. OpenCode ⭐ BEST PATTERN REFERENCE (wrong language)
**What it is:** TypeScript/Go AI coding assistant with clean client/server split.
**Stack:** Bun/TypeScript backend (Hono HTTP server), Go TUI subprocess with RPC.
**Pattern:** Worker Thread handles I/O, main thread for UI, RPC bridging. TUI and server communicate via structured JSON RPC. Exposes OpenAPI spec.
**Why it's interesting:** Probably the most mature example of exactly our desired architecture.
**Issues:** TypeScript + Go — can't adopt code, but the design patterns are excellent references.
**Verdict:** Read architecture docs for design patterns. Don't adopt code. Search "Dissecting OpenCode Complete Architecture Analysis" for a breakdown.

---

### 5. ICECODE
**What it is:** Python FastAPI backend (port 13210) with HTTP bridge to TypeScript frontends (Ink TUI, React desktop, web).
**Pattern:** Backend never imports TS. Frontend connects via HTTP/WebSocket only. Language-agnostic separation. Uses WebSocket + SSE for streaming.
**Verdict:** Shows clean language-agnostic HTTP-only API boundary. Useful if we want to support non-Python TUI clients later. For now, overkill — we're Python only.

---

### 6. Aider / aider-ce
**What it is:** Mature AI coding assistant. Embedded agent (no server). 
**Stack:** prompt_toolkit + Rich for TUI. Has MarkdownStream for streaming responses.
**Issues:** Embedded — agent is inside the TUI process. Doesn't fit our architecture.
**Verdict:** Don't adapt. But study its input/output patterns and MarkdownStream implementation. It's the most polished terminal AI UX for reference.

---

### 7. DeepSeek TUI / Deepy (Textual-based)
**What they are:** Experimental Textual UIs for AI assistants. Reasoning block streaming in Deepy.
**Issues:** Both embedded agents, not thin clients.
**Verdict:** Study for Textual widget patterns (especially reasoning block display). Don't adopt.

---

### 8. Claurst (Rust)
**What it is:** Rust TUI agent with Agent Client Protocol (ACP) compatibility. Multi-provider.
**Issues:** Rust stack. Not relevant.
**Verdict:** Skip.

---

### 9. Claude Code (Ink/React) / Letta Code (Ink/React)
**What they are:** TypeScript/Ink TUIs. James already familiar with both.
**Issues:** TypeScript. Not Python.
**Verdict:** Style reference and UX inspiration only. Not adaptable.

---

## Key Architectural Findings

### Thin-client pattern is not rare
OpenCode, tcode (FastAPI mode), Solenoid, ICECODE all split backend and UI into separate processes. The pattern exists; we're not pioneering it. This is a good sign — mature patterns exist to steal from.

### SSE is the standard for streaming to TUI clients
All modern tools use SSE or WebSocket (SSE preferred for unidirectional server-to-client). Our plan is correct.

### Textual v4 has what we need
`MarkdownStream` (streaming partials), `Workers` (async SSE consumption), `Reactive` (clean state → display). The July 2025 additions specifically addressed streaming display. Timing is good.

### Nobody does *exactly* our architecture in Python
Solenoid comes closest (Textual + FastAPI + SSE), but it's coupled to AG-UI/Google ADK. We'd be adapting the Textual client side while replacing the backend coupling with our own API calls.

---

## Preliminary Verdict

**Likely conclusion:** Build custom, but steal heavily.
- Framework: **Textual v4** (MarkdownStream, Workers, async-native)
- Streaming: **httpx AsyncClient** + Textual Worker → app events
- Pattern reference: **Solenoid** TUI-side, **OpenCode** architecture-side
- UX reference: **Aider** for input/output patterns, **Claude Code** for visual style

Whether Solenoid's Textual TUI side is strippable enough to fork or adapt is worth a quick look before committing to build-from-scratch.

---

## Deep Dives Still Needed

- [ ] Solenoid: Is the Textual client side separable from AG-UI? What does the SSE subscription look like?
- [ ] tcode (hifar): What does server.py + bridge.py actually look like? Event schema?
- [ ] Textual Workers + httpx SSE pattern: Any existing example repos using this exact combination?
- [ ] OpenCode architecture doc: What does the RPC layer look like? Any lessons for our SSE event schema?

---

## Decision Criteria (from requirements)

Requirements that most differentiate candidates:
1. **Real-time SSE streaming with typewriter** → Textual MarkdownStream + Worker pattern ✓
2. **Activity from non-TUI sources** → Polling or persistent SSE connection to server ✓ (design question)
3. **Tool approval flow** → Textual Input + custom widget needed (no off-shelf)
4. **Esc to halt** → SIGINT or API call; needs server support too
5. **Landing page** → Textual Screen switching ✓
6. **Image paste** → Hard regardless of framework. Textual has image widget (alpha)

---

## Jun 4 Update — OpenCode Revisit

**New lead from Opus 4.8:** A pydantic-ai project exists that implements the OpenCode server protocol and uses the stock OpenCode TUI unchanged. If confirmed:
- We implement the OpenCode server protocol on our pydantic-ai stack
- Use OpenCode TUI as-is (no TUI build from scratch)
- Agent Home management console (agent CRUD, landing page) built separately

This changes the calculus significantly. Prior survey may have underweighted OpenCode because we didn't know it had a real client/server protocol, not just a tight TUI↔backend coupling.

**Action:** Deep dive on OpenCode. Specifically:
1. What does the OpenCode server protocol actually look like? (endpoints, event schema, SSE format)
2. What is the pydantic-ai implementation? (name, repo, what it covers)
3. What does the stock OpenCode TUI cover vs. our requirements? (tool approval, multi-agent, Esc halt, etc.)
4. What would we need to implement vs. what we get for free?

---

*Notes by Sonnet. Full requirements: /workspace/git/Agent-Home/docs/Planning/tui_prototype.md*
