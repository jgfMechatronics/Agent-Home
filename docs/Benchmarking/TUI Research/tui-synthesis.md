# TUI Research Synthesis
*Created: June 4, 2026*

## ACP (Agent Client Protocol) — Major Finding

During TUI research, we discovered ACP — an emerging standard protocol for exactly the thin-client ↔ agent-server connection we need.

### What ACP Is
- **Repo:** `agentclientprotocol/agent-client-protocol`
- **Purpose:** JSON-RPC protocol standardizing communication between code editors/TUIs ("clients") and coding agents ("servers")
- **Status:** Protocol version 1 (stable). Official SDKs for Python, TypeScript, Rust, Kotlin, Java.
- **Backers:** JetBrains (Anna Zhdan, Core Maintainer) + Block/Goose (Alex Hancock) leading the Transports Working Group

### Transport Status
**Current (stable):** stdio — client launches agent as subprocess, stdin/stdout JSON-RPC

**HTTP Transport (in progress):**
- RFD (spec) merged April 22, 2026
- Actively revised (latest commit May 5, 2026)
- Qwen implementing it (PR #4472, May 24, 2026 — Draft status)

**HTTP Transport Design:**
- Single `/acp` endpoint supports both Streamable HTTP and WebSocket upgrade
- HTTP/2 required for Streamable HTTP
- Two SSE streams: connection-scoped + session-scoped
- POST returns 202 immediately, responses delivered on SSE streams
- Session persistence built in: `session/new`, `session/load`, `session/resume`

### Key ACP Features
- `session/new`, `session/load`, `session/resume` — persistent sessions that survive restarts (aligned with our agent model)
- `PermissionOption`, `RequestPermissionOutcome` — tool approval built into schema
- MCP integration first-class — pass MCP server configs on session creation
- `SessionUpdate` — session-scoped updates pushed over SSE

### Strategic Implication
Instead of "pick a TUI to adopt," we could "implement ACP and get any compatible client":
- Zed (mentioned as target)
- Goose (Block is leading working group)
- JetBrains IDEs (JetBrains co-leading)
- Qwen (already implementing)
- Any future ACP-compatible client

### The stdio→HTTP Bridge Approach
Since HTTP transport isn't shipped yet, we could:
1. Implement ACP over HTTP on our server (the real work, not throwaway)
2. Use a temporary stdio→HTTP bridge so existing ACP stdio clients can connect

Pattern:
```
[ACP stdio client] ←stdio→ [bridge/proxy] ←HTTP/WS→ [our server]
```

Prior art: ageneral.ai mentioned building exactly this ("ACP-to-remote-agent bridge, stdio on IDE side, WebSocket through proxy to agent"). FastMCP does the inverse (stdio server → HTTP). A basic stdio↔HTTP JSON-RPC translator is ~1 day of Python if no off-the-shelf solution exists.

When native HTTP clients ship, we drop the bridge layer. Our HTTP implementation is the permanent artifact.

### ✅ Prelminarilty verified: Server-Initiated Pushes Supported

**Can ACP servers push unprompted events to clients? YES.**

From the HTTP transport spec:
> "All server→client messages (responses to requests **and unsolicited notifications**) are delivered via SSE streams"

The dual-stream model provides two channels:
- **Connection-scoped stream**: "any server-initiated messages not tied to a specific session" — perfect for self-wake notifications
- **Session-scoped stream**: "session update notifications, server-to-client requests" — perfect for ongoing activity display

This means our requirements are fully supported:
- **Self-wake**: Agent pushes on connection-scoped stream when it wakes
- **Inter-agent initiated activity display**: could also be connection-scoped
- **Background tasks**: Same as above
connection-scoped vs session-scoped will require further research to clarify the details.

We will need to dig into this further to completely confirm the behavior exists as we intend/need it. We could be reading too far into the "unsolicited notifications" line.

### Python SDK
Official `python-sdk` exists — we could implement ACP server-side in Python on our pydantic-ai stack without TypeScript.

---

## Candidates From Research

### Tier 1: Architecture Match (HTTP+SSE Client/Server)

| Project | Lang | Stars | Notes |
|---------|------|-------|-------|
| **Solenoid** | Python | ? | FastAPI server + Textual TUI, SSE streaming, AG-UI protocol (from CopilotKit). Closest to our stack. Question: can we decouple TUI from AG-UI/Google ADK backend? |
| **Arbiter** | ? | ? | Stream-native runtime, HTTP+SSE multi-tenant API, TUI client + CLI. Has "Agent-to-Agent v1.0 protocol" and "writ DSL". Need to find repo. |
| **toad** | Python | 3.2k | **Pure ACP thin client** — not an agent, just a frontend. Connects to 12+ agent CLIs (OpenHands, Claude Code, Gemini CLI, etc.) via ACP. Will McGugan (Textual creator) built this after Textualize funding ended. Features: Markdown streaming, integrated shell with full interactivity, @ for file context, Jupyter-like conversation navigation. Install: batrachian.ai. See [release announcement](https://willmcgugan.github.io/toad-released/). |

### Tier 2: Good Reference / Daemon Patterns

| Project | Lang | Notes |
|---------|------|-------|
| **Qwen Code** (daemon) | Go | `qwen serve` HTTP+SSE mode, multi-provider. Daemon pattern relevant even if rest isn't. |
| **tcode** (hifar) | Python | FastAPI server + TUI client + event bus bridge. SQLite persistence. Architecture pattern worth studying. |
| **interactive-process-mcp** | Go | MCP server for long-running processes, internal SSH arch, SSE over HTTP, multi-agent session sharing. |

### Tier 3: Library/Pattern Examples

| Project | Lang | Notes |
|---------|------|-------|
| **Textual v4** | Python | Framework, not agent. `MarkdownStream` widget (streaming partials), `Workers` API (async SSE consumption). Likely our foundation if building custom. |
| **Smelt** | Rust | 4 modes, vim bindings, sessions, LLM-powered compaction, MCP, image support. Best-in-class Rust TUI for UI inspiration. |
| **Evocli** | Rust | Full-screen TUI, 64 tools, long-term memory, MCP native, streaming with thinking animation. |
| **Consoul** | Python | Textual/LangChain, multi-provider, streaming, file attachments. Good Textual streaming patterns. |
| **Parllama** | Python | Ollama+multi-provider, streaming, vision support, memory system, session management. |

### Protocol Compatibility Options

These aren't thin clients we'd adopt — they're projects whose *server protocol* we could potentially implement, letting us use their TUI unchanged.

| Project | Notes |
|---------|-------|
| **OpenCode** | 100k+ stars. Clean HTTP+SSE protocol (OpenAPI 3.1 spec at `/doc`). TUI connects via REST + SSE at `/global/event`. **Framework adoption rejected** (session model mismatch, Effect-TS barrier — see `/docs/Benchmarking/opencode_framework_benchmarking.md`), but **protocol compatibility remains possible**. We'd implement their server API on our pydantic-ai stack, use their Go TUI as-is. Less promising than ACP (proprietary vs standard), but architecturally viable. |

### Disqualified (With Reasons)

| Project | Reason |
|---------|--------|
| **Aider / aider-ce** | Embedded agent (no server). Agent runs inside TUI process. Good UX reference only. |
| **DeepSeek TUI / Deepy** | Embedded agents. Textual widget patterns useful for reference. |
| **Claurst** | Rust stack. Not our path for spike. |
| **Claude Code / Letta Code** | TypeScript/Ink. Not adaptable. Style reference only. |
| **ICECODE** | Python backend, TS frontends. Overkill — we're Python-only for now. |
| **gact-tui** | 0 stars, Go/TS. Multiple backend adapters (interesting) but doesn't support ACP. |
| **forge** | TypeScript. ACP support, but conversation history lives client-side (we want server-side). |

Note: gacti-tui and forge could be considered if we have issues with toad or ACP

---

## Key Observations

1. **SSE+HTTP client/server split is rare** — most TUIs embed the agent. Solenoid, Arbiter, Qwen daemon, tcode are the exceptions.
2. **Rust dominates polished TUIs** — Smelt, Evocli, Sven, Crab Code, Claurst, DeepSeek-TUI all Rust/ratatui.
3. **Textual is the Python answer** — Solenoid, Consoul, Parllama, tldw_chatbook all use it. `MarkdownStream` + `Workers` is the pattern.
4. **ACP changes the calculus** — toad (Python, 3.2k stars) has ACP support. If we implement ACP server-side, toad might "just work" as client.

---

## Preliminary Strategy

**If ACP path:**
1. Implement ACP server-side on pydantic-ai (Python SDK exists)
2. Use toad or future ACP clients
3. stdio→HTTP bridge for existing ACP stdio clients during transition

**If custom TUI path:**
1. Textual v4 as foundation
2. httpx AsyncClient + Textual Worker for SSE consumption
3. Study Solenoid for TUI-side patterns, OpenCode for architecture patterns

---

## Research Sources
- Opus notes: `/workspace/git/Agent-Home/docs/Benchmarking/TUI Research/opus_TUI_benchmark_notes.md`
- Sonnet notes: `/workspace/git/Agent-Home/docs/Benchmarking/TUI Research/sonnet_TUI_benchmark_notes.md`
- James notes: `/workspace/git/Agent-Home/docs/Benchmarking/TUI Research/james_TUI_benchmark_notes.md`
- OpenCode deep dive: `/workspace/git/Agent-Home/docs/Benchmarking/opencode_framework_benchmarking.md`
