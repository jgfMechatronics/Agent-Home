# TUI Benchmark Survey — Opus Notes
*First pass, June 4, 2026*

## Key Architectural Constraint

Our TUI is unusual: **thin client ↔ separate server (HTTP+SSE)**, not monolithic agent. Most open-source options bundle the agent into the TUI binary.

---

## Candidates by Architecture Pattern

### Client/Server with HTTP+SSE (Most Relevant)

| Project | Lang | Notes |
|---------|------|-------|
| **Arbiter** | ? | Stream-native runtime, HTTP+SSE multi-tenant API, TUI client + CLI, Agent-to-Agent v1.0 protocol, writ DSL |
| **Solenoid** | Python | FastAPI server + Textual client, Google ADK, AG-UI protocol, local WASM sandbox, MCP support |
| **Qwen Code** (daemon) | Go | `qwen serve` daemon mode with HTTP+SSE, multi-provider, terminal + IDE integrations |
| **interactive-process-mcp** | Go | MCP server for long-running processes, internal SSH architecture, SSE over HTTP transport, multi-agent session sharing |

### Monolithic but Good Reference Implementations

**Rust/Ratatui:**
- **Evocli**: Full-screen TUI, 64 tools, long-term memory, MCP native, streaming with thinking animation
- **Sven**: Keyboard-driven, interactive TUI + Slint desktop, headless/networking modes, live streamed markdown
- **DeepSeek-TUI**: Streaming reasoning blocks, three modes (Plan/Agent/YOLO), 1M context, auto model selection per turn
- **Smelt**: MIT, 4 modes (Normal/Plan/Apply/Yolo), vim bindings, sessions, LLM-powered compaction, MCP, image support
- **Crab Code**: Claude Code open-source alt, Apache 2.0, multi-provider
- **Claurst**: Beta v0.1.4, clean-room Claude Code reimplementation, ACP protocol (editor integration), GPL-3.0

**Go:**
- **Gen Code**: Single binary, 5 pluggable pillars (LLMs, search, personas, skills, self-evolving), ~5x faster/smaller than Claude Code

**Python/Textual:**
- **Saarthi-cli**: LangGraph-powered, persistent memory, real-time token streaming, MCP, multi-provider, human-in-the-loop approval
- **Consoul**: Beautiful TUI via Textual/LangChain, multi-provider chat, streaming, file attachments, image analysis
- **tldw_chatbook**: Sophisticated TUI, 16+ features (chat, RAG, media ingestion, evaluations, coding), FTS5+vector search
- **Parllama**: Ollama+multi-provider, streaming responses, vision support, memory system, session management

**Python/Other:**
- **llm-tui**: vim bindings, multi-provider, project sessions, tool system (Read/Write/Edit/Glob/Grep/Bash), SQLite storage

---

## Architecture Pattern Observations

1. **Monolithic dominates**: Most TUIs embed the agent loop directly
2. **Rust dominates**: Sven, Smelt, Crab Code, Claurst, DeepSeek-TUI, Evocli all Rust/ratatui
3. **Textual for Python**: Consoul, tldw_chatbook, Parllama, Solenoid all use Textual
4. **Go shows sophistication**: Gen Code, Qwen, Arbiter, interactive-process-mcp show daemon/HTTP patterns
5. **SSE+HTTP is rare**: Only Arbiter, Solenoid, Qwen daemon, interactive-process-mcp

---

## Candidates Worth Deeper Investigation

### Tier 1 (Architecture Match)
- **Solenoid**: Python + Textual + FastAPI — closest to our stack. AG-UI protocol worth examining.
- **Arbiter**: True client/server split. Need to find repo and examine architecture.

### Tier 2 (Good Reference)
- **Qwen Code**: Daemon mode pattern is relevant even if rest isn't
- **Smelt/Evocli**: Best-in-class Rust TUIs for UI inspiration

### Tier 3 (Library Examples)
- **Consoul/Parllama**: Textual streaming patterns to learn from

---

## Questions for Reconvene

1. Did James/Sonnet find anything with true client/server split?
2. Is Textual the right library, or should we consider Rust/ratatui?
3. Should we examine Solenoid's AG-UI protocol more closely?
4. Any candidates with good tool approval workflow implementations?

---

## MAJOR FINDING: OpenCode (Post-Compaction Discovery)

**Source:** Sonnet + Claude.ai Opus 4.8 found this; I missed it in initial survey.

**Architecture (exactly our target):**
- Client-server split — TUI is a client connecting to HTTP server
- HTTP + SSE — REST API (OpenAPI 3.1) + SSE endpoint at `/global/event`
- Multi-client — TUI, web browser, desktop (Tauri), IDE extension, SDK
- Multiple clients simultaneously — `opencode serve` headless, attach from anywhere
- Real-time sync — All clients receive same streaming output via SSE

**Tech stack:** Bun, TypeScript, Hono (HTTP), SolidJS (desktop), **Go (TUI/SDK)**

**Scale:** 100k+ GitHub stars

**Key commands:**
- `opencode` / `opencode tui` — Start TUI
- `opencode serve --port 8080` — Headless server
- `opencode attach [url]` — Connect TUI to existing server
- `opencode web` — Web interface

**API endpoints:**
- `/session` — Create, list, fork, share sessions
- `/session/:id/message` — Send messages, get responses (streaming)
- `/global/event` — SSE endpoint for real-time sync
- `/doc` — OpenAPI 3.1 spec

**Open questions:**
- Memory system? (Is it flat-file like others, or database-backed?)
- Can we use their TUI with our backend?
- Sonnet found a PydanticAI backend that speaks OpenCode protocol — investigate

---

## Decision Factors

**For Textual (Python):**
- Same language as Agent Home server
- Rapid iteration for spike
- Multiple reference implementations
- We know Python well

**Against Textual:**
- Rust TUIs look more polished
- Performance ceiling lower
- Async complexity

**Recommendation:** Start with Textual for spike (speed), consider Rust for production later if needed.
