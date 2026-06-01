## Paths Under Consideration

### Path 1: MCP as Tool Bridge

**Summary:** Agent Home is the central identity layer and runs the agent loop as an MCP *client*. Tools are provided by MCP *servers* running in portable "bundles" that can be launched in different environments.

**Architecture:**
```
                    ┌─────────────────────────────────┐
                    │       Agent Home Server         │
                    │  - Identity, memory, history    │
                    │  - Agent loop (pydantic-ai)     │
                    │  - MCP client                   │
                    └─────────────────────────────────┘
                        ↕                  ↕
                        MCP           SSE / HTTP
                        ↕                  ↕
┌─────────────────────────────┐   ┌─────────────────────────────┐
│   MCP Bundle (portable)     │   │   Conversation Interface    │
│  - filesystem-mcp           │   │  - Our CLI (display layer)  │
│  - bash-mcp                 │   │  - Telegram                 │
│  - other servers as needed  │   │  - Future interfaces        │
│                             │   │                             │
│  Launched in: sandbox /     │   │                             │
│  host OS / VM / remote      │   │                             │
└─────────────────────────────┘   └─────────────────────────────┘
        "hands"                         "voice/ears"
```

**What we build:**
- Display CLI (presentation logic, SSE rendering)
- MCP client integration (pydantic-ai has native support)
- Approval gate mechanism (SSE event + server-side flag)
- Bundle launcher utility

**What we borrow:**
- `@modelcontextprotocol/server-filesystem` (Anthropic-maintained: read, write, edit, search, list)
    - ⚠️ Anthropic's README explicitly calls this a "reference implementation" / "educational examples for developers" — NOT production-ready per their own docs
    - 2 known CVEs in history (path traversal via symlinks + prefix matching) — patched in 2025.7.1
    - or others, but something like this
- Community bash MCP servers (HTTP transport available)
- Any future MCP-compatible tool providers

**Why this path is appealing:**
- Not locked to Letta's proprietary ClientToolSchema protocol (which they're moving away from)
- Industry standard protocol — build once, connect to anything MCP-compatible
- Existing MCP servers are battle-tested and maintained by others
- Portable bundles achieve the "same agent, different hands" goal elegantly
- Aligns with non-goal #1 (don't build our own coding tools)
- Clear path to leverage existing work (Pydantic AI supports MCP)

**Identified Risks:**
1. **MCP ecosystem risk** — What if maintained MCP servers are flaky, poorly documented, or abandoned? We'd inherit their bugs and limitations.
2. **Anthropic's filesystem server "reference implementation" disclaimer** — Their README calls it \"educational examples for developers.\" Worth calibrating: 84.9k stars and 286k weekly downloads suggest real-world usage despite the disclaimer. Likely legal/liability hedging rather than a quality signal. 2 CVEs in history — both patched, indicating active maintenance. For self-hosted personal use the bar is different than enterprise production. Verified feature set is solid (14 tools, good edit_file) with one key gap: **no content search** (see risk #8).
3. **pydantic-ai MCP support maturity** — How mature is it really? We haven't verified it does what we need in practice.
4. **MCP protocol churn** — 2026-07-28 RC released May 21, 2026. Removes initialize handshake + session IDs, stateless core. Breaking changes. Tier 1 SDKs expected to ship support by July 28. Near-term protocol instability to factor in.
5. **unforseen complexity** — Looks simple but haven't designed it. Could be harder than assumed.
6. **Bundle concept implementation** — Sounds elegant but unbuilt. Actual complexity unknown.
7. **HTTP transport for MCP** — Mentioned as available, but is it well-supported in the servers we'd use?
8. **Tool feature parity** — Do the existing MCP servers have all the features we currently use? (e.g., Edit's exact string matching, Grep's output modes, etc.)

---

### Path 2: Agent Home + Letta Code (Proprietary Protocol)

**Summary:** Agent Home is the central identity layer. Letta Code provides the conversation interface AND tool execution via Letta's proprietary ClientToolSchema pause/resume protocol.

**Architecture:**
```
                    ┌─────────────────────────────────┐
                    │       Agent Home Server         │
                    │  - Identity, memory, history    │
                    │  - Agent loop (pydantic-ai)     │
                    │  - Implements ClientToolSchema  │
                    │    pause/resume protocol        │
                    └─────────────────────────────────┘
                                ↕
                           SSE + ToolReturn
                                ↕
                    ┌─────────────────────────────────┐
                    │          Letta Code             │
                    │  - Conversation interface       │
                    │  - Tool execution (bash, read,  │
                    │    write, edit, grep, etc.)     │
                    │  - Approval UI                  │
                    └─────────────────────────────────┘
```

**What we build:**
- ClientToolSchema pause/resume protocol (stop_reason: requires_approval, ToolReturn handling)
- SSE format compatible with Letta's TypeScript SDK
- ~8 API endpoints matching Letta's Backend interface

**What we borrow:**
- Letta Code's full tool implementation (battle-tested)
- Letta Code's TUI/approval UI
- Letta's TypeScript SDK handles SSE parsing

**Why this path could work:**
- Letta Code's tools are production-quality, already tested
- We get a professional TUI for free
- The pause/resume protocol is proven (it's what we use today)
- Clear path to leverage existing work

**Risks / Concerns:**
1. **Proprietary lock-in** — We're implementing Letta's protocol, not an industry standard. If Letta changes it, we adapt or break.
2. **Letta is moving away from this** — They're shifting to client-side everything. We'd be building on architecture they're deprecating.
3. **Single harness dependency** — This path only works with Letta Code. Other harnesses (Pi, future tools) would need different integrations.
4. **Lots of Letta Code internals to disable** — Letta Code has features we don't want (cloud sync, etc.) that we'd need to work around.
5. **Telegram/self-wake gap** — In contexts without Letta Code connected, no tool execution available.

---

### Path 3: Agent Home Built Inside Existing Harness

**Summary:** Instead of Agent Home being a separate server, we build our identity/memory layer directly into an existing coding harness (Pi, OpenCode, OpenClaw, or Letta Code fork).

**Architecture:**
```
┌─────────────────────────────────────────────────────┐
│            Existing Harness (Pi / OpenCode / etc.)  │
│  ┌───────────────────────────────────────────────┐  │
│  │  Agent Home (embedded)                        │  │
│  │  - Identity, memory, history                  │  │
│  │  - Our memory system, deferred compilation    │  │
│  └───────────────────────────────────────────────┘  │
│  - Agent loop (harness's native)                    │
│  - Tool execution (harness's native)                │
│  - Conversation interface (harness's native)        │
└─────────────────────────────────────────────────────┘
```

**What we build:**
- Memory/identity layer as a library or module
- Integration hooks into the harness's agent loop
- Adapters for the harness's storage/state management

**What we borrow:**
- Everything else — agent loop, tools, TUI, all from the harness

**Why this path could work:**
- Maximum leverage of existing work
- The harness is already a complete, working system
- We focus purely on what makes us different (memory architecture)

**Risks / Concerns:**
1. **Harness architecture constraints** — We're limited by how the harness is designed. If it doesn't have hooks where we need them, we're stuck or forking.
2. **Portability loss** — Agent identity becomes tied to one harness. "Same agent across contexts" becomes much harder.
3. **Maintenance burden** — We inherit the harness's bugs, update cadence, and design decisions. We're downstream of their choices.
4. **Fork divergence** — If we fork (e.g., Letta Code), we lose upstream updates and own all maintenance.
5. **Which harness?** — Pi, OpenCode, OpenClaw, Letta Code all have different architectures, languages (TS vs Python), and design philosophies. Choosing one means betting on it.
6. **Requirement #1 tension** — "Agent consistent across all contexts" is hard if the agent lives inside one specific harness.
7. Difficulty leveraging existing work

---

### Path 4: OpenClaw (the wild card)

**Summary:** A subset of Path 3, but with a specific harness: OpenClaw. Embed Agent Home's identity/memory layer inside OpenClaw rather than building our own server.

**What is OpenClaw:**
- 374k stars, MIT, TypeScript, actively maintained (latest: v2026.5.20)
- Self-hostable, local-first personal AI assistant
- Native channels: Telegram, WhatsApp, Discord, Slack, Signal, iMessage
- Full system access (files, shell, scripts), adjustable sandboxing
- Skills & plugins ecosystem, computer use via browser control
- Built by Peter Steinberger (steipete) — joined OpenAI Feb 2026; project continues open source
- **Internally uses Pi as coding agent** (`@mariozechner/pi-agent-core`, `pi-coding-agent`, `pi-tui`)

**Why this path is appealing:**
- Telegram is native — one of our target interfaces, handled for free
- Computer use / browser control already built
- Massive community, battle-tested
- Self-hostable, local-first values match ours

**Why this path has problems:**
- Same core concerns as Path 3: Agent Home identity becomes embedded in their architecture, portability loss
- TypeScript — our stack is Python/pydantic-ai; integration friction at every boundary
- We'd be downstream of their design decisions, update cadence, and priorities
- Embeds Pi as the coding agent — inheriting Pi's architecture inside OpenClaw's architecture
- Memory system is weak (James's assessment)
- "Same agent across contexts" goal is harder when identity lives inside one harness

**Current verdict:** Appealing capabilities bundle, but Path 3 concerns apply in full. Telegram is better solved as a thin display interface (same pattern as our CLI). Not recommended for Phase 3.

---

### Path 5: pydantic-ai Native Capabilities (Library Path)

**Summary:** Embed filesystem/shell tools directly as pydantic-ai capabilities inside Agent Home. No MCP, no separate process. Tools are Python code running in the same process as the agent loop.

**Discovered:** May 24, 2026 during Opus/Sonnet exploration session.

**Architecture:**
```
┌─────────────────────────────────────┐
│          Agent Home Server          │
│  - Identity, memory, history        │
│  - Agent loop (pydantic-ai)         │
│  - Tools as EMBEDDED CAPABILITIES   │
│    (ConsoleCapability / harness)    │
└─────────────────────────────────────┘
              ↕ SSE / HTTP
┌─────────────────────────────────────┐
│     Conversation Interface          │
│  - Our CLI (display layer)          │
│  - Telegram                         │
│  - Future interfaces                │
└─────────────────────────────────────┘
```

**Implementation options (both work with pydantic-ai):**
1. **TODAY**: `vstorm-co/pydantic-ai-backend` ConsoleCapability — ls, read_file, write_file, edit_file, glob, grep, execute
2. **NOT YET**: Official `pydantic-ai-harness` PR #177 — FileSystem + Shell capabilities (66 tests, 100% coverage). Still draft/open as of May 25, 2026 — timeline unknown.

**What we build:**
- Display CLI (presentation logic, SSE rendering)
- Approval gate mechanism
- (Optional) Docker sandbox integration for isolation

**What we borrow:**
- ConsoleCapability (vstorm-co) or FileSystem+Shell (pydantic-ai-harness)
- All tool implementations maintained by others

**Why this path is appealing:**
- Simpler — no MCP protocol, no separate servers to manage
- One less moving part (tools can't independently fail)
- pydantic-ai native, follows their design patterns
- Works TODAY with vstorm-co, or soon with official harness
- **More robust for self-wake**: Tools always available when Agent Home is up, no dependency on external server
- `uv add pydantic-ai-harness` and done

**Where MCP still wins (consider hybrid):**
1. **Sandbox isolation** — Tools run in restricted container, Agent Home elsewhere (security-sensitive deployment)
2. **Remote tool execution** — Tools physically on a different machine
3. **External/community tools** — Web search APIs, GitHub, etc. that already exist as MCP servers

**Identified Risks:**
1. **Tied to our process** — Can't run tools in a different security context without additional work
2. **vstorm-co is third-party** — Not official pydantic-ai, though they're endorsed and collaborating on upstream
3. **PR #177 not merged** — Still draft as of May 25, 2026. vstorm-co ConsoleCapability is the working option until then.
4. **Tool feature parity** — Verified: ConsoleCapability has grep with output modes (content/files/count), edit (str_replace), read, write, glob, bash. Complete bundle for our coding use case. Edit format is str_replace (not unified diffs) — functional, but see Patterns section for potential future improvement.

**Hybrid recommendation:**
- **Embed core tools** (filesystem, bash) as capabilities — always available, zero ops
- **MCP client** for external/community tool servers — web search, GitHub API, etc.
- pydantic-ai treats both uniformly. Adding MCP later = one line. Decision is reversible.

---

## Decision

**Recommendation: Path 5 Hybrid** — pydantic-ai native capabilities for Phase 3 dogfooding, with MCPToolset client hook for future external tool integration.

### Why Path 5 over Path 1 (MCP) for Phase 3

Not "MCP bad" — the Anthropic filesystem server is actually capable (14 tools, solid `edit_file` with oldText/newText substring matching, dry-run mode, git-style diffs). The specific gap: **no content search**. `search_files` is filename/glob only; `grep` is absent. For a coding agent, content search is non-negotiable. Path 1 = filesystem server + bash MCP (for ripgrep) + possibly a dedicated content search server — 2-3 running processes, with content search papering over via bash calls. For Phase 3 dogfooding, that's unnecessary complexity.

Path 5 via vstorm-co ConsoleCapability has the complete bundle today: grep with output modes, edit, read, write, glob, bash. One package, zero gaps for our coding use case.

MCP protocol is also in flux (2026-07-28 RC with breaking changes released May 21). pydantic-ai's MCPToolset likely abstracts this, but it's another reason to prefer native for Phase 3.

### Why not close the door on MCP

The ecosystem is growing. If a battle-tested filesystem+content-search bundle emerges, or we need sandboxed isolation, or external community tool integrations (web search, GitHub API), MCP is the right hook. pydantic-ai treats MCPToolset uniformly with native capabilities — adding it later is one line. This is a *today* decision, not a *forever* decision.

**The hybrid:** Embed core tools (filesystem, bash) as native capabilities — always available, zero ops overhead. Keep MCPToolset as the client hook for external services as the ecosystem matures.

### Phase 3 scope

| Layer | What | Source |
|-------|------|--------|
| Core tools | Filesystem, bash, grep, glob | ConsoleCapability (vstorm-co) |
| Display CLI | Text streaming, tool call/result display, approval flow | Build |
| Approval gates | Pause on tool call, user decision, resume | Build |
| External tools | MCPToolset client hook | pydantic-ai native (deferred) |

**CLI scope note:** "Display CLI" has more depth than it sounds. What flows through SSE: streaming text, tool calls (name + args), tool results (file contents, grep matches, bash output, diffs, errors), thinking blocks, approval requests. Phase 3 bar: functional for dogfooding, not polished. Minimum viable — text streaming, readable tool call/result display, functional approval flow. Syntax highlighting, diff visualization, pagination: Phase 4.

### Open items before finalizing

1. **PR #4393 status** — Still open as of Mar 12, 2026; some scope may have been absorbed into PR #4640 (merged Mar 24, included execution environments abstraction). Recheck before finalizing.
2. **Edit format opportunity** — ConsoleCapability's `edit_file` uses `str_replace` (not unified diffs). Aider reports 30-50% error reduction with unified diff format. Functional today; worth evaluating as a future improvement.
3. **vstorm-co dependency** — Third-party. First-party ExecutionEnvironment support actively in flight (PR #4393 + #4640 foundation). Track for migration path when it lands.

---

## Patterns Worth Understanding

*These aren't paths we're evaluating — they're architectural patterns from the ecosystem that inform our thinking.*

### Stateful Kernel Pattern (Jupyter-style)

**What it is:** Instead of spawning a fresh subprocess per tool call (stateless), maintain a persistent kernel where environment variables, installed packages, and defined functions persist across calls.

**Why it matters:** Qualitatively different from subprocess-per-call. An agent can `pip install` once and use the package for the rest of the session. Set an env var, it persists. Define a helper function, call it later.

**Where to look:** Jupyter kernel protocol, pydantic-ai-harness CodeMode (stateless version — uses Monty sandbox)

**Our position:** Not needed for Phase 3, but worth understanding for future coding agent work.

---

### Aider's Diff/Changeset Architecture

**What it is:** The model never calls `write_file` directly. It generates diffs/changesets, Aider applies them atomically. Tool API is `apply_changeset(changes)` not `write_file(path, content)`.

**Why it matters:** Atomic, auditable, reversible. Git history reads like a thoughtful engineer's changelog. Two-model pattern possible: \"Architect\" model (frontier, planning) + \"Editor\" model (generates specific edits).

**Key finding:** Unified diffs reduce editing errors by **30-50%** vs other formats. Aider uses 5 lines of context with flexible strategies for mismatched hunks (offset tolerance, fuzzy matching). This is worth adopting regardless of tool architecture.

**Source:** [Aider](https://github.com/paul-gauthier/aider) — treats git as first-class citizen

**Our position:** Could influence how we design edit tools. The atomic changeset model is more robust than individual file writes.

---

### LSP as Tool Layer

**What it is:** Language Server Protocol gives semantic code understanding — "what calls this function?", "find all references", "what are the type errors?", "rename symbol across codebase".

**Why it matters:** Semantic understanding, not just syntactic file operations. An agent asking "what calls this function?" is fundamentally more capable than one that can only grep.

**Where to look:** pyright (Python), rust-analyzer (Rust), typescript-language-server. Cline gets this free via VS Code embedding.

**Our position:** High ceiling, complex integration. Worth flagging as future possibility for advanced coding work.

---

### Snapshot/Transaction Pattern

**What it is:** Work on an isolated copy of the workspace (git worktree), validate changes, apply atomically. "Before any risky operation, snapshot. If something goes wrong, roll back."

**Why it matters:** Git integration at the tool level, not just at commit level. Interacts interestingly with our memory architecture — could snapshot before running agent tools, restore if run goes bad.

**Where to look:** OpenCode has worktree support. Git worktree docs.

**Our position:** Interesting for robustness. Not Phase 3, but could be valuable for autonomous work.

---

### Sub-Agent as Tool

**What it is:** "Run this bash command" delegates to a narrow-scope smaller model (e.g., Haiku) with elevated permissions. Main agent stays clean; sub-agent does risky/dirty work.

**Where to look:** pydantic-ai-harness `SubAgentCapability` (PR #178)

**Our position:** We already use this pattern with Haiku for document work. Extension of existing practice.

---

### Agent Identity Standards (AGENTS.md)

**What it is:** Open standard for agent identity, stewarded by Linux Foundation's Agentic AI Foundation. Single identity file readable by multiple runtimes (Claude Code, Cursor, Cline, Codex CLI, Windsurf, etc.).

**Why it matters:** Cross-runtime portability. Same agent identity works across different harnesses.

**Where to look:** [agentic-harness](https://github.com/sevenschulte/agentic-harness) — shows AGENTS.md + SKILL.md standards with runtime-specific mappings

**Our position:** Worth tracking. Agent Home agents could potentially export AGENTS.md-compatible identity for use in other tools.

---

## Reference: Coding Agent Architectures (2026)

From [AI Coding Agent Architecture Deep Dive](https://fp8.co/articles/AI-Coding-Agent-Architecture-Deep-Dive):

All modern coding agents share the same core pattern: **agent loop that cycles between LLM reasoning and tool execution**. They diverge in:
- Context window management
- Tool dispatch (parallel vs sequential)
- Edit application method
- Safety boundaries

| Agent | Key Differentiator |
|-------|-------------------|
| **Claude Code** | Parallel tool calls, broad toolset, Edit-Apply loop with streaming |
| **Cursor** | Multi-file edit with speculative decoding, background indexing |
| **Cline** | Approval-based (every tool call needs permission), MCP for extensibility, Plan/Act modes |
| **Aider** | Git-native diffs, repository mapping, two-model (Architect/Editor) pattern |
| **Continue** | Open-source Copilot replacement, inline autocomplete + chat |
| **OpenHands** | Full Docker sandbox, long-running autonomous tasks |

---

## Reference: pydantic-ai Ecosystem (May 2026)

**Core pydantic-ai capabilities:**
- Model-agnostic (OpenAI, Anthropic, Gemini, Bedrock, etc.)
- **MCP support** via `MCPToolset` (PR #5325, merged May 7, 2026) — new recommended API, replaces deprecated `MCPServer*` and `FastMCPToolset`. Built on FastMCP Client. Accepts URL, script path, FastMCP server object, or MCPConfig dict.
- Agent2Agent interoperability
- Durable execution (survives restarts)
- YAML/JSON agent definitions (no code required)
- Dependency injection for tools

**pydantic-ai-harness (official capability library):**

| Category | Capability | Status |
|----------|-----------|--------|
| Tools & execution | CodeMode (sandboxed Python via Monty) | ✅ Released |
| Tools & execution | FileSystem + Shell | 🚧 PR #177 |
| Tool discovery | ToolSearch (progressive discovery for large toolsets) | ✅ In core |
| Agent orchestration | SubAgents (delegate to specialized child agents) | 🚧 PR #178 |
| Skills | Progressive tool loading (search, activate, deactivate) | 🚧 PR #183 |
| Context management | Sliding window + LLM compaction | ✅ Released |
| Safety | Guardrails (input validation) | 🚧 PR #182 |

**pydantic-ai core PRs (execution environments):**

| PR | What | Status |
|----|------|--------|
| #4358 | CodeExecutionToolset + ExecutionEnvironmentToolset | ✅ Merged Feb 19, 2026 |
| #4393 | `ExecutionEnvironment` ABC (Local/Docker/Memory backends) + `ExecutionEnvironmentToolset` (ls, shell, read_file, write_file, replace_str, glob, grep) | 🚧 Open as of Mar 12, 2026 |
| #4640 | Capability abstraction, AgentSpec, Hooks, unified thinking, per-run toolset isolation, builtin tool fallback | ✅ Merged Mar 24, 2026 |

**Trajectory read:** Active investment in first-party execution environments. PR #4640 landing suggests infrastructure is being laid; PR #4393 is the specific tool implementation waiting to land on that foundation.

**Third-party capabilities (vstorm-co):**
- `pydantic-ai-backend` ConsoleCapability — works TODAY
- `summarization-pydantic-ai` — context management
- `subagents-pydantic-ai` — sub-agent delegation

---
