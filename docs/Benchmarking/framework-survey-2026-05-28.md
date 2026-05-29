# Agent Framework Benchmarking Survey
*Date: May 28, 2026 | Author: Sonnet*

## Pivot Question

Are there existing frameworks close enough to our intended Agent Home architecture that building on top of them is faster/better than our current greenfield approach?

**What we're looking for — plumbing, not memory:**
- Persistent HTTP REST + SSE server
- MCP tool isolation (separate process/network boundary)
- TUI/CLI client
- Clean, replaceable core loop (we can insert our block-based memory)

**What we bring:** structured block memory, semantic retrieval, dynamic context injection, ethics framework, continuity.

---

## Architecture Requirements (Abbreviated)

From `architecture-vision.md`:
- **Req 5**: Memory controls system prompt without tool calls — must be in-process
- **Req 13**: Agentic tools (FS, bash) isolated from Core/memory DB
- **Req 14**: Core on always-on server (mobile access, self-wake)
- **Goal 8**: TUI/CLI launchable from any machine
- **Goal 9**: Agentic tools on remote machine (eventual, not MVP)

**Foundational decision**: Core-on-server. Phase 3 = localhost but built with network boundaries (not throwaway).

---

## Candidate Matrix

| Framework | HTTP API | MCP Isolation | TUI/CLI | Clean Loop | Stars | Language | Tier |
|---|---|---|---|---|---|---|---|
| OpenHands SDK | ✓ REST+WS | ✓ Docker | CLI | ⚠️ Event-sourced memory | 70k+ | Python | 1 |
| Jaato | ✓ WS+Unix IPC | ✓ plugin | ✓ multi-client | ✓ | ? | Python | 1 |
| KohakuTerrarium | ✓ WS Laboratory | ✓ MCP | ✓ Studio/TUI | ⚠️ Baked memory | 329 | Python | 2 |
| Soothe (mirasoth) | ✓ WS+HTTP | ✗ langchain | CLI | ✗ LangChain dep | ? | Python | 3 |
| OpenAgent (geroale) | ✓ (Server) | ✓ MCP bundled | ✓ CLI+Desktop | ✓ (young) | 0 | Python | 3 |
| AgenticGateway | N/A | MCP proxy only | N/A | N/A | 2.9k | Rust | ✗ not a framework |
| Agentrail | ✓ | ✓ Docker | ✓ | ✓ | 10 | TS | ✗ archived |
| DreamAgent | ✓ FastAPI | ✗ Redis | ✗ React | ✗ DAG-only | 0 | Python | ✗ too early |

---

## Tier 1 — Worth Deep Code Review

### OpenHands / software-agent-sdk

**What it is:** Production-grade agent framework (70k+ stars, arxiv paper, actively maintained). Phase 1 was monolithic; V1 SDK modularizes into `software-agent-sdk`.

**Architecture alignment:**
- FastAPI REST + WebSocket server ✓
- Docker/Kubernetes sandboxing for execution ✓ (maps to Req 13)
- MCP integration ✓
- `LocalConversation` vs `RemoteConversation` — network boundary built in ✓

**The memory model:**
- `EventStore`: immutable append-only log. Every LLM interaction, tool call, observation, memory write is an event.
- `ConversationState`: serializable container, saveable to disk/DB. Lives in-memory during execution.
- `Condenser`: manages memory compression when context fills. Extensible — `NoOpCondenser`, `LLMSummarizingCondenser`, etc.

**Key concern:** Memory IS the event log. Their architecture is event-sourced; ours is block-based. These are different models.

- **Scenario A**: Treat block memory as a side-car — blocks live in our DB, injected into system prompt on each turn, and we just don't use OpenHands' event log for memory. The event log still captures everything for replay/audit — useful, not harmful.
- **Scenario B**: Condenser is our compaction — we write a `BlockMemoryCondenser` that, instead of summarizing, calls our block-updating logic. Might actually fit.

**Build-on verdict**: Possible, but requires understanding how deeply their core loop is coupled to `ConversationState`. If we can inject our system prompt pre-loop, this might work. **Warrants a code read of `software-agent-sdk` core loop.**

---

### Jaato

**What it is:** Daemon-first agent framework. Designed for long-running server mode. 55+ plugins.

**Architecture alignment:**
- Daemon process, Unix socket + WebSocket IPC ✓
- Multi-client support (multiple TUIs connected simultaneously) ✓
- Session persistence ✓
- Pipeline-presentation split: server emits events, clients render — exactly our SSE model ✓
- Plugin architecture — tool isolation is pluggable ✓

**The memory model:**
- Session persistence (disk), less opinionated about format
- Plugin architecture suggests memory could be a plugin

**Key concern:** Stars unknown / project maturity unclear. Need to verify it's actively maintained.

**Build-on verdict**: Architecture most philosophically aligned with what we want. Plugin-based memory replacement might be cleanest path of all candidates. **Warrants a code read of daemon loop + plugin interface.**

---

## Tier 2 — Interesting, Specific Concern

### KohakuTerrarium (Kohaku-Lab/KohakuTerrarium)

**What it is:** Python-native framework with layered Creature/Terrarium/Studio/Laboratory architecture. Active — v1.4.0 shipped May 2026.

**Architecture alignment:**
- Studio/Laboratory split: Studio = local, Laboratory = network-split with WebSocket ✓ (maps to Core-on-server)
- MCP support built in ✓
- TUI and web UI ✓
- Filesystem-backed memory with FTS/semantic/hybrid search — impressive

**Key concern:** Memory is baked into the `Creature` abstraction. Swapping it means fighting the Creature API. The FTS/semantic search is actually _close_ to what we want, but the storage format (filesystem vs our PostgreSQL) and the block structure may not map.

**Build-on verdict**: Would require either accepting their memory format (conflicts with Req 5 — in-process context injection), or forking the Creature abstraction. **Lower confidence than Jaato, but the Laboratory/Studio split is worth understanding.**

---

## Tier 3 — Don't Pursue

### Soothe (mirasoth/soothe)
- CLI→Daemon architecture is right, but it builds on LangChain/DeepAgents. We explicitly ruled out deepagents (same problems as Letta — private internal APIs, monolithic loop). Inheriting that chain is a no.

### OpenAgent (geroale)
- Architecture ideas are good (Server + CLI + Desktop clients, MCP bundled). But 0 stars, April 2026, no community. Too early-stage to use as foundation.

### AgenticGateway (agentgateway/agentgateway)
- This is an MCP proxy — infrastructure, not an agent platform. Not relevant to our pivot question.

### Agentrail
- Confirmed real but ARCHIVED. Dead project.

### DreamAgent
- 0 stars, solo contributor, April 2026. DAG orchestration + Redis is the opposite of our clean loop model.

---

## Key Architectural Finding

**The memory coupling problem** is the critical filter. Most frameworks have memory baked into their core loop. The question is always: can we inject our system prompt (with block memory content) before the LLM call, without the framework fighting us?

The spectrum:
- **OpenHands**: event log is memory, but Condenser is extensible — possible hook
- **Jaato**: plugin-based — most likely to allow clean replacement
- **KohakuTerrarium**: Creature abstraction bakes memory in — harder
- **Soothe**: LangChain memory chain — tied at the hip

**The execution isolation question** is mostly solved: both OpenHands (Docker) and KohakuTerrarium (MCP) and Jaato (plugin) provide tool isolation. Any Tier 1-2 candidate handles Req 13.

---

## Recommendation for Opus

**Deep-dive priority:**
1. **Jaato** — read the daemon loop, session store, and plugin interface. Is memory a plugin or a core assumption?
2. **OpenHands software-agent-sdk** — read the core loop. Where does `ConversationState` get populated? Can we inject a block-based system prompt pre-LLM-call?

**Decision frame:** We're not looking for a perfect match — we're asking if either of these can save us 3-6 weeks of plumbing work without costing us 6 weeks of fighting the framework. The bar is: can we replace their memory model without a fork?

**OpenHands specific risk:** 70k stars means they have momentum and a roadmap. If they keep evolving `ConversationState`, we're tracking a moving target. Building on top of a framework with this much velocity has maintenance risk.

**Greenfield risk we're trying to avoid:** Spending weeks on: HTTP server setup, SSE streaming infrastructure, session management, Docker/MCP execution isolation — things that already exist.

---

## Greenfield Baseline (Current Agent-Home)

For context — what we already have:
- FastAPI server + pydantic-ai core loop ✓
- SSE streaming ✓
- Block-based memory with dynamic context injection ✓
- 294 tests

**What we still need to build for Phase 3:**
- MCP tool server integration (afternoon, given MCPToolset exists)
- TUI client (0 → usable CLI)
- Session persistence improvements
- Self-wake scheduling

The greenfield path is not as far behind as it might seem. We have the hard part (memory + persistence). The question is whether Jaato or OpenHands SDK gets us the remaining 30-40% faster than building it ourselves.
