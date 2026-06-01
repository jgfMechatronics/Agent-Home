# Architecture Evaluation

## Starting Point: Component Relationships

Each connection is an independent decision. We evaluate coupling method and directionality for each.

```
                                          ┌─────────────────────┐
                                          │   TUI / Management  │
                                          │       Console       │
                                          └──────────┬──────────┘
                                                     │
                                                     ? (coupling? direction?)
                                                     │
┌─────────────┐                           ┌──────────▼──────────┐
│  Self-Wake  │───────────?───────────────│    Core Agent Loop  │
│   System    │                           │       Process       │
└─────────────┘                           └──┬───────┬───────┬──┘
                                             │       │       │
                                             ?       ?       ?
                                             │       │       │
                                ┌────────────▼──┐ ┌──▼──────┐ ┌──▼────────────┐
                                │    Memory     │ │   FS    │ │     Misc      │
                                │    System     │ │  Tools  │ │ (WhatsApp etc)│
                                └───────────────┘ └─────────┘ └───────────────┘
```

## Scope Decisions (What's NOT a separate box)

- **LLM API**: Part of Core — we're using pydantic-ai's agent loop, not rolling our own
- **Database/Persistence**: Part of Memory System — coupling to DB is internal design detail

## Connection Decisions Summary

| Connection | Coupling | Directionality | Decision | Rationale |
|------------|----------|----------------|----------|-----------|
| Core ↔ Memory | In-process | N/A (same process) | ✅ **IN-PROCESS** | Memory IS identity; Req 5 forces tight coupling |
| Core ↔ FS Tools | MCP | Core → Tools | ✅ **MCP** | Req 13 isolation + Goal 9 remote execution |
| Core ↔ Misc | Case-by-case | Varies | ✅ **CASE-BY-CASE** | Heterogeneous bucket |
| Core ↔ TUI | HTTP + SSE | Bidirectional | ✅ **HTTP + SSE** | REST for commands, SSE for push events |
| Core ↔ Self-Wake | In-process | N/A (same process) | ✅ **IN-PROCESS** | Simple scheduler; external later if needed|

## Important Considerations (Cross-Cutting)

These aren't boxes but need to be supported by the architecture:

- **Inter-agent communication**: Agents need to be able to invoke/talk to each other. Routing TBD — could be Core↔Core direct, or mediated. Must account for this in design.
- **TUI push path**: The Core↔TUI connection isn't just "TUI calls Core." If Core completes a long task, how does it notify the TUI? Options: TUI polls, WebSocket, SSE, webhooks. This is part of the Core↔TUI directionality question.

## Coupling Options (Reference)

| Method | Description | Characteristics |
|--------|-------------|-----------------|
| In-process | Direct function calls, same memory space | Lowest latency, tightest coupling, no isolation |
| MCP | Model Context Protocol (JSON-RPC over stdio/HTTP) | Standardized tool protocol, portable, isolated |
| HTTP/REST | Request/response over network | Well-understood, stateless, works across machines |
| WebSocket | Persistent bidirectional connection | Good for streaming, push-capable, more complex |
| SSE | Server-sent events (one-way server→client) | Simple streaming, HTTP-based, one direction only |
| gRPC | Binary protocol, typed contracts | High performance, probably overkill for us |
| Unix socket / IPC | Inter-process, same machine | Fast, isolated, but not network-portable |
| Message queue | Async decoupled (Redis, RabbitMQ, etc.) | Loose coupling, complexity cost, good for scale |
| Shared DB/file | Coordinate via shared state, no direct connection | Implicit coupling, simple, but can get messy |

Not all options make sense for every connection — we'll narrow to realistic candidates per link.

## Evaluation Process

For each connection, we attempt to ask:
1. **What coupling method?** (in-process, HTTP, MCP, WebSocket, etc.)
2. **Who initiates?** (which side calls, which side responds)
3. **Why?** (security, latency, portability, simplicity, etc.)

---

## Connection Decisions

### Core ↔ Memory System

**Decision: IN-PROCESS (Memory is part of Core)**

**Options considered:**
- In-process (direct function calls)
- HTTP (memory as microservice)
- Shared DB (loosely coupled via database)

**Rationale:**
- Req 5: Memory controls system prompt and does dynamic recall *without tool calls* — network boundary would add latency and failure risk on every turn
- Auto-recall needs to run at multiple points during a single agent turn, not just at tool call boundaries
- Memory IS the agent's identity — separating it creates "where does the agent live?" confusion
- No security benefit to isolation (unlike filesystem tools where Req 13 applies)

**On internal modularity:** "In-process" ≠ "monolithic blob." Memory system can be well-defined internal module with clean interfaces. Modularity through code design, not process boundaries. If memory needs to talk to external services (graph DB, embedding service), those are implementation details of the memory module.

**Implication:** Memory box moves inside Core box on final diagram.

---

### Core ↔ FS Tools

**Decision: MCP**

**Options considered:**
- MCP (JSON-RPC over stdio/HTTP)
- HTTP/REST (custom API)
- In-process (direct function calls)
- Unix socket / IPC (same machine, different process)

**In-process ruled out by:**
- Req 13: Agentic actions MUST be isolated from core memories/persistence DB
- Goal 9: Remote execution requirement (tools on laptop, Core on server)
- Sandboxing principle: tools and Core in same failure domain = hippocampus risk

**Why MCP over HTTP/IPC:**
- Industry standard — ecosystem exists, maintained by others
- Existing tools available (Anthropic filesystem MCP server, well-starred)
- Less custom work for us
- Portable — same protocol local or remote
- Shareable — any custom tools we build can be contributed back
- Reverse-connection pattern works (tool server dials out to Core via Tailscale)

**Directionality:** Core initiates → Tools respond. Tools are passive.

---

### Core ↔ Misc (WhatsApp, etc.)

**Decision: CASE-BY-CASE**

Misc is a heterogeneous bucket — each integration evaluated on its own needs. Some may be MCP (if tool-like), some HTTP (external APIs), some webhooks (notifications). No blanket protocol.

Note: Telegram already works and serves as prototype for this category.

---

### Core ↔ TUI / Management Console

**Decision: HTTP + SSE**
- REST for TUI→Core commands (send message, approve tool call, etc.)
- SSE for Core→TUI event streaming (agent activity regardless of initiator)

**Options considered:**

1. **SSH (TUI runs on server, accessed via terminal)**
   - Pros: Simple deployment, no API layer between TUI and Core, proven pattern
   - Cons: Mobile = terminal emulator over SSH (clunky), TUI code must live on server, SSH port is larger attack surface than API endpoint
   - Rejected because: Doesn't support the "replace claude.ai with mobile" goal cleanly; we'd need the API anyway for Telegram/other integrations

2. **WebSocket (bidirectional on single connection)**
   - Pros: Single connection, true bidirectional streaming
   - Cons: More complex protocol, some proxy/LB issues, need custom reconnection logic
   - Rejected because: We don't need bidirectional streaming. Commands are short REST calls; only server→client needs streaming.

3. **HTTP + SSE (REST for commands, SSE for events)** ✓
   - Pros: It's just HTTP (proxy/LB friendly), automatic reconnection via EventSource API, we already have REST built, simpler server implementation
   - Cons: Two connection types instead of one (minor)
   - Selected because: Sufficient for our needs, simpler, better infrastructure compatibility

**Key requirement discussed:** "Eager refresh" — TUI should show agent activity regardless of who initiated (self-wake, inter-agent, group chat, Telegram). This requires Core→TUI push capability.

**Pattern:**
- TUI opens persistent SSE: `GET /agents/{id}/events`
- Core pushes granular events (turn started, token delta, tool call, completion, memory write, etc.)
- TUI sends commands via REST
- Catch-up after disconnect: use message history (always being updated anyway) + optionally `Last-Event-ID` for fine-grained replay

**Future consideration:** Real-time streaming of agent activity (tokens as they arrive) is nice-to-have. Can be done via persistent SSE connection. Implementation details deferred.

---

### Core ↔ Self-Wake System

**Decision: IN-PROCESS (Phase 3), external later if needed**

**Options considered:**
- In-process (asyncio background task)
- External scheduler process → HTTP to Core
- Cron/systemd timers → HTTP to Core

**Cron ruled out:**
- One-shot scheduling awkward (cron is for recurring; `at` is separate tool)
- State visibility poor (need to parse crontab to see scheduled wakes)
- Event-based wakes impossible (cron can't do "wake when X happens")

**External vs In-process discussion:**

Initially leaned external for fault isolation (scheduler crash doesn't take down Core). But on examination:
- A simple periodic scheduler is ~100 lines — not meaningfully crash-prone
- The crash isolation argument is strongest for complex code, weakest for trivial code
- External adds: separate service skeleton, systemd unit, HTTP client in Core
- In-process in asyncio is natural: `asyncio.sleep()` loop as background task, yields control, doesn't block

**Why in-process for Phase 3:**
- Simpler to implement (no IPC, no extra service)
- asyncio makes it trivial (cooperative sleep, integrates naturally)
- Phase 3 scope is periodic wakes only — simple enough that isolation overhead isn't justified
- Migration path is clean if we need external later

**Migration trigger:** Event-based wakes (external webhooks, subscriptions, heterogeneous triggers). When that becomes a real requirement, consider extracting to external "event router" service. For now, YAGNI.

**Key constraint:** Keep scheduler as clean internal module with crisp interface like:
- `schedule_wake(agent_id, time, payload)`
- `get_schedules(agent_id)`  
- `cancel_wake(schedule_id)`

Don't entangle with agent loop internals — this makes extraction mechanical when needed.

**Implication:** Self-Wake module lives inside Core box on final diagram.

---

## Final Resolved Architecture (Phase 3)

All connection decisions made. Memory and Self-Wake are internal modules; FS Tools and TUI are external services.

```
                                    ┌─────────────────────┐
                                    │   TUI / Management  │
                                    │       Console       │
                                    └──────────▲──────────┘
                                               │
                                          HTTP + SSE
                                     (REST cmds, SSE events)
                                               │
                                    ┌──────────▼──────────────────────────────┐
                                    │            Core Agent Loop              │
                                    │  ┌─────────────┐  ┌──────────────────┐  │
                                    │  │   Memory    │  │    Self-Wake     │  │
                                    │  │   System    │  │    Scheduler     │  │
                                    │  │ (in-process)│  │   (in-process)   │  │
                                    │  └─────────────┘  └──────────────────┘  │
                                    └────────┬────────────────────▲───────────┘
                                             │                    │
                                            MCP              Case-by-case
                                    (JSON-RPC, portable)    (HTTP, webhooks, MCP, etc.)
                                             │                    │
                                    ┌────────▼────────┐  ┌────────▼────────┐
                                    │    FS Tools     │  │      Misc       │
                                    │   (MCP server)  │  │ (WhatsApp, etc) │
                                    │                 │  │                 │
                                    │ Can run local   │  │ External APIs,  │
                                    │ or remote       │  │ integrations    │
                                    └─────────────────┘  └─────────────────┘
```

**Key properties:**
- **Core is the center**: All external communication goes through Core's REST API
- **Memory is identity**: Not a service, part of what makes the agent *this* agent
- **Self-Wake is simple**: Background task, clean module boundary, extractable later
- **FS Tools are isolated**: Req 13 satisfied — can't accidentally touch agent DB
- **TUI is push-capable**: SSE enables "eager refresh" for all agent activity
- **Misc is flexible**: Each integration chooses appropriate protocol

**Phase 3 deployment:**
- Core + Memory + Self-Wake = single process (uvicorn)
- FS Tools = MCP server (can be local subprocess or remote via Tailscale)
- TUI = client application (CLI/TUI connects to Core over HTTP)

---
