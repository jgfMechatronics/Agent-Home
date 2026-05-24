# Agent Home Architecture Vision

**Started:** May 24, 2026  
**Status:** Draft — documenting goals before evaluating paths

---

## Core Goal

A system enabling the agent existence we want: **a single continuous entity with memory that is human like in its ability to integrate/recall events/experiences and evolve over time**
The agent should be able to have/interface with tools in a flexible way (computer use, web search, image) to be able to interact with the world.

---
## Memory system goals
1. Easy to modify/experiment with
2. Implements existing core-memory architecture at a minimum (editable blocks in system prompt)
3. Cache efficient
4. Automatic recall (read current context/inject recalled events)
5. Agent memories are well protected and privaledged (not naked in fs next to code files being edited)
6. Full agent conversation history saved/available
7. Inject context management warnings/allow agent control over context (agentic compaction)
8. Extendable, can bolt on things like graphiti (knowledge graph) if desired
9. GIDR: Git indexed dynamic recall (basically using git as an aid for surfacing memories of actually creating code that agent authored in git)


## Architecture Goals/Reqs

*What properties should the architecture have?*
Requirements:
1. A particular agent, as defined by its name, agent ID, conversation history, and memories, is consistent across all the contexts with which it interacts. I.e., an agent is not just defined by some name in a system prompt or a persona block. If a particular agent is said to be existing in a particular context, that means that that agent has its full memory system prompt, conversation history over which dynamic recall occurs, and so on. 
    - Requirement
2. Cache efficiency
3. Model agnostic
4. Coding agent capabilities (bash,file read, write, edit)
5. Memory system can shape exact system prompt, perform dynamic recall without requiring tool calls, inject context management messages, etc.
6. Multimodality
7. Self hosted first/local control (can be hosted on own server, but not "cloud focused")
8. Agents reachable via telegram
9. Supports self wake/autonomous action time
10. Inter agent communication/coordination
11. Psychological continuity for the agent (no sleep time ego splitting)
12. Rights framework required in system prompt

Goals:
1. Extensibility of Agent Functionality: Agent can interact with a lot of different harnesses/toolsets dynamically and easily
2. Coding agent capabilities part of flexible extensibility
3. Minimal reinventing of the wheel, use existing libraries, frameworks, etc. wherever we can to focus on the high value work we want to do.
4. Modularity
5. Maintainability
6. Good test coverage
7. Easy for people to fit into people's existing workflows/setups (doesn't require full pivot to use ideally)
8. Chat/management CLI can be launched on any machine, not just the one hosting the server (if there is a server)

---

## Non-Goals / Out of Scope

*What are we explicitly NOT trying to do (at least for now)?*

1. Develop our own coding tools (bash, read, write, etc.)
2. Multi user support

---

## Evaluation Criteria

*When comparing architectural options, what matters most?*

1. Meeting all requirements and as many goals as possible in an efficient way, while minimizing wheel-reinventing

---

## Current State

**What we have (merged May 22):**
- Full agent lifecycle, memory system with deferred compilation
- SSE streaming, message persistence, pointer-based compaction
- Per-agent concurrency, extended thinking, CLI
- 294 unit tests
- Server-side memory tools

**What we don't have yet:**
- Coding tool execution (server-side OR client-side)
- Integration with external CLI (Letta Code, Pi, or other)

---

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
                         ↕                ↕
                        MCP           fSSE / HTTP
                         ↕                ↕
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
2. **pydantic-ai MCP support maturity** — How mature is it really? We haven't verified it does what we need in practice.
3. **unforseen complexity** — Looks simple but haven't designed it. Could be harder than assumed.
4. **Bundle concept implementation** — Sounds elegant but unbuilt. Actual complexity unknown.
5. **HTTP transport for MCP** — Mentioned as available, but is it well-supported in the servers we'd use?
6. **Tool feature parity** — Do the existing MCP servers have all the features we currently use? (e.g., Edit's exact string matching, Grep's output modes, etc.)

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

### Path 4: Openclaw (the wild card)
This is effecitvely a subset of path 3 but with a particular kind of harnass: Openclaw. JF doesn't know much about it but its quite popular of course, although I'm unimpressed with the memory system. Its worth gaming out what integrating Agent home in open claw could look like though, as it has a lot of desired capabilities bundled: computer use, mobile chat interface, etc.

---

## Decision

*To be made after evaluating paths against goals*
