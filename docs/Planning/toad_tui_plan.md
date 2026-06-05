# Toad TUI Integration Plan for Agent Home

## Executive Summary

**Recommendation: Adopt toad as our TUI client.**

Toad natively supports 11 of 12 requirements, with the remaining one (image paste) on a reasonable path. The only required work on our side is an ACP stdio-to-HTTP bridge CLI.

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
| 7 | Esc to halt | Double-tap Esc → `session/cancel` RPC |
| 8 | Landing page | Agent picker + project directory selector |
| 9 | Two display modes | Sidebar (ctrl+B) with extensible panels |
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
5. **Session management:** Handle initialize, new_session, load_session, prompt, cancel

### RPC Methods to Implement

| Method | Direction | Agent Home Equivalent |
|--------|-----------|----------------------|
| `initialize` | toad→bridge | Return capabilities |
| `session/new` | toad→bridge | `POST /agents/{id}/sessions` |
| `session/load` | toad→bridge | `GET /agents/{id}/sessions/{session_id}` |
| `session/prompt` | toad→bridge | `POST /sessions/{id}/messages` (SSE response) |
| `session/cancel` | toad→bridge | `POST /sessions/{id}/cancel` |
| `session/set_mode` | toad→bridge | Custom extension |
| `session/update` | bridge→toad | Forward from SSE stream |
| `session/request_permission` | bridge→toad | Forward from SSE, wait for response |

### fs/terminal Callbacks

Toad can handle `fs/read_text_file`, `fs/write_text_file`, and `terminal/*` locally (it has implementations). For remote execution via MCP, we'd either:
- Have toad handle locally (simpler, current MCP proxy approach)
- Forward to Agent Home which forwards to MCP (more complex, true remote)

**Recommend:** Local handling for prototype. Toad already implements these.

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

[actions."*".install]
command = "uv tool install agent-home"
bootstrap_uv = true
description = "Install Agent Home CLI"
```

---

## Implementation Phases

### Phase 1: Spike (Prove the Concept)
- [ ] `agent-home acp` CLI skeleton
- [ ] `initialize` → return hardcoded capabilities
- [ ] `session/new` → create session via Agent Home API
- [ ] `session/prompt` → POST message, subscribe SSE, forward chunks
- [ ] `session/update` forwarding from SSE
- [ ] Basic error handling
- [ ] **Separate console for Agent Home status** (no fork yet)

### Phase 2: Full Session Support
- [ ] `session/load` for resume
- [ ] `session/cancel` 
- [ ] `session/request_permission` flow (pause SSE, prompt toad, resume)
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
| Toad development direction diverges from our needs | AGPL allows forking; toad is well-structured |
| Bridge complexity underestimated | Start with spike, iterate. Core RPC is ~6 methods. |
| Image support becomes blocking | Fork toad to add (Phase 3), contribute upstream |
| ACP protocol changes | Protocol is versioned; toad maintains compatibility |
| Custom panels need fork | Phase 1 uses separate console; Phase 3 fork adds panels properly |

---

## Decision

**Proceed with toad integration.**

The bridge is bounded work (~6 RPC methods for MVP). Toad handles all the hard TUI problems (streaming, diffs, permissions, resize, styling). We focus on our differentiator: the memory system and agent loop.

Next step: Implement Phase 1 bridge as spike to validate the approach.
