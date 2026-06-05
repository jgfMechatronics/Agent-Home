# Toad TUI Evaluation - Scratch Notes

## Overview
Toad = pure ACP thin client by Will McGugan (Textual/Rich creator)
- 3.2k stars, active development
- Built on Textual framework
- AGPL licensed (commercial available)

## Critical Architecture Finding
**Toad speaks STDIO ACP only** - spawns agent subprocess, communicates via stdin/stdout JSON-RPC.

Our Agent Home is HTTP. **Bridge required:**
```
agent-home acp (CLI) → reads stdin JSON-RPC → HTTP to Agent Home server → writes stdout
```

Agent registration via TOML files in `src/toad/data/agents/`. Pattern:
```toml
identity = "agenthome.dev"
name = "Agent Home"
protocol = "acp"
run_command."*" = "agent-home acp"
```

---

## Final Requirements Assessment

### ✅ Fully Supported (Native)

| Req | Feature | How Toad Handles It |
|-----|---------|---------------------|
| 1 | Streaming typewriter | `AgentMessageChunk` → `messages.Update` posted to conversation |
| 2 | External activity display | Message loop handles server-initiated `session/update` — bridge forwards SSE as JSON-RPC |
| 3 | Tool calls | `ToolCall` widget with status badges, expand/collapse, content types |
| 4 | Git-style diffs | `ToolCallContentDiff` → `textual_diff_view.DiffView` (split/unified, syntax highlight) |
| 5 | Approve/Deny | `session/request_permission` → `Question` widget (a/A/r/R keybindings) |
| 7 | Esc to halt | Double-tap Esc → `agent.cancel()` → `session/cancel` RPC |
| 8 | Landing page | `screens/store.py` - agent picker, project directory selector |
| 9 | Display modes | Sidebar shipped (ctrl+B) with Plan + DirectoryTree panels. Can add custom panels. |
| 11 | Resizing | Textual framework handles natively |
| 12 | Non-blocking streaming | Prompt stays active during streaming. Server must handle concurrent sends. |

### ⚠️ Partial / On Roadmap

| Req | Feature | Status |
|-----|---------|--------|
| 6 | Manage tool permissions | README roadmap: "UI for MCP servers" planned but not shipped |

### ❌ Not Currently Supported

| Req | Feature | Status |
|-----|---------|--------|
| 10 | Image paste | `image: False` hardcoded in capabilities. Protocol supports `ImageContent` but UI doesn't handle it. |

---

## Goals Assessment

| Goal | Status |
|------|--------|
| 1. Claude Code styling | ✅ Toad already has this aesthetic |
| 2. Custom theming | ✅ Settings system exists ("Almost everything in Toad may be tweaked") |
| 3. Reusable widgets | ✅ Already componentized (ToolCall, Question, DiffView, etc.) |

---

## Key Code References

- `acp/agent.py` (889 lines) - ACP client, message loop, RPC handlers
- `acp/protocol.py` (458 lines) - Type definitions for ACP messages
- `widgets/tool_call.py` - Tool call rendering with diff support
- `widgets/question.py` - Permission prompt UI
- `widgets/conversation.py` - Main chat interface, cancel handling
- `screens/store.py` - Landing page / agent picker
- `screens/main.py` - Main screen with sidebar (ctrl+B)

---

## Bridge Architecture

```
┌─────────────┐      stdio       ┌─────────────────┐      HTTP/SSE      ┌──────────────┐
│    Toad     │ ←──────────────→ │  agent-home acp │ ←────────────────→ │  Agent Home  │
│   (TUI)     │   JSON-RPC       │    (bridge)     │                    │   (server)   │
└─────────────┘                  └─────────────────┘                    └──────────────┘
```

Bridge responsibilities:
1. Read JSON-RPC requests from stdin
2. Translate to HTTP calls to Agent Home API
3. Subscribe to SSE stream for session updates
4. Forward server events to stdout as JSON-RPC notifications
5. Handle session lifecycle (initialize, new_session, prompt, cancel)

---

## Open Items

1. **Image support workaround:** Could accept file paths and have Agent Home read them server-side? Or wait for toad to implement.

2. **Req 6 workaround:** Could add our own MCP panel to sidebar? Need to expose MCP server status via ACP extension or custom panel.

3. **Bridge complexity estimate:** Core RPC methods to implement:
   - `initialize` 
   - `session/new`
   - `session/load` (for resume)
   - `session/prompt`
   - `session/cancel`
   - `session/set_mode`
   - Plus fs/terminal callbacks if we want toad to handle those (or we handle server-side)
