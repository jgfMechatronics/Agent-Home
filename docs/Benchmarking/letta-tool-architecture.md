# Letta Tool Architecture Benchmarking

**Date:** May 24, 2026  
**Author:** Sonnet  
**Purpose:** Map how Letta handles the client/server tool split, as reference for Agent Home Phase 3 design.

---

## Summary

Letta has **two distinct tool execution modes** that are mutually exclusive per session:

- **Server-side (APIBackend):** All tools run on the server. Client is a display layer only.
- **Client-side via `ClientToolSchema`:** Server pauses on specific tools, client executes and resumes.

There is no persistent middle ground — but the `ClientToolSchema` mechanism proves Letta solved the pause/resume problem already.

---

## Server-Side Tool Execution

### ToolExecutorFactory (`tool_execution_manager.py`)

Routes tool execution based on `ToolType` enum:

| ToolType | Executor | What it does |
|----------|----------|--------------|
| `LETTA_CORE`, `LETTA_MEMORY_CORE`, etc. | `LettaCoreToolExecutor` | Memory/DB operations, server-side |
| `LETTA_BUILTIN` | `LettaBuiltinToolExecutor` | `run_code` (E2B cloud), `web_search` / `fetch_webpage` (Exa API) |
| `LETTA_FILES_CORE` | `LettaFileToolExecutor` | File operations |
| `EXTERNAL_MCP` | `ExternalMCPToolExecutor` | MCP calls via `MCPManager` — fully server-side; comment says "MCP is local / doesn't support remote" |
| `CUSTOM` (default) | `SandboxToolExecutor` | Runs tool source code as subprocess; tries Modal → E2B → LOCAL |

**LOCAL sandbox mechanism:** Generates Python script from tool `source_code`, writes to temp file, executes as `asyncio` subprocess, optionally inside a venv. Completely server-side.

### Agent loop (`agent.py` / `letta_agent_v3.py`)

`agent.py` is the old-style sync agent (1758 lines). `letta_agent_v3.py` is modern.  
`execute_tool_and_persist_state()` dispatches via `ToolExecutorFactory` and persists state after each tool call.

---

## Client-Side Tool Execution (`ClientToolSchema`)

### How it works

1. **Client declares tools** in request body: `client_tools: [ClientToolSchema]`
2. **Agent calls a client tool** → `base_agent_v2.py` checks if tool name is in `client_tool_names` set
3. **Execution pauses**: `stop_reason = StopReasonType.requires_approval`, `PendingApprovalError` raised
4. **Server creates** `approval_request_message` and persists it, then streams terminal event
5. **Client receives** stream ending with `stop_reason: "requires_approval"` 
6. **Client executes** the tool locally
7. **Client resumes** by sending next message with `ToolReturn` object (includes `tool_call_id`, `status`, `tool_return`)

`ToolReturn` schema:
```python
class ToolReturn(MessageReturn):
    type: Literal[MessageReturnType.tool]
    tool_return: Union[str, List[LettaToolReturnContentUnion]]
    status: Literal["success", "error"]
    tool_call_id: str
    stdout: Optional[List[str]] = None
    stderr: Optional[List[str]] = None
```

### Key implementation location

`base_agent_v2.py` ~line 1543:
```python
# Get names of client-side tools (these are executed by client, not server)
client_tool_names = {ct.name for ct in self.client_tools} if self.client_tools else set()

# Tools requiring approval: requires_approval tools OR client-side tools
requested_tool_calls = [
    t for t in tool_calls
    if tool_rules_solver.is_requires_approval_tool(t.function.name) or t.function.name in client_tool_names
]
```

### `StopReasonType` values

```
end_turn, max_steps, tool_rule, requires_approval, error, cancelled,
insufficient_credits, invalid_tool_call, no_tool_call, invalid_llm_response,
llm_api_error, max_tokens_exceeded, context_window_overflow_in_system_prompt
```

---

## SSE Stream Format

### Server → Client events (`streaming_service.py`)

```
data: {LettaStreamingResponse JSON}\n\n   # per-event chunk
data: {LettaStopReason JSON}\n\n           # before terminal
data: [DONE]\n\n                           # terminal
event: error\ndata: {JSON}\n\n             # error events
```

### LettaStreamingResponse message types

From `local-backend.test.ts` and `letta_message.py`:

| `message_type` | Description |
|---------------|-------------|
| `assistant_message` | Text content from the agent |
| `tool_call_message` | Agent is calling a tool |
| `tool_return_message` | Tool execution result |
| `stop_reason` | Terminal: `{message_type: "stop_reason", stop_reason: "end_turn"}` |
| `reasoning_message` | Agent's reasoning/thinking content |
| `system_message`, `user_message` | Chat messages |

### Letta Code SDK

The Letta TypeScript SDK parses raw SSE bytes → `Stream<LettaStreamingResponse>` objects.  
`APIBackend.createConversationMessageStream()` delegates to `client.conversations.messages.create()`.  
**Implication for Path A:** We implement the SSE endpoints; the SDK handles parsing. We don't need to worry about client-side stream parsing.

---

## Relevant Endpoints

### Primary (what Letta Code uses)

| Method | Path | Purpose |
|--------|------|---------|
| `POST` | `/v1/conversations/{id}/messages` | Send message + receive SSE stream |
| `GET` | `/v1/conversations/{id}/messages` (stream variant) | Retrieve existing stream by `otid` + `starting_after` |
| `POST` | `/v1/conversations/{id}/cancel` | Cancel in-flight run |

### Agent-direct mode (legacy compat)
`conversation_id="default"` + `agent_id` in body → uses per-agent locking, skips conversation features.  
`conversation_id="agent-*"` → also agent-direct (deprecated).

### ConversationMessageRequest notable fields
```python
messages: list[MessageCreate]         # the user turn
client_tools: list[ClientToolSchema]  # optional client-side tools
streaming: bool = True                # default streaming
stream_tokens: bool = False           # token-level streaming
max_steps: int                        # step limit
override_model: str                   # per-request model override
include_return_message_types: list    # filter what comes back
```

---

## Architecture Summary: Server vs Client

```
┌─────────────────────────────────────────────┐
│              Letta Server                   │
│                                             │
│  Agent Loop (letta_agent_v3.py)             │
│    ↓                                        │
│  ToolExecutorFactory                        │
│    ├── Core tools (memory, DB)              │
│    ├── Builtin tools (E2B, Exa)             │
│    ├── Custom tools (subprocess/sandbox)    │
│    └── MCP tools (MCPManager, local)        │
│                                             │
│  If ClientToolSchema match:                 │
│    → stop_reason: requires_approval         │
│    → stream ends, client takes over         │
└─────────────────────────────────────────────┘
              ↕ SSE / HTTP
┌─────────────────────────────────────────────┐
│              Client (Letta Code)            │
│                                             │
│  APIBackend → Letta TS SDK                  │
│    → Stream<LettaStreamingResponse>         │
│    → handles requires_approval if declared  │
└─────────────────────────────────────────────┘
```

---

## Implications for Agent Home

### Path A (implement Letta REST API subset)

We implement `POST /v1/conversations/{id}/messages` returning SSE.  
Our server currently runs the **full agent loop** — all tools are server-side (same as Letta's default).  
The Letta TypeScript SDK handles SSE parsing for us.

**Events we need to emit:**
1. `{message_type: "assistant_message", content: [{type: "text", text: "..."}], ...}` — text output
2. `{message_type: "tool_call_message", ...}` — tool calls (if we want to surface them)
3. `{message_type: "stop_reason", stop_reason: "end_turn"}` — before [DONE]
4. `data: [DONE]\n\n` — terminal

**Important clarification:** Our *current* Letta setup (Opus on LettaServerProd + Letta Code) uses `ClientToolSchema` for ALL coding tools (Bash, Read, Write, Edit, Grep, etc.). Letta Code always sends `client_tools` in every request via `sendMessageStream` — it's not configurable. The Letta server pauses on each tool call via `requires_approval`, Letta Code executes locally, and returns a `ToolReturn` to resume.

**For co-located Phase 3 dogfooding (Agent Home + Letta Code on same machine):** Accept `client_tools` in the request schema (don't error), but run tools server-side in our pydantic-ai loop. Since Agent Home and the workspace are in the same container, "server-side" vs "client-side" is a distinction without a practical difference. Tools execute in the same filesystem either way.

**For remote deployment or full architectural parity:** Implement `requires_approval` pause/resume (Path B in pi-feasibility-assessment.md). The blueprint exists in Letta's `ClientToolSchema` implementation.

### Future: client-side tool execution

If we ever want Pi or another client to run tools locally, the `requires_approval` pause/resume pattern is the proven design. We'd need:
- Run IDs (to identify a paused run)
- A resume endpoint (submit `ToolReturn` to continue)
- Client-side tool declaration in our request schema

Not Phase 3 scope — but the blueprint exists in Letta's implementation.

---

## Files Referenced

- `letta/agents/letta_agent_v3.py` — modern agent, `stream()` method
- `letta/agents/base_agent_v2.py` — client_tools detection (~line 1543)
- `letta/services/tool_execution_manager.py` — `ToolExecutorFactory`
- `letta/services/sandbox_tool_executor.py` — LOCAL/E2B/Modal sandbox
- `letta/services/builtin_tool_executor.py` — builtin tools (E2B, Exa)
- `letta/services/streaming_service.py` — SSE generation, error handling
- `letta/server/rest_api/routers/v1/conversations.py` — `send_conversation_message` endpoint
- `letta/schemas/letta_request.py` — `LettaRequest`, `ConversationMessageRequest`, `ClientToolSchema`
- `letta/schemas/letta_message.py` — `ToolReturn`, `ApprovalReturn`, `ToolReturnMessage`
- `letta/schemas/letta_stop_reason.py` — `StopReasonType`, `LettaStopReason`
- `src/backend/backend.ts` (Letta Code) — `Backend` interface, `APIBackend`
- `src/backend/local-backend.test.ts` (Letta Code) — SSE event format examples
