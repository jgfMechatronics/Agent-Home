# Squash Merge Scratchpad — E-LLM Agent Server (Development → main)
*Sonnet exploration notes for Opus to synthesize commit message from*
*May 22, 2026*

---

## Repository Structure

```
E-LLM_Agent_Server/
├── main.py                          # 4-line entry point — re-exports `app` for uvicorn
├── agent/
│   ├── types.py                     # AgentConfig, AgentDeps
│   ├── factory.py                   # AgentFactory (per-request, lock management)
│   ├── tools.py                     # memory_replace, memory_insert
│   ├── compaction.py                # is_compaction_needed, compact
│   └── crud.py                      # get_agent_record, create_agent_record, agent_exists
├── api/
│   ├── app.py                       # FastAPI app factory, lifespan, exception handlers
│   ├── routes.py                    # All HTTP routes (APIRouter prefix /agents)
│   ├── fastapi_deps.py              # FastAPI yield dependencies
│   └── schemas.py                   # Pydantic request/response schemas
├── db/
│   ├── models.py                    # SQLAlchemy ORM models
│   ├── connection.py                # Engine creation, init_db, get_session
│   └── session.py                   # (empty)
├── memory/
│   ├── block_crud.py                # Block CRUD (read = session, write = deps/lock)
│   └── system_prompt_compilation.py # compile_system_prompt, get_system_prompt
├── messages/
│   └── messages.py                  # persist_messages, load_messages, deserialize_messages
├── cli/
│   └── cli.py                       # Interactive + headless CLI (~700 lines)
└── tests/
    └── [22 test files, ~4757 lines, 290+ tests passing]
```

---

## Module Summaries

### agent/types.py
- `AgentConfig` (Pydantic BaseModel, `extra=forbid`): `model_name`, `tool_names`, `soft_compaction_limit`, `compaction_target_percentage` (default 0.25), `is_deletable` (default False), `retries` (default 4), `thinking_enabled` (default False)
- `AgentDeps`: dataclass wrapping `AsyncSession` + `AgentRecord`. Properties provide typed access to config fields. `commit_changes_refresh_agent_record()` — commits session and refreshes agent record (prevents ORM expiry after write)

### agent/factory.py
- `AgentFactory`: per-request factory, takes `lock_reg` (app-wide dict) + `session`
- `build_deps(agent_id)`: async context manager — acquires per-agent `asyncio.Lock` (60s timeout → `AgentLockedError`), fetches `AgentRecord` (not found → `AgentNotFoundError`), yields `AgentDeps`, releases lock on exit regardless of outcome
- `build_agent_and_deps(agent_id)`: wraps `build_deps`, constructs pydantic-ai `Agent` with: model, tools from registry, `cache_context=True`, thinking config (`budget_tokens=10000`, `max_tokens=16000` when `thinking_enabled`), `output_type=[str, DeferredToolRequests]`
- Domain exceptions: `AgentNotFoundError`, `AgentLockedError`

### agent/tools.py
- `memory_replace`, `memory_insert` — Letta-style memory tools callable by agents
- **All failure paths use `raise ModelRetry(...)` — never plain exceptions**
- Occurrence-based matching (1-indexed, not regex)
- Returns edit context snippet on success
- `TOOL_REGISTRY` dict + `get_tools_for_agent(tool_names)`

### agent/compaction.py
- `is_compaction_needed(input_tokens, config)`: simple threshold check against `soft_compaction_limit`
- `compact(deps, input_tokens)`: estimates avg tokens/msg, advances `context_window_start` pointer to trim oldest messages, **never deletes messages** (pointer-only), minimum 4 messages always kept, calls `compile_system_prompt` after advancing

### agent/crud.py
- `get_agent_record(session, agent_id)` — `session.get()` by PK
- `agent_exists(session, agent_id)` — `select + exists`
- `create_agent_record(session, name, system_instructions, config)` — `session.add() + flush()`

### api/app.py
- FastAPI lifespan: creates SQLite engine at `/data/db.sqlite`, `init_db`, stores `engine` + `agent_lock_reg={}` on `app.state`
- App-level exception handlers (centralized, prevents drift across routes):
  - `AgentNotFoundError` → 404
  - `AgentLockedError` → 503
  - `Exception` → 500 (detail exposed — intended for self-hosters)
- `_exc_detail(exc)` → `"TypeName: msg"` format, used by all handlers
- `_create_app()` factory for test isolation
- `/health` endpoint returning `{"status": "ok"}`

### api/routes.py
- `APIRouter(prefix="/agents")`
- Routes:
  - `POST /` — create agent (201)
  - `GET /{id}` — agent metadata
  - `POST /{id}/messages` — SSE streaming chat (EventSourceResponse)
  - `POST /{id}/recompile_system_prompt` — manual recompile trigger
  - `GET /{id}/memory/blocks` — read core memory
  - `POST /{id}/memory/blocks` — create memory block (201)
  - `GET /{id}/messages` — message history (?full=true for all)
- `map_to_sse(event)`: event type name in SSE `event:` field; `AgentRunResultEvent` → `data={}` (stream-end signal only)
- `send_message` flow: load history → `run_stream_events` → persist on `AgentRunResultEvent` → commit → maybe compact
- Error in `send_message` → rollback + yield SSE `Error` event

### api/fastapi_deps.py
- `get_lock_reg(request)` — returns `app.state.agent_lock_reg`
- `get_session_dep(request)` — yields session from `app.state.engine` via `get_session()`
- `get_agent_and_deps(agent_id, session, lock_reg)` — yields `(Agent, AgentDeps)`, holds lock
- `get_deps_dep(agent_id, session, lock_reg)` — yields `AgentDeps` only (no Agent), holds lock — "best function name in the entire codebase"
- Domain exceptions propagate to app-level handlers (not caught here)

### api/schemas.py
- Requests: `MessageRequest` (non-empty str), `CreateAgentRequest`, `CreateMemoryBlockRequest`
- Responses: `AgentMetadataResponse` (with `from_record` classmethod), `MemoryBlockResponse` (with `from_record`), `CoreMemoryResponse`, `MessagesResponse` (`list[Any]` — format deferred), `HealthResponse`

### db/models.py
- `utcnow()` — returns naive UTC datetime (strips tzinfo for SQLite compatibility)
- `AgentConfigType` — `TypeDecorator` storing `AgentConfig` as JSON, with migration hook in `process_result_value`
- `AgentRecord`: `id` (uuid str), `name`, `agent_config`, `system_instructions`, `compiled_system_prompt`, `sys_prompt_compiled_at`, `context_window_start`, `created_at`, `updated_at`. Cascade-deletes memory_blocks and messages.
- `MemoryBlockRecord`: `agent_id` FK, `label`, `description`, `content`, `char_limit`, `position`. Unique constraints on `(agent_id, label)` and `(agent_id, position)`. Indexed on `(agent_id, position)` and `(agent_id, label)`.
- `MessageRecord`: `agent_id` FK, `type` (`ModelRequest`/`ModelResponse`), `content` (TEXT — raw serialized JSON), `input_tokens` (nullable, set on final row only), `timestamp`. Indexed on `(agent_id, timestamp)`.

### db/connection.py
- `create_sqlite_engine(db_path)` — `NullPool`, FK PRAGMA via sync_engine event listener
- `init_db(engine)` — `Base.metadata.create_all`
- `get_session(engine)` — asynccontextmanager: commit-on-success / rollback-on-exception / close-in-finally

### memory/block_crud.py
- **Read ops take `(session, agent_id)`** — no lock required, concurrent reads ok
- **Write ops take `(deps: AgentDeps)`** — proves lock held, enables `commit_changes_refresh_agent_record()`
- `get_blocks`, `get_block`, `update_block` (char_limit check), `create_block` (auto-position at end), `delete_block`, `reorder_blocks` (two-phase: negative positions → final, avoids unique constraint collision)
- `DuplicateBlockError` exception for duplicate labels
- `_persist(deps, commit, record)` — flush or commit+refresh helper

### memory/system_prompt_compilation.py
- `compile_system_prompt(deps)`: fetches blocks in position order, assembles XML:
  - `<system_instructions>\n{instructions}\n</system_instructions>`
  - Per block: `<{label}>\n<description>…</description>\n<metadata>\nchars_current: N\nchars_limit: M\n</metadata>\n<content>…</content>\n</{label}>`
  - Joined with `\n`, stored in `deps.compiled_system_prompt`, flushes
- `get_system_prompt(ctx_or_deps)` — returns cached compiled prompt, **does NOT recompile** (deferred compilation principle)

### messages/messages.py
- `persist_messages(deps, messages, input_tokens)`:
  - Runs `_replace_orphaned_tool_messages` first
  - Per-message serialization with `_handle_serialization_error` on failure (injects error ModelResponse)
  - `_bump_timestamp_if_needed` — bumps 1µs if out-of-order, logs warning
  - Summary warnings appended at END of chain (not buried in history)
  - `input_tokens` set on final row only
  - Flushes (caller commits)
- `_replace_orphaned_tool_messages(messages)`:
  - Orphaned call: ModelResponse with ToolCallPart not immediately followed by matching return
  - Orphaned return: ModelRequest with ToolReturnPart OR RetryPromptPart not immediately preceded by matching call
  - **RetryPromptPart is a valid response to ToolCallPart** (handles ModelRetry case — live test bug fix)
  - `_is_valid_tool_pair`: `call_ids == return_ids` (equality, not subset)
  - `_make_orphan_replacement`: uses `match/case` on part type name, handles `RetryPromptPart.tool_name` nullable
- `load_messages(session, agent_id, start_timestamp)` — chronological, optional start filter (compaction window)
- `deserialize_messages(records)` — pure function, wraps each record in `[…]` for TypeAdapter, raises `ValueError` on failure

### cli/cli.py
- ~700 lines. Interactive and headless (`--headless`) modes.
- Commands: `/create` (wizard or `-q` quick), `/use`, `/chat`, `/history`, `/info`, `/memory`, `/newblock`, `/recompile`
- Default input (no `/` prefix) → chat
- `_StreamState` dataclass — tracks `in_thinking`/`had_thinking`/`response_started` across SSE events
- Thinking display: `[Thinking]` header + content + `─×40` separator line on transition to text response
- `--invoker` auto-resolved from `AGENT_ID` env var → agents.json lookup (for agent-to-agent invocation)
- Headless mode outputs JSON for all events (structured for agent consumption)
- History display handles thinking, tool-call, tool-return, retry-prompt parts

---

## Key Architecture Decisions

1. **Deferred compilation**: `get_system_prompt` returns cached compiled prompt — does NOT recompile on every call. Recompile is explicit (route or after compaction).

2. **Per-agent locking**: `asyncio.Lock` per agent_id in app-wide registry. Routes requiring write access acquire lock via `get_agent_and_deps` or `get_deps_dep`. Lock prevents concurrent mutation of same agent.

3. **Flush-then-commit pattern**: Tools and persistence functions call `session.flush()` only. Single atomic commit at end of request (in `get_session` or `commit_changes_refresh_agent_record`). Rollback on any exception.

4. **Message storage**: One row per `ModelMessage`, content as raw serialized JSON (not JSONB). `input_tokens` on final row only. Never deletes messages — compaction advances pointer.

5. **Orphan handling + RetryPromptPart**: Live test revealed ModelRetry produces `RetryPromptPart` (not `ToolReturnPart`). Fixed to treat both as valid tool responses. Errors don't crash persistence — injected as `ModelResponse(TextPart)`.

6. **App-level exception handlers**: `AgentNotFoundError`→404, `AgentLockedError`→503, `Exception`→500. Applied consistently from any raise site. Detail strings exposed (self-hoster intent).

7. **Extended thinking**: Controlled by `AgentConfig.thinking_enabled`. When true: `budget_tokens=10000`, `max_tokens=16000`. `output_type=[str, DeferredToolRequests]` works with thinking (pydantic-ai sets `tool_choice='auto'` not `'required'`).

8. **Block write ops require deps**: Proof of lock held. Read ops take plain session. Enforced by function signatures.

---

## Test Coverage

22 test files, ~4757 lines, **290+ tests passing** (5 pre-existing xfail).
Mirrors source structure exactly:
- `tests/agent/` — types, crud, factory, tools, compaction
- `tests/api/` — app (exception handlers), fastapi_deps, map_to_sse, routes, schemas  
- `tests/db/` — connection, models, ORM expiry
- `tests/memory/` — block_crud, system_prompt_compilation
- `tests/messages/` — full persist/load/deserialize + orphan detection + RetryPromptPart

---

## What Shipped (Development branch highlights)

The Development branch represents the complete build of the E-LLM Agent Server from scratch, replacing Letta as the foundation for persistent AI agents. Key deliverables:

- **Full agent lifecycle**: create, configure (model, tools, thinking, compaction params, retries), persist, converse
- **Memory system**: named blocks with char limits, XML-compiled system prompts, tools to read/write blocks in-context
- **Streaming chat**: SSE endpoint with pydantic-ai `run_stream_events`, full event type pass-through
- **Message persistence**: robust orphan detection (tool call/return/retry), serialization error injection, timestamp ordering, compaction-aware loading
- **Pointer-based compaction**: soft limit threshold, target percentage, minimum context floor, never deletes history
- **Per-agent concurrency control**: asyncio.Lock registry, 60s timeout → 503
- **Extended thinking**: config-driven, haiku-4-5+ compatible
- **CLI**: interactive wizard + headless mode for agent use, thinking-aware display, auto-invoker identity
- **290+ tests** covering all layers, TDD throughout
- **Live tested**: two concurrent agents, persistence confirmed, first contact made
