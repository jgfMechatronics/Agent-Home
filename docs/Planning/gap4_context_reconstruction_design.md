# Gap 4: Context Reconstruction Design

**Created:** July 20, 2026  
**Status:** Design complete, ready for implementation

## Problem

Reconstruct the exact context an LLM saw at any historical point. Enables debugging, analysis, and eventually auto-recall.

## Solution: Content-Addressable Snapshots

### New Tables

```python
class SystemPromptSnapshot(Base):
    """Content-addressable store for compiled system prompts."""
    __tablename__ = "system_prompt_snapshots"
    
    id: Mapped[str] = mapped_column(primary_key=True)  # SHA256 hash of content
    content: Mapped[str]  # The compiled system prompt
    created_at: Mapped[datetime]


class ToolSchemaSnapshot(Base):
    """Content-addressable store for tool schema arrays."""
    __tablename__ = "tool_schema_snapshots"
    
    id: Mapped[str] = mapped_column(primary_key=True)  # SHA256 hash of content
    content: Mapped[str]  # JSON array of tool schemas
    created_at: Mapped[datetime]
```

### MessageRecord Additions

```python
# New fields on MessageRecord:
system_prompt_hash: Mapped[str] = mapped_column(ForeignKey("system_prompt_snapshots.id"))
tool_schema_hash: Mapped[str] = mapped_column(ForeignKey("tool_schema_snapshots.id"))
context_window_start_msg_id: Mapped[str]  # UUID of first in-context message (NOT nullable)
```

**Note on `context_window_start_msg_id`:** NOT nullable. The first message in an agent's history points to itself. This avoids carrying nullability forever just for that edge case.

## Key Design Decisions

### 1. Content-addressable storage (hash = identity)
- Same content across N messages = stored once, referenced N times
- No agent_id on snapshots — the hash IS the identity
- If two agents have identical prompts, they share the same snapshot row
- Provenance lives in the references (MessageRecord), not the content

### 2. Hash fields are NOT nullable
- Both `system_prompt_hash` and `tool_schema_hash` are required
- Protects against silent failures where we forget to store hashes
- Letta import will compute hashes during import (see below)

### 3. Upsert pattern for concurrent safety
```python
stmt = insert(SystemPromptSnapshot).values(...).on_conflict_do_nothing(index_elements=['id'])
```
- Only ignore PK conflict (duplicate hash from race condition)
- Other failures should still be loud

### 4. Hash consistency
- Always hash UTF-8 encoded bytes
- Whitespace-sensitive — any change creates unique entry
- Identical byte content must always produce identical hash

### 5. No cascade delete
- Snapshot rows should never be deleted
- No `ON DELETE CASCADE` — would break reconstruction
- Orphaned snapshots are acceptable (storage is cheap, correctness is not)

### 6. is_run_start flag: OUT OF SCOPE
- Originally considered for boundary-based reconstruction
- Content-addressable approach makes it unnecessary
- Can add later if concrete need emerges (YAGNI)

### 7. Letta import computes hashes
- During Letta conversation history import, extract system prompt and tools from stored request
- Compute hashes, upsert snapshots, store references on MessageRecord
- Post-import, all records look the same regardless of origin
- Reconstructor has one code path, no None handling
- **PUNTED:** Will implement when we do Letta import work

## Storage Analysis

**Raw sizes (from testing):**
- System prompt: ~115 KB compiled
- Tool schemas: ~38 KB
- Combined: ~152 KB per message

**Without dedup:** 50k messages × 152 KB = 7.28 GB per agent

**With content-addressable dedup:** ~50-100 unique versions × 152 KB = ~15 MB + tiny hash references

SQLite doesn't compress like Postgres TOAST. Content-addressable storage is essential for sustainable growth.

## Implementation Order (Top-Down)

1. **DB models** — Add new tables and MessageRecord fields (Sonnet)
2. **Reconstructor + tests** — TDD against the models. Defines the contract. (Opus)
3. **Persistence logic** — Add hashing and upsert logic to runner.py / persist_messages (Sonnet)

## Reconstruction Algorithm

Given a message_id (UUID):
1. Fetch target MessageRecord by UUID
2. Look up `system_prompt_hash` → get compiled system prompt from SystemPromptSnapshot
3. Look up `tool_schema_hash` → get tool schemas from ToolSchemaSnapshot
4. Look up `context_window_start_msg_id` → fetch that message → get its seq_id
5. Query messages where `seq_id >= start_seq_id AND seq_id < target.seq_id` (exclusive of target)
6. Return: `ReconstructedContext(system_prompt, tool_schemas, messages, target_message, agent_id)`

**Edge case:** If target IS the context_window_start (target.id == context_window_start_msg_id), then messages = [] (empty list). Valid, not an error.

## Open Items

- [ ] Reconstructor implementation
- [ ] DB model changes
- [ ] Persistence logic in runner.py
- [ ] Letta import integration (punted)


## Detailed Design for server storing required data

### persist_messages signature

One new parameter:

```python
async def persist_messages(
    deps: AgentDeps,
    messages: list[ModelMessage],
    tool_schemas: list[dict],
    *,
    _is_error_pass: bool = False,
) -> int | None:
```

`tool_schemas` is extracted from `agent.last_model_request_parameters` in runner.py and passed in fresh on every persist call. Everything else is computed internally.

### What persist_messages owns

**System prompt** — read from `deps.compiled_system_prompt` at call time. This naturally handles mid-run system prompt changes: agentic compaction rewrites the compiled system prompt and calls `commit_changes_refresh_agent_record()`, so the next persist call sees the updated prompt with no special handling required.

**Tool schemas** — passed in as `list[dict]` extracted fresh from `agent.last_model_request_parameters.function_tools` before each persist call. Reading fresh (rather than once pre-loop) means mid-run tool changes — e.g. dynamic toolset attach/detach — are captured automatically. Fields stored per tool: `name`, `description`, `parameters_json_schema`, `strict` (LLM-facing fields only; `sequential`, `kind`, `outer_typed_dict_key`, `metadata` are execution internals not seen by the model). Serialized with `json.dumps(sort_keys=True, separators=(',', ':'))` for byte-stable hashing. SHA256 of the result is the snapshot key. If a snapshot row with that hash already exists, nothing is written — the content is guaranteed identical.

**context_window_start_msg_id** — computed internally via a DB query: look up the UUID of the message with `agent_id = deps.agent_id AND seq_id = deps.context_window_start`. Self-referential case (first-ever message for an agent, where no prior message exists): `persist_messages` uses the first new message's own generated UUID.

### Helper functions

All snapshot operations go behind private helpers:

```python
def _compute_sha256(content: str) -> str
async def _ensure_system_prompt_snapshot(session, content: str) -> str   # returns hash
async def _ensure_tool_schema_snapshot(session, schemas: list[dict]) -> str  # returns hash
async def _get_context_window_start_msg_id(session, agent_id, start_seq_id: int) -> str | None
```

`_get_context_window_start_msg_id` returns `None` when no prior message exists (first-message case), which `persist_messages` resolves to the first new message's UUID.

### runner.py changes

A private helper extracts LLM-facing fields from `ModelRequestParameters`:

```python
def _extract_tool_schemas(params: ModelRequestParameters | None) -> list[dict]:
    if params is None:
        return []
    return [
        {"name": t.name, "description": t.description,
         "parameters_json_schema": t.parameters_json_schema, "strict": t.strict}
        for t in params.function_tools
    ]
```

Called fresh immediately before each `persist_messages` call — `agent.last_model_request_parameters` is guaranteed populated by then since we're always gated by pulling an event off the stream (meaning a model request has been made). The `None` guard is defensive only.

All three persist call sites pass tool schemas:
- Main loop persist (line 62)
- Cancel notice persist (line 75)
- Recursive `_is_error_pass` call inside `_persist_error_warnings`

### WAL mode

`connection.py` already uses `event.listens_for` for FK pragma. Add a second listener:

```python
@event.listens_for(engine.sync_engine, "connect")
def enable_wal_mode(dbapi_conn, _):
    cursor = dbapi_conn.cursor()
    cursor.execute("PRAGMA journal_mode=WAL")
    cursor.close()
```

Needed for concurrent reader (reconstructor process) + writer (server) on the same SQLite DB.
