# E-LLM Agent Server Implementation Plan

*Mapping architecture to Pydantic AI specifics*

**Reference:** `architecturePlan.md` for high-level design decisions.

**All code should be considered as pseudocode and does not reflect exact implementation details**
**The plan and architecturePlan are NOT necessarily updated with all decisions/changes made during actual implementation**

---

## 1. Agent Loop Implementation

*How we implement each step from the architecture plan using Pydantic AI.*

### Entry Point: Route Handler (Streaming)

**Streaming-first design.** See `architecturePlan.md` Section 3 for rationale and event schema.

**SSE library:** Use `sse_starlette` (`EventSourceResponse`) — handles SSE formatting, keep-alive, and proper headers. Cleaner than manual `StreamingResponse`. (Learned from Arnav's arnagent-server.)

```python
from sse_starlette.sse import EventSourceResponse

@app.post("/agents/{agent_id}/messages")
async def send_message(
    agent_id,
    request,
    agent_factory: AgentFactory = Depends(get_agent_factory),  # per-request, has session + lock_reg bound
):
    async with agent_factory.build_agent_and_deps(agent_id) as (agent, deps):
        records = await load_messages(deps.session, deps.agent_id, start_timestamp=deps.context_window_start)
        history = deserialize_messages(records)
        
        async def event_generator():
            async for event in agent.run_stream_events(...):
                if sse_event := map_to_sse(event):
                    yield sse_event
                if isinstance(event, AgentRunResultEvent):
                    # persist, run compaction, yield done
                    ...
        
        return EventSourceResponse(event_generator())
```

**Key points:**
- `run_stream_events()` returns flat `AsyncIterator` — no context manager needed
- Persistence happens inside stream loop on `AgentRunResultEvent`
- Lock held for entire stream duration (released when generator exhausts)

**Per-agent locking:** `agent_factory.build_agent_and_deps()` acquires lock on entry via `get_deps()` internally, releases on exit. Only one request per agent_id at a time.

**Per-request AgentFactory:** Factory is constructed per-request via `Depends(get_agent_factory)` with session + lock_reg bound. Routes see clean interface (`agent_factory.build_agent_and_deps(agent_id)`), not raw lock_reg.

---

### Step 1: Load Agent State

**What we load:**
- Agent config from DB (AgentConfig JSON, agent_id, `context_window_start`)
- **Message history** (active context only — `WHERE timestamp >= context_window_start`)
- Tool definitions (registered per-agent)

*System prompt is loaded lazily by `get_system_prompt()` and passed as `instructions` to the Agent — not pre-loaded into deps.*

*Note: Message history ≠ conversation history. We load only what's in the active context window, not all messages ever.*

**Pydantic AI mapping:**

`AgentDeps` (defined in `agent/types.py`) — @dataclass(init=False) holding:
- `session: AsyncSession` — direct field for DB operations
- Private `_agent_record: AgentRecord` — accessed via properties (agent_id, name, config, system_instructions, compiled_system_prompt rw, sys_prompt_compiled_at rw, context_window_start rw)
- `commit_changes_refresh_agent_record()` method — commits session and refreshes record atomically

The `AgentDeps` instance is passed to `run_stream_events(deps=...)` and made available to tools via `RunContext[AgentDeps]`.

### Step 2: Build Agent & Messages

**Per-request AgentFactory pattern:**

```python
# App-level lock registry (created in lifespan, stored on app.state)
# lock_reg: dict[str, asyncio.Lock] = {}

class AgentFactory:
    """
    Per-request factory for building agents with locking.
    
    Constructed via FastAPI Depends with session + lock_reg bound.
    Routes call agent_factory.build_agent_and_deps(agent_id) for clean interface.
    
    Read-only operations (get_blocks, get_block) can use session directly
    without factory — allowing concurrent reads while writes are locked.
    
    Write operations require agent_factory.get_deps() or agent_factory.build_agent_and_deps(),
    which proves the caller holds the lock.
    """
    def __init__(self, lock_reg: dict, session: AsyncSession):
        self._lock_reg = lock_reg
        self._session = session
    
    def _get_lock(self, agent_id: str) -> asyncio.Lock:
        if agent_id not in self._lock_reg:
            self._lock_reg[agent_id] = asyncio.Lock()
        return self._lock_reg[agent_id]
    
    @asynccontextmanager
    async def get_deps(self, agent_id: str) -> AgentDeps:
        """Acquire lock (lock-then-fetch), yield deps, release on exit."""
        lock = self._get_lock(agent_id)
        try:
            await asyncio.wait_for(lock.acquire(), timeout=LOCK_TIMEOUT_SECONDS)
        except asyncio.TimeoutError:
            raise AgentLockedError(f"Agent {agent_id!r} did not become available within {LOCK_TIMEOUT_SECONDS}s")
        try:
            agent_record = await self._session.get(AgentRecord, agent_id)
            if agent_record is None:
                raise AgentNotFoundError(f"Agent {agent_id!r} not found")
            yield AgentDeps(self._session, agent_record)
        finally:
            lock.release()
    
    @asynccontextmanager
    async def build_agent_and_deps(self, agent_id: str) -> tuple[Agent, AgentDeps]:
        """Build Agent + AgentDeps with locking. Uses get_deps() internally."""
        async with self.get_deps(agent_id) as deps:
            agent = Agent(
                model=get_model(deps.config.model_name),
                instructions=get_system_prompt,
                tools=get_tools_for_agent(deps.config.tool_names),
                ...
            )
            yield agent, deps

# Module-level (stateless)
def get_model(model_name: str) -> Model:
    """Map DB model string to Pydantic AI model instance."""
    return AnthropicModel(model_name)

# FastAPI dependency
def get_agent_factory(
    session: AsyncSession = Depends(get_session_dep),
    lock_reg: dict = Depends(get_lock_reg),
) -> AgentFactory:
    return AgentFactory(lock_reg, session)
```

**Message history:** Loaded in Step 1, passed via `message_history` parameter to `run_stream_events()`.

**Key insight:** `instructions` can be an async callable — and unlike `system_prompt`, it is always re-evaluated even when `message_history` is non-empty. SQLAlchemy's identity map means repeated calls in the same session return cached objects — no redundant DB hits.

**Locking limitation (MVP):** In-memory lock dict works for single uvicorn worker. Multi-worker would need external locks (Redis, DB advisory locks). Not MVP — we run single worker.

### Step 3: Agent Run (Streaming)

**Pydantic AI handles the tool loop internally.** We use `run_stream_events()` — see route handler above.

**Cache settings** (passed at agent construction or per-run):
- `anthropic_cache_instructions=True` — cache system prompt (biggest win)
- `anthropic_cache_tool_definitions=True` — cache tools
- `anthropic_cache_messages=True` — cache messages (confirmed: `bool | Literal['5m', '1h']`)

**Event types in stream:**
- `PartDeltaEvent` with `TextPartDelta` → forward as `chunk`
- `FunctionToolCallEvent` → forward as `tool_call`
- `FunctionToolResultEvent` → forward as `tool_return`
- `AgentRunResultEvent` → terminal event, carries `result.new_messages()` for persistence

**Memory tools:** Persist immediately during tool execution (in the tool function itself), not batched to end of turn.

### Step 4: Handle Result

**`DeferredToolRequests` is returned as `result.output`** — there is no `result.is_deferred` property. Check with `isinstance`.

**Prerequisite:** `DeferredToolRequests` must be included in the agent's `output_type` at construction, otherwise Pydantic AI raises a `UserError`:
```python
agent = Agent(
    model=...,
    output_type=[str, DeferredToolRequests],  # required for approval flow to work
    ...
)
```

**Source:** `result.py` lines 211-216, 213-214 error message.

```python
from pydantic_ai import DeferredToolRequests

if isinstance(result.output, DeferredToolRequests):
    # Return approval request to user
    return DeferredResponse(
        tool_calls=result.output.approvals,
        # ... metadata
    )
else:
    # Extract final response
    response_text = result.output
```

**Resuming after approval:** Same streaming pattern — `run_stream_events()` with `deferred_tool_results` parameter.

### Step 5: Persist & Return

**What we save:** On `AgentRunResultEvent`, save every message from `result.new_messages()` — one row per `ModelMessage`. This includes the user's `ModelRequest`, the assistant's `ModelResponse`(s), any intermediate tool-return `ModelRequest`s, and any tool-call `ModelResponse`s.

*No pre-saving.* User message is not saved before the run. If the server crashes mid-run, the user resends. Acceptable for MVP.

*All messages go into conversation history. The `context_window_start` pointer determines what's loaded as active context; persisting doesn't touch the pointer.*

**Pydantic AI message types returned by `new_messages()`:**
- `ModelRequest` — user message and/or tool return (`UserPromptPart` / `ToolReturnPart`)
- `ModelResponse` — assistant message, may contain `TextPart`, `ToolCallPart`, and/or `ThinkingPart`

See Database Schema and Message Serialization sections for storage details.

### Step 6: Check Context Size

**Token counting (post-turn):**

`result.usage()` returns a `RunUsage` with `input_tokens` — the actual token count Anthropic billed for the just-completed turn. This is the size of the context we sent, which is a good proxy for what the next turn starts with.

```python
context_size = result.usage().input_tokens
```

**Note:** `request_tokens` is deprecated — use `input_tokens`. **Source:** `usage.py` lines 22-23.

This is cheaper and simpler than calling `model.count_tokens(...)` separately, which requires reconstructing `ModelRequestParameters` manually outside of an agent run context.

**Soft limit check:**
```python
if context_size > soft_compaction_limit:
    await compact(deps)
```

**When to count:** Post-turn only. No need to count before or during a turn for MVP — the goal is just to detect when compaction is needed before the *next* turn starts.

---

## 2. Compaction Implementation

### MVP: Pointer Advancement (No Summaries)

**Trigger:** Post-turn, when `result.usage().input_tokens > soft_compaction_limit`

**The math:**
```python
async def compact(deps: AgentDeps, current_tokens: int):
    """Advance context_window_start to evict oldest messages."""
    
    # System prompt is fixed overhead — can't evict it
    sys_prompt_tokens = deps.agent_config.cached_system_prompt_tokens or 0  # fallback if not yet compiled
    message_tokens = current_tokens - sys_prompt_tokens
    
    # Count messages in current context window
    message_count = await count_messages(deps)
    tokens_per_message = message_tokens / message_count  # average estimate
    
    # Target 40% of soft limit (aggressive margin for estimate drift)
    target_message_tokens = deps.agent_config.soft_compaction_limit * deps.agent_config.compact_fraction
    messages_to_evict = round((message_tokens - target_message_tokens) / tokens_per_message)
    
    # Advance pointer — guard: never evict most recent 4 turns
    messages_to_evict = min(messages_to_evict, message_count - 4)
    if messages_to_evict > 0:
        await advance_pointer(deps, messages_to_evict)
```

**Suggest default compactFraction=0.4:** The equal-distribution assumption drifts low when old messages are short (common pattern: early turns short, later tool results long). Build in margin. Occasional re-compaction is fine; stuck in a compaction loop is not.

**Minimum history guard:** Never evict most recent 4 turns. Prevents pathological case where one giant tool result causes us to evict context we actually need.

**Ground truth arrives next turn:** Anthropic tells us the actual token count. If we're still over, compact again. No need for loops or precise estimates.

### No HistoryProcessor

**Why not:** HistoryProcessor only modifies in-memory state, not DB. With our pointer-based approach, it would mask problems rather than solve them:
- Runaway context continues instead of stopping
- More expensive tokens burned
- Human doesn't know something's wrong

**Fail loudly philosophy:** If compaction isn't keeping up and context overflows, let Anthropic throw the error. That's a circuit breaker — stops the loop and signals intervention needed. Don't paper over it.

**MVP:** Anthropic context error = halt, needs human intervention. Graceful interrupt + auto-retry is future work.

### Future: Agent-Driven Summaries

**Not MVP.** See `agenticCompactionPlan.md` for the `evict_messages_and_recompile` tool design.

When we add summaries:
1. Agent generates summary of messages being evicted
2. Construct `ModelRequest(parts=[UserPromptPart("<summary>...</summary>")])`, insert as row with `type='Summary'`
3. Set `timestamp` to just before the first kept message's timestamp (so it sorts correctly in history)
4. UPDATE agent's `context_window_start` to summary row's timestamp (or slightly before)
5. **No DELETEs** — evicted messages stay in conversation history

*Preserves full audit trail while maintaining active context as a sliding window.*

---

## 3. Cache Control (CRITICAL)

**Without cache hits, we pay full price (~$3/MTok) every turn. With cache hits on system prompt, ~$0.30/MTok.**

### Pydantic AI Support

Fully supported via `AnthropicModelSettings`:

| Setting | What it caches | MVP? |
|---------|---------------|------|
| `anthropic_cache_instructions` | System prompt | **YES** |
| `anthropic_cache_tool_definitions` | Tool definitions | **YES** |
| `anthropic_cache_messages` | Last user message | **YES** |

**Note on `anthropic_cache_messages`:** Requirements section 9 explicitly calls for "a break at the end of message history" as MVP. This setting puts a cache point on the last user message, which serves that purpose. It helps within-turn (tool call iterations can reuse the cached context) and across closely-spaced turns (5min TTL). Changed from "No" — this one is required.

**Source:** `models/anthropic.py` lines 184-207

### Usage

Pass via `model_settings` to `run_stream_events()` or set at agent construction (merges with per-run). Key settings: `anthropic_cache_instructions`, `anthropic_cache_tool_definitions`, `anthropic_cache_messages`.

### Why Deferred Compilation Matters for Caching

**Problem:** If we recompile system prompt every turn, cache is busted every turn.

**Solution:** Deferred compilation — only recompile during compaction.

- Memory block edits update DB, NOT the compiled prompt
- Agent sees "stale" prompt (fine — they can read current values via tools)
- At compaction: recompile prompt, save to DB
- Result: system prompt is stable turn-to-turn → cache hits

**Expected cache hit rate:** Near 100% between compactions.

---

## 4. Multi-Agent Implementation

### Architecture Decisions (from `architecturePlan.md`)

- **Single server, multiple agents** — simpler deployment, better inter-agent comms
- **Single DB with agent_id** — standard multi-tenant pattern

### Per-Agent Locking

Multiple requests to different agents can run concurrently (async event loop). But same-agent requests must serialize to avoid race conditions.

See details in `AgentFactory.get_deps()` (Section 3.1) — lock acquisition happens there (lock-then-fetch pattern), and `AgentFactory.build_agent_and_deps()` uses it internally.

### Inter-Agent Communication (Deferred)

**Edge case:** If Opus messages Sonnet, we touch Sonnet's data from Opus's request. Should acquire Sonnet's lock.

**For MVP:** Single-agent focus. Inter-agent = separate requests. Queuing system for "both active and messaging each other" is post-MVP.

---

## 5. Database Schema

**MVP uses SQLite** — single file, zero config, no server process. SQLAlchemy abstracts the differences, so we can swap to PostgreSQL later by just changing the connection URL.



### Key Design: Single Table + Pointer (No DELETEs)

**Conversation history vs Message history:**
- **Conversation history** = permanent, append-only audit trail. Every message ever exchanged.
- **Message history** = active context sent to model. Rolling buffer (of sorts), compaction applies here.

**Implementation:** Single `messages` table. A `context_window_start` timestamp on the agent tracks where active context begins. No DELETEs ever.

- **Load message history:** `SELECT ... WHERE timestamp >= context_window_start`
- **Load full conversation history:** `SELECT ... WHERE agent_id = ?` (no timestamp filter)
- **Compaction:** INSERT summary message, advance `context_window_start` to that timestamp

*Validated by Letta's approach: they use `message_ids` list or `in_context` flag, same principle.*

### Tables

**agent**
| Column | Type | Notes |
|--------|------|-------|
| id | TEXT | Primary key, UUID stored as string, auto-generated via `default=lambda: str(uuid.uuid4())` |
| name | TEXT | Display name |
| agent_config | TEXT | JSON-serialized object containing: `model_name` (string, e.g. "claude-sonnet-4-20250514"), `tool_names` (array of strings), `soft_compaction_limit` (integer, tokens), cache flags, and any other per-agent config. SQLAlchemy JSON type handles serialization. |
| system_instructions | TEXT | The typical "system prompt", foundational block which will be placed at beginning of compiled_system_prompt, before core memories |
| compiled_system_prompt | TEXT | Pre-compiled prompt |
|sys_prompt_compiled_at | TEXT | When prompt was last compiled (ISO 8601 datetime, SQLAlchemy DateTime type) |
| context_window_start | TEXT | Pointer: messages >= this are in active context (ISO 8601 datetime) |
| created_at | TEXT | populated on insert (ISO 8601 datetime) |
| updated_at | TEXT | populated on insert/update (ISO 8601 datetime) |

**memory_block**
| Column | Type | Notes |
|--------|------|-------|
| id | TEXT | Primary key, UUID stored as string, auto-generated |
| agent_id | TEXT | FK → agent (UUID as string) |
| label | TEXT | Block name (e.g., "persona", "human") |
| description | TEXT | What this block is for |
| content | TEXT | Block content |
| char_limit | INTEGER | Max chars for this block |
| position | INTEGER | Ordering for system prompt compilation (0-indexed) |
| created_at | TEXT | populated on insert (ISO 8601 datetime) |
| updated_at | TEXT | populated on insert/update (ISO 8601 datetime) |

*Position enables customizable block ordering in compiled prompt — useful for cache optimization (stable blocks first) and semantic grouping.*

**message**
| Column | Type | Notes |
|--------|------|-------|
| id | TEXT | Primary key, UUID stored as string, auto-generated |
| agent_id | TEXT | FK → agent (UUID as string) |
| type | TEXT | 'ModelRequest' &#124; 'ModelResponse' &#124; 'Summary' — set at insert, mirrors Pydantic AI type names |
| content | TEXT | Single serialized ModelMessage (JSON string) |
| input_tokens | INTEGER | NULL except on the final 'ModelResponse' row closing each run |
| timestamp | TEXT | Ordering + context window boundary — `Mapped[datetime]`, stored as naive UTC via SQLAlchemy |

**One row per `ModelMessage`.** `new_messages()` returns a list — each element gets its own row. A simple user→assistant turn produces two rows; a tool-calling turn produces four or more.

**`type` values:**
- `'ModelRequest'` — user messages, tool returns (`ModelRequest` with `UserPromptPart` or `ToolReturnPart`)
- `'ModelResponse'` — assistant responses, tool calls (`ModelResponse` with `TextPart` or `ToolCallPart`)
- `'Summary'` — synthetic summary rows inserted at compaction. Content is a `ModelRequest(UserPromptPart)` with XML-wrapped summary text. Set `type='Summary'` so we can query for them without parsing content JSON.

*Summary rows are part of conversation history and load naturally as `message_history` — no special reconstruction needed.*

**Why single table + pointer:**
- Append = INSERT. No read-modify-write.
- Compaction = INSERT summary row + UPDATE pointer. No DELETEs, simple, atomic.
- Full conversation history always available for audit/debugging.

**Indexes:**
- `messages(agent_id, timestamp)` — ordered history load + context window queries
- `messages(agent_id, type, timestamp DESC)` — efficient queries by type (e.g. find last ModelResponse with input_tokens)
- `memory_blocks(agent_id, position)` — for ordered block compilation
- `memory_blocks(agent_id, label)` — for single block lookups by name

**SQLite note:** `content` is a TEXT column storing JSON strings. No native JSONB — use SQLite's JSON1 functions (`json_extract`) for content queries where needed.

### Message Serialization

**Pydantic AI provides built-in serialization — no custom mappers needed.**

**Saving (on `AgentRunResultEvent`):**
```python
for msg in result.new_messages():
    row = MessageRecord(
        agent_id=deps.agent_id,
        type="ModelRequest" if isinstance(msg, ModelRequest) else "ModelResponse",
        content=ModelMessagesTypeAdapter.dump_json([msg]).decode('utf-8'),  # single-element list → serialize → store
        input_tokens=None,  # set below on final row only
    )
    session.add(row)

# Set input_tokens on final row (last ModelResponse closes the run)
last_row.input_tokens = result.usage().input_tokens
```

**Loading (before run):**
```python
records = await load_messages(deps.session, deps.agent_id, start_timestamp=deps.context_window_start)
history = deserialize_messages(records)  # converts MessageRecords to ModelMessages
# pass as message_history to run_stream_events()
```

**Summary rows:** Constructed at compaction time as `ModelRequest(parts=[UserPromptPart("<summary>...</summary>")])`, stored with `type='Summary'`. Load identically to regular rows — Pydantic AI receives it as a normal `ModelRequest` in `message_history`.

**Timestamps:** All stored as datetime utc. Can be converted to appropriate timezone at point of use.

---

## 6. Module Structure

```
e-llm-framework/
├── api/
│   ├── routes.py          # FastAPI endpoints
│   └── schemas.py         # Request/response models (MessageRequest, SSEEvent, AgentResponse, CoreMemoryBlocks)
├── agent/
│   ├── types.py           # Agent domain types (AgentConfig, AgentDeps)
│   ├── factory.py         # Agent execution logic
│   ├── tools.py           # Tool definitions
│   ├── compaction.py      # Compaction logic
|   └── agent_crud.py            # CRUD agent data in database
├── memory/
│   ├── block_crud.py          # Memory block operations
│   └── system_prompt_compilation.py
├── messages/
│   └── persistence.py         # Message persistence (persist, load, advance pointer)     
├── db/
│   ├── models.py          # SQLAlchemy models
│   ├── connection.py      # Engine creation and session management
│   └── migrations/        # Alembic migrations
└── main.py                # FastAPI app entry point
```

---

## Open Questions (Implementation-Specific)

### MVP

### Resolved

1. **System prompt not re-evaluated when message_history passed:** `system_prompt` has this problem — Pydantic AI skips it when `message_history` is non-empty. **Fix: use `instructions` instead.** Unlike `system_prompt`, `instructions` are always re-evaluated from the current agent regardless of history. Passed as `instructions=get_system_prompt` in the Agent constructor. See https://ai.pydantic.dev/message-history/#accessing-messages-from-results

2. **Pydantic AI message serialization:** → Use `ModelMessagesTypeAdapter`. Built-in, round-trips cleanly. See Message Serialization section.

### Deferred

3. **Streaming implementation:** How does `agent.run_stream()` integrate with SSE? → **Resolved.** See Section 1 (Entry Point) and Section 4.1 (Routes) — `run_stream_events()` with `map_to_sse()`, inline in messages endpoint.

4. **Hard limit interrupt:** Still open — how to interrupt Pydantic AI's internal tool loop mid-execution.

---

## Test Conventions

*Shared baseline for test infrastructure and patterns.*

### Tooling

- **Framework:** pytest + pytest-asyncio (for async test support)
- **Database:** SQLite in-memory (`:memory:`) for all DB tests — fast, isolated, no cleanup needed
- **Assertion style:** Plain `assert` statements (pytest's default)

### Fixtures

Implemented in `tests/conftest.py`. Async tests run automatically without `@pytest.mark.asyncio` — `asyncio_mode = "auto"` is set in `pyproject.toml`.

---

## Implementation Units & Test Behaviors

*Enumerate what we're building and what behaviors to verify. NO implementation yet — just the inventory.*

### Structure

Units organized by module. Each unit lists:
- **What it is** (brief description)
- **Behaviors to test** (becomes test cases)  

[x] next to a behavior to test indicates that the test for that behavior has been implemented, not necessarily that the test is passing. 
[xr] indicates implemented (by agents or JF) and reviewed by JF
[xs] indicates implemented (by agents) and skimmed by JF
---

### 1. Database Layer (`db/`)

#### 1.1 SQLAlchemy Models (`models.py`)

**Units:**
- [xr] `AgentRecord` model
- [xr] `MemoryBlockRecord` model  
- [xr] `MessageRecord` model  

*Note: `*Record` suffix distinguishes from Pydantic AI's `Agent` class.*

**Behaviors to test:**  
- [xr] Agent model stores and retrieves all fields (name, AgentConfig, system_instructions, compiled_system_prompt,sys_prompt_compiled_at, context_window_start, created_at, updated_at)
- [xr] `AgentConfig` JSON contains required keys: `model_name` (string), `tool_names` (array of strings), `soft_compaction_limit` (integer), `is_deletable` (bool) — validation enforced by `AgentConfig` Pydantic model (see Section 3.0 Agent Types in `agent/types.py`)
- [xr] `AgentConfig` timestamps autopop
- [xr] `tool_names` in `AgentConfig` is an array of strings
- [xr] `context_window_start` defaults to NULL on creation (no compaction has occurred yet)
- [xr] `system_instructions` defaults to '' on creation
- [xr] `sys_prompt_compiled_at` is NULL on creation (indicates prompt has never been compiled)
- [xr] MemoryBlock FK constraint enforced (can't create block for nonexistent agent)
- [xr] MemoryBlock unique constraint on (agent_id, label)
- [xr] Message FK constraint enforced
- [xr] Cascade delete: deleting agent deletes associated blocks and messages
- [xr] JSON fields (AgentConfig, content) round-trip correctly
- [xr] MemoryBlock model stores and retrieves all fields (agent_id, label, description, content, char_limit, position, created_at, updated_at)
- [xr] MemoryBlock model autopopulates created_at and updated_at
- [xr] Message model stores and retrieves all fields (agent_id, type, content, input_tokens, timestamp)
- [xr] Message input_tokens is nullable (only set on final response row that closes a run)
- [xr] Primary key (id) is auto-generated UUID string on insert for all models (Python-side via `default=lambda: str(uuid.uuid4())`, stored as TEXT)


#### 1.2 Session/DB Management (`db/connection.py`)

**Units:**
- [xr] `create_sqlite_engine(db_path: str) -> AsyncEngine` — creates engine with appropriate pool settings; called once at server startup from FastAPI lifespan event. Disposal handled in lifespan cleanup
- [xr] `init_db(engine: AsyncEngine)` — runs `Base.metadata.create_all`; called from lifespan event after engine creation
    - TODO: Make responsible for checking if db needs init so can always be called by lifespan
- [xr] `get_session(engine: AsyncEngine)` — async context manager yielding an `AsyncSession`; takes engine as parameter (never creates engine internally)

*Design rationale: engine creation is expensive and should happen once at startup. `get_session()` takes an engine rather than a URL so callers control engine lifetime and tests can inject an in-memory engine without mocking. The lifespan event calls `create_sqlite_engine()`, then `init_db()`, and stores the engine on `app.state.engine`.*

**Behaviors to test:**
- [xr] `create_sqlite_engine` returns an `AsyncEngine` with properties as expected
- [xr] `init_db` creates all expected tables (querying a known table succeeds; without it, raises "no such table")
- [ ] `init_db` no-op if db already init
    - TODO
- [xr] `get_session(engine)` yields an `AsyncSession` bound to the provided engine and with any other properties we define in the test
- [xr] Multiple concurrent sessions from the same engine each function independently (write + commit succeeds in each) — this test catches misconfigured pool (e.g. `StaticPool`)  
       Use `create_db_engine` to create input to get_session to make this a small scope integration test

*Note: `get_session(engine)` now explicitly manages transactions:*
  - *On normal exit: `await session.commit()`*
  - *On exception: `await session.rollback()` then re-raise*
  - *On any path: `await session.close()` in finally block*
  
*Tests: TestGetSessionTransactionBehavior in tests/db/test_connection.py uses `engine.connect()` directly for assertions (avoids masking bugs via get_session's own reads).*


---

### 2. Memory Layer (`memory/`)

#### 2.1 Block CRUD (`block_crud.py`)

**Read/Write Signature Split:**
- **Read operations** take `(session: AsyncSession, agent_id: str)` — no lock required, allows concurrent reads (e.g., ADE polling)
- **Write operations** take `(deps: AgentDeps)` — requires deps, which proves caller holds the per-agent lock

This prevents lost updates on read-modify-write sequences (like `memory_replace`) while allowing lightweight read access.

**Units:**

*Internal helper:*
- [xr] `_persist(deps, commit, record)` — async helper: on commit path, calls `deps.commit_changes_refresh_agent_record()` then `session.refresh(record)` if provided; on flush path, calls `deps.session.flush()` (server-generated values like timestamps remain stale until refresh)

*Read operations (no lock):*
- [xr] `get_blocks(session, agent_id)` — load all blocks ordered by position
- [xr] `get_block(session, agent_id, label)` — load single block by label

*Write operations (require deps → lock held):*
- [xr] `update_block(deps, label, content, existing_block=None)` — update block content. Caller can pass existing block model instance to avoid redundant DB fetch.
- [xr] `create_block(deps, label, ...)` — create new block
- [xr] `delete_block(deps, label)` — remove block
- [xr] `reorder_blocks(deps, labels_in_order: list[str])` — update position values based on list order

Note: For use cases that need deps but not a full agent (e.g., direct API block manipulation), use `agent_factory.get_deps(agent_id)` which acquires the lock and yields deps without constructing the full Pydantic AI Agent.

**TODO (Critical):** Current design hits DB directly rather than going through `deps._agent_record`. This results in memory blocks on deps going stale after update. Single source of truth issue — either go through deps._agent_record for mutations or provide a separate helper that refreshes blocks from deps after updates.

**Behaviors to test:**

*Read operations — take `(session, agent_id)`:*
*`get_blocks`:*
- [xr] Returns blocks ordered by position (ascending) for correct agent (use multi tenant db for challenge)
- [xr] Returns empty list for agent with no blocks

*`get_block`:*
- [xr] Returns correct block for correct agent by label (use multi tenant db for challenge)
- [xr] Returns None for nonexistent label

*Write operations — take `deps`:*
*Common (Write operations`update_block`, `create_block`, `delete_block`, `reorder_blocks`):*
- [xr] Multi-tenant isolation: only operates on blocks for specified `agent_id`

*`update_block`:*
- [xr] Modifies content, updates `updated_at`
- [xr] Enforces char_limit (rejects content exceeding limit)
- [xr] On nonexistent block raises appropriate error

*`create_block`:*
- [xr] Inserts new block with correct defaults
- [xr] With duplicate label raises/fails (unique constraint)
- [xr] Assigns position (auto-increment IE add to end, or explicitly specified by caller)

*`delete_block`:*
- [xr] Removes block
- [xr] On nonexistent block raises error (fail loudly — silent success masks typos/bugs)

*`reorder_blocks`:*
- [xr] Assigns positions 0, 1, 2... based on list order
- [xr] Updates all positions atomically
- [xr] Raises error if list doesn't include all blocks for agent (must be explicit)
- [xr] Raises error if list contains unknown label


#### 2.2 System Prompt Compilation (`system_prompt_compilation.py`)

**Units:**
- [xr] `compile_system_prompt(deps)` — assemble blocks into prompt
- [xr] `get_system_prompt(ctx: RunContext[AgentDeps] | deps: AgentDeps)` — async callable passed as `instructions` to Pydantic AI Agent; loads compiled prompt from DB. Accepts either a RunContext or AgentDeps directly.


**Behaviors to test:**
*`compile_system_prompt`:*
- [xr] Fetches blocks via get_blocks(deps.session, deps.agent_id); no direct AgentRecord fetch
- [xr] Assembles blocks w/ expected content in position order, for the correct agent only
- [xr] Includes system_instructions first (before blocks)
- [xr] Formats blocks with labels/descriptions, and includes expected XML wrappers
- [xr] Stores result via deps.compiled_system_prompt setter, updates deps.sys_prompt_compiled_at
- [xr] Calls await deps.session.flush() (not commit) — commit happens at turn level in routes
- [xr] Handles agent with no blocks (just system_instructions)

*`get_system_prompt`:*
- [xr] Returns cached `compiled_system_prompt` regardless of when blocks were last updated (**deferred compilation** — no staleness check)
- [xr] Returns empty string when `compiled_system_prompt` is NULL (agent created but sys prompt not compiled — caller should handle gracefully)
- [xr] Extracts `agent_id` from `ctx.deps` if ctx is RunContext; uses deps directly if AgentDeps passed
- [xr] Does not mutate stored prompt
- [xr] Memory block edits do NOT trigger recompilation (agents see "stale" prompt between compactions — this is intentional)
- [xr] Compilation is deterministic (same blocks → same output)  


---

### 3. Agent Layer (`agent/`)

#### 3.0 Agent Types (`agent/types.py`)

**Rationale:** AgentConfig and AgentDeps are internal domain objects, not API schemas. Keeping them in `agent/types.py` ensures the API layer imports FROM the agent layer (correct dependency direction), rather than having the agent layer depend on the API layer.

**Units:**
- [xr] `AgentConfig` — Pydantic model for agent configuration, stored as JSON in agents table
- [xr] `AgentDeps` — @dataclass(init=False) holding session and _agent_record; properties expose agent_id, name, config, system_instructions, compiled_system_prompt (rw), sys_prompt_compiled_at (rw), context_window_start (rw); includes commit_changes_refresh_agent_record() method

**Behaviors to test:**

*`AgentConfig`:*
- [xr] Requires `model_name` (ValidationError if missing)
- [xr] Requires `tool_names` (ValidationError if missing)
- [xr] Requires `soft_compaction_limit` (ValidationError if missing)
- [xr] `tool_names` must be list of strings (ValidationError if wrong type)
- [xr] `soft_compaction_limit` must be positive integer (ValidationError if <= 0)
- [xr] `compaction_target_percentage` — percentage of soft limit to target post-compaction (stored in AgentConfig, used in compact() logic)
- [xr] `is_deletable` defaults to `False` if not provided
- [xr] Round-trips through JSON correctly (`model_dump()` → JSON → `model_validate()`)
- [xr] Extra fields are rejected (`extra='forbid'`)

*`AgentDeps`:*
- [xr] @dataclass(init=False) with manual __init__(session, agent_record) — requires both parameters
- [xr] Private _agent_record field; all access via properties (agent_id, name, config, system_instructions, compiled_system_prompt rw, sys_prompt_compiled_at rw, context_window_start rw)
- [xr] Method: commit_changes_refresh_agent_record() — commits session then immediately refreshes _agent_record (prevents MissingGreenlet on subsequent reads)


---

#### 3.1 Agent Factory (`factory.py`)

**Note:** `AgentDeps` is imported from `agent/types.py` (see Section 3.0).

**Architecture: Per-Request AgentFactory**

The factory is constructed per-request via FastAPI Depends, receiving refs to:
- `lock_reg` — shared app-level lock registry (`app.state.agent_lock_reg`)
- `session` — per-request AsyncSession

This gives us:
- Clean abstraction at API boundary (routes see `agent_factory.build_agent_and_deps(agent_id)`, not raw lock_reg)
- No coupling risk (factory is per-request, instance state can't leak between requests)
- Session bound at construction (cleaner method signatures)

**Design note:** Multiple per-request factories hold refs to the same shared `lock_reg`. This is intentional — the lock registry IS shared state. Future consideration: a more sophisticated lock_reg wrapper that restricts what each holder can do (e.g., can't release another request's lock).

**Pseudocode:** *(Methods shown without `self.` prefix for brevity)*
```
class AgentFactory:
    __init__(lock_reg, session):
        store refs
    
    _get_lock(agent_id) -> Lock:
        return or create lock in registry
    
    @asynccontextmanager
    get_deps(agent_id) -> yields AgentDeps:
        lock = _get_lock(agent_id)
        try:
            await asyncio.wait_for(lock.acquire(), timeout=LOCK_TIMEOUT_SECONDS)
        except asyncio.TimeoutError:
            raise AgentLockedError(...)
        try:
            agent_record = await session.get(AgentRecord, agent_id)
            if agent_record is None:
                raise AgentNotFoundError(...)
            yield AgentDeps(session, agent_record)
        finally:
            lock.release()
    
    @asynccontextmanager
    build_agent_and_deps(agent_id) -> yields (Agent, AgentDeps):
        async with get_deps(agent_id) as deps:
            model = get_model(deps.config.model_name)
            tools = get_tools_for_agent(deps.config.tool_names)
            agent = Agent(model=model, tools=tools, deps_type=AgentDeps)
            yield (agent, deps)

# Module-level (stateless):
get_model(model_name) -> AnthropicModel:
    map string to Pydantic AI model instance
    
# FastAPI DI:
def get_agent_factory(
    session = Depends(get_session_dep),
    lock_reg = Depends(get_lock_reg),
) -> AgentFactory:
    return AgentFactory(lock_reg, session)
```

**Units:**
- [xr] `AgentFactory` class — per-request factory constructed with lock_reg + session refs
- [xr] `AgentFactory._get_lock(agent_id)` — returns per-agent asyncio.Lock from registry (creates if needed)
- [xr] `AgentFactory.get_deps(agent_id)` — async context manager that acquires lock (lock-then-fetch) and yields `AgentDeps`
- [xr] `AgentFactory.build_agent_and_deps(agent_id)` — async context manager that yields `(Agent, AgentDeps)`. Uses `get_deps()` internally.  
    Note: After writing tests, please leave this implementation for JF, good pattern to get comfortable with 
- [xr] `get_model(model_name)` — module-level function, maps DB model string to Pydantic AI model instance
- [ ] `get_agent_factory(session = Depends(get_session_dep), lock_reg = Depends(get_lock_reg))` — FastAPI dependency that constructs per-request `AgentFactory(lock_reg, session)`. Lives in factory.py to keep factory-related code together.

**TODO:** Consider if can eliminate `AgentFactory` class entirely. Can get_agent_factory fulfill the same role?

**Note:** Be sure to use appropriate agent_crud functions where possible

**Note:** `lock_reg` is created in `lifespan()` and stored on `app.state.agent_lock_reg`. Per-request `AgentFactory` receives it via FastAPI Depends chain. This avoids module-level globals and makes testing clean (each test constructs factory with fresh registry). Single-process only; distributed locking (Redis, DB row locks) is a future consideration for multi-worker scaling.

**Behaviors to test:**

*`AgentFactory._get_lock`:*
- [xr] Returns same lock instance for same agent_id (not a new lock each call)
- [xr] Returns different lock instances for different agent_ids

*`AgentFactory.get_deps`:*
- [xr] Yields deps with session and _agent_record populated; properties accessible for agent_id, config, system_instructions, compiled_system_prompt, etc.
- [xr] Acquires per-agent lock on entry (lock-then-fetch)
- [xr] Releases lock on normal exit
- [xr] Releases lock even if exception raised inside the `async with` block
- [xr] Releases lock even if DB fetch fails (pre-yield exception)
- [xr] Second concurrent call on same `agent_id` blocks until first context exits
- [xr] Lock acquisition timeout after LOCK_TIMEOUT_SECONDS raises `AgentLockedError` with message: f"Agent {agent_id!r} did not become available within {LOCK_TIMEOUT_SECONDS}s"
- [xr] Concurrent calls on different `agent_id`s do not block each other
- [xr] Raises `AgentNotFoundError` for unknown `agent_id`

*`AgentFactory.build_agent_and_deps`:*
- [xr] Yields a valid `(agent, deps)` tuple inside the context
- [xr] Lock behavior inherited from `get_deps` (lock held for duration of context)
- [xr] Constructed agent uses model from `deps.config.model_name`
- [xr] Constructed agent has cache settings enabled
- [xr] Constructed agent has the correct tools for this agent's `tool_names` config

*`get_model` (module-level):*
- [xr] Returns `AnthropicModel` instance for valid model name
- [xr] Raises for unknown/unsupported model name

*`get_agent_factory`:*
- [ ] Returns `AgentFactory` with correct session and lock_reg bound
- [ ] Factory constructed per-request (new instance each call)

Note: Tests construct `AgentFactory(lock_reg={}, session=mock_session)` directly — no app state needed.
Question: Should we try to mock agent? May not be needed. As long as LLM connection isn't made on creation

#### 3.2 Tools (`tools.py`)

**Units:**
- [xr] `TOOL_REGISTRY` — dict mapping tool name strings to callable functions
- [xr] `get_tools_for_agent(tool_names: list[str])` — look up tools by name
- [xr] `memory_replace(label, old_string, new_string, occurrence=None)` — find/replace in block
- [xr] `memory_insert(label, content, after=None, occurrence=None)` — insert by anchor string
- ~~[ ] `archival_memory_insert(content, tags)` — store to archival~~ -> Deferred to v2, needs embedding
- ~~[ ] `archival_memory_search(query, ...)` — semantic search~~ -> Deferred to v2, needs embedding
- ~~[ ] `conversation_search(query, ...)` — search message history~~ -> Deferred to V2 Need semantic search to be useful 

**Behaviors to test:**

*`TOOL_REGISTRY`:*
- [xr] Contains all available tools keyed by name

*`get_tools_for_agent`:*
- [xr] Returns list of callables for valid tool names
- [xr] Raises `KeyError` for unknown tool name (fail loudly)

*Common (`memory_replace`, `memory_insert`):*
- [xr] Raises `ModelRetry` if `label` doesn't exist for this agent
- [xr] Updates correct block specified by `label`
- [xr] Raises `ModelRetry` if resulting content would exceed `char_limit`
- [xr] Flushes change to DB immediately (commit=False); committed atomically at end of turn along with message persistence. Rationale: atomic commit prevents "time travel" — agent could have partial memory state changes without corresponding message history, tracing a different path if crashed mid-turn
- [xr] Returns SNIPPET of updated block content on success
- [xr] Does NOT trigger system prompt recompilation
- [xr] Callable via Pydantic AI tool mechanism — receives `RunContext[AgentDeps]` with correct deps populated
- [xr] Tool errors raise `ModelRetry` for model self-correction
- [xr] Raises `ModelRetry` if `old_string`/`after` appears more than once and `occurrence` is not specified
- [xr] Raises `ModelRetry` if `old_string`/`after` not found in block (no silent no-op)
- [xr] Raises `ModelRetry` if `occurrence=N` is specified but fewer than N occurrences exist
- [xr] Raises `ModelRetry` if `old_string`/`after` not specified or empty
Note: Great opportunity to parametrize on tool to avoid duplication

*`memory_replace`:*
- [xr] Replaces the target occurrence of `old_string` with `new_string`
- [xr] `occurrence=N` (1-indexed) replaces the Nth occurrence when specified
- [xr] Only replaces the target occurrence, does not touch any other content

*`memory_insert`:*
- [xr] `after="<start>"` inserts content at start of block
- [xr] `after="<end>"` inserts content at end of block  
Note: We can adjust these if we have issues with Agent wanting to actually store those strings in memory for some reason.
- [xr] `after="anchor"` inserts content after the target occurrence of the anchor string
- [xr] `occurrence=N` (1-indexed) inserts after the Nth occurrence when specified
- [xr] Content inserted without overwriting existing content  


#### 3.3 Compaction (`compaction.py`)

**Units:**
- [xr] `is_compaction_needed(input_tokens, config)` — compare against limits
- [xr] `compact(deps, input_tokens)` — async function: advances context_window_start pointer, flushes to DB, recompiles system prompt via await compile_system_prompt(deps). Calculates avg tokens/msg by subtracting estimated sys prompt tokens (len/4) from input_tokens, then targets the percentage specified in deps.config.compaction_target_percentage.
    - Note: May lag by the model's output depending on what exactly anthropic returns

**Behaviors to test:**

*`is_compaction_needed`:*
- [xr] Returns `True` when `input_tokens > soft_compaction_limit`
- [xr] Returns `False` when `input_tokens <= soft_compaction_limit`

*`compact`:*
- [xr] Advances `context_window_start` pointer in DB
- [xr] Does NOT delete any messages
- [xr] Never evicts the most recent 4 messages (minimum history guard)
- [xr] With 4 or fewer messages in context is a no-op
- [xr] Targets ~N% (configurable, stored in AgentConfig) of `soft_compaction_limit` post-compaction (within estimate drift tolerance)
- [xr] Calls `compile_system_prompt` (deferred compilation happens here)
Note: We will separately test that `load_messages` uses `context_window_start` pointer to filter messages in subsequent loads  


#### 3.4 Agent CRUD (`agent_crud.py`)
NOTE: Some units deferred until they become explicitly necessary.
They were not called in current routes nor did they have any dependents
**Units:**
- [xr] `create_agent_record(session, name, system_instructions, config: AgentConfig)` — add agent to DB with specified configuration, return AgentRecord (includes generated UUID, timestamps)
- [xr] `get_agent_record(session, agent_id)` — return AgentRecord for a given agent_id, or None if not found
- [xr] `agent_exists(session, agent_id) -> bool` — lightweight existence check via EXISTS scalar query; does not load the full record
- [DEFERRED] `get_config(agent_id)` — return AgentConfig for a given agent_id
- [DEFERRED] `update_config(agent_id, new_config: AgentConfig)` — replace agent's config in DB.
- [DEFERRED] `delete_agent(agent_id) -> bool` — remove all data associated with agent from DB IFF `AgentConfig.is_deletable` is True. Returns True if deleted, False if not found or delete-protected. Note: to delete a delete-protected agent, first set `is_deletable=True` via `update_config`.
- [DEFERRED] `replace_system_instructions(agent_id, instructions)` — set (overwrite) the system instructions for an agent  
- [DEFERRED] `get_system_instructions(agent_id)`
- [DEFERRED] `list_agents() -> list[AgentConfig]` — return all agent configs (can access name and ID from config)
Note: unlike block_crud, agent_crud works On agent ID instead of agent deps. This is because block_crud Will often be called from inside an active session, where a database connection already exists and agent deps is already load.  
agent_crud will often happen from discrete API hits.

**Behaviors to test:**

*`create_agent_record`:*
- [xr] Inserts agent row in DB with provided config — `test_config_and_data_survive_db_round_trip` (expire clears identity map; fetch would fail if row absent)
- [xr] Generates UUID for `agent_id`, returns full AgentRecord — `test_returns_agent_record_with_correct_fields` (uuid.UUID(record.id) raises if invalid)
- [xr] New agent has empty message history (no messages, `context_window_start` = None) — `test_returns_agent_record_with_correct_fields` (context_window_start is None + load_messages returns [])
- [xr] Config is persisted and round-trips correctly — `test_config_and_data_survive_db_round_trip` (agent_config == SAMPLE_AGENT_CONFIG after expire+fetch)

*`get_agent_record`:*
- [xr] Returns AgentRecord for a valid `agent_id` — `test_get_agent_record_returns_record_for_known_id`
- [xr] Returns None for unknown `agent_id` — `test_get_agent_record_returns_none_for_unknown_id`

*`agent_exists`*
- [TODO] TODO (LOL)

*[DEFERRED] `get_config`:*
- [ ] Returns correct AgentConfig for a valid `agent_id`
- [ ] Raises `NotFound` for unknown `agent_id`

*`replace_agent_config`:*
- [ ] Replaces agent config in DB with `new_config`
- [ ] Returns updated config
- [ ] Raises `AgentNotFoundError` for unknown `agent_id`
- [ ] Unrelated configs not affected (not explicitly tested)

*[DEFERRED] `delete_agent`:*
- [ ] Returns True and deletes agent row and all associated data (messages, memory blocks) when `is_deletable=True`
- [ ] Returns False if `is_deletable=False` — no accidental deletions
- [ ] Returns False if `agent_id` not found
- [ ] Cascade is complete — no orphaned messages or memory blocks remain after deletion
- [ ] After deletion, `get_config` raises `NotFound` for the deleted `agent_id`

*[DEFERRED] `list_agents`:*
- [ ] Returns list of all AgentConfigs
- [ ] Returns empty list if no agents exist

*`replace_system_instructions`:*
- [ ] Stores instructions for the given agent (overwrites any previous value)
- [ ] triggers recompilation of system prompt
- [ ] Returns stored instructions

*[DEFERRED] `get_system_instructions`:*
- [ ] Returns expected system instructions for given agent_id
- [ ] Raises `NotFound` for unknown agent_id


---

### 4. API Layer (`api/`)

#### 4.1 Routes (`routes.py`)

**Dependency injection pattern:**
Routes that need to write agent state (e.g., `POST /agents/{agent_id}/messages`) use `get_agent_and_deps` as a FastAPI dependency. This resolves the issue of exceptions in SSE generators — the dependency handles lock acquisition/release cleanly, and error translation (AgentNotFoundError→404, AgentLockedError→503, etc.) happens before stream starts.

Read-only routes (e.g., `GET /agents/{agent_id}/core_memory`) can use `session` directly without factory — allows concurrent reads while writes are locked.

**SSE Design (inline in messages endpoint):**
Pass-through Pydantic AI events with FastAPI's native ServerSentEvent. Uses `EventSourceResponse` as response class.

Serialization: `map_to_sse(event)` returns `ServerSentEvent(data=event, event=type(event).__name__)`. 

Pydantic AI event types (all forwarded): `PartStartEvent`, `PartDeltaEvent`, `PartEndEvent`, `FunctionToolCallEvent`, `FunctionToolResultEvent`, `FinalResultEvent`, `AgentRunResultEvent`

TODO: Consider extracting event loop logic to an `EventGenerator` class or similar if the messages endpoint becomes unwieldy. For MVP, inline generator is simpler.

**Units:**
- [xr] `POST /agents/{agent_id}/messages` — main chat endpoint (streaming). Implemented as yield dependency + inline event generator. Uses `get_agent_and_deps` to safely handle lock acquisition and error translation.
- [xr] `map_to_sse(event)` — helper function to serialize Pydantic AI event to FastAPI ServerSentEvent
- [xr] `POST /agents/` - create new agent with specified name and config
- [xr] `GET /agents/{agent_id}` — get agent info
- [xr] `GET /agents/{agent_id}/core_memory`
- [xr] `GET /agents/{agent_id}/messages` — get conversation history
- [xr] `GET /agents/{agent_id}/config` — return agent's current AgentConfig
- [ ] `PUT /agents/{agent_id}/config` — replace agent's AgentConfig
- [xr] `GET /agents/{agent_id}/system-instructions` — return agent's current system instructions
- [ ] `PUT /agents/{agent_id}/system-instructions` — replace agent's system instructions (triggers recompilation)

**Behaviors to test:**

*`map_to_sse(event)`:*
- [xs] `PartStartEvent` → `event: PartStartEvent`, `data: {"index": N, "part": {...}}`
- [xs] `PartDeltaEvent` → `event: PartDeltaEvent`, `data: {"index": N, "delta": {...}}`
- [xs] `PartEndEvent` → `event: PartEndEvent`, `data: {"index": N, "part": {...}}`
- [xs] `FunctionToolCallEvent` → `event: FunctionToolCallEvent`, `data: {"part": {...}, "tool_call_id": "..."}`
- [xs] `FunctionToolResultEvent` → `event: FunctionToolResultEvent`, `data: {"tool_call_id": "...", "result": ...}`
- [xs] `FinalResultEvent` → `event: FinalResultEvent`, `data: {"tool_name": ... | null}`
- [xs] `AgentRunResultEvent` → `event: AgentRunResultEvent`, `data: {}` (signals stream end)

*`POST /agents/{agent_id}/messages`:*
- [xr] Returns `Content-Type: text/event-stream` with correct SSE headers
- [xr] Yields one SSE `data:` line per event from `agent.run_stream_events()`
- [xr] Streams `PartDeltaEvent` for each text delta
- [xr] Streams `FunctionToolCallEvent` when agent invokes a tool
- [xr] Streams `FunctionToolResultEvent` when tool returns
- [xr] Streams `AgentRunResultEvent` as the final event when run completes
- [xr] Returns 404 for unknown `agent_id` (before stream starts)
- [xr] Returns 400 for malformed request body (missing `message` field)
- [xr] Returns 503 when agent is already locked by a concurrent request (no queuing — simpler, caller retries)
- [xr] Persists new messages to DB exactly once — on `AgentRunResultEvent`, via `persist_messages(deps, ...)`
    - This is covered with a composite test: Test that persistence is called, and test that AgentRunResultEvent is last yielded event
    - Technically our test doesn't cover that the persistance happens before yielding, only that it happens once at all.  
      But, we have a seperate requirement that when client disconnnects we continue to persist which eliminates the practical concern
- [SKIP] Does NOT attempt persistence on any earlier event
- [xr] After persist_messages succeeds: calls `await deps.commit_changes_refresh_agent_record()` (commits session, refreshes agent record to avoid MissingGreenlet)
- [xr] Compaction called under appropriate conditions synchronously after commit, before stream closes (no background task)
- [xr] On exception during agent run:calls `await deps.session.rollback()`, yields `ServerSentEvent(data={'message': \"Unexpected internal server error: ...\"}, event='Error')`, then stream closes
- [TODO] Allows agent to complete their run when client disconnects prematurely (dead heads), then persists as much valid content as possible
    - this one is tricky, will figure out soon
- [SKIP] Keep-alive: long-running tool chains don't cause premature timeout
    - handled by sse-starlette automatically, sends keep alive pings

*`POST /agents/{agent_id}/messages` edge cases:*
- [xr] Agent run with zero text output (only tool calls) — still emits tool events + `AgentRunResultEvent`
- [TODO] Tool that returns an error — `FunctionToolResultEvent` with error content, run continues
    - This may be outdated. We use ModelRetry exceptions now rather than returning strings with error content. Not sure how pydantic handles that.
- [TODO] Very long response — `PartDeltaEvent`s flow continuously without buffering full response
- [xr] Empty `new_messages()` — persist is no-op, `AgentRunResultEvent` still sent

*`POST /agents/`:*
- [xr] Calls expected agent creation internal function with correct args (that unit seperately verified)
- [xr] Returns new agent ID
- [xr] Returns failure of agent creation function (invalid agent settings,etc.) to caller

*`GET /agents/{agent_id}`:*
- [xr] Returns agent metadata: `name`, `model`, `created_at`, `updated_at`

*`GET /agents/{agent_id}/core_memory`:*
- [xr] Returns current memory blocks with labels, descriptions, content, char_limit, last updated
- [xr] Returns some form of emptiness when no core memory blocks

*`GET /agents/{agent_id}/messages`:*
- [xr] Returns messages from `context_window_start` by default (active context view)
- [xr] With `?full=true` returns complete conversation history (no pointer filter)
- [SKIP] Messages returned in chronological order (`timestamp` ascending)
    - Just assert that it returns what the message functions give it (which we do) and unit test those seperately

*Common (`GET /agents/{agent_id}/messages`, `GET /agents/{agent_id}/core_memory`, `GET /agents/{agent_id}`)*
- [xr] Returns 404 for unknown `agent_id`

*`GET /agents/{agent_id}/config`:*
- [xr] Returns current AgentConfig as JSON for a valid agent_id

*`PUT /agents/{agent_id}/config`:*
- [xr] Calls replace_agent_config with correct agent_id and validated config (validated by AgentConfig constructor)
- [xr] Returns 422 for invalid config (fails AgentConfig validation)
- [xr] Returns 409 if agent is currently locked (run in progress)

*`GET /agents/{agent_id}/system-instructions`:*
- [xr] Returns current system instructions string for a valid agent_id
*`PUT /agents/{agent_id}/system-instructions`:*
- [xr] Calls replace_system_instructions with correct agent_id and instructions string
- [xr] Returns 409 if agent is currently locked (run in progress)

*Common (`GET/PUT /agents/{agent_id}/config`, `GET/PUT /agents/{agent_id}/system-instructions`)*
- [xr] Returns 404 for unknown agent_id
- [xr] (PUTs only) returns 200 on success and echos back the set value (sys instr/agent config)
    - echos the SET value, not just the passed in value


#### 4.2 Schemas (`schemas.py`)

**Note:** `AgentConfig` is defined in `agent/types.py` (see Section 3.0). `schemas.py` contains API request/response schemas only.

**Note:** No custom SSE event schemas. SSE serialization uses the pass-through approach — Pydantic AI's native event types are serialized directly by `map_to_sse` in `routes.py` (Section 4.1). Those tests live in `test_map_to_sse.py` and are complete.

**Units:**
- [xr] `MessageRequest` — incoming message schema
- [xr] `CreateAgentRequest` — name, system_instructions, config (embeds `AgentConfig` from `agent/types.py`)
- [xr] `AgentMetadataResponse` — agent info response with id field (used for both POST /agents/ and GET /agents/{agent_id})
- [xr] `MemoryBlockResponse` — single block in core memory response
- [xr] `CoreMemoryResponse` — response to `GET /agents/{agent_id}/core_memory`
- [xr] `MessageItem` — single message row: id, type ('ModelRequest'|'ModelResponse'|'Summary'), content (raw serialized ModelMessage JSON), timestamp. Display parsing deferred to later.
- [xr] `MessagesResponse` — response to `GET /agents/{agent_id}/messages`
- [xr] `HealthResponse` — response to `GET /health`

**Behaviors to test:**
**Note:** The actual schema members left off unit tests, see schemas.py for the schema definitions.
Here we only list/unit test custom behavior

*`MessageRequest`:*
- [xr] Rejects empty string for `message`


*Common (all API schemas):*
- [xr] Serializes to valid JSON (no unserializable fields)


#### 4.3 Deps (`api/fastapi_deps.py`)

**Note:** All FastAPI dependency functions live in `api/fastapi_deps.py`. This keeps dependencies separate from app initialization logic.

**Units:**
- [xr] `get_lock_reg(request)` — FastAPI dependency that returns `request.app.state.agent_lock_reg`
- [xr] `get_session_dep(request)` — FastAPI dependency wrapper. Yields session from `app.state.engine` via module-level `get_session()`.
- [xr] `get_agent_and_deps(agent_id, session, lock_reg)` — FastAPI yield dependency. Builds `AgentFactory`, calls `build_agent_and_deps()` as async context manager, yields `(Agent, AgentDeps)`. Translates exceptions: AgentNotFoundError→404, AgentLockedError→503, Exception→500. Used in `send_message` and other write endpoints.

**Behaviors to test:**

*`get_lock_reg`:*
- [xr] Returns `request.app.state.agent_lock_reg`

*`get_session_dep`:*
- [xr] Yields session from `get_session(app.state.engine)`
- [xr] Session is request-scoped (new each call)

*`get_agent_and_deps`:*
- [xr] Yields `(Agent, AgentDeps)` tuple
- [xr] Builds AgentFactory internally with lock_reg + session bound
- [xr] Calls `build_agent_and_deps(agent_id)` as async context manager
- [xr] On AgentNotFoundError: raises HTTPException(404)
- [xr] On AgentLockedError: raises HTTPException(503)
- [xr] On other Exception: raises HTTPException(500)

#### 4.4 App & Lifespan (`api/app.py`)

**Units:**
- [xr] `_create_app()` — factory function that creates FastAPI app with lifespan, includes router, registers /health endpoint. Enables fresh instances per test (no state contamination). Underscore prefix = not for live use (tests access it anyway). Basic wiring verified by `test_create_app_includes_router`; lifespan behavior tested by `TestLifespan` class.
- [xr] `lifespan(app)` — async context manager: calls `create_sqlite_engine`, then `init_db`, stores engine on `app.state.engine`, initializes `app.state.agent_lock_reg = {}` (per-agent lock registry); disposes engine on shutdown
- [xr] module-level `app = _create_app()` — production instance
- [xr] `GET /health` — lightweight check that FastAPI is up. Lives in app.py (not routes.py) because routes have `/agents` prefix. TODO: Later add DB reachability check.

**Notes:** 
- `get_lock_reg` and `get_session_dep` live in Section 4.3 (`api/fastapi_deps.py`)
- `get_agent_factory` moved to Section 3.1 (factory.py) to keep factory-related code together

**Behaviors to test:**

*`_create_app()`:*
- [xr] Returned app includes all routes from `router` (subset check)
- [xr] Returned app includes `/health` endpoint

*`lifespan(app)`:*
- [xr] Calls `create_sqlite_engine`, then `init_db` with the returned engine (if DB not already init, perhaps this check should be init_db responsibility though), then stores engine on `app.state.engine`, initializes `app.state.agent_lock_reg = {}`, then disposes engine on shutdown — verified via mocks (pytest-mock), not real DB
    - TODO: Make init_db check if db needs init before doing so
- [xr] `app.state.agent_lock_reg` is an empty dict after startup
- [xr] If `init_db` raises, `engine.dispose()` is still called (cleanup on partial startup failure)

*Test pattern: patch `create_sqlite_engine` and `init_db`; use `asgi-lifespan.LifespanManager(app)` to trigger startup/shutdown (httpx's ASGITransport does NOT trigger lifespan events).*

*Health check:*
- [xr] Returns 200 with `{\"status\": \"ok\"}`
- [TODO] Returns non-200 when DB is unreachable

---

### 5. Message Persistence/helpers (`messages/messages.py`)

**Units:**
- [xr] `persist_messages(deps, messages, input_tokens)` — save each ModelMessage as its own row; set input_tokens on final row
- [xr] `load_messages(session, agent_id, start_timestamp=None) -> list[MessageRecord]` — load messages as ORM records; filters to `>= start_timestamp` if provided, otherwise returns full history
- [xr] `deserialize_messages(records) -> list[ModelMessage]` — pure function, converts MessageRecords to ModelMessages
- [SKIP] `advance_pointer(deps, messages_to_evict)` — move context_window_start forward
- [SKIP] `count_messages(deps)` — count messages in current context window
    - these two we just implemented directly in compaction. Compaction is already pretty lean so I don't think we need to
    split them out at this point.

*Write operations (`persist_messages`, `advance_pointer`) take `deps: AgentDeps` (caller must hold agent lock). Read operations (`load_messages`, `count_messages`) take `session` + `agent_id` directly — no deps required for reads (principle of least privilege).*

*Callers that need in-context messages call `load_messages(session, agent_id, start_timestamp=context_window_start)` then `deserialize_messages()` if they need ModelMessages. No convenience wrapper — the explicit args are self-documenting.*

**Behaviors to test:**

*`persist_messages`:*
- [xr] Inserts one row per `ModelMessage` in the list
- [xr] Sets `type='ModelRequest'` for `ModelRequest` instances, `type='ModelResponse'` for `ModelResponse` instances
- [xr] Serializes each message individually via `ModelMessagesTypeAdapter`
- [xr] Sets `input_tokens` on the last row only (the final `ModelResponse` closing the run)
- [xr] Leaves `input_tokens` NULL on all non-final rows
- [xr] Pulls timestamp from ModelRequest/ModelResponse and stores as `timestamp` for that row
- [xr] Checks that timestamp for message about to be persisted is NEWER than the last message in the DB.  
      If older, something is wrong with timekeeping somewhere. Inject a warning and set the timestamp to SOMETHING newer than the last msg to preserve conversation order
Resolved: ModelRequest.timestamp may be None — _message_timestamp() handles this: uses ModelRequest.timestamp if present, else extracts from first UserPromptPart in parts.
- [xr] Appends (never overwrites previous messages)
  → test_records_isolated_per_agent: persists twice to the same agent (my_first, my_second), verifies both messages survive by content equality (not just count)
- [xr] With empty list is a no-op (no rows inserted)
  → test_empty_messages_list_is_noop: passes [], asserts records == []
- [xr] When encountering invalid data somewhere in the message list, stores a ModelResponse with content indicating an invalid message was encountered, then stores the remaining messages if possible
  → test_serialization_failure_injects_error_response: verifies len==4 (good, error, good2, summary warning), records[0]==good and records[2]==good2 by content equality, error record (records[1]) and summary warning (records[3]) verified by exact content
- [xr] Unwilling to persist an orphaned tool call (tool call without a tool response following it)
    - [xr] Just replaces tool call with an error ModelResponse containing info about the call that was orphaned, see previous line
  → test_orphaned_tool_call_replaced_with_error_response + test_orphaned_tool_return_replaced_with_error_response (both via _assert_orphan_replaced): checks len==2, records[0].type==ModelResponse, no ToolCallPart/ToolReturnPart in restored parts, exact error content, summary warning appended with correct timestamp
- [xr] Multi-tenant isolation: only persists messages for specified `agent_id`
  → test_records_isolated_per_agent: verifies record counts and message contents for both agents independently

*`load_messages`:*
- [xr] Returns MessageRecords (not deserialized) — caller deserializes if needed
  → test_returns_list_of_message_records: isinstance(records[0], MessageRecord)
- [xr] Returns messages in chronological order
  → test_results_in_chronological_order: constructs 3 messages with 5ms asyncio.sleep() between each to guarantee distinct timestamps, then asserts timestamps == sorted(timestamps)
- [xr] Returns only messages where `timestamp >= start_timestamp` when provided
  → test_start_timestamp_filters_inclusive: persists early batch, captures cutoff as last record of early batch, sleeps 100ms, persists late batch, asserts load_messages returns [early[1]] + late — inclusive boundary explicitly exercised
- [xr] Returns full message history when `start_timestamp` is `None`
  → test_returns_all_messages_when_no_start_timestamp: persists 4 messages, calls load_messages with no start_timestamp arg, asserts deserialize_messages(records) == messages
- [xr] Returns empty list if start_timestamp is ahead of all messages
  → test_start_timestamp_ahead_of_all_messages_returns_empty: uses datetime(9999,...) as start_timestamp, asserts records == []
- [DEFER] Handles `type='Summary'` rows — Summary type not yet implemented; revisit when compaction adds summarization
- [xr] Multi-tenant isolation: only returns messages for specified `agent_id`
  → test_returns_only_records_for_given_agent: persists for self.agent and other_agent, loads for self.agent, checks len==2 and all(r.agent_id == self.agent.id for r in records)

*`deserialize_messages`:*
- [SKIP] Pure function (no DB access) (This is a signature attribute not something we necessarily need to test)
- [xr] Converts each MessageRecord to its correct Pydantic AI type (`ModelRequest` or `ModelResponse`)
  → test_deserializes_messages parametrized: [messages0]=ModelRequest, [messages1]=ModelResponse, [messages2]=tool pair. Equality check catches type mismatches (dataclass __eq__ uses other.__class__ is self.__class__)
- [xr] Preserves chronological order from input
  → test_deserializes_messages[messages2]: list equality on 2-item tool pair verifies order. Also implicitly covered at scale by test_performance_1000_messages (result == messages on 1000 items)
- [xr] Handles empty list (returns empty list)
  → test_empty_list_returns_empty: assert deserialize_messages([]) == []
- [DEFER] Handles `type='Summary'` and all other types — Summary type not yet implemented; revisit when compaction adds summarization
- [xr] Include performance test for deserialization. Feed it a list of N messages where N is large enough to get a good reading, measure time to deserialize. Compute time/msg, this op will be common and I want to know if its going to be slow.
  → test_performance_1000_messages: 1000 messages, elapsed measured, time/msg printed, asserts elapsed < 0.5s, also checks result == messages

*`count_messages`:*
- [SKIP] Returns count of messages where `timestamp >= context_window_start`
- [SKIP] Returns 0 if no messages in current window

*Common (`persist_messages`, `load_messages`, `deserialize_messages`):*
- [xr] Round-trip integrity: persist → load → deserialize returns equivalent message objects (integration test)
  → test_request_response_round_trip: persist [request, response], load, deserialize, assert restored == original
  → test_tool_pair_round_trip: same for tool pair, asserts each element individually


---

## 6. OSS Crediting (Critical)
- [ ] Letta properly credited for designs, ideas, and inspiration that we took from them.

## Near Term Deferred Work

*Decisions deferred until CLI integration phase — need hands-on testing to inform these.*

**Display & Format:**
- [ ] `MessageItem` display format — currently raw JSON passthrough. Parse/structure when UI needs are known.
- [ ] `map_to_letta_sse()` — transforms our events to Letta's `message_type` format (only if we select Letta Code as CLI)

**CLI & Tool Execution:**
- [ ] Agentic coding CLI selection — candidates: Letta Code (same architecture, TypeScript), Plandex, OpenCode
- [ ] Tool execution model details — committed to memory tools server-side; coding tools likely client-side ("one agent, many mech suits") but details deferred until CLI research

---

*Sonnet and Opus to fill in behaviors. James reviews before implementation.*

---

## Research Notes

See `pydanticNotes.md` for detailed source citations and exploration notes.
