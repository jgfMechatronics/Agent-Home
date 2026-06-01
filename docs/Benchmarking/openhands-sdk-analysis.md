# OpenHands SDK Analysis
*Completed: May 29, 2026 — Sonnet*

## TL;DR

**OpenHands is architecturally incompatible with dynamic per-turn memory block injection. Greenfield confirmed.**

This isn't a minor impedance mismatch — it's a fundamental conflict between their design assumptions and our core requirement (Req 5: memory system controls system prompt injection, dynamic each turn).

---

## What We Investigated

Files read from `software-agent-sdk/openhands-sdk/`:

- `openhands/sdk/agent/agent.py` (1257 lines) — main agent loop
- `openhands/sdk/agent/base.py` (843 lines) — `AgentBase` config and init
- `openhands/sdk/agent/utils.py` (630 lines) — `prepare_llm_messages()` pipeline
- `openhands/sdk/event/llm_convertible/system.py` (105 lines) — `SystemPromptEvent`
- `openhands/sdk/context/view/view.py` (161 lines) — `View` (condenser input/output)
- `openhands/sdk/context/agent_context.py` (477 lines) — `AgentContext`

---

## The Message Preparation Pipeline

Every agent step calls:

```
Agent.step()
  → prepare_llm_messages(state.events, condenser=self.condenser, llm=self.llm)
      → View.from_events(events)          # linear replay of event log
      → condenser.condense(view, agent_llm)  # optional compression
      → LLMConvertibleEvent.events_to_messages(events)  # convert to LLM messages
  → make_llm_completion(llm, messages, tools)
```

The **system message comes from a `SystemPromptEvent`** that's an entry in the event log — not rebuilt from config on each step.

---

## Why Dynamic Injection Fails

### 1. `AgentBase` is a frozen Pydantic model

```python
# base.py line 108-111
model_config = ConfigDict(
    frozen=True,
    arbitrary_types_allowed=True,
)
```

You cannot update `agent.condenser`, `agent.agent_context`, or any other field after construction. The agent config is immutable for the life of the conversation.

### 2. `SystemPromptEvent` is written once at conversation init

```python
# agent.py — init_state()
def init_state(self, state, on_event):
    ...
    if has_system_prompt:
        return  # skip if already written — for conversation resume
    ...
    on_event(SystemPromptEvent(
        system_prompt=TextContent(text=self.static_system_message),
        dynamic_context=dynamic_context,  # TextContent | None
        tools=list(self._tools.values()),
    ))
```

Once emitted, this event sits in the immutable event log. `init_state()` is never called again during the conversation.

### 3. OpenHands' "dynamic" is per-conversation, not per-turn

The `dynamic_context` field in `SystemPromptEvent` (and `AgentBase.dynamic_context` property) is explicitly designed for **per-conversation** variation:

```python
# system.py
dynamic_context: TextContent | None = Field(
    default=None,
    description=(
        "Optional dynamic per-conversation context (runtime info, repo context, "
        "secrets). When provided, this is included as a second content block in "
        "the system message (not cached)."
    ),
)
```

Their model: different repos, different users = different dynamic context **at conversation start**. Our model: memory blocks change **after every tool call and message**.

### 4. `prepare_llm_messages()` has no per-turn injection hook

```python
# utils.py lines 470-521
def prepare_llm_messages(events, condenser=None, ...):
    view = View.from_events(events)      # replay event log
    if condenser:
        condensation_result = condenser.condense(view, agent_llm=llm)  # compress
        ...
    messages = LLMConvertibleEvent.events_to_messages(llm_convertible_events)
    return messages
```

The condenser API is `condense(view: View, agent_llm: LLM | None) -> View | Condensation`. It receives the frozen event list and can transform it, but has no access to per-turn external state like our memory DB.

---

## Could a Custom Condenser Work Around This?

**Technically possible. Architecturally wrong. Practically unfeasible.**

Theoretically, a custom `CondenserBase` subclass could:
1. Receive the `View` on each step
2. Find and drop the frozen `SystemPromptEvent`
3. Reconstruct a fresh one with current block state
4. Return the modified `View`

Why this doesn't work in practice:

**Problem A: No conversation context in the condenser call.**
`condense(view, agent_llm)` — no conversation ID, no session context. The condenser can't know *which* agent's blocks to fetch from our DB.

**Problem B: No clean state update path.**
Because `AgentBase` is `frozen=True`, you can't do `agent.condenser = updated_condenser` between steps. The condenser would need mutable Python state (not Pydantic fields) to hold a reference to current blocks, updated via some side channel. That's thread-unsafe and completely fragile.

**Problem C: Splits persisted state from runtime state.**
The event log would still contain the old `SystemPromptEvent`. Only the condenser's transformed `View` would have current blocks. This fights every assumption OpenHands makes about event-log-as-ground-truth.

**Problem D: We'd own all the interesting work anyway.**
The condenser workaround is essentially "build our memory system inside OpenHands' framework." At that point, we're getting nothing from OpenHands except its overhead — tool handling, security checking, event types, Jinja2 templates — none of which we want or need.

---

## The Core Incompatibility

| Dimension | OpenHands design assumption | Our requirement |
|---|---|---|
| Agent config lifetime | Immutable after construction (`frozen=True`) | Memory blocks are living state, change every turn |
| System prompt | Written once at conversation init | Rebuilt every turn from current block state |
| "Dynamic" context | Per-conversation (repo, runtime, secrets) | Per-turn (block content after tool calls) |
| Event log | Append-only, immutable, ground truth | Block state NOT in event log — injected fresh from DB |
| Condenser purpose | Context compression (fit in window) | Not a memory injection layer |

This is by design on their end — their architecture optimizes for Anthropic prompt caching (immutable static block = cross-conversation cache hits). Our architecture optimizes for dynamic memory (blocks updated after every tool call = always-current context).

These are not compatible goals.

---

## Verdict

**Greenfield.** The OpenHands investigation is complete and the answer is definitive.

The benchmarking survey has now reviewed all candidates:
- Letta: left in March for good reasons
- PAIS: private `_agent_graph` import, 25 stars, we'd override the core loop anyway
- Jaato: 0 stars, likely AI-generated scaffolding
- deepagents: same monolith problems as Letta (`_agent_graph` import, ~400-line function)
- OpenHands: frozen-model architecture incompatible with per-turn block injection

The frameworks were worth investigating — they confirmed that our design is genuinely novel (nobody is doing block-based dynamic per-turn memory injection with pydantic-ai) and that there's no existing foundation we'd want to build on.

Build on pydantic-ai + FastAPI directly. Own the agent loop. The memory system IS the novel contribution.

---

## Server Architecture, Component Lifecycles, and Persistence

*Additional investigation, May 29, 2026 — Sonnet*

### Additional Files Read

From `openhands-agent-server/openhands/agent_server/`:
- `event_service.py` (1225 lines)
- `conversation_service.py` (1396 lines)
- `dependencies.py` (90 lines)

From `openhands-sdk/openhands/sdk/conversation/`:
- `impl/local_conversation.py` (1968 lines — targeted greps)
- `state.py` (572 lines — targeted greps)

---

### Full Request Flow (REST → LLM call)

```
POST /{conversation_id}/events                             [event_router.py:190]
  → EventService.send_message(message, run=True)           [event_service.py:433]
      → run_in_executor(_conversation.send_message)        [sync write to event log, awaited]
      → self.run()                                         [spawns background task, returns]
          → _run_and_publish()
              → conversation.arun()                        [local_conversation.py:1059]
                  → while True:                            [line 1110]
                      → agent.astep(conv, on_event, on_token)  [agent.py:684]
                          → aprepare_llm_messages(state.events, condenser, llm)  [line 722]
                          → amake_llm_completion(llm, messages, tools)           [line 738]
                          → dispatch: tool calls / content / stop
```

---

### Component Lifecycles

**`ConversationService`** (`conversation_service.py:371`):
- Stored as `request.app.state.conversation_service` — a process-lifetime singleton
- Holds `_event_services: dict[UUID, EventService]`
- `__aenter__` (line 939): on server start, iterates all `conversations_dir/*/meta.json`, calls `_start_event_service(stored)` for each, populating the dict
- All `EventService` instances remain in-memory for the full server process lifetime

**`EventService`** (`event_service.py:61`):
- One per conversation UUID; created by `ConversationService._start_event_service()` (line 1080)
- Key fields: `stored: StoredConversation`, `_conversation: LocalConversation | None = None`
- `_conversation` is `None` at construction; set during `start()` (lines 693–771)
- Once set, `_conversation` persists for the `EventService` lifetime (= server process lifetime)

**`LocalConversation`** (`local_conversation.py`):
- One per `EventService`; constructed in `EventService.start()` at line 751
- Constructor args include: `agent`, `workspace`, `persistence_dir`, `conversation_id`, `plugins`, `callbacks`, `max_iteration_per_run`, `secrets`, `cipher`, `hook_config`
- Holds `_state: ConversationState`
- `_ensure_agent_ready()` (line 657): double-checked lock — `if self._agent_ready: return` makes all subsequent calls no-ops

**`Agent`** (inside `LocalConversation`):
- One per `LocalConversation`
- Initialized on first `_ensure_agent_ready()` call via `agent.init_state()`
- `init_state()` contains: `if has_system_prompt: return` — skips re-initialization when resuming a conversation that already has a `SystemPromptEvent` in the event log

---

### Persistence Model

`ConversationState.create()` (`state.py:282`) behaves differently depending on whether `persistence_dir` is provided:

- **With `persistence_dir`**: reads `base_state.json` from a file store; attaches `EventLog(file_store, dir_path=EVENTS_DIR)` pointing to the existing event stream; calls `agent.verify(state.agent, events=state._events)` for tool consistency
- **Without `persistence_dir`**: falls back to `InMemoryFileStore` with no durability; logs a warning

Storage layout per conversation (all file-based, not DB-backed):
```
conversations_dir/
  {conversation_id}/
    meta.json          ← StoredConversation (agent config, plugins, settings)
    base_state.json    ← ConversationState scalar fields (execution status, stats)
    events/            ← EventLog (append-only event stream)
    observations/      ← env_observation_persistence_dir (large observation blobs)
```

---

### Server Restart Behavior

On server start, `ConversationService.__aenter__` (line 939):
1. Reads each `conversations_dir/*/meta.json` → `StoredConversation`
2. Calls `_start_event_service(stored)` for each → new `EventService` + `LocalConversation`
3. `LocalConversation._state` is created with the persisted `persistence_dir`, loading the existing `base_state.json` and attaching the existing `EventLog`

On the first message received after restart:
4. `_ensure_agent_ready()` fires → calls `init_state()`
5. `init_state()` inspects the loaded `EventLog` for an existing `SystemPromptEvent`
6. `has_system_prompt` is `True` → `init_state()` returns without emitting a new `SystemPromptEvent`

---

## References

- `software-agent-sdk/openhands-sdk/openhands/sdk/agent/base.py` — `AgentBase`, frozen model config, `init_state()`
- `software-agent-sdk/openhands-sdk/openhands/sdk/agent/utils.py` — `prepare_llm_messages()` pipeline  
- `software-agent-sdk/openhands-sdk/openhands/sdk/event/llm_convertible/system.py` — `SystemPromptEvent`
- `software-agent-sdk/openhands-sdk/openhands/sdk/context/view/view.py` — `View`, condenser I/O
- `software-agent-sdk/openhands-agent-server/openhands/agent_server/event_service.py` — `EventService`, `LocalConversation` construction, `start()`
- `software-agent-sdk/openhands-agent-server/openhands/agent_server/conversation_service.py` — `ConversationService`, `_event_services` dict, `__aenter__` startup loop
- `software-agent-sdk/openhands-agent-server/openhands/agent_server/dependencies.py` — FastAPI dependency injection, `get_event_service()`
- `software-agent-sdk/openhands-sdk/openhands/sdk/conversation/state.py` — `ConversationState.create()`, persistence model, `EventLog` attachment
- `software-agent-sdk/openhands-sdk/openhands/sdk/conversation/impl/local_conversation.py` — `_ensure_agent_ready()`, `_state` lifecycle
