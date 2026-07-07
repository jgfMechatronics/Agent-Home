# test_routes.py structural map
**File:** `tests/api/test_routes.py` (1222 lines)  
**Branch:** `Development/EnableCancellationImprovePersistence`  
**Author:** Sonnet — June 12, 2026

---

## DEDUP CANDIDATES

| # | Lines (A) | Lines (B) | Nature | Commonization idea |
|---|-----------|-----------|--------|--------------------|
| 1 | 334–348 | 397–411 | **Near-identical tests** | Both configure `PartStartEvent + raises_mid_stream=RuntimeError`, both assert `sse_events[-1]["event"] == "Error"`. B is strictly weaker (no count, no message check). Merge into A or drop B. **Strong candidate.** |
| 2 | 221–224 | 524–530 | **Mock session construction** | `TestSendMessage.mock_agent_dep` manually builds `mock_session` with 3 AsyncMock attrs; `_make_mock_session()` already exists at 524. Replace 221–224 with `self.mock_session = _make_mock_session()`. **Easy win.** |
| 3 | 617–641 | 774–789 | **`real_agent_dep` fixtures** | Both: set `self.agent_record`, call `_make_mock_session()`, install `get_agent_and_deps` override, pop on yield. Differ in agent construction and extra attrs (`set_stream_fn` vs `tool_entered/release`). Could extract the install/cleanup skeleton into `_PersistSpyMixin`, but unique parts are meaningful — factoring would add abstraction overhead. **Low priority.** |
| 4 | 560–565 | 724–729 | **Stream function structure** | `_two_step_stream` and `_blocking_tool_stream` are structurally identical (`_has_tool_return` branch, same shape). Differ only in tool name and terminal text. `_tool_call_delta` (553–557) was already extracted for this reason. Could unify via a factory, but readability would suffer. **Skip unless more stream fns are added.** |
| 5 | 226–238 | 1143–1152 | **`_configure` dep-override factories** | Both define an inner `_configure(raise_exc=None)` that installs a fake `async def _mock_dep()` into `app.dependency_overrides`. Differ in the dep key and what the dep yields. Conceptually the same; practically, the yield contents differ enough that extraction would need a `yield_fn` callback — marginal gain. **Note only.** |
| 6 | 206–241 | 617–641 | **Autouse mock-dep fixtures** | `TestSendMessage.mock_agent_dep` and `TestPersistenceAcrossInterruptions.real_agent_dep` both install `get_agent_and_deps` with a mock/real agent + mock session. The former uses a configurable mock; the latter uses a real pydantic-ai Agent. Difference is intentional (unit vs integration) — no merge, but note their symmetry. |

---

## MODULE-LEVEL TEST DATA (55–71)

| Name | Line | Purpose |
|------|------|---------|
| `TOOL_CALL_PART` | 57 | Fixed `ToolCallPart` used in TOOL_STREAM |
| `TOOL_RETURN_PART` | 58 | Matching `ToolReturnPart` |
| `MINIMAL_STREAM` | 61 | Lambda → `[AgentRunResultEvent]` only |
| `TEXT_STREAM` | 62–66 | Lambda → PartStartEvent + PartDeltaEvent + AgentRunResultEvent |
| `TOOL_STREAM` | 67–71 | Lambda → FunctionToolCallEvent + FunctionToolResultEvent + AgentRunResultEvent |

---

## MODULE-LEVEL FIXTURES (74–153)

| Name | Lines | Scope | autouse | Purpose |
|------|-------|-------|---------|---------|
| `app` | 76–80 | function | No | Fresh `FastAPI` instance per test via `_create_app()` |
| `client` | 83–94 | function | No | `AsyncClient` over `ASGITransport`; `raise_app_exceptions=False` |
| `override_db_session` | 115–126 | function | **Yes** | Injects test `AsyncSession` into `get_session_dep` for all tests |
| `_base_route_patches` | 129–152 | function | No | Patches `load_messages`, `deserialize_messages`, `is_compaction_needed`, `compact`; shared by `TestSendMessage` and `_PersistSpyMixin` |

---

## MODULE-LEVEL HELPERS (97–113, 155–196)

| Name | Lines | Purpose |
|------|-------|---------|
| `make_mock_agent(events, raises_mid_stream)` | 97–112 | Build a mock `Agent` whose `run_stream_events` yields given events; optionally raises after |
| `VALID_SSE_PREFIXES` | 157 | Tuple of legal SSE line prefixes for the assertion in `collect_sse_events` |
| `collect_sse_events(response)` | 159–182 | Parse SSE response into `[{event, data}]`; asserts every line is a valid SSE field |
| `stream_and_collect(client, agent_id, message)` | 185–195 | POST to messages endpoint, assert 200, return parsed SSE events |

---

## PERSISTENCE SECTION HELPERS (477–580)

| Name | Lines | Purpose |
|------|-------|---------|
| `_union_of_persisted(spy)` | 481–487 | Concatenate messages from all `persist_messages` call_args in order |
| `_select(union, msg_type, part_type)` | 490–495 | Filter `union` to messages of `msg_type` containing at least one `part_type` part |
| `_assert_no_duplicates(union)` | 498–504 | Assert each message object appears at most once |
| `_assert_no_orphans(union)` | 507–521 | Assert every `ToolCallPart` has a matching `ToolReturnPart` or `RetryPromptPart` |
| `_make_mock_session()` | 524–530 | Build `Mock` session with `AsyncMock` commit/rollback/refresh |
| `_make_function_agent(stream_fn)` | 533–541 | Build real `Agent(FunctionModel)` + register `record_thing` tool |
| `_has_tool_return(messages)` | 544–550 | True when messages contain a `ToolReturnPart` (stream is in step 2) |
| `_tool_call_delta(tool_name, tool_call_id)` | 553–557 | Build a `DeltaToolCalls` with one tool call entry |
| `_two_step_stream(messages, info)` | 560–565 | Step 1 → emit `record_thing` call; step 2 → yield `"Turn complete."` |
| `_exception_after_tool_return_stream(messages, info)` | 568–580 | Step 1 → emit tool call; step 2 → yield leading text then raise `RuntimeError` |

---

## CANCELLATION HELPERS (724–757)

| Name | Lines | Purpose |
|------|-------|---------|
| `_blocking_tool_stream(messages, info)` | 724–729 | Step 1 → emit `blocking_tool` call; step 2 → yield `"Done."` |
| `_make_blocking_agent(tool_entered, release)` | 732–746 | Build Agent with async tool that signals `tool_entered` and awaits `release` |
| `_PYDANTIC_AI_VERSION` | 753 | Runtime pydantic-ai version string |
| `_RENDEZVOUS_MIN_VERSION` | 757 | `"1.104.0"` — minimum version with confirmed rendezvous semantics |

---

## BASE CLASS

| Name | Lines | Purpose |
|------|-------|---------|
| `_PersistSpyMixin` | 587–603 | Autouse `persist_spy` fixture: patches `persist_messages` (autospec) + exposes `_base_route_patches` attrs as `self.*`. Inherited by `TestPersistenceAcrossInterruptions` and `TestCancellation`. |

---

## TEST CLASSES

### `TestSendMessage` (200–411)
**Tests:** `POST /agents/{agent_id}/messages` — main streaming endpoint.

**Fixtures:**
| Fixture | Lines | autouse | Purpose |
|---------|-------|---------|---------|
| `mock_agent_dep` | 206–241 | **Yes** | Injects mock Agent + mock session via `get_agent_and_deps`; exposes `self.configure_mock_get_agent_and_deps()`, `self.mock_session`, `self.agent_record`. Default stream: `MINIMAL_STREAM`. |
| `mock_route_side_effects` | 243–256 | **Yes** | Adds `persist_messages` mock on top of `_base_route_patches`; exposes `self.mock_persist`, `self.mock_load_messages`, etc. |

**Tests:**
| Name | Line | What it asserts |
|------|------|----------------|
| `test_event_types_are_forwarded_as_sse` | 262 | Parametrized (text/tool): SSE event type sequence matches pydantic-ai event names |
| `test_dep_http_exception_returns_appropriate_status` | 278 | Parametrized (404/503): HTTPException from dep propagates to correct HTTP status |
| `test_content_type_is_event_stream` | 291 | Response Content-Type contains `text/event-stream` |
| `test_returns_400_for_malformed_body` | 301 | Missing `message` field → 400 or 422 |
| `test_persists_messages_on_agent_run_result_event` | 312 | `persist_messages` called exactly once (on `AgentRunResultEvent`) |
| `test_compaction_called_based_on_check` | 323 | Parametrized: `compact` called iff `is_compaction_needed` returns True |
| `test_yields_error_event_on_exception` | 334 | Partial stream + mid-stream exception → PartStartEvent + Error SSE; checks exact error message |
| `test_persist_called_when_new_messages_empty` | 350 | `persist_messages` still called when `new_messages()` returns `[]` |
| `test_commits_session_on_happy_path` | 365 | Session committed once, no rollback |
| `test_rollback_and_error_sse_on_persist_failure` | 372 | Persist failure → rollback + Error SSE, no commit |
| `test_commits_before_compaction_failure_so_turn_is_preserved` | 382 | Compaction failure → commit already called, Error SSE |
| `test_yields_error_sse_on_mid_stream_exception` | 397 | Mid-stream exception → last SSE event is Error *(weaker duplicate of line 334 — see DEDUP #1)* |

---

### `TestCreateAgent` (413–475)
**Tests:** `POST /agents/` — create agent.

**Fixtures:**
| Fixture | Lines | autouse | Purpose |
|---------|-------|---------|---------|
| `mock_create_agent_deps` | 428–432 | **Yes** | Patches `create_agent_record`; exposes `self.mock_create_agent_record` |

**Tests:**
| Name | Line | What it asserts |
|------|------|----------------|
| `test_creates_agent_and_returns_metadata` | 434 | 201 + `AgentMetadataResponse` matches mock record fields |
| `test_returns_500_when_create_agent_fails` | 461 | Unexpected exception → 500 with detail |
| `test_returns_400_for_invalid_config` | 468 | Missing required fields → 400 or 422 |

---

### `TestPersistenceAcrossInterruptions` (606–721) — inherits `_PersistSpyMixin`
**Tests:** Persistence contract using real pydantic-ai Agent + FunctionModel.

**Fixtures:**
| Fixture | Lines | autouse | Purpose |
|---------|-------|---------|---------|
| `persist_spy` (from mixin) | 594–603 | **Yes** | Spy on `persist_messages`; expose base patches as `self.*` |
| `real_agent_dep` | 617–641 | **Yes** | Real `Agent(FunctionModel)` + mock session; `self.set_stream_fn(fn)` swaps stream fn |

**Tests:**
| Name | Line | What it asserts |
|------|------|----------------|
| `test_happy_path_persists_full_message_union` | 644 | Union contains UserPrompt, ToolCall, ToolReturn, Text in causal order; no dups/orphans; history excluded; commit called |
| `test_persist_survives_mid_run_exception` | 682 | **RED** — persist called on crash, union has ToolCall+Return but no Text, commit not rollback *(contract-defining)* |

---

### `TestCancellation` (764–872) — inherits `_PersistSpyMixin`
**Tests:** Cancellation contract using blocking-tool Agent for deterministic timing.

**Fixtures:**
| Fixture | Lines | autouse | Purpose |
|---------|-------|---------|---------|
| `persist_spy` (from mixin) | 594–603 | **Yes** | Same as above |
| `real_agent_dep` | 774–789 | **Yes** | Blocking Agent + `self.tool_entered`/`self.release` events + mock session |

**Tests:**
| Name | Line | What it asserts |
|------|------|----------------|
| `test_graceful_cancel` | 799 | **xfail(strict=True)** — cancel → 200, tool completes, pair persisted, no post-tool text, cancel notice, commit not rollback |

---

### Standalone tests (875–974)

| Name | Line | Status | What it asserts |
|------|------|--------|----------------|
| `test_rendezvous_tool_does_not_start_before_event_consumed` | 884 | GREEN | pydantic-ai rendezvous: tool does NOT start before `FunctionToolCallEvent` consumed; `asyncio.skipif` on versions < 1.104.0 |
| `test_TODO_decide_cancel_orphan_tool_call_handling` | 945 | `pytest.fail()` | Decision marker: trim vs. let `persist_messages` sanitizer eat orphaned ToolCallPart on cancel |

---

### `TestGetAgent` (977–999)
**Tests:** `GET /agents/{agent_id}` — agent metadata. No class fixtures.

| Name | Line | What it asserts |
|------|------|----------------|
| `test_returns_agent_metadata` | 980 | 200 + `AgentMetadataResponse` matches `agent_record` fields |

---

### `TestGetMemoryBlocks` (1002–1034)
**Tests:** `GET /agents/{agent_id}/memory/blocks`. No class fixtures.

| Name | Line | What it asserts |
|------|------|----------------|
| `test_returns_memory_blocks` | 1005 | 200 + blocks in position order match schema |
| `test_returns_empty_blocks_list_when_no_blocks` | 1026 | 200 + `blocks == []` when agent has none |

---

### `TestGetMessages` (1037–1087) — class-level `@pytest.mark.xfail`
**Tests:** `GET /agents/{agent_id}/messages` (format TBD).

**Fixtures:**
| Fixture | Lines | autouse | Purpose |
|---------|-------|---------|---------|
| `mock_message_loaders` | 1044–1055 | **Yes** | Patches `load_messages`; exposes `self.mock_load_messages` |

**Tests:**
| Name | Line | What it asserts |
|------|------|----------------|
| `test_default_loads_context_window_and_returns_messages` | 1057 | `load_messages` called with `context_window_start`; response contains messages |
| `test_full_true_returns_complete_history` | 1070 | `?full=true` → `load_messages` called with `start_timestamp=None` |
| `test_returns_reasonable_format` | 1083 | `pytest.fail()` — stub, format not yet defined |

---

### `TestHealthCheck` (1090–1104)
**Tests:** `GET /health`. No class fixtures.

| Name | Line | Status | What it asserts |
|------|------|--------|----------------|
| `test_returns_200_ok` | 1093 | GREEN | 200 + `{"status": "ok"}` |
| `test_returns_503_when_db_unreachable` | 1100 | xfail | 503 when DB unreachable (not yet implemented) |

---

### `TestNotFound` (1107–1119)
**Tests:** 404 behavior across GET endpoints. No class fixtures.

| Name | Line | What it asserts |
|------|------|----------------|
| `test_get_endpoints_return_404_for_unknown_agent` | 1115 | Parametrized 3 paths: all return 404 for unknown UUID |

---

### `TestCreateMemoryBlock` (1122–1222)
**Tests:** `POST /agents/{agent_id}/memory/blocks`.

**Fixtures:**
| Fixture | Lines | autouse | Purpose |
|---------|-------|---------|---------|
| `mock_create_block_dep` | 1133–1158 | **Yes** | Overrides `get_deps_dep` (configurable raise_exc); patches `create_block`; exposes `self.configure_mock_get_deps_dep()`, `self.mock_create_block`, `self.agent_record` |

**Tests:**
| Name | Line | What it asserts |
|------|------|----------------|
| `test_calls_create_block_and_returns_201` | 1160 | 201 + `MemoryBlockResponse` matches created record |
| `test_returns_404_for_unknown_agent` | 1176 | `AgentNotFoundError` from dep → 404, `create_block` not called |
| `test_returns_400_for_duplicate_block` | 1191 | `DuplicateBlockError` → 400 with label in detail |
| `test_returns_500_for_unexpected_error` | 1209 | Unexpected exception → 500 with detail |

---

## OPUS RECOMMENDATIONS (Jun 12)
Verdicts after targeted reads of every flagged section. Tiered by value/risk.

### Tier 1 — clear wins, low risk
- **Drop `test_yields_error_sse_on_mid_stream_exception` (397–411).** Strictly subsumed by
  `test_yields_error_event_on_exception` (334–348): identical partial-stream + mid-stream-`RuntimeError`
  setup; 397 asserts only `sse_events[-1] == "Error"`, while 334 already asserts that PLUS `len == 2`,
  the leading `PartStartEvent`, AND the exact error message. The only difference is the RuntimeError's
  text — which 334 verifies precisely. **Zero coverage lost.** [Sonnet #1 — confirmed]
- **Use `_make_mock_session()` at 221–224.** The manual 4-line Mock+AsyncMock build in `mock_agent_dep`
  is byte-identical to `_make_mock_session()` (524–530) → `self.mock_session = _make_mock_session()`.
  (Optional: bare `Mock()` at 1141 in `mock_create_block_dep` could adopt it too for consistency, but it
  doesn't exercise the async methods, so not required.) [Sonnet #2 — confirmed]

### Tier 2 — worthwhile, small judgment call (DRY-positive, slight abstraction cost)
- **Extract `_override_dependency(app, dep_key, dep_factory)` context manager** for the 4 manual
  install-then-pop sites: `mock_agent_dep` (232/241), `real_agent_dep`×2 (639/641, 787/789),
  `mock_create_block_dep` (1149/1158). Pairs install+teardown so `.pop()` can't be forgotten or drift.
  **This is the single biggest *conceptual* repetition in the file — spans 4 classes.**
  [synthesis across Sonnet #3/#5/#6 — elevated]
- **Lift the `real_agent_dep` skeleton into `_PersistSpyMixin`** (both subclasses already inherit it).
  The two fixtures (617–641 / 774–789) share an identical skeleton — set `agent_record`,
  `_make_mock_session()`, install override yielding `(agent, make_deps(...))`, pop on teardown — differing
  only in agent construction + per-class attrs. Provide `self._install_agent_override(app, agent_record,
  agent_factory)` (built on the context manager above); each subclass keeps a thin fixture for its unique
  attrs (`set_stream_fn` vs `tool_entered`/`release`). The agent-factory **callback** preserves
  late-binding of `self._stream_fn`. [Sonnet #3 — elevated from "low priority": callback param is
  idiomatic, not forced, and the home (`_PersistSpyMixin`) already exists]

### Tier 3 — note only / skip (agree with Sonnet)
- **Stream-fn factory** for `_two_step_stream`/`_blocking_tool_stream` (+ variant
  `_exception_after_tool_return_stream`): only 2–3 instances and the named fns read better than factory
  calls; `_tool_call_delta` already removed the real boilerplate. Skip unless more appear. [Sonnet #4]
- **`_configure` raise-guard factories** (226–238 / 1143–1152): a genuine conceptual twin (raise_exc-guarded
  dep override) but the yielded payloads differ structurally (agent+deps vs deps); a generic helper needs a
  yield-callback for marginal gain — and is largely absorbed if we adopt `_override_dependency` above.
  [Sonnet #5]

### Not pursued
- `mock_agent_dep` vs `real_agent_dep` symmetry (Sonnet #6): intentional unit-vs-integration split. Leave separate.
