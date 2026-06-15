# Req:

- [ ] Doesn't duplicate during persistence across persistence pts
- [ ] Doesn't persist message history (messages loaded from the DB and passed in at agent
  construction must NOT be re-persisted). Technically a sub-case of "doesn't duplicate
  across persistence pts," but called out explicitly because it's an easy gotcha.
- [ ] Persists every complete Model Message as you go, **except** orphaned tool calls. A
  ModelResponse carrying a ToolCallPart must NOT be persisted until its matching
  ToolReturnPart exists — i.e. defer the persistence point until the tool returns, so the
  call+return go in as a well-formed unit. (persist_messages' orphan sanitizer is a
  failsafe against invalid histories; we should never actually be feeding it an orphan.)
- [ ] Persists even on mid-run exception or cancellation
- [ ] Cancellation allows any active tool execution to complete
  - May be sensitive to PydanticAI internals w/o `agent.iter()` whatever
- [ ] Cancellation ends persisted chain w/ Cancellation notice (user msg containing `<system_message>content</system_message>` or similar)
- [ ] Cancellation achieved through PydanticAI recommended method.
  - For `run_stream_events`: exit the async context manager
  - Pyd AI #5313 is relevant
  - The complexity of testing this one in an impl agnostic way may not be worth it. We can talk about it
- [ ] Cancellation works through /agents/{id}/cancel (which we will map to ACP session/cancel. Either at the server level, at the adapter level, etc. This is effectively a naming concern so OK to defer that detail)

# Test Coverage (tests/api/test_routes.py)

Tests are written first (TDD). Persistence-survival and cancel tests define the
contract for the not-yet-written implementation, so they are RED/xfail until impl lands.

| # | Requirement | Test(s) | State |
|---|-------------|---------|-------|
| 1 | No duplication across persistence points | `TestPersistenceAcrossInterruptions::test_happy_path_persists_full_message_union` (asserts `_assert_no_duplicates` on the persisted union) | GREEN (teeth come from the future as-you-go multi-point impl, where the identity check catches cursor re-persist) |
| 1b | Doesn't re-persist pre-loaded message history | `test_happy_path_persists_full_message_union` (injects fake history via `deserialize`, asserts it's excluded from the union) | GREEN |
| 2 | Persists every complete Model Message as you go, except orphaned tool calls | `test_happy_path...` (well-formed call+return pair, `_assert_no_orphans`); as-you-go incremental behavior is contract for impl | GREEN (happy path); incremental asserted at impl time |
| 3 | Persists even on mid-run exception or cancellation | `test_persist_survives_mid_run_exception` (exception path) + `TestCancellation::test_graceful_cancel` (cancel path) | RED (exception, hard) / xfail (cancel) |
| 4 | Cancellation allows active tool to complete | `test_graceful_cancel` (toolcall+toolreturn pair persisted, no final TextPart) + `test_rendezvous_tool_does_not_start_before_event_consumed` (guards the pyd rendezvous property the strategy relies on) | xfail (graceful) / GREEN (rendezvous guard) |
| 5 | Cancellation notice persisted (`<system_message>` user msg) | `test_graceful_cancel` (asserts a `UserPromptPart` containing `<system_message>` in the union) | xfail |
| 6 | Cancellation via PydanticAI recommended method | Mechanism is impl-specific (not asserted directly per plan); `test_rendezvous...` guards the library behavior our `run_stream_events` deferred-CM-exit strategy depends on | GREEN (guard) |
| 7 | Cancellation via `/agents/{id}/cancel` | `test_graceful_cancel` (POSTs to the route, asserts 200) | xfail (route unimplemented → 404) |

RED vs xfail convention: **hard RED** when existing code violates a contract
(`test_persist_survives_mid_run_exception` — persist-at-end discards on exception);
**xfail(strict=True)** when the infrastructure doesn't exist yet (cancel route +
`_cancel_signals`). strict=True self-cleans: an unexpected pass errors, forcing marker removal.

## Open decision: cancel-orphan tool-call handling (`pytest.fail` marker)

When a cancel lands on a tool call that was **generated but never run** (the rendezvous
property guarantees the tool didn't start), the tail of the captured/persisted messages is
a lone `ToolCallPart` with no matching `ToolReturnPart`. Two viable handlings:

- **(A) Trim** the lone `ToolCallPart` in the cursor/persist logic (tool re-requested next
  turn). Preserves sibling text/thinking parts on that `ModelResponse`. Needs a unit test of
  the trim/cursor logic.
- **(B) Let `persist_messages` eat it** — its orphan sanitizer already replaces an unmatched
  `ToolCallPart` with an `[Orphaned tool call(s) dropped]` record, keeping history API-valid.
  Cheaper. **NOTE (verified Jun 12):** the sanitizer is **whole-message** —
  `_replace_orphaned_tool_messages` (`messages.py`) skips the entire orphaned `ModelResponse`
  and replaces it with `ModelResponse(parts=[TextPart(error_text)])`, so **any accompanying
  text/thinking parts are discarded**. If we care about preserving that prose, (B) requires
  *also* changing the sanitizer to do part-level (not message-level) replacement. As-is, (A)
  preserves the most.

Both yield API-valid history (no dangling `tool_use`), so this is a narrative-cleanliness
call, **deliberately deferred**. It is parked as a standing `pytest.fail` marker —
`test_TODO_decide_cancel_orphan_tool_call_handling` — so the impl can't ship without
resolving it. On resolution: make the decision, update this table, and replace the marker
with the real persistence unit test.

> Note: a `layer-1` regression guard (pinning that pyd appends the `ModelResponse` *before*
> the event is consumed) was considered and **rejected** — it would false-alarm if a future
> pyd change *eliminated* the orphan possibility (i.e., fail when things got safer). The
> rendezvous guard is kept because it fails on a genuinely *dangerous* change.
