# Req:

- [ ] Doesn't duplicate during persistence across persistence pts
- [ ] Doesn't persist message history (messages loaded from the DB and passed in at agent
  construction must NOT be re-persisted). Technically a sub-case of "doesn't duplicate
  across persistence pts," but called out explicitly because it's an easy gotcha.
- [ ] Persists every complete Model Message as you go, **except** orphaned tool calls. A
  ModelResponse carrying a ToolCallPart must NOT be persisted until its matching
  ToolReturnPart exists — i.e. defer the persistence point until the tool returns, so the
  call+return go in as a well-formed unit. (persist_messages' orphan sanitizer is a
  failsafe against invalid histories; we should never actually be feeding it an orphan. If we persisted tool calls as we go w/ no return, persist would sanitize both the tool call and the return)
- [ ] Persists even on mid-run exception or cancellation
- [ ] Cancellation allows any active tool execution to complete
  - May be sensitive to PydanticAI internals w/o `agent.iter()` whatever
  - We may find that this is an issue for longer running tools. If I press cancel and have to wait 2 minutes for the agent to be available again from a blocking tool call, that is probably not ideal
    Consider this out of scope for now, complex to deal with. Ideally we would also abort the running tool rather than just unblocking the agent and let it run in the background. Not necessarily trivial with MCP considerations
- [ ] Cancellation ends persisted chain w/ Cancellation notice (user msg containing `<system_message>content</system_message>` or similar)
- [ ] Cancellation achieved through PydanticAI recommended method.
  - For `run_stream_events`: exit the async context manager
  - Pyd AI #5313 is relevant
  - The complexity of testing this one in an impl agnostic way may not be worth it. We can talk about it
- [ ] Cancellation works through /agents/{id}/cancel (which we will map to ACP session/cancel. Either at the server level, at the adapter level, etc. This is effectively a naming concern so OK to defer that detail)

# Test Coverage (tests/api/test_routes.py)
FILL IN

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
