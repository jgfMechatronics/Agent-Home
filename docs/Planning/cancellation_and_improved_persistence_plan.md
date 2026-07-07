# Req:
[xr] indicates covered w/ test and reviewed

- [xr] Doesn't duplicate during persistence across persistence pts
- [xr] Doesn't persist message history (messages loaded from the DB and passed in at agent
  construction must NOT be re-persisted). Technically a sub-case of "doesn't duplicate
  across persistence pts," but called out explicitly because it's an easy gotcha.
- [xr] Persists every complete Model Message as you go, **except** orphaned tool calls. A
  ModelResponse carrying a ToolCallPart must NOT be persisted until its matching
  ToolReturnPart exists — i.e. defer the persistence point until the tool returns, so the
  call+return go in as a well-formed unit. (persist_messages' orphan sanitizer is a
  failsafe against invalid histories; we should never actually be feeding it an orphan. If we persisted tool calls as we go w/ no return, persist would sanitize both the tool call and the return)
- [xr] Persists even on mid-run exception or cancellation
  - Here it can go ahead and feed an orphaned tool call to persist_messages and let it just strip out the bad parts. This gives a more complete (and valid) record of what happened.
- [xr] Cancellation allows any active tool execution to complete
  - May be sensitive to PydanticAI internals w/o `agent.iter()` whatever
  - We may find that this is an issue for longer running tools. If I press cancel and have to wait 2 minutes for the agent to be available again from a blocking tool call, that is probably not ideal
    Consider this out of scope for now, complex to deal with. Ideally we would also abort the running tool rather than just unblocking the agent and let it run in the background. Not necessarily trivial with MCP considerations
- [xr] Cancellation ends persisted chain w/ Cancellation notice (user msg containing `<system_message>content</system_message>` or similar)
- [PENDING IMPL] Cancellation achieved through PydanticAI recommended method.
  - For `run_stream_events`: exit the async context manager
  - Pyd AI #5313 is relevant
  - testing this is not worth it. We just need to make sure that we use the recommended method when implementing
- [xr] Cancellation works through /agents/{id}/cancel (which we will map to ACP session/cancel. Either at the server level, at the adapter level, etc. This is effectively a naming concern so OK to defer that detail)


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
UPDATE: We chose option B. This means this is no longer the routes concern.

Both yield API-valid history (no dangling `tool_use`), so this is a narrative-cleanliness
call, **deliberately deferred**. It is parked as a standing `pytest.fail` marker —
`test_TODO_decide_cancel_orphan_tool_call_handling` — so the impl can't ship without
resolving it. On resolution: make the decision, update this table, and replace the marker
with the real persistence unit test.
