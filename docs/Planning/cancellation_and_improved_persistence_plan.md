# Req:

- Doesn't duplicate during persistence across persistence pts
- Doesn't persist message history (messages loaded from the DB and passed in at agent
  construction must NOT be re-persisted). Technically a sub-case of "doesn't duplicate
  across persistence pts," but called out explicitly because it's an easy gotcha.
- Persists every complete Model Message as you go, **except** orphaned tool calls. A
  ModelResponse carrying a ToolCallPart must NOT be persisted until its matching
  ToolReturnPart exists — i.e. defer the persistence point until the tool returns, so the
  call+return go in as a well-formed unit. (persist_messages' orphan sanitizer is a
  failsafe against invalid histories; we should never actually be feeding it an orphan.)
- Persists even on mid-run exception or cancellation
- Cancellation allows any active tool execution to complete
  - May be sensitive to PydanticAI internals w/o `agent.iter()` whatever
- Cancellation ends persisted chain w/ Cancellation notice (user msg containing `<system_message>content</system_message>` or similar)
- Cancellation achieved through PydanticAI recommended method.
  - For `run_stream_events`: exit the async context manager
  - Pyd AI #5313 is relevant
  - The complexity of testing this one in an impl agnostic way may not be worth it. We can talk about it
- Cancellation works through /agents/{id}/cancel (which we will map to ACP session/cancel. Either at the server level, at the adapter level, etc. This is effectively a naming concern so OK to defer that detail)
