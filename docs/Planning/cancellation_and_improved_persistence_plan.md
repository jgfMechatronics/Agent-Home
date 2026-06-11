# Req:

- Doesn't duplicate during persistence across persistence pts
- Persists every complete Model Message as you go (how to test?)
- Persists even on mid-run exception or cancellation
- Cancellation allows any active tool execution to complete
  - May be sensitive to PydanticAI internals w/o `agent.iter()` whatever
- Cancellation ends persisted chain w/ Cancellation notice (user msg containing `<system_message>content</system_message>` or similar)
- Cancellation achieved through PydanticAI recommended method.
  - For `run_stream_events`: exit the async context manager
  - Pyd AI #5313 is relevant
  - The complexity of testing this one in an impl agnostic way may not be worth it. We can talk about it
- Cancellation works through /agents/{id}/cancel (which we will map to ACP session/cancel. Either at the server level, at the adapter level, etc. This is effectively a naming concern so OK to defer that detail)
