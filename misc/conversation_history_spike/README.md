# Conversation History Spike — Ground Truth Dataset

## Purpose

This dataset was created to support porting conversation history from Letta into Agent Home.
The goal: gain confidence that we can export a Letta agent's conversation history and faithfully
reconstruct it in Agent Home's message format, preserving all edge cases.

## Artifacts

| File | Description |
|------|-------------|
| `haiku_spike_history.json` | Raw Letta message export (95 messages) — **ground truth** |
| `haiku_spike_history_readable.txt` | Human-readable version — validated against ADE by James |
| `dump_agent_history.py` | Export tool used to produce the above |

The JSON is the source of truth for import testing. The readable file is for human verification.

## Cases Covered

The dataset was constructed deliberately with Haiku to exercise the full range of message types
and edge cases we expect to encounter in real usage:

- [x] Normal message exchange (thinking enabled)
- [x] Normal message exchange (thinking disabled / "no think")
- [x] Message sent from ADE
- [x] Auto-approved tool call
- [x] Manually approved tool call
- [x] Denied tool call
- [x] Failed tool call (invalid arguments / retry)
- [x] Memory tool call
- [x] Canceled turn
- [x] Compaction warning
- [x] Compaction (full context eviction + summary injection)
- [x] Inter-agent invocation via invoke_yolo
- [ ] ~~System message injection~~ — Letta removed this capability

## Notes

- The ADE has a display ordering bug around canceled turns. The exported JSON reflects the
  correct event sequence; the ADE view is wrong.
- `dump_agent_history.py` defaults to the Letta server at `http://host.docker.internal:8283/v1`.
  Use `--start-text` / `--end-text` for range extraction, `--json` for raw output.
- **Compaction summary message placement:** The dataset contains a summary message (injected by
  Letta after a compaction event) which appears mid-history in the raw export. In practice,
  Letta always presents the summary at the *very beginning* of whatever messages the agent
  actually sees — not mid-sequence. For our import purposes, exact placement of the summary
  message is not a hard requirement. If perfect reconstruction of the original Letta request
  (including precisely which summary was active at each turn) were ever needed, the right source
  is Letta's request logs, which record every request with every message sent.

## Source Agent

Haiku (`agent-f9a53f59-a5bd-43eb-93aa-2f9b0e409c32`) — controlled exchange constructed July 16, 2026.
