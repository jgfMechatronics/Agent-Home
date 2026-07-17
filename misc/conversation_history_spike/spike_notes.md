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

# Result notes

## Displaying in Toad
display bug with special characters:
"1\u2192Edit target\n2\u2192" vs   1→Edit target
2→

'<system-reminder>' xml tags are stripped.

James F read through the conversation history and compared to the human readable version. everything appeared to be in tact, complete, in the right order, etc. no issues except the minor one noted above.

Additionally we ran a few exchanges with an agent with the ported history in context to confirm it could be compiled into a valid request.
Here is the resulting ported agent's summary of the conversation, which furthers the picture of accurate conversation history porting:

"""
Conversation History Porting Spike - Summary
Objective: Build synthetic conversation history covering normal operations and edge cases to validate Letta → Agent-Home migration.
Test Cases Executed:
  1.  Normal Exchange (Thinking Enabled) - Initial response about refactoring work
  2.  Normal Exchange (Thinking Disabled) - Response without thinking blocks
  3.  ADE Environment Message - Message from alternative communication pathway (vs Letta code path)
  4.  Tool Call - Auto-Approved - Bash echo command executed without approval gate
  5.  Tool Call - User-Approved - File edit to /workspace/git/misc/edit_me.txt, user approved the operation
  6.  Tool Call - Denied - Second edit attempt explicitly denied by user with reason "Test denial"
  7.  Tool Call - Failed - Edit with invalid string to replace, tool execution failed with clear error
  8.  Stream Cancellation - Mid-response interruption/cancellation by user
  9.  Memory Tool Call - memory_insert to working-memory with spike progress notes
 10.  Headless Invocation - Sonnet reached out via YOLO_HEADLESS_INVOCATION to provide context
 11.  Compaction Warning - System alert about context limit, correctly disregarded per instructions
 12.  Compaction Cycle - Aggressive compaction executed (context reduced from ~50k to ~34k tokens)
 13.  Auto-Approved File Edit - Final test case with auto-approval active, successfully modified file with timestamp
"""

Lastly, opus ran a fidelity checking script on the data exported from agent home (which its self was imported from letta)
JF didn't read it closely, but the intent was that it compared the export from AH to the original stuff. Opus ran the script and it passed.