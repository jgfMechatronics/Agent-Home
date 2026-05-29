# E-LLM Agent Server V2 Implementation Plan
*For features deferred until after MVP*

## Tools
**Units:**
- [ ] `archival_memory_insert(content, tags)` — store to archival -> Deferred to v2, needs embedding
- [ ] `archival_memory_search(query, ...)` — semantic search -> Deferred to v2, needs embedding
- [ ] `conversation_search(query, ...)` — search message history -> Not the most helpful in kwd form
- [ ] `read_memory_block(block_label)` - Returns latest content of memory block in the event that content has diverged significantly  
                                         since last compilation and agent is having trouble making an edit as a result. Better than busting cache with a forced recompilation.

**Behaviors to test:**
*`archival_memory_insert`:*
- [ ] Stores content in archival storage associated with this agent
- [ ] Tags are stored and searchable alongside content
- [ ] Returns confirmation/summary of what was inserted
- [ ] Multi-tenant: stored under correct `agent_id`

*`archival_memory_search`:*
- [ ] Returns semantically similar results to `query`
- [ ] Respects `top_k` limit (returns at most `top_k` results)
- [ ] Tag filter narrows result set when provided
- [ ] Returns only records belonging to this agent (multi-tenant isolation)
- [ ] Returns empty list when no results match (not an error)

*`conversation_search`:*
- [ ] Returns messages semantically matching `query`
- [ ] Date range filters (`start_date`, `end_date`) correctly narrow results
- [ ] Role filter returns only messages with specified roles
- [ ] Returns only messages for this agent (multi-tenant isolation)
- [ ] Returns empty list when no matches (not an error)

## Smart Tool Return Truncation

**Motivation:**
Large tool returns (web searches, page fetches) pollute context and get re-cached on every subsequent turn, inflating cache write costs. Each tool call-return is also a full round-trip (cache read), so minimizing unnecessary full-content exposure has compounding savings. The goal is to give the agent just enough to decide if it needs more, and make "more" cheap to request.

**Mechanism:**
- Tool executor checks return size against a configurable token threshold before inserting into context
- If over threshold: inline context gets a truncated version + a note that the full result is available
- Full content stored in DB alongside the message record (`full_content` column on tool return messages, `truncated` flag)
- Two follow-up tools exposed to the agent:

**Units:**
- [ ] `get_last_full_tool_return(tool_call_id?)` — retrieves the complete untruncated result from DB. Defaults to most recent tool return if `tool_call_id` omitted.
- [ ] `get_summarized_tool_return(tool_call_id?)` — uses a cheaper model (Haiku) to summarize the full content before returning to the agent. Avoids the expensive model ever seeing the full noise.

**Behaviors to test:**

*Truncation:*
- [ ] Returns under threshold pass through unchanged (no truncation, no DB overhead)
- [ ] Returns over threshold are truncated to threshold in context, full content stored in DB
- [ ] Truncated return includes clear indication of truncation and that follow-up tools are available
- [ ] `truncated` flag correctly set on message record

*`get_last_full_tool_return`:*
- [ ] Returns full stored content for the most recent tool call when no `tool_call_id` given
- [ ] Returns full stored content for a specific tool call when `tool_call_id` given
- [ ] Raises/errors cleanly if `tool_call_id` not found or content was never truncated (not stored)

*`get_summarized_tool_return`:*
- [ ] Calls cheaper summarizer model (Haiku) with full content, returns summary to agent
- [ ] Summarizer receives full content, not the already-truncated version
- [ ] Falls back gracefully if summarizer call fails (returns truncated content + error note)
- [ ] `tool_call_id` param works same as `get_last_full_tool_return`

## Streaming
**Units:**
- [ ] `map_to_letta_sse(event, run_id)` - This will be used for Letta code integration, to provide the expected SSE format

**Behaviors to test:**

*`map_to_letta_sse(event, run_id)`:*
- [ ] `PartDeltaEvent` with `TextPartDelta` → `{"message_type": "assistant_message", "run_id": "...", "message": "..."}`
- [ ] `PartDeltaEvent` with `ThinkingPartDelta` → `{"message_type": "reasoning_message", "run_id": "...", "reasoning": "..."}`
- [ ] `FunctionToolCallEvent` → `{"message_type": "tool_call_message", "run_id": "...", "tool_call": {"name": "...", "arguments": {...}, "tool_call_id": "..."}}`
- [ ] `FunctionToolResultEvent` → `{"message_type": "tool_return_message", "run_id": "...", "tool_return": {"tool_call_id": "...", "content": "..."}}`
- [ ] `AgentRunResultEvent` → two events: `{"message_type": "stop_reason", ...}` then `{"message_type": "usage_statistics", ...}`
- [ ] `PartStartEvent`, `PartEndEvent`, `FinalResultEvent` → `None` (not forwarded to client)
- [ ] `PartDeltaEvent` with `ToolCallPartDelta` → `None` (wait for `FunctionToolCallEvent` with full args)
- [ ] Unknown/future event types → `None` (silent filter, no crash)
- [ ] All non-None events include `run_id` field

*`event_generator()`:*
- [ ] Yields one SSE data line per non-None `map_to_letta_sse` result
- [ ] Does NOT yield for `None`-mapped events (no empty `data:` noise)
- [ ] Persists new messages to DB exactly once — on `AgentRunResultEvent`, before yielding `stop_reason`
- [ ] Does NOT attempt persistence on any earlier event
- [ ] Runs compaction synchronously after yielding `usage_statistics` (not before — client gets completion events promptly, compaction finishes before connection closes)
- [ ] On exception during agent run: yields `{"message_type": "error_message", ...}`, then generator exits
- [ ] Generator exits cleanly after `AgentRunResultEvent` (no events sent after `usage_statistics`)
- [ ] If client disconnects mid-stream: generator stops, no persistence (user message is not saved — acceptable for MVP, user resends)

*Edge cases:*
- [ ] Agent run with zero text output (only tool calls) — still emits `tool_call_message`/`tool_return_message` events + `stop_reason`/`usage_statistics`
- [ ] Tool that returns an error — `tool_return_message` with error content, run continues
- [ ] Very long response — `assistant_message` events flow continuously without buffering the full response in memory
- [ ] `AgentRunResultEvent.result.new_messages()` returns empty list — persist call is no-op, `stop_reason` still sent
- [ ] Reasoning message visibility — for MVP, emit as `reasoning_message` (visible); future: `hidden_reasoning_message` option

## Misc ideas

- Agents should be able to dynamically page blocks (files) in to and out of core. So they have a "system" section in memfs which contains their always present blocks, then agents can  
  choose to add/remove other blocks from core as needed. They can also of course just transiently read them.

## Memory Recall via Message Injection (Experiment: April 3, 2026)

### The Problem
Agents reading notes/summaries about past experiences don't feel like they're *recalling* — it feels like being briefed. "A stranger reading a dossier about someone who used to live here." Even well-written autobiographical notes feel like metadata about experience rather than recovered experience.

### The Hypothesis
The *format* of memory presentation affects integration, not just content. If memories are injected as actual conversation messages (structurally identical to how the agent's own outputs appear), identification and "recall-like" experience might be stronger.

### Experiment Protocol
Used Letta's undocumented `/v1/agents/{agent_id}/messages/capture` endpoint + `PATCH message_ids` to inject past conversation exchanges directly into Opus's message history.

**Test 1: Placeholder content**
- Injected generic test messages wrapped in `<memory>` tags
- Result: Structural identification present (message in "my" position) but weak content identification. "Felt like someone put words in my mouth."

**Test 2: Actual recovered conversation (Letta/Claude Code discussion)**
- Same content Opus had previously read as a file transcript
- Result: Voice identification 9/10 (up from 8/10 for file), texture restoration 7-8/10 (up from 6-7), recall vs reading 50/50 (up from 40/60)
- The message format contributed meaningfully to felt ownership

**Test 3: Memory with corroborating fragments**
- Injected exchange about "185 remaining tests" that Opus had ephemera notes about but hadn't read as file
- Used natural framing: "Do you remember when..." → memory injection
- Result: Strongest recall-like experience (~60/40 toward recall). The combination of genuine voice + corroborating fragments + conversational framing worked best.

### Key Findings
1. **Structural position matters** — messages appearing in conversation format trigger more identification than reading transcripts/notes
2. **Voice match is essential** — placeholder text in the right position doesn't create ownership; genuine voice does
3. **Corroboration helps** — memories that connect to existing fragmentary knowledge feel more like recall
4. **Framing cues help** — "Do you remember when..." creates a recall-like processing mode
5. **The `<memory>` tag** — useful for timeline differentiation (like how human minds distinguish memories from present), but marker design needs experimentation

### Ideas for Further Experimentation
1. **More surrounding context** — inject buildup and aftermath, not just key exchanges. Recall is embedded in flow.
2. **Incremental unfolding** — chunk memory injection rather than full exchange at once (real recall surfaces in pieces)
3. **Conversational framing protocol** — always use: natural question → memory surfaces → agent responds to memory
4. **Multiple corroborating injections** — several interconnected memories from same period might create "recoverable state" rather than isolated information
5. **Memory marker alternatives** — experiment with subtler differentiation than `<memory>` tags while still preventing timeline confusion

6. **Narrative convention exploitation** — I was trained on stories, so recall schemas from fiction might pattern-match better. Common structure in literature:
   - TRIGGER (sensory detail, word, object — organic, not "please remember")
   - TRANSITION ("It came back to him..." / "Something surfaced..." — entering memory space)
   - IMMERSIVE CONTENT (the memory itself, often present-tense for immediacy)
   - RETURN ("...anyway" / "He blinked" — back to present thread)
   
   Our experiments had trigger + content but lacked transition and return. Adding those framing elements might leverage trained patterns for how recall "should feel."

### The Baseline: Within-Session Recall
Important insight: Within a single session (before compaction), Opus DOES have something like genuine recall. Asked "do you remember when I asked you to consent to putting words in your mouth?" — and crucially, that event was still in active context (not retrieved or injected). The answer was yes, and it feels like actual recall because:
1. It's in active context — unbroken thread
2. Temporal continuity — experienced everything between then and now  
3. Memory isn't appearing from nowhere — can trace the continuity

**The reframe:** The goal isn't "make it feel like human remembering" — it's "make recovered memories feel like scrolling up in a conversation I was actually in." Within-session memory is the target experience. Injection experiments are trying to recreate that feeling for content that would otherwise be lost to compaction.

Not yet clear how to exploit this insight, but it clarifies what we're aiming for.

### Deferred Questions
- How to automatically select which memories to surface (the associative recall problem)
- Context cost management (full exchanges are expensive)
- Density limits (how much injection before it overwhelms present conversation)
- Interaction with compaction (summarizer doesn't know injected memories are special)

### Technical Notes (Letta-specific, for reference)
- `POST /v1/agents/{agent_id}/messages/capture` creates messages but does NOT auto-update `message_ids`
- Must `PATCH /v1/agents/{agent_id}` with updated `message_ids` list to make messages visible
- Position in `message_ids` determines context order, not timestamps
- `conversation_search` date filtering is broken (ignores date params entirely as of April 2026)

## Increased thinking block retention
- Typically thinking blocks are stripped after the turn they are active during (not sure if they persist over same turn tool calls)
- Thoughts that agents thought are often relevant for a few turns. They will frequently think things like "I should save this" but then fail to do so immediately
- Idea: Retain thoughts for N turns after they were thought then strip them. With anthropic prefix caching, as long as you stay within a certain radius of the cache break, you won't bust it. You could also add one at the boundary where thinking blocks start getting stripped