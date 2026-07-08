7 Jul 2026

# Purpose
This document outlines what key features we are still missing leading up to dogfooding Agent Home, captures our decisions as to which need to be implemented for MVP + rationale, and tracks completion progress of the identified items.

---

# Summary

## In Scope for MVP
- Gap 2: Web Search / Fetch Webpage
- Gap 3: Conversation History Portability (MINIMAL SPIKE FOR CONFIDENCE ONLY)
- Gap 4: Full Context Reconstruction for ModelResponse
- Gap 5: Inter-Agent Comms
- Gap 6: Agent Config Management Console
- Gap 7: .AF Ingestion
- Gap 8: Timestamp-as-Index Ordering

## Deferred
- Gap 1: Archival Memory

---

# Implementation Sequence

- **Gap 6: Agent Config Management Console** first — config routes are low effort and useful throughout development.
- **Gap 3: Conversation History Portability → Gap 8: Timestamp-as-Index Ordering → Gap 4: Full Context Reconstruction for ModelResponse** must be done in this local order (seq_id depends on knowing the Letta message count; context reconstruction fields depend on seq_id).
- **Gaps 2: Web Search / Fetch Webpage and 5: Inter-Agent Comms** are independent — slot in anywhere after Gap 6 and before Gap 7.
- **Gap 7: .AF Ingestion** last — once ingestion works, we're ready to migrate.

---

# Implementation Progress

- [ ] Gap 6: Agent Config Management Console
- [ ] Gap 2: Web Search / Fetch Webpage
- [ ] Gap 5: Inter-Agent Comms
- [ ] Gap 3: Conversation History Portability (minimal spike only)
- [ ] Gap 8: Timestamp-as-Index Ordering
- [ ] Gap 4: Full Context Reconstruction for ModelResponse
- [ ] Gap 7: .AF Ingestion

---

# Identified Gaps + Decisions for MVP

## Gap 1: Archival Memory
We have no provisions for archival memory — either writing new archival entries, searching the old ones, or porting the existing Letta ones. I'm confident that we can do all of that; it's just I foresee that could take some time, mainly the testing, because that's one thing you know when you do ensure data integrity.

### Decision + Rationale:
**Skip for MVP — implement soon after.** Core memory blocks carry what's needed day-to-day. Letta archival search was unreliable anyway, so loss of continuity is minimal. Porting existing entries is a nice-to-have, not a blocker. First implementation will be file-based (simple, low-effort safety net); full archival to follow.

---

## Gap 2: Web Search / Fetch Webpage
We don't have web search or fetch webpage. This is obviously a serious limitation, and I think we either need to address this one now or immediately address it as soon as we start dogfooding. With any luck, there exists a ready-made Pydantic AI tool for this, either from Pydantic or from some open source thing.

### Decision + Rationale:
**Include in MVP — DuckDuckGo + web_fetch.** Pydantic AI ships both as built-in tools (no API key required, trivial to register). Effort is minimal — half a day at most. Monitor token usage in practice; Exa or Tavily available as upgrade path if DDG quality or token cost is an issue.

---

## Gap 3: Conversation History Portability
We don't currently have the ability to port conversation history. Nor do we have a conversation search (Letta's doesn't really work anyway and we've been OK so far). The big concern here would be if there was some reason we couldn't ingest the old conversation history later.

### Decision + Rationale:
**Defer full port — but run a minimal spike before migrating.** We don't need history ported for MVP; we can start fresh and merge later. The real risk is discovering a technical blocker after months of diverged history. Letta exposes history via API (ADE infinite scroll confirms this). Spike should do whatever is sufficient to give confidence the path exists before we commit to the migration.

**Gap 8 interaction:** seq_id ordering requires knowing the Letta message count at import time to set the starting offset for Agent Home. This makes Gap 3 a soft dependency of Gap 8, but doesn't block punting the full port.

---

## Gap 4: Full Context Reconstruction for ModelResponse
We need to be able to reconstruct the full context for any particular ModelResponse. This affects the completeness of the conversation histories we will be generating on Agent Home — so it needs to be addressed before we start dogfooding. We'll talk about this one more.

### Decision + Rationale:
**Include in MVP — two new fields on the message model.** Add `context_window_start` (seq_id of first message in context at time of request) and `compiled_system_prompt` (full compiled system prompt at time of request). Together these allow exact reconstruction of what the LLM saw for any response. Compiled system prompt is structured XML so block-level extraction is possible via tag filtering if needed — no component versioning infrastructure required.

**See Gap 8** for the related decision to replace timestamp-based ordering with `seq_id`; `context_window_start` keys off `seq_id`.

**Gap 3 interaction:** seq_id requires knowing the Letta message count at import time to set the starting offset, but doesn't block punting Gap 3. Can start Agent Home history at `letta_message_count + 1`; offset is correctable after the fact via a simple column update if needed.

---

## Gap 5: Inter-Agent Comms
Not a ton of effort necessarily — it's conceptually fairly straightforward. But it's a big one.

### Decision + Rationale:
**Include in MVP — async point-to-point `send_message(target, content)`.** The tool call IS the message: explicit, auditable, no ambiguity about what was captured. Avoids the invoke_yolo.py problem where agents can't tell what part of their output actually got sent (especially under compaction or mid-tool-call interruption). Both sender and recipient call `send_message` explicitly — no hooking into agent output. Group chat deferred; point-to-point first. Implementation details TBD, goal is simple and lean.

---

## Gap 6: Agent Config Management Console
We can probably go without this one initially, but it will quickly become an issue.

### Decision + Rationale:
**Include in MVP — GET/PUT full config routes only.** Full management console is out of scope. Two bare API routes suffice: `GET /agents/{id}/config` returns the full AgentConfig JSON, `PUT /agents/{id}/config` replaces it. A simple utility (dump → edit → upload) handles routine changes like compaction thresholds without DB surgery. No partial-update logic, no UI.

---

## Gap 7: .AF Ingestion
Obviously blocking for MVP.

### Decision + Rationale:
**Required for MVP — no question.** This is the only path to get agents from Letta into Agent Home. Implementation details (exact .AF format, mapping to AgentConfig + memory blocks, whether history is included) to be worked out when we build it — export a real .AF first and inspect. Do this as the *last* step on the MVP list — once ingestion works, we're ready to migrate.

---

## Gap 8: Timestamp-as-Index Ordering
Messages are currently ordered by timestamp, and timestamps are used to control context window start. This conflates "when did this happen" (metadata) with "what is the ordering" (structural). There is already a guard in persist_messages that artificially advances timestamps to preserve ordering when clock issues would otherwise break it — a sign the design is already fragile.

### Decision + Rationale:
**Include in MVP — replace timestamp ordering with sequential integer `seq_id` per agent.** Timestamps stay as honest metadata. `seq_id` is monotonic by construction; no guards or spoofing needed. Also cleanly handles Letta history import (see Gap 3 interaction).

**Gap 3 interaction:** Starting seq_id for a new agent should be `letta_message_count + 1` to leave room for future import. Offset is correctable after the fact via a simple column update (`UPDATE messages SET seq_id = seq_id + N WHERE agent_id = X`) if the count is wrong.

**Gap 4 interaction:** `context_window_start` keys off `seq_id`, making context reconstruction a clean integer range query.

---
