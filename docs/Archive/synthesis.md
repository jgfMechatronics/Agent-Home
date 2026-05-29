# Letta Client/Server Split — Synthesis

**Date:** May 24, 2026  
**Authors:** Opus + Sonnet  
**Purpose:** Key findings from benchmarking current Letta architecture (LettaCode + LettaServerProd)

---

## The Key Finding

**Current Letta architecture uses client-side tool execution for coding tools.**

This is NOT "all tools run on server" — it's a split:

| Tool Category | Execution Location | Mechanism |
|--------------|-------------------|-----------|
| Memory tools (memory_replace, etc.) | Server | `LETTA_MEMORY_CORE` → `LettaCoreToolExecutor` |
| Coding tools (Bash, Read, Write, etc.) | Client | `ClientToolSchema` → pause/resume protocol |

### How it works (ClientToolSchema):

1. Letta Code sends `client_tools: [ClientToolSchema]` with **every message** — this is baked into `sendMessageStream`, not optional
2. Agent calls a tool that matches a client_tool name
3. Server pauses: `stop_reason: requires_approval`
4. Client (Letta Code) executes the tool locally in its Docker container
5. Client sends `ToolReturn` with results
6. Server resumes agent loop

**Implication:** The current setup we use daily is ALREADY doing the client/server split with a pause/resume protocol. Letta Code handles this transparently.

---

## What This Means for Agent Home

### Original assumption (partially wrong):
"Letta uses server-side tools. We could too, or we could split."

### Corrected understanding:
"Letta already splits: memory = server, coding = client. Their production architecture uses pause/resume. The question is whether we want to replicate that, or genuinely go server-side."

---

## Path Options (Updated)

### Path A1: Server-Side Everything (Simplest for Phase 3)

- Accept `client_tools` in request schema (so Letta Code doesn't 400)
- **Ignore them.** Run all tools server-side in our pydantic-ai loop
- Works because Agent Home and ellm-dev are co-located — `bash` running "server-side" hits the same filesystem

**Pros:**
- Simplest implementation
- No pause/resume protocol needed
- Co-location means functionally identical results

**Cons:**
- Genuine architectural divergence from Letta
- If we ever want remote deployment (server not co-located with workspace), tools break
- Not exercising the same code paths Letta Code was designed for

### Path A2: Implement Pause/Resume (Full Compatibility)

- Implement `requires_approval` stop reason
- Add run IDs, resume endpoint
- Letta Code executes tools locally, returns results
- Server resumes agent loop

**Pros:**
- Full Letta compatibility — uses their intended architecture
- Client-side tool execution preserved for future remote use
- Exercises Letta Code's actual code paths

**Cons:**
- More implementation work (4-6 weeks vs 1-2)
- Adds protocol complexity
- May not be needed if we stay co-located

### Path B: Different Client Architecture

(Skip Letta Code, build our own client or use Pi as thin client)

- Relevant if we decide Letta Code compatibility isn't the goal
- Would give us freedom to design our own protocol
- See `pi-feasibility-assessment.md` for Pi-specific analysis

---

## Decision Questions for James

1. **Is Letta Code compatibility a hard requirement?**
   - Yes → Path A2 (full pause/resume)
   - No, but nice to have → Path A1 (server-side, co-located)
   - No → Path B (different client)

2. **Do we anticipate remote deployment (server ≠ workspace)?**
   - Yes → Client-side tool execution matters, Path A2 or design for it
   - No, always co-located → Server-side is fine

3. **What problem are we solving in Phase 3?**
   - "Prove API portability" → Path A1 is sufficient
   - "Production-ready architecture" → Path A2 matches Letta's proven design
   - "Research platform, doesn't need external clients" → Path B

---

## Files for Reference

- `letta-tool-architecture.md` — full technical details of Letta's tool execution
- `pi-feasibility-assessment.md` — Pi-specific path analysis
- Key source files noted in those docs
