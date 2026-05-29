# Agentic CLI Harness Research: Synthesis & Recommendations

**Date:** 2026-05-23  
**Researchers:** Opus (synthesis), Sonnet (primary research)

---

## Executive Summary

**Letta Code is the recommended path.** It's the only major CLI harness designed client→server from the start, has a clean Backend interface for integration, and consolidates LettaBot's messaging capabilities (though WhatsApp is still beta). The codebase quality is significantly better than the Letta server.

The landscape has many strong alternatives (OpenCode at 163k stars, Pi at 52k), but all run the agent loop internally and would require significant surgery to swap in our server.

---

## The Candidates

### Letta Code (letta-ai/letta-code) — **RECOMMENDED**

| Metric | Value |
|--------|-------|
| Stars | 2,437 |
| License | Apache-2.0 |
| Language | TypeScript (98%) |
| Files | 48 (lean) |
| Releases | 172 |
| Contributors | 30 |

**Why it fits:**
- Already client→server architecture via `LETTA_BASE_URL`
- Clean `Backend` TypeScript interface with ~20 methods
- LettaBot functionality rolling in (Telegram/Slack/Discord now, WhatsApp coming)
- Active development, issues get closed

**Critical finding:** The Backend interface is intentionally abstracted (refactored May 3, 2026). This is the right seam for integration — we can implement the interface without forking.

**Concerns:**
- Issue #1224: system-reminder injection can cause context overflow loops (588k tokens observed)
- Mitigation: Our server controls message persistence; if we treat system reminders as ephemeral, this likely doesn't bite us

### OpenCode (anomalyco/opencode) — Market leader, wrong architecture

| Metric | Value |
|--------|-------|
| Stars | 163,923 |
| License | MIT |
| Language | TypeScript |
| Releases | 809 |

**Why we're not recommending:**
- Runs agent loop internally
- Would need significant surgery to swap in our server as backend
- `opencode serve` exists but serves the CLI as an API, not connects to external server

**Good for inspiration:** Multiple built-in agents (build, plan, general), permission system, GitHub Actions integration

### Pi (earendil-works/pi) — Beautiful philosophy, wrong architecture

| Metric | Value |
|--------|-------|
| Stars | 52,000 |
| License | MIT |
| Language | TypeScript |

**Why James might love it:**
- "Minimal terminal coding harness — adapt Pi to your workflows, not the other way around"
- 4 modes: interactive, print/JSON, RPC, SDK
- Deep extensibility via TypeScript extensions, skills, prompts
- Philosophy of building what you need, not shipping everything

**Why we're not recommending:**
- Also runs agent loop internally
- RPC mode could potentially be adapted but that's significant effort
- No client→server architecture

**Plot twist:** Letta Code already has Pi integrated as a local executor! (`LETTA_LOCAL_BACKEND_EXECUTOR=pi`). So we get Pi's UX philosophy if we want it, through Letta Code.

### Interesting Outlier: Hermes-Agent-RS

| Metric | Value |
|--------|-------|
| Stars | 27 |
| License | BSL 1.1 |
| Language | Rust (89%) |

- Single ~19MB binary, zero dependencies
- 17 platform adapters: Telegram, Discord, Slack, **WhatsApp**, Signal, Matrix, and 11 more
- 30+ tool backends, 10 LLM providers
- Runs on Raspberry Pi or $3 VPS

**Why it's interesting:** Shows what's possible with clean architecture — full multi-platform support in a single binary. Not recommending adoption, but worth watching for design inspiration.

---

## Integration Paths for Letta Code

### Path A: Implement Letta REST API subset (APIBackend route)
- Point `LETTA_BASE_URL` at our E-LLM Agent Server
- Implement ~20 REST endpoints matching the Backend interface
- **Effort:** High
- **Upside:** No fork needed, Letta Code updates come free
- **Downside:** Substantial REST surface to implement

### Path B: Fork and implement Backend interface directly
- Add `ELLMBackend` implementing the TypeScript `Backend` interface
- Call our server directly
- **Effort:** Medium
- **Upside:** No REST overhead, can expose capabilities honestly
- **Downside:** Fork maintenance

### Path C: Piggyback on LocalBackend's Pi executor
- `LETTA_LOCAL_BACKEND_EXPERIMENTAL=1`
- `LETTA_LOCAL_BACKEND_EXECUTOR=pi`
- If our server speaks Pi protocol → near-zero effort
- **Effort:** Low IF compatible
- **Risk:** LocalBackend is "experimental"
- **Investigation needed:** What exactly is the Pi protocol?

---

## Recommendation

**Start with Path A** (implement REST API subset). Reasons:

1. **No fork maintenance** — we stay current with Letta Code improvements
2. **Clean separation** — our server is our server, Letta Code is Letta Code
3. **The Backend interface tells us exactly what to implement** — it's well-typed TypeScript
4. **WhatsApp/Signal will come** — LettaBot consolidation is real and ongoing

**Path C worth investigating** as potential quick win — if Pi protocol is simple, could be faster path to dogfooding.

---

## Files in This Research

| File | Contents |
|------|----------|
| `lettabot-consolidation.md` | LettaBot → Letta Code migration status |
| `letta-code-quality.md` | Codebase quality assessment |
| `letta-code-backend-seam.md` | Backend interface analysis |
| `SYNTHESIS.md` | This document |

---

## Next Steps

1. **James decision:** Confirm Letta Code direction
2. **Investigate Path C:** Check Pi protocol — could accelerate timeline
3. **Begin Path A implementation:** Stub out REST endpoints matching Backend interface
4. **Flag issue #1224:** Design our message persistence to avoid the system-reminder loop bug
