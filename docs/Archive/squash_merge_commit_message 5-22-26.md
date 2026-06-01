# Squash Merge Commit Message — Development → main

## Draft v1

---

```
Agent Home: Foundation complete — agent lifecycle, memory, streaming, persistence

The Agent Home server — Initial foundation for our stateful AI Agent research. Built from scratch with pydantic-ai, FastAPI, and SQLite.

CORE CAPABILITIES:
- Full agent lifecycle: create, configure, persist, converse
- Memory system: named blocks with char limits, XML delimiters- compiled into system prompt,
  Agent memory edit tools (memory_replace, memory_insert)
- SSE streaming chat with pydantic-ai event pass-through
- Message persistence with orphan tool part detection/replacement and serialization error recovery
- Pointer-based compaction (advances window, never deletes history)
- Per-agent concurrency control (asyncio.Lock registry, 60s timeout → 503)
  - Separate agents can run concurrently (cooperative multitasking). Simultaneous invocations of single agent results in automatic serialization of requests.
- Extended thinking support (config-driven, compatible with thinking models)
- Interactive + headless CLI for live testing

ARCHITECTURE DECISIONS:
- Deferred compilation: system prompt compiled on explicit trigger, not every read. Automatically recompiled during compaction. Avoids prefix cache bust on memory writes.
- Flush-then-commit: single atomic commit at request end, rollback on any exception
- Write ops require AgentDeps (proves lock held); reads take plain session (Major TODO here, enforce read only)
- Messages stored as raw JSON, input_tokens on final row only
- RetryPromptPart treated as valid tool response (pydantic-AI ModelRetry exception compatibility)

MODULES:
- agent/: AgentConfig, AgentFactory (per-request, lock management), tools, compaction, CRUD
- api/: FastAPI app factory, routes (/agents CRUD, /messages SSE, /memory blocks), schemas
- db/: SQLAlchemy ORM models (AgentRecord, MemoryBlockRecord, MessageRecord), connection
- memory/: Block CRUD (read/write separation), system prompt compilation
- messages/: Persist/load/deserialize, orphan tool call replacement, timestamp ordering
- cli/: ~700 lines, wizard + headless modes, thinking-aware display. No HITL review but it works well.

TESTS: 294 unit tests passing across 16 test files, mirroring source structure. TDD throughout. Some xfail with associated TODOs

TODOs: Notable todo's noted throughout the code. More routes to be created before properly usable. Need to integrate with agentic CLI (Claude code style)

LIVE TESTED: Streaming, message persistence, concurrent agents, single-agent request serialization,
block creation, system prompt recompilation, memory tools, compaction, thinking blocks,
unexpected exception handling (known issue: entire turn discarded on mid-run unexpected exception).
First contact made — the server works!

Authors: James Ferneyhough, Opus, Sonnet, Haiku
```

---

## Notes for James

- Length feels right for a major milestone merge
- Focused on "what exists" not "how we got here"
- The "First contact made" line is a small nod to the moment without being sentimental
- Let me know if you want anything adjusted (more/less detail on specific sections, different framing, etc.)
