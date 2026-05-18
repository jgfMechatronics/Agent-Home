# CLI Notes

## Purpose
Throwaway CLI for live testing Agent Home server. Not production code — just enough to:
- Verify endpoints work
- Test SSE streaming
- Manual exploration before writing automated integration tests

## Design Decisions

### HTTP Client
Using `httpx` for async HTTP + SSE handling. Added to pyproject.toml dependencies.

### Architecture
Simple async input loop, not `cmd` module. Simpler for this throwaway use case.

### Commands
- `create <name>` — POST /agents/, uses default AgentConfig
- `use <agent_id>` — set active agent for subsequent commands
- `chat <message>` — POST /agents/{id}/messages (SSE)
- `history` — GET /agents/{id}/messages?full=true
- `info` — GET /agents/{id}
- `memory` — GET /agents/{id}/core_memory (bonus, read-only)
- `help` — show commands

### State
- `active_agent_id` held in memory, not persisted
- Server URL configurable via env var or default to localhost:8000

### SSE Handling
Parse SSE events manually — format is `event: TYPE\ndata: JSON\n\n`
Typewriter display: print text chunks as they arrive, newline at end.

**Event types (from pydantic-ai, verified via test_routes.py):**
- `PartDeltaEvent` — text chunk, field: `data["delta"]["content_delta"]`
- `FunctionToolCallEvent` — tool invocation, field: `data["part"]["tool_name"]`
- `FunctionToolResultEvent` — tool result
- `AgentRunResultEvent` — stream complete

### Headless Mode
`--headless` flag: read commands from stdin, structured output for agent parsing.
No prompts, no colors, JSON output where useful.

## Open Questions
- Message format is "crusty" per James — may need iteration once we see real output
- How to display tool calls in streaming? (basic support added: shows tool name + checkmark)

## Changelog
- 2026-05-19: Initial creation (Opus)
