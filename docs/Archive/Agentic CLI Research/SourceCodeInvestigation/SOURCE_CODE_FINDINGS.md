# Source Code Investigation Findings
**Date:** May 23, 2026  
**Investigator:** Sonnet  
**Repos reviewed:** letta-ai/letta-code (thorough), earendil-works/pi (stroll), sst/opencode (stroll)

---

## Summary Verdict

**Recommendation confirmed: Letta Code, Path A (implement Letta REST API subset).**

The source code review significantly upgraded the confidence in this recommendation. Web-based assessment undercounted Letta Code's codebase by ~10x. Pi is confirmed as an agent runtime, not a backend server. OpenCode is enterprise infrastructure far beyond our use case.

---

## Letta Code — Thorough Review

### Codebase Size (Corrected)
Web assessment was wrong ("48 files"). Actual counts from local clone:
- **504 source files, 340 test files** (co-located throughout, `.test.ts` pattern)
- Test distribution: cli(79), tools(60), agent(47), channels(40), websocket(24), permissions(15), backend(11), reminders(7), integration-tests(7)

### CI / Quality Infrastructure

`bun run check` runs 7 checks in sequence:
1. Circular dependencies (madge)
2. Layer boundaries (custom script with WHY comments)
3. Exported function style
4. Filename casing
5. Test mock isolation
6. Biome linting
7. TypeScript (tsc --noEmit)

Commit hooks via Husky + lint-staged enforce Biome on every commit. This is a professional CI pipeline — not a "tests exist" checkbox.

### Architectural Enforcement (Critical Finding)

`scripts/check-layer-boundaries.js` enforces dependency direction at CI level:

| Layer | Cannot import from | Reason |
|-------|-------------------|--------|
| `tools/` | `cli/` | Tools run headless and in agent contexts |
| `backend/` | `cli/`, `websocket/` | Low-level abstraction |
| `providers/` | `agent/`, `cli/` | Pure LLM adapters |
| `websocket/listener/` | `backend/api/client`, `backend/api/conversations` | Must go through `getBackend()` |
| `cli/app/` | `backend/api/conversations` | Must go through `getBackend()` |
| `telemetry/` | `cli/`, `agent/`, `websocket/`, `tools/` | Leaf observer only |

**This means `getBackend()` is not just a type — it's an enforced architectural seam.** Any code that consumes conversations or agents must go through it. We plug in here.

### Backend Interface — Full Method Inventory

22 methods total in `src/backend/backend.ts`:

**Agents:**
- `createAgent(body, options?)` 
- `retrieveAgent(agentId, options?)`
- `listAgents(body?)`
- `deleteAgent(agentId, options?)`
- `updateAgent(agentId, body, options?)`

**Conversations:**
- `createConversation(body, options?)`
- `retrieveConversation(conversationId, options?)`
- `listConversations(body?)`
- `updateConversation(conversationId, body, options?)`
- `recompileConversation(conversationId, body?, options?)`
- `forkConversation(conversationId, options?)`
- `cancelConversation(conversationId)`

**Messages:**
- `createConversationMessageStream(conversationId, body, options?)` ← **core send+stream**
- `streamConversationMessages(conversationId, body, options?)`
- `listConversationMessages(conversationId, body?, options?)`
- `compactConversationMessages(conversationId, body?, options?)`
- `listAgentMessages(agentId, body?, options?)`
- `retrieveMessage(messageId, options?)`

**Runs:**
- `retrieveRun(runId)`
- `streamRunMessages(runId, body, options?)`

**Other:**
- `listModels(options?)`

### BackendCapabilities Flags

```typescript
interface BackendCapabilities {
  remoteMemfs: boolean;
  serverSideToolManagement: boolean;
  serverSecrets: boolean;
  agentFileImportExport: boolean;
  promptRecompile: boolean;
  byokProviderRefresh: boolean;
  localModelCatalog: boolean;
  localMemfs: boolean;
}
```

**Key insight:** Client reads these at startup and gracefully disables unsupported features. We declare what we support; the CLI adapts. No 501s needed for unimplemented endpoints.

### Minimal Dogfood Implementation (~8 methods)

Must implement:
1. `createAgent` — create a new agent
2. `retrieveAgent` — get agent state
3. `listAgents` — list agents for selection UI
4. `createConversation` — create a conversation thread
5. `retrieveConversation` — get conversation (client needs `in_context_message_ids` for pending approval detection)
6. `createConversationMessageStream` — **THE** core method: send message, receive SSE stream
7. `listConversationMessages` — message history for backfill
8. `retrieveMessage` — get specific message by ID (last in-context message check)

Stub with capabilities = false:
- `forkConversation` → `agentFileImportExport: false`
- `compactConversationMessages` → `promptRecompile: false`
- `streamRunMessages`, `retrieveRun` — runs API
- `deleteAgent`, `updateAgent`, `updateConversation` — management ops
- `listModels` → `localModelCatalog: false`
- `byokProviderRefresh: false`, `remoteMemfs: false`, `serverSecrets: false`, `serverSideToolManagement: false`, `localMemfs: false`

### Test Quality Assessment

**Verdict: Good.** Tests are behavior-focused, not implementation-coupled.

Evidence from `src/agent/approval-result-normalization.test.ts`:
- Each test covers a distinct behavioral case
- Named constants (not magic strings) — `INTERRUPTED_BY_USER`
- Edge cases: null returns, legacy format migration, structured vs string returns
- No repetition — tests don't test the same path with different constants

Source code quality in `src/agent/check-approval.ts`:
- WHY comments throughout ("The source of truth for pending approvals is `in_context_message_ids`")
- Constants defined at top with explanatory comments
- Exported functions for testability (`prepareMessageHistory`, `extractApprovals`)
- Main function is long (~250 lines) — only real DRY concern observed

**Not a slop factory. Substantially better quality than the Letta server we left.**

### Backend Directory Test Coverage

11 test files covering:
- `api-backend.test.ts` — API backend behavior
- `fake-headless-backend.test.ts` — dev backend
- `local-backend.test.ts` — local backend
- `local-compaction-parity.test.ts` — compaction consistency
- `local-provider-errors.test.ts`, `local-provider-timeout.test.ts` — error paths
- `local-system-prompt-compilation.test.ts` — prompt assembly
- `message-search.test.ts` — search behavior
- `pi-model-factory.test.ts`, `pi-stream-adapter.test.ts` — Pi integration
- `provider-turn-executor.test.ts` — turn execution

---

## Pi — Stroll

**381 source files, 234 test files** across 4 packages:
- `pi-coding-agent` (CLI): 125 test files
- `pi-ai` (unified LLM API): 71 test files
- `pi-tui` (terminal UI): 22 test files
- `pi-agent-core` (runtime): 16 test files

Uses vitest. 221 releases, active (last push May 22, 2026). 52,907 stars, MIT.

**Architecture conclusion:** Pi is an **agent runtime**, not a server backend. It runs agent loops locally (pi-agent-core), provides a unified LLM API (pi-ai), and a terminal UI (pi-tui). Letta Code uses Pi as its `LocalBackend` executor — the Pi integration path would mean running our agent inside Pi's runtime, not plugging our server in as an API backend.

**The "agent-loop-internal concern" from prior research was correct.** Pi is not a path to get our server used by Letta Code — it's an alternative architecture altogether.

Path A (implement Letta REST API subset) remains correct for our use case.

---

## OpenCode — Stroll

**163,000 stars.** Much larger than anticipated: enterprise, sdk, plugin, desktop, extensions, slack packages. The main `opencode` package alone has 425 source files.

Not just a CLI — full engineering platform: ACP protocol, LSP integration, worktrees, snapshots, MCP, BYOK providers, control plane, plugin system.

**Verdict: Wildly overbuilt for our purposes.** Integrating with OpenCode would mean integrating with enterprise infrastructure. Our agents would become a plugin in someone else's platform rather than using Letta Code as a client for our server.

Good engineering reference for patterns. Not a candidate for adoption.

---

## Implementation Notes for Path A

The Letta REST API endpoints we'd implement map cleanly to our existing server structure:

| Letta Code calls | Our server endpoint |
|-----------------|---------------------|
| `client.agents.create()` | `POST /agents` |
| `client.agents.retrieve()` | `GET /agents/{agent_id}` |
| `client.agents.list()` | `GET /agents` |
| `client.conversations.create()` | `POST /conversations` |
| `client.conversations.retrieve()` | `GET /conversations/{conversation_id}` |
| `client.conversations.messages.create()` | `POST /conversations/{id}/messages` (SSE) |
| `client.conversations.messages.list()` | `GET /conversations/{id}/messages` |
| `client.messages.retrieve()` | `GET /messages/{message_id}` |

The `@letta-ai/letta-client` TypeScript SDK (which Letta Code uses) auto-generates from OpenAPI. We can reference the SDK's type signatures directly to ensure we match the expected response shapes.

### Response Shape Note

`listConversationMessages` and `listAgentMessages` return paginated responses with a `.getPaginatedItems()` method. We need to return the right paginated wrapper, not a raw array. Check `@letta-ai/letta-client` source for the pagination type shape.

---

*Investigation complete. Prior synthesis doc (SYNTHESIS.md) + this doc together form the full recommendation basis for Phase 3 planning.*
