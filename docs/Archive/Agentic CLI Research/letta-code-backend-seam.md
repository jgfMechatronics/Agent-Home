# Letta Code: Backend Seam Analysis

**Source:** `src/backend/backend.ts` (main branch, verified May 23 2026)  
**Key commit:** d596277 "refactor(api): add backend seam for lifecycle calls" (May 3 2026)

## Summary

Letta Code does NOT hardcode the Letta REST API. It has a clean `Backend` interface with swappable implementations. This is the most important architectural finding for our integration decision.

---

## The Backend Interface

`Backend` is a TypeScript interface with ~20 well-typed methods:

**Agents:**
- `retrieveAgent(agentId)` → `agents.retrieve`
- `listAgents(body?)` → `agents.list`
- `createAgent(body)` → `agents.create`
- `updateAgent(agentId, body)` → `agents.update`
- `deleteAgent(agentId)` → `agents.delete`

**Conversations:**
- `retrieveConversation(id)` → `conversations.retrieve`
- `listConversations(body?)` → `conversations.list`
- `createConversation(body)` → `conversations.create`
- `updateConversation(id, body)` → `conversations.update`
- `recompileConversation(id, body?)` → `conversations.recompile`
- `cancelConversation(id)` → `conversations.cancel`
- `forkConversation(id, options?)` → `conversations.fork`
- `listConversationMessages(id, body?)` → `conversations.messages.list`
- `compactConversationMessages(id, body?)` → `conversations.messages.compact`
- `createConversationMessageStream(id, body)` → `conversations.messages.create` (streaming)
- `streamConversationMessages(id, body)` → `conversations.messages.stream`

**Agents (messages):**
- `listAgentMessages(agentId, body?)` → `agents.messages.list`

**Messages:**
- `retrieveMessage(id)` → `messages.retrieve`

**Runs:**
- `retrieveRun(id)` → `runs.retrieve`
- `streamRunMessages(runId, body)` → `runs.messages.stream`

**Models:**
- `listModels(options?)` → `models.list`

**BackendCapabilities flags:** `remoteMemfs`, `serverSideToolManagement`, `serverSecrets`, `agentFileImportExport`, `promptRecompile`, `byokProviderRefresh`, `localModelCatalog`, `localMemfs`

---

## Two Production Implementations

### 1. `APIBackend` (default)
- Routes all calls through `@letta-ai/letta-client` SDK pointed at `LETTA_BASE_URL`
- Full Letta REST API surface required
- This is what runs when you point Letta Code at `api.letta.com` or a self-hosted Letta server

### 2. `LocalBackend` (experimental)
- Enabled via `LETTA_LOCAL_BACKEND_EXPERIMENTAL=1`
- Executor selected via `LETTA_LOCAL_BACKEND_EXECUTOR` = `"pi"` or `"deterministic"`
- Storage: local filesystem at configurable `storageDir`
- **Pi is already integrated** as a local executor option (`PiStreamAdapter`, `ProviderTurnExecutor`)
- This bypasses the HTTP API entirely

---

## Dev Backends (for internal testing)

All live at `src/backend/dev/`:
- `FakeHeadlessBackend` — in-memory fake, accepts optional executor
- `PiStreamAdapter` — wraps Pi (the inference provider)
- `ProviderTurnExecutor` — generic provider wrapper  
- `DeterministicToolCallExecutor` — scripted tool calls for testing

Selected via `--dev-backend` flag:
- `fake-headless`
- `fake-headless-tool-call`
- `fake-headless-provider`
- `fake-headless-pi` ← Pi as backend

---

## Implications for Our Integration

Three paths, ordered by effort:

### Path A: Implement Letta REST API subset (APIBackend route)
- Point `LETTA_BASE_URL` at our E-LLM Agent Server
- We implement the ~20 REST endpoints the Backend interface wraps
- No fork of Letta Code needed
- Effort: High — it's a well-defined interface but substantial REST surface
- Upside: Clean separation, Letta Code updates come free

### Path B: Implement the Backend interface directly (LocalBackend route)
- Fork Letta Code, add an `ELLMBackend` implementation
- Implement the `Backend` interface — pure TypeScript, no REST layer
- Call our server directly (or in-process)
- Effort: Medium — fork required, but the interface is the exact right abstraction
- Upside: No REST protocol overhead, can expose capabilities honestly via `BackendCapabilities`

### Path C: Piggyback on LocalBackend's Pi executor
- Set `LETTA_LOCAL_BACKEND_EXPERIMENTAL=1`, `LETTA_LOCAL_BACKEND_EXECUTOR=pi`
- LocalBackend already knows how to call Pi
- If our server exposes a Pi-compatible interface, zero fork needed
- Effort: Low IF Pi protocol compatibility exists — needs investigation
- Risk: LocalBackend is explicitly "experimental"

---

## Recommendation

**Path A is safest for staying current.** The Backend interface is the right seam — Letta Code already abstracted it. Implementing ~20 REST endpoints against our server is more work upfront but means no fork maintenance.

**Path B is architecturally cleanest** if we're willing to maintain a fork. The Backend interface is exactly the boundary we'd want anyway.

**Path C needs more research** — what is the Pi protocol? If it's close to what we emit, this could be near-zero effort.

---

## Auth note

Credential validation calls `client.agents.list({ limit: 1 })`. For local/self-hosted use, this just needs a real response — not Letta Cloud OAuth. BYOK via `LETTA_BASE_URL` + `LETTA_API_KEY` works for APIBackend.
