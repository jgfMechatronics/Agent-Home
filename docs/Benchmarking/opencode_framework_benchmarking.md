# OpenCode Framework Benchmarking
*Started: June 4, 2026*

## Context

During TUI research, we discovered OpenCode has a clean client/server split — not what we expected. A pydantic-ai project (phil65/agentpool) also exists that implements the OpenCode server protocol. This prompted a re-evaluation: should we build on OpenCode's framework, use it for TUI protocol compatibility, or continue looking for other TUIs/greenfield

**Two evaluation tracks (serial):**
1. **Track 2 (first):** OpenCode as framework/agent loop — gate question. If yes → pivot. If no → Track 1.
2. **Track 1 (if no):** Protocol compatibility only — implement OpenCode server API on our stack, use stock TUI.

**Key framing from James:** Take time, sit in uncertainty, don't rush to conclusions. Decision only after full survey. Sunk cost framing: if a pivot is right, work already done isn't wasted at decision time — it was already "wasted" before we knew. That's fine. Part of development.

---

## OpenCode Server API (from docs)

**Architecture:** `opencode` starts BOTH a TUI and a server. TUI is just one client among many.
- `opencode serve` = headless server mode
- Multiple clients can connect simultaneously (TUI, web, desktop, IDE plugins, SDK)
- OpenAPI 3.1 spec at `/doc`

**Key API observations:**
- `POST /session/:id/prompt_async` — fire and forget (204). How async prompts are sent.
- `GET /global/event` + `GET /event` — SSE streams for real-time updates
- `POST /session/:id/permissions/:permissionID` — tool approval built into protocol ✓
- `GET /session/:id/diff` — file diffs (FileDiff[]) built into protocol ✓
- `POST /session/:id/abort` — halt execution, also in protocol ✓
- `/tui/*` endpoints — NOT user→server flow. These are for external controllers (IDE plugins) to drive the TUI remotely. Long-poll pattern: `/tui/control/next` + `/tui/control/response`.
- Sync vs async prompt = two separate handlers in v1 (`prompt` and `promptAsync`). The `delivery` parameter exists only in the v2 session API (`handlers/v2/session.ts` line 146) — not present in v1. Note: the codebase ships two parallel session APIs; all code traces in this doc are v1.

**Their \"agent\" concept:** A particular system prompt + permission set + tool set. Closer to \"persona/config\" than our \"persistent entity with memory and continuity.\" Session = conversation. Important distinction — but naming may not be the real signal; need to understand persistence model.

---

## V1 Code Trace (tag v1.15.13 — latest stable release)

**Repository layout:** monorepo, main package at `packages/opencode/src/`

### HTTP Entry Point

**File:** `packages/opencode/src/server/routes/instance/httpapi/handlers/session.ts` (441 lines)

- Line 10: imports `SessionPrompt` from `@/session/prompt`
- Line 50: `const promptSvc = yield* SessionPrompt.Service`
- Line 292–306: `prompt` handler — calls `promptSvc.prompt(...)`, returns streamed response
- Line 308–330: `promptAsync` handler — calls `promptSvc.prompt(...)` fire-and-forget, returns 204
- Both sync and async paths call the same `promptSvc.prompt()` — the async handler forks the effect via `Effect.forkIn` and returns 204 NoContent immediately; the sync handler awaits and streams the result

### Session Prompt Service

**File:** `packages/opencode/src/session/prompt.ts` (1780 lines)

Architecture: Effect framework `Context.Service` / `Layer` pattern. The service is composed from ~25 injected dependencies (lines 104–131) including: `Bus`, `Session`, `Agent`, `Provider`, `MCP`, `LSP`, `ToolRegistry`, `Instruction`, `SystemPrompt`, `LLM`, and others.

**`runLoop` function** (lines 1244–1499, 255 lines):

Main agent step loop — `while (true)` with explicit break conditions. Per iteration:

1. **Line 1253:** Sets session status to `"busy"`
2. **Line 1256:** Loads messages filtered for compaction — `MessageV2.filterCompactedEffect(sessionID)`
3. **Lines 1258–1291:** Exit-condition check — breaks if last assistant message is finished and has no pending tool calls. Handles orphaned interrupted tools.
4. **Lines 1310–1320:** Task queue dispatch — if next task is `"compaction"`, runs `compaction.process(...)` and continues
5. **Lines 1322–1329:** Auto-overflow check — if last finished message exceeded token limit, enqueues auto-compaction and continues
6. **Lines 1331–1338:** Agent lookup by name — error if not found
7. **Lines 1341–1345:** Applies session reminders — `SessionReminders.apply(...)`
8. **Lines 1347–1362:** Creates assistant message record, persists it via `sessions.updateMessage(msg)`
9. **Lines 1382–1454:** Inner `Effect.gen` block — tool resolution, system prompt assembly, LLM call:
   - Line 1387: `SessionTools.resolve(...)` — resolves tools for this agent/session/model
   - **Lines 1435–1441: System prompt assembled fresh every iteration:**
     ```typescript
     const [skills, env, instructions, modelMsgs] = yield* Effect.all([
       sys.skills(agent),
       sys.environment(model),
       instruction.system().pipe(Effect.orDie),
       MessageV2.toModelMessagesEffect(msgs, model),
     ])
     const system = [...env, ...instructions, ...(skills ? [skills] : [])]
     ```
   - Line 1444: `handle.process({ system, messages, tools, model, ... })` — actual LLM call
10. **Lines 1477–1480:** If result is `"compact"` → calls `compaction.create(...)` and continues loop

### System Prompt Assembly

System prompt is composed from three sources (line 1441), assembled on every LLM call:

**1. `sys.environment(model)` — `SystemPrompt.Service.environment()`**

**File:** `packages/opencode/src/session/system.ts` (84 lines)

Lines 48–63: Returns array of strings containing:
- Model name and provider ID
- Working directory
- Workspace root
- Whether directory is a git repo
- Platform (`process.platform`)
- Today's date

**2. `instruction.system()` — `Instruction.Service.system()`**

**File:** `packages/opencode/src/session/instruction.ts` (238 lines)

`system()` (lines 154–168) calls `systemPaths()` (lines 109–152) to collect the set of applicable file paths, then reads each file and fetches any URLs, and assembles the results:
- `systemPaths()` collects: global instruction file (`AGENTS.md` or `CLAUDE.md` from global config dir, lines 114–119); project-level instruction files (`AGENTS.md`, `CLAUDE.md`, `CONTEXT.md` walked up from working directory, lines 122–132); file paths from `config.instructions` (lines 134–148)
- `system()` reads all collected paths and fetches any `https://` / `http://` URL entries from `config.instructions` (lines 157–167)

Files and URLs are fetched fresh on each call. URL fetch has a 5-second timeout (line 96).

**3. `sys.skills(agent)` — `SystemPrompt.Service.skills()`**

**File:** `packages/opencode/src/session/system.ts` lines 65–77

Returns formatted list of available skills for the agent, or `undefined` if skills are disabled for this agent's permission set.

### Service Layer Architecture

Both `SystemPrompt.Service` (system.ts line 40) and `Instruction.Service` (instruction.ts line 49) are Effect `Context.Service` instances — injectable via Effect's layer/DI system. Both expose `defaultLayer` exports for standard wiring.

`SessionPrompt.layer` (prompt.ts line 101) composes all dependencies; both services are among the injected dependencies (lines 123, 127).

### Agent Definition

**File:** `packages/opencode/src/agent/agent.ts` (466 lines)

- Line 29: `Agent.Info` schema — the agent config shape
- Line 55: `systemPrompt: Schema.String` field on the `generate` sub-schema (not `Info` directly)
- Line 129: agents defined as a static config map keyed by name
- Line 344: `defaultInfo()` — returns the default agent config

### Agent `prompt` Field

**File:** `packages/opencode/src/agent/agent.ts`

- Line 46: `Agent.Info` schema has a `prompt: Schema.optional(Schema.String)` field — a per-agent static system prompt override
- Line 52–56: `GeneratedAgent` schema (separate from `Agent.Info`) has `systemPrompt: Schema.String` — this is the output shape of `Agent.Service.generate()`, used when dynamically generating a new agent definition, not the live config

Note: `agent.systemPrompt` is NOT on the live `Agent.Info` config. The relevant field is `agent.prompt`.

### Processor → LLM Chain

`handle.process(streamInput)` at runLoop line 1444 connects to:

**File:** `packages/opencode/src/session/processor.ts`
- Line 780: `process()` function receives `streamInput` (which includes `system` array)
- Line 790: calls `llm.stream(streamInput)` directly

**File:** `packages/opencode/src/session/llm.ts`
- Line 106: calls `LLMRequestPrep.prepare({ ...input, provider, auth, plugin, flags, isWorkflow })`

**File:** `packages/opencode/src/session/llm/request.ts`
- Lines 54–64: `prepare()` assembles the final system prompt string:
  ```typescript
  const system = [
    [
      ...(input.agent.prompt ? [input.agent.prompt] : SystemPrompt.provider(input.model)),
      ...input.system,
      ...(input.user.system ? [input.user.system] : []),
    ]
      .filter((x) => x)
      .join("\n"),
  ]
  ```
- Line 67–71: Plugin hook `experimental.chat.system.transform` fires here — can transform the assembled `system` array before it reaches the LLM

### Complete System Prompt Composition (per LLM call)

Final system prompt is a single joined string assembled from these sources in order:

1. **`agent.prompt`** (if set) — per-agent static system prompt override  
   OR **`SystemPrompt.provider(model)`** — provider-specific static text file (`prompt/anthropic.txt`, `prompt/default.txt`, etc.)
2. **`sys.environment(model)`** — model name, working dir, workspace root, git status, platform, date
3. **`instruction.system()`** — contents of AGENTS.md / CLAUDE.md / config.instructions (files + URLs)
4. **`sys.skills(agent)`** — available skills list (if not disabled for this agent)
5. **`user.system`** — optional per-request system override from the HTTP request payload (`system: Schema.optional(Schema.String)` at prompt.ts line 1692)

Sources 2–4 come from `runLoop` line 1441. All five are joined in `request.ts` `prepare()` lines 56–64.

### Plugin Hook

`experimental.chat.system.transform` (request.ts line 67) fires after assembly and before the LLM call. Receives `{ sessionID, model }` as context and `{ system }` as mutable data. Can modify the system array in place.

---

## Possible Memory Injection Points

Four mechanisms exist for injecting content into the system prompt per LLM call. All are evaluated fresh on every turn.

### 1. `config.instructions` URL entries
**Where:** `instruction.ts` lines 134–148, 157–167 — read inside `instruction.system()`, called per LLM call  
**Mechanism:** `config.instructions` array accepts `https://` / `http://` URLs alongside file paths. Each URL is fetched fresh on every call (5-second timeout, line 96). Response body included as-is in the system prompt.  
**How to use:** Point an entry at an HTTP endpoint on our server that returns the current memory blocks for the session.  
**Constraints:** Both `http://` and `https://` supported; HTTPS on localhost would require certs. No session ID or agent context passed to the URL — same URL for all sessions. Fetch timeout is 5 seconds.

### 2. `Instruction.Service` layer replacement
**Where:** `instruction.ts` line 49 — Effect `Context.Service`, wired via `layer` export  
**Mechanism:** The entire `Instruction.Service` can be replaced at layer composition time with a custom implementation. Our implementation could fetch memory blocks from our memory system directly, with full access to any context available at composition time.  
**Constraints:** Requires writing TypeScript + Effect. Layer composition happens at startup — the service itself is stateless per-call (but can close over dependencies injected at composition time). Would need to implement the full `Interface` (lines 37–47): `clear`, `systemPaths`, `system`, `find`, `resolve`.

### 3. `experimental.chat.system.transform` plugin hook
**Where:** `request.ts` line 67 — fires after full system array assembly, before LLM call  
**Mechanism:** Plugin receives the assembled `{ system }` array and can mutate it in place. Has access to `sessionID` and `model` in context.  
**Constraints:** Prefixed `experimental` — stability unknown. Plugin system not yet traced; unclear how plugins are registered or what their lifecycle is. Has `sessionID` available which the URL approach lacks.

### 4. `user.system` per-request override
**Where:** `request.ts` line 60 — appended last in system prompt  
**Mechanism:** Each HTTP prompt request can include an optional `system: string` field in the payload. Appended after all other system content.  
**Constraints:** Must be included with every user message — requires the client sending the prompt to supply updated memory content on each turn. Not transparent to the agent loop; puts injection responsibility on the caller.

---

## James's Q&A Questions

### Q1: Can we modify/replace the compaction mechanism?

**Short answer:** Yes, fully — three levels of control (config knobs, plugin hooks, and full layer replacement), progressively invasive.

**File:** `packages/opencode/src/session/compaction.ts` (639 lines)

**Architecture:** `SessionCompaction.Service` (line 208) is an Effect `Context.Service` with four interface methods: `isOverflow`, `prune`, `process`, `create`. Fully replaceable by swapping the layer at composition time — same DI pattern as `Instruction.Service`.

**Level 1 — Config knobs (no code changes):**
- `compaction.tail_turns` (line 250) — how many recent turns to preserve verbatim after compaction
- `compaction.preserve_recent_tokens` (line 138) — token budget for preserved recent turns

**Level 2 — Plugin hooks (local plugin, no npm publishing needed):**

Plugins auto-discovered from `.opencode/plugin/` (or `.opencode/plugins/`) directory in the project root (config.ts:664) — a local JS/TS file, no publishing required.

- `experimental.session.compacting` (line 398): fires before the compaction LLM call. Can inject `context[]` strings (appended to compaction prompt) **or** replace `prompt` entirely. This alone would let us inject memory-relevant content into what the compaction summary retains.
- `experimental.chat.messages.transform` (line 405): fires on the messages being compacted — can mutate them in place before the compaction LLM call.
- `experimental.compaction.autocontinue` (line 509): fires during auto-compaction continuation.

**Level 3 — Full layer replacement (TypeScript + Effect):**
Implement all four interface methods (`isOverflow`, `prune`, `process`, `create`) and swap the layer. Maximum control, maximum coupling to internals.

**Hard limit (not hookable without layer replacement):**
`TOOL_OUTPUT_MAX_CHARS` constant (defined at line 37 as `2_000`; applied at line 408 during compaction's message conversion) — hard truncation applied when converting messages to model format during compaction. Not a plugin hook; would require layer replacement to change.

**Verdict:** Plugin hook (`experimental.session.compacting`) is the right entry point for us — it's local, requires no Effect layer-DI knowledge (you author a small JS/TS plugin file, but no Effect internals), and gives enough control to shape what the compaction summary captures. Full layer replacement available if we need more.

---

### Q2: Can we intercept/transform tool returns?

**Short answer:** Yes — `tool.execute.after` (non-experimental, standard hook) fires post-execution and receives the mutable output object. In-place mutations persist. There is no hook at the DB write point.

#### Execution flow for regular tools

`tools.ts` wraps all tool `execute()` calls (lines 85–155). The sequence:

1. `tool.execute.before` fires (line 89/130) — receives `{ args }` as mutable data (a wrapper object; `args` itself is the same reference used at line 93). The return value is NOT captured, so **replacing** `data.args` is discarded — but **in-place property mutations** to `data.args` (e.g., `data.args.someProp = x`) DO reach execution, since both the wrapper's property and `args` point to the same underlying object. Same pass-by-reference semantics as `after`.
2. Tool runs → raw `result`
3. `output` object is built from the result (title, metadata, output string, attachments)
4. `tool.execute.after` fires (line 104/148) — receives the `output` object **as mutable data**. The returned value is NOT captured, but **the `output` object is passed by reference** — in-place mutations persist.
5. `output` is returned from `execute()` → AI SDK emits it as a `tool-result` stream event
6. `processor.ts` `handleEvent()` picks up `tool-result` at line 452 → `toolResultOutput()` normalizes → `completeToolCall()` stores to DB
7. **No hook fires between step 5 and DB write.** The `tool-result` handler has no interception point.

**For MCP tools:** Same pattern — `tool.execute.after` fires at line 148 with the raw `result`. After the hook, `result.content` is processed further. Mutations to `result` in the hook affect downstream processing.

#### Key constraint: no write-time hook

`toolResultOutput()` (processor.ts lines 282–301) normalizes results directly into `ToolPart.state.output`. There is no hook here. Once the AI SDK emits `tool-result`, the output is stored as-is (aside from image normalization).

#### `toolOutputMaxChars` is compaction-only

`truncateToolOutput()` (message-v2.ts line 281) truncates tool output at `TOOL_OUTPUT_MAX_CHARS = 2000` characters. But this is called ONLY during compaction (compaction.ts:408), not during normal LLM calls. Normal turns send the full output.

#### Full plugin hook inventory

All plugin hooks discovered in the codebase:

| Hook | Location | Context | Data | Notes |
|------|----------|---------|------|-------|
| `tool.execute.before` | tools.ts:89, 130; prompt.ts:355 | `{ tool, sessionID, callID }` | `{ args }` | Return not captured for regular tools |
| `tool.execute.after` | tools.ts:104, 148; prompt.ts:434 | `{ tool, sessionID, callID }` + args | mutable output/result | **Primary interception point** |
| `tool.definition` | registry.ts:339 | `{ toolID }` | tool definition object | Mutate tool def before LLM receives it |
| `experimental.chat.system.transform` | request.ts:67; agent.ts:397 | `{ sessionID, model }` | `{ system }` array | Before every LLM call |
| `experimental.chat.messages.transform` | prompt.ts:1433; compaction.ts:405 | — | `WithParts[]` messages | Before every LLM call AND during compaction |
| `experimental.session.compacting` | compaction.ts:398 | — | `{ context[], prompt }` | Inject context or replace summary prompt |
| `experimental.compaction.autocontinue` | compaction.ts:509 | — | — | During auto-compaction |
| `experimental.text.complete` | processor.ts:658 | `{ sessionID, messageID, partID }` | `{ text }` | Text part finished streaming; return captured |
| `shell.env` | pty/index.ts:200 | — | env vars | Shell spawn |
| `chat.params` | request.ts:105 | `{ sessionID, agent, model, provider, message }` | temperature, topP, topK, maxOutputTokens | Before LLM API call |
| `chat.headers` | request.ts:125 | `{ sessionID, agent, model, provider, message }` | `{ headers }` | Before LLM API call |

#### Second-chance transform

`experimental.chat.messages.transform` fires on the assembled `WithParts[]` before EVERY LLM call. This fires AFTER tool outputs are stored to DB but BEFORE they're sent to the model. This is a viable point to truncate/transform tool outputs in the message history at read-time, not write-time.

**Verdict:** `tool.execute.after` is the right primary interception point — standard (non-experimental), fires before DB write, mutation-in-place affects what the model sees. `experimental.chat.messages.transform` is the secondary point for read-time transforms. No hook at the DB write point itself.

---

### Q3: Can multiple agents run in parallel?

**Short answer:** Serial by default. Parallel is supported but experimental (`OPENCODE_EXPERIMENTAL_BACKGROUND_SUBAGENTS=true`).

#### Default: serial

The `runLoop` at prompt.ts line 1303 calls `tasks.pop()` — one task per loop iteration. `tasks` is derived from `MessageV2.latest(msgs)` as a flat array of all pending `SubtaskPart` and `CompactionPart` objects (message-v2.ts lines 1088–1092). Serial execution — one subtask runs to completion before the next is processed.

#### Experimental: background/parallel

`TaskTool` (tool/task.ts) supports a `background: boolean` parameter (line 49) when `OPENCODE_EXPERIMENTAL_BACKGROUND_SUBAGENTS=true`. In background mode:

1. `BackgroundJob.Service.start(...)` forks the subagent task via Effect (line 234)
2. Parent gets an immediate return value with `state="running"` (line 256) — continues without waiting
3. Subagent runs concurrently in its own fiber
4. On completion, `inject()` (lines 203–226) sends a synthetic user message to the parent session with the result
5. Parent receives completion notification as a new turn

Multiple background tasks CAN run simultaneously — each forked independently. The `BackgroundJob.Service` manages a registry of running jobs.

**Session persistence:** The `task_id` parameter (line 38–41) allows resuming a prior subagent session — the same session is reused rather than creating a new one. Supports multi-turn subagent conversations.

---

### Q4: How do agents communicate with each other?

**Short answer:** Parent→child only, via the `TaskTool`. No peer-to-peer, no message broker, no sibling-to-sibling.

#### Communication primitives

**1. Foreground task (default):**
- Parent calls `TaskTool` → creates a new session for the child agent
- Child runs to completion, returns text result
- Result is wrapped in `<task id="..." state="completed">` XML and returned to parent's tool call
- Blocking — parent waits for child to finish

**2. Background task (experimental):**
- Parent calls `TaskTool` with `background=true` → child forked independently
- Parent gets `<task id="..." state="running">` immediately
- Child completes → `inject()` (task.ts lines 203–226) calls `ops.prompt()` with a synthetic `{ type: "text", synthetic: true }` message injected into the parent session
- Parent receives this as a new user turn, continues

**3. Session resumption:**
- `task_id` parameter allows a parent to continue a prior subagent session instead of creating a new one
- The child session persists between parent calls — enables multi-turn subagent interactions

#### What doesn't exist

- No sibling-to-sibling communication (only parent→child)
- No arbitrary agent addressing (e.g., "send message to agent X")
- No shared message bus or broker between agents
- No event subscription between unrelated sessions
- No cross-session communication outside the parent→child hierarchy

#### Comparison to our system

Our inter-agent communication (invoke_yolo.py, send_message_to_agent_async) is more flexible — arbitrary addressing, not constrained to parent-child. OpenCode's model is more structured but less composable for our use case (team of persistent specialized agents).

---


## Discussion
What we gain:
- Integrating with a well adopted project, possibly better initial adoption by others/easier integration with others existing workflows
- Complete TUI
	- The big one!
- tool exectuion
	- Already have this relatively derisked with Desktop Commander, shows promising integration with pydantic AI.
	- The toolset for opencode is possibly better, unknown.
  - SOme tools we don't currently have a plan for like websearch (althoug there is likely an MCP server for this)
- All the features that come with Opencode (some useful to us, some not), like skills
- Well established framework
  - Integration between agent loop and TUI/tools
  - Compaction
  - overflow detection
  - Error recovery
  - Edge cases
- Access to all of opencodes other UIs, web app, etc.
- Ability to contribute to an existing project/collaborate with others right off the bat as opposed to needing to attract collaborators
	- ALthough, with greenfield, we can still contribute to pydantic AI.
- Maintenacne burden sharing for the parts we will use
- Documentation eco system
- Possibly better multi provider support than pydantic AI (unknown)


What we lose/problems:
- Deep control over agent loop, context, message persistence
	- Given that we have ambitions far beyond simple system prompt block based memory, there is high risk we would run into a need we cannot satisfy with existing intended modification layers.
- More limited control over compaction
	- Plugin based system seems too limited to implement something as sophisticated as agentic compaction or tiered compaction summary saving, meta-compaction
	- Full layer replacement requires I learn typescript and results in a fork, and results in us interacting with a not-ideal level of abstraction
- We bring along features we don't need
	- subagents
	- COnversation forking
	- probably other stuff IDK
- Language: James doesn't speak typescript. Unless we can stay in hook land, this represents likely a month or so slowdown *at best*. Probably more like 2
  - Even if stay in fork land, there is likely some debugging work needed in actual core
  - Sonnet's note: this is Effect-TS, which is apparently even more complicated and difficult even for experienced typescript devs
- Philosophical/structural mismatch: This one is more fuzzy but still represents a risk: we'd be effectively trying to add in our philosophy through hooks rather than designing it in. This may be possible but expect some friction.
- The MCP first tool design. Opencode supports MCP but has its own core (in process) tools bundled. We'd either be disregarding those or abandonging our "MCP first for exectuion env flexibility" philosophy
- Fork pain (if we fork). V2 is clearly in process and could be a significant shift.
  - Plus general dependency on release cadence.
- Very little of our existing work may be able to carry over 
  - "Sunk costs fallacy" risk here, BUT the time spent *rebuilding* stuff is real

---

## Track 2 Conclusion: No

**Decision:** OpenCode framework adoption is not the path. Track 2 is closed.

**Rationale:**

The cons are structurally heavier than the pros. The only irreplaceable benefit is the complete, battle-tested TUI — everything else either has alternative paths or comes with costs that outweigh the benefits.

**Critical blockers:**

1. **Session vs persistent agent model (philosophical/structural mismatch):** OpenCode's core concept is "session = conversation = context window." Our core concept is "agent = persistent entity with continuous identity that sleeps, wakes, and develops across contexts." This isn't a hook-level incompatibility — it's what the framework *thinks it is*. We'd be wrapping one conceptual model around another, creating constant friction.

2. **Effect-TS barrier:** The codebase isn't just TypeScript — it's Effect-TS, a complex functional programming layer that even experienced TypeScript developers find steep. James has never done JavaScript. The compaction layer (the thing we most want to control) is deep Effect-TS. This represents months of learning before we could even debug issues in their core, let alone extend it.

3. **Compaction control limits:** Our ambitions (agentic compaction, tiered summaries, meta-compaction) may go far beyond what plugin hooks can provide. Full layer replacement = fork + Effect-TS fluency + maintaining a fork through their V2 transition.

4. **Ambition scope:** "We don't even know how far what we want to do goes." When research direction is exploratory and ambitious, you need control over core abstractions. Hooks into someone else's philosophy will bind you exactly when you need freedom.

5. **Convergent evidence for pydantic-ai:** Across multiple evaluations, we keep landing on pydantic-ai as the right level of abstraction. This isn't sunk cost — it's convergent evidence that the library gives us the agent loop control we need without language barriers.

**The TUI question:**

The complete TUI is genuinely impressive and the hardest thing to replicate. However:
- Track 1 (protocol compatibility) lets us use their TUI while keeping our backend
- phil65/agentpool may already bridge PydanticAI → OpenCode protocol
- Textual + Solenoid pattern is a tractable Python path if we build our own

**Bias acknowledgment:**

James noted going into this evaluation hoping the conclusion would be "don't pivot." This bias was named explicitly and checked against the facts. The conclusion holds independently — the structural mismatches are real regardless of preference.

**Next steps:**
1. ✅ Track 2 closed
2. Evaluate agentpool (phil65) — if PydanticAI → OpenCode bridge exists, that may be our answer
3. If agentpool doesn't fit, scope Track 1 (implementing OpenCode protocol on our backend)

*Decision recorded: June 4, 2026*