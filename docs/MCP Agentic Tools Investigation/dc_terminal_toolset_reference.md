Document from Claude.ai Opus 4.8
# Desktop Commander — Terminal Toolset Reference

Extracted from `wonderwhy-er/DesktopCommanderMCP` source (`src/server.ts` descriptions +
`src/tools/schemas.ts` param shapes). This is the verbatim agent-facing text plus the real
Zod schemas — the closest thing DC has to a tool spec.

---

## Schemas (param shapes)

```ts
StartProcessArgsSchema = {
  command: string,
  timeout_ms: number,          // REQUIRED — also the backgrounding lever (see below)
  shell?: string,
  verbose_timing?: boolean,
}

ReadProcessOutputArgsSchema = {
  pid: number,
  timeout_ms?: number,
  offset?: number,             // 0 = new since last read; positive = absolute line; negative = tail
  length?: number,             // max lines (default: config.fileReadLineLimit, 1000)
  verbose_timing?: boolean,
}

InteractWithProcessArgsSchema = {
  pid: number,
  input: string,
  timeout_ms?: number,         // default 8000
  wait_for_prompt?: boolean,   // default true
  verbose_timing?: boolean,
}

ForceTerminateArgsSchema = { pid: number }
ListSessionsArgsSchema   = {}                 // no args
ListProcessesArgsSchema  = {}                 // no args
KillProcessArgsSchema    = { pid: number }

// Filesystem read (the Claude Code `Read` analog), for comparison:
ReadFileArgsSchema = {
  path: string,
  isUrl?: boolean,             // default false
  offset?: number,             // default 0  (0-based start line; negative = tail)
  length?: number,             // default 1000 lines
  sheet?: string,              // xlsx
  range?: string,
  options?: object,
}
```

---

## Agent-facing descriptions (verbatim)

### start_process
Start a new terminal process with intelligent state detection.
Returns `Process started with PID N` plus initial output, then one of three states when
`timeout_ms` elapses:
- finished in time → completed, output inline (one-shot blocking)
- still running → "Process is running. Use read_process_output to get more output."
- waiting on stdin → ready for interact_with_process

Pitched in-tool as the PRIMARY tool for local file analysis (it insists you NOT use the
native analysis/REPL tool, which "cannot access local files and WILL FAIL"). Common patterns
it advertises:
- `start_process("python3 -i")` → Python REPL (recommended for data work)
- `start_process("node -i")` / `start_process("node:local")` → JS REPL / stateless node
- `start_process("wc -l file.csv")`, `start_process("head -10 file.csv")` → quick one-shots
Smart detection: REPL prompts (>>>, >, $), waiting-for-input, completion-vs-timeout, early exit.

### read_process_output
Read output from a running process with file-like pagination.
- offset=0 → new output since last read (default); positive → absolute line; negative → tail
- length → max lines (default 1000 via fileReadLineLimit)
- Examples: `offset:0,length:100` (first 100 new); `offset:500,length:50` (lines 500–549);
  `offset:-20` (last 20)
- For offset=0, waits up to timeout_ms for new output to arrive.
- Output protection: status like `[Reading 100 lines from line 0 (total: 5000, 4900 remaining)]`.
- Reports detection state: waiting for input / finished / timeout-reached-may-still-be-running.

### interact_with_process
Send input to a running process and automatically receive the response.
Heavily pitched as THE primary tool for local file analysis via REPLs. Params: pid, input,
timeout_ms (default 8000), wait_for_prompt (default true), verbose_timing (default false).
Auto-waits for REPL prompt, detects errors/completion, strips prompts from output.
Supported REPLs: python3 -i, node -i, R, julia, bash/zsh, mysql/postgres.
verbose_timing exposes exit reason + full output timeline for latency debugging.

### force_terminate
Force terminate a running terminal session by pid.

### list_sessions
List all active terminal sessions. Shows PID, Blocked (waiting-for-input), Runtime.
"Blocked: true" usually means a REPL is waiting on input; long runtime + blocked = possibly stuck.

### list_processes
List all running processes. Returns PID, command name, CPU%, memory.

### kill_process
Terminate a running process by PID. Forceful — use with caution.

---

## Backgrounding model (the non-obvious part)

There is **no `background: true` flag**. Backgrounding is emergent from `timeout_ms`:

```
start_process("npm run dev", timeout_ms=3000)
   → finishes <3s?  returns output, done            (blocking one-shot)
   → still running? returns PID + "running" status; process persists as a session
read_process_output(pid)          # poll; offset/length paging
interact_with_process(pid, input) # if it wants stdin
list_sessions                     # what's alive
force_terminate(pid) / kill_process(pid)
```

Lever: big timeout = effectively blocking; small timeout = detach-and-hand-me-the-PID.

---

## Desktop Commander  ↔  Claude Code tool mapping

| Need            | Claude Code (trained schema)                              | Desktop Commander                                         | Shape match? |
|-----------------|-----------------------------------------------------------|-----------------------------------------------------------|--------------|
| Range read      | `Read(path, offset, limit)` — token-paged, 1-based lines  | `read_file(path, offset, length)` — line-paged, 0-based   | CLOSE (rename limit→length; index base differs) |
| Edit            | `Edit(old_string, new_string, replace_all)` — EXACT match, fail-on-miss, read-before-edit guard | `edit_block(SEARCH/REPLACE block)` — **fuzzy fallback** on miss | DEVIATES (format + semantics; fuzzy is the risk) |
| Write           | `Write(path, content)` — read-before-overwrite guard      | `write_file(path, content)`                               | CLOSE |
| One-shot exec   | `Bash(command, timeout, run_in_background)`               | `start_process(command, timeout_ms)`                      | DEVIATES (no bg flag; bg via timeout) |
| Background exec | `Bash(run_in_background:true)` → `BashOutput` / `KillShell` | `start_process(short timeout)` → `read_process_output` / `force_terminate` | DEVIATES (different mental model) |
| Content search  | `Grep(pattern, output_mode, glob, type, multiline)` — ripgrep | `start_search` / `code_search` (ripgrep)                  | PARTIAL |
| File find       | `Glob(pattern)`                                           | `search_files` / `list_dir`                               | PARTIAL |
| Interactive REPL| (no native equiv; Bash + manual)                          | `interact_with_process(pid, input)`                       | DC EXTRA |

**Reading of the risk:** the filesystem read is nearly on-distribution. The two surfaces most
likely to confuse a CC-trained model are (1) the **exec model** — it knows `Bash` with a
`run_in_background` flag, not "start_process + tune the timeout," and (2) **edit_block**'s
block format + fuzzy matching vs the exact-match `Edit` it was drilled on. DC's ALL-CAPS,
"NEVER use the analysis tool" descriptions are themselves evidence of fighting the model's
priors in-context.

**Normalization strategy (smart-server move):** keep DC as the execution/IO backend, but
re-expose a CC-shaped surface to the model at your server layer:
- present a `bash`-shaped tool (`command`, `timeout`, `run_in_background`) → translate
  `run_in_background:true` to a short `timeout_ms` + return the PID; map a `bash_output` tool
  to `read_process_output`.
- present an exact-match `edit`/`str_replace` tool (`old_string`/`new_string`/`replace_all`)
  → either disable edit_block's fuzzy path or implement exact-match yourself and use DC only
  for the file IO; enforce read-before-edit in your harness.
- alias `read_file(length)` → `read(offset, limit)` to match `Read`.
- replace DC's verbose coaching descriptions with terse CC-style ones — a CC-trained model
  already knows these tools, so the coaching is closer to off-distribution noise than help.

This decouples backend choice from the surface the model sees: pick DC for its solid process
management, but the model never has to leave its trained ergonomics.
