Created: Jun 1 2026

# Research/Plan
Goal: Connect and test MCP file system tools to Agent Home Core.

Current target servers:
https://github.com/modelcontextprotocol/servers/tree/main/src/filesystem
Anthropic's official FS MCP

https://github.com/patrickomatik/mcp-bash
Most starred MCP bash server I've found on github, but still only 32 so will probably need some vetting

**Problem**:
Anthropic's Anthropic's MCP server only supports stdio. We need HTTP, SSE, or streamable HTTP to play nicely with our intended architecture.
Options:
- Find another that natively supports HTTP
- Wrap it

Option for wrapping STDIO MCP servers to stream SSE:
https://github.com/sparfenyuk/mcp-proxy#about
Decent stars and is python, good option.

**FastMCP (the pythonic way to work with MCP servers apparently)**

## FastMCP Findings (Jun 1 2026)

**Key discovery:** FastMCP's `create_proxy()` handles everything — npm package pulling, subprocess spawning, stdio management, and HTTP exposure. No manual installation or subprocess code needed.

### The Pattern

```python
from fastmcp.server import create_proxy

proxy = create_proxy({
    "mcpServers": {
        "default": {
            "command": "npx",
            "args": ["-y", "@modelcontextprotocol/server-filesystem", "/workspace"]
        }
    }
}, name="filesystem")

proxy.run(transport="streamable-http", port=8080)
```

Then pydantic-ai connects with:
```python
from pydantic_ai.mcp import MCPToolset

toolset = MCPToolset("http://localhost:8080/mcp")
# Pass to Agent via toolsets=[toolset]
```

Note: `MCPToolset` was added in pydantic-ai 1.97.0 (May 15, 2026). Earlier versions used `MCPServerStreamableHTTP` which is now deprecated.

### What FastMCP Handles
- `npx -y` auto-installs the npm package if not cached
- Spawns and manages the subprocess lifecycle
- Bridges stdio ↔ HTTP transport
- Session isolation (each request gets fresh backend session)
- Concurrent request handling
- Component caching (tools, resources, prompts) with configurable TTL

### Key Docs
- Composing Servers (mounting external): https://gofastmcp.com/servers/composition
- Proxy Provider (session isolation, caching): https://gofastmcp.com/servers/providers/proxy
- create_proxy() API: https://gofastmcp.com/python-sdk/fastmcp-server-server

### Open Questions
- Grep gap: Anthropic's FS server has no content search. Options: bash fallback, supplemental tool, or custom extension.
- Error handling: How does FastMCP surface subprocess crashes / npm failures?


## Spike Plan:
- Use Anthropic official FS MCP Server via NPM
- Wrap it with FastMCP, 
    - Have FastMCP auto pull the NPM and host the subprocess
    - Use StreamableHTTP mode (NOT SSE)
    - FastMCP should be running in a seperate process from Agent Core. For the first test, both AH core and the MCP server can run in ellm-dev. We will later want to test cross container with MCP server running in seperate container.
- Connect with Pydantic AI using toolset = MCPToolset()
- Pass the toolset to the agent constructor in the agent factory, pass it in parallel to the existing tools (pydantic AI internally combines them)
- Start a new folder in Agent-Home Base "mcp_tools"

# Implementation Notes

## Spike Implementation (Jun 1 2026)

### What We Built

1. **`mcp_tools/fs_proxy.py`** — FastMCP proxy wrapper for Anthropic's filesystem MCP server
   - Uses `create_proxy()` with Claude Desktop config format
   - Streamable HTTP transport on port 8080
   - Exposes `/workspace` by default (configurable via `--workspace` arg)
   - Run with: `uv run python -m mcp_tools.fs_proxy`

2. **Factory integration** — Modified `agent/factory.py`
   - Added `MCPToolset("http://localhost:8080/mcp")` to toolsets
   - Pydantic-ai merges `tools=` (memory callables) + `toolsets=` (MCP) automatically
   - TODO: Make URL configurable via deps.config or env

3. **Test script** — `mcp_tools/test_connection.py`
   - Verifies connection and lists discovered tools
   - Run with proxy active: `uv run python -m mcp_tools.test_connection`

### Tools Discovered (14 total)

| Tool | Purpose |
|------|---------|
| `read_file` | Read file as text (deprecated, use read_text_file) |
| `read_text_file` | Read file as text |
| `read_media_file` | Read image/audio as base64 |
| `read_multiple_files` | Batch read |
| `write_file` | Create/overwrite file |
| `edit_file` | Line-based edits (replace exact matches) |
| `create_directory` | Create dir (mkdir -p behavior) |
| `list_directory` | List contents |
| `list_directory_with_sizes` | List with file sizes |
| `directory_tree` | Recursive JSON tree |
| `move_file` | Move/rename |
| `search_files` | **Glob pattern on filenames** (NOT content search) |
| `get_file_info` | File metadata |
| `list_allowed_directories` | Security: show exposed paths |

### Confirmed Gap: No Content Search

`search_files` is glob-style filename matching, NOT grep. For coding agents, content search is load-bearing. Options:
- Add bash MCP server and use `grep`/`rg` via shell
- Build custom MCP extension for content search
- Live without for now (spike scope)

### Key Decisions

1. **Streamable HTTP** (not SSE) — Modern standard, pydantic-ai auto-detects
2. **Separate process** — MCP proxy runs independently from Agent Home server
3. **Hardcoded URL for spike** — `http://localhost:8080/mcp` (TODO: make configurable)
4. **No tests yet** — Per spike instructions, rapid prototype phase

### Dependencies Added

- `fastmcp` — Added to pyproject.toml
- `pydantic-ai` — Upgraded from **1.72.0** to **1.104.0** for `MCPToolset` support (available in 1.97.0+)

### Next Steps

- [ ] Cross-container test (MCP in separate container)
- [ ] Make MCP URL configurable (env or agent config)
- [ ] Evaluate bash MCP for grep/shell commands
- [ ] Consider MCPServerStdio for in-process option (simpler for single-machine deploy)

**NOTE: Testing described in this file generally based on agent self report and thus is not completely rigorous**
However, we are mostly testing behavior of existing modules. This is primarily exploratory to see if the behaviors match our needs and work within our system.

### Live testing completed and passed:
- Read dir contents
- Create file
- Read file contents
    - **Major gap:** can get first or last N lines of a file but cannot specify an arbitrary line range
- Edit file (append)
- Edit file (replace)
- Search files (glob)
    - The agent reported some funniness with the recursive behavior, it required funny syntax
- directory_tree on small dir

### Failed
- read media (png in this case)
    - Tool call showed in CLI but it just silently had an issue. Agent didn't get to follow up and the turn doesn't persist
- directory_tree on large dir
    - There seems to be no limits to how large a response it will return. Overflowed agent context.


## Jun 2: Switched to Desktop Commander MCP
Swapped `@modelcontextprotocol/server-filesystem` for `@wonderwhy-er/desktop-commander@latest` (6.1k stars, active maintenance).

**Why:** Anthropic server lacks arbitrary line-range reads (head/tail only). Desktop Commander's `read_file` supports `offset` + `length` for arbitrary ranges — closes the critical gap. Also includes `code_search` (ripgrep-based), terminal process management, and better directory listing with depth/overflow controls.

**allowedDirectories scoping:** Desktop Commander doesn't accept path as CLI arg — it uses a `config.json` in its working directory. FastMCP's `StdioMCPServer` config supports `cwd`, so we pre-write a `config.json` in a temp dir and pass that as `cwd`. Config is created fresh each proxy start; allowedDirectories restricts filesystem ops (note: terminal commands bypass this restriction — acceptable for our container-isolated setup).

**Fuzzy edit:** `edit_block` has a fuzzy fallback when exact match fails. For strict edit semantics we can fork to disable it, but leaving it on for now to evaluate in dogfooding.

**DEFAULT_WORKSPACE changed to `/workspace/git/misc/test`** for initial testing.

### Testing

#### Tested and passed:
- Search (grep/glob style)
    - Interesting behavior. Will return results instantly if search completes quickly and will require retrieval of results from a background process if search does not complete in some short time window. Timeout behavior doesn't seem to control whether or not it blocks, it seems to just determine how long the background search is willing to run.
- start_process
    - Here the timeout parameter determines whether or not the process is run in the background. If the command does not complete within the specified timeout, it will run in the background. If it completes within the specified timeout, it runs synchronously in blocks. So the timeout parameter can really be best thought of as block duration.
    - Can interact with background processes with continuity of env
    - Retrieval of background results (including pagination)
- read_file 
    - pagination
    - Tail reads
- edit
    - exact matching replaces content
    - fuzzy matchign does not replace content, but informs model where they missed.
- write_file
    - create
    - append

#### Tested and failed:
- list_directory
    - didn't return contents of the test dir. Could have been a syntax or config issue.
    - Debatable if we even want this tool. I'm skeptical of tools that are just a convenience wrapper for things that can be run with Bash. The model already knows how to ls.

#### General thoughts
The behavior and name and arguments of these tools does deviate some from Claude code style. That being said, all the underlying capabilities seem to still be there, just exposed in a slightly different way. I've seen research that suggests that models performance on agent decoding tasks is slightly degraded by using harnesses other than their native provider harness, in Anthropix case, Claude Code. The question is, how far do you have to deviate from the native harness before you get degradation? It's possible that the differences in desktop commanders' way of interacting with tools could be a problem.
What we might end up wanting to do is wrap the interfaces to make them more Claude code-like, which all the existing functionality seems sufficient to do that with a simple wrapper as opposed to having to reinvent stuff. But this is a problem that we should wait to see if actually exists before trying to solve it. It should be obvious in dogfooding if models are trying to call Claude code named tools or they're struggling with the way that the tools are presented.

It seems that remapping inputs and even doing some basic tool behavior modification is quite trivial with FastMCP wrappers. See claude.ai conversation.

There are some tools I will likely want to filter out, stuff that is easily achievable with basic bash commands like ls.
I'm skeptical its worth it to have things like list_directory taking up context and polluting the tool list.
