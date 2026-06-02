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

### Live testing completed and passed:
- Read dir contents
- Create file
- Read file contents
- Edit file (append)
- Edit file (replace)