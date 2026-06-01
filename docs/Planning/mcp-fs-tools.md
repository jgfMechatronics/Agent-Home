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

proxy.run(transport="http", port=8080)
```

Then pydantic-ai connects with:
```python
from pydantic_ai.mcp import MCPToolset

tools = MCPToolset("http://localhost:8080/mcp")  # auto-detects transport
```

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