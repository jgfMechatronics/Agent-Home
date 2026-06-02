"""FastMCP proxy wrapping Anthropic's filesystem MCP server.

Run this as a separate process before starting the Agent Home server.
Uses Streamable HTTP transport on port 8080.

Usage:
    python -m mcp_tools.fs_proxy

Or with uv:
    uv run python -m mcp_tools.fs_proxy
"""
import argparse

from fastmcp.server import create_proxy


DEFAULT_HOST = "127.0.0.1"
DEFAULT_PORT = 8080
DEFAULT_WORKSPACE = "/workspace"


def create_fs_proxy(workspace_path: str = DEFAULT_WORKSPACE):
    """Create a FastMCP proxy for the Anthropic filesystem server.
    
    Args:
        workspace_path: Root path to expose via the filesystem server.
    
    Returns:
        FastMCP proxy server instance.
    """
    return create_proxy(
        {
            "mcpServers": {
                "filesystem": {
                    "command": "npx",
                    "args": [
                        "-y",
                        "@modelcontextprotocol/server-filesystem",
                        workspace_path,
                    ],
                }
            }
        },
        name="filesystem-proxy",
    )


def main():
    parser = argparse.ArgumentParser(description="Run MCP filesystem proxy server")
    parser.add_argument(
        "--host",
        type=str,
        default=DEFAULT_HOST,
        help=f"Host to bind to (default: {DEFAULT_HOST}, use 0.0.0.0 for cross-container)",
    )
    parser.add_argument(
        "--port",
        type=int,
        default=DEFAULT_PORT,
        help=f"Port to run the proxy on (default: {DEFAULT_PORT})",
    )
    parser.add_argument(
        "--workspace",
        type=str,
        default=DEFAULT_WORKSPACE,
        help=f"Workspace path to expose (default: {DEFAULT_WORKSPACE})",
    )
    args = parser.parse_args()

    proxy = create_fs_proxy(args.workspace)
    print(f"Starting filesystem MCP proxy on http://{args.host}:{args.port}/mcp")
    print(f"Exposing workspace: {args.workspace}")
    proxy.run(transport="streamable-http", host=args.host, port=args.port)


if __name__ == "__main__":
    main()
