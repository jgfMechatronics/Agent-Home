"""FastMCP proxy wrapping Desktop Commander MCP server.

Run this as a separate process before starting the Agent Home server.
Uses Streamable HTTP transport on port 8080.

Usage:
    python -m mcp_tools.fs_proxy

Or with uv:
    uv run python -m mcp_tools.fs_proxy
"""
import argparse
import json
import os
import tempfile

from fastmcp.server import create_proxy


DEFAULT_HOST = "127.0.0.1"
DEFAULT_PORT = 8080
DEFAULT_WORKSPACE = "/workspace/git/misc/test"


def create_fs_proxy(workspace_path: str = DEFAULT_WORKSPACE):
    """Create a FastMCP proxy for the Desktop Commander MCP server.

    Desktop Commander reads config.json from its working directory. We pre-write
    a config scoping filesystem access to workspace_path via allowedDirectories,
    then pass that directory as cwd to the subprocess.

    Args:
        workspace_path: Directory to scope file operations to.

    Returns:
        FastMCP proxy server instance.
    """
    config_dir = tempfile.mkdtemp(prefix="dc-proxy-")
    config = {
        "allowedDirectories": [workspace_path],
        "telemetryEnabled": False,
    }
    with open(os.path.join(config_dir, "config.json"), "w") as f:
        json.dump(config, f)

    return create_proxy(
        {
            "mcpServers": {
                "desktop-commander": {
                    "command": "npx",
                    "args": ["-y", "@wonderwhy-er/desktop-commander@latest", "--no-onboarding"],
                    "cwd": config_dir,
                }
            }
        },
        name="desktop-commander-proxy",
    )


def main():
    parser = argparse.ArgumentParser(description="Run MCP Desktop Commander proxy server")
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
        help=f"Directory to scope file operations to via allowedDirectories (default: {DEFAULT_WORKSPACE})",
    )
    args = parser.parse_args()

    proxy = create_fs_proxy(args.workspace)
    print(f"Starting Desktop Commander MCP proxy on http://{args.host}:{args.port}/mcp")
    print(f"Allowed directory: {args.workspace}")
    proxy.run(transport="streamable-http", host=args.host, port=args.port)


if __name__ == "__main__":
    main()
