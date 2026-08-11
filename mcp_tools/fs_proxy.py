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


DEFAULT_HOST = "0.0.0.0"
DEFAULT_PORT = 8080
DEFAULT_WORKSPACE = "/workspace/git/misc/test"

# Allowlist of Desktop Commander tools to expose.
# Everything else is excluded — keeps context lean and avoids junk tools.
# TODO: Remove read_multiple_files once parallel tool calls are re-enabled in Agent Home
#       (currently disabled due to orphan remover incompatibility).
_ALLOWED_TOOLS = {
    "read_file",
    "read_multiple_files",
    "write_file",
    "edit_block",
    "get_file_info",
    "start_process",
    "interact_with_process",
    "read_process_output",
    "force_terminate",
    "list_sessions",
    "start_search",
    "get_more_search_results",
    "stop_search",
    "list_searches",
}


def create_fs_proxy(workspace_path: str = DEFAULT_WORKSPACE):
    """Create a FastMCP proxy for the Desktop Commander MCP server.

    Desktop Commander reads config.json from its working directory. We pre-write
    a config scoping filesystem access to workspace_path via allowedDirectories,
    then pass that directory as cwd to the subprocess.

    Only tools in _ALLOWED_TOOLS are exposed — the rest are hidden via allowlist
    to keep context lean and exclude DC-internal/junk tools.

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

    proxy = create_proxy(
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
    proxy.enable(names=_ALLOWED_TOOLS, only=True)
    return proxy


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
