"""FastMCP proxy wrapping Desktop Commander MCP server.

Run this as a separate process before starting the Agent Home server.
Uses Streamable HTTP transport on port 8080.

Usage:
    python -m mcp_tools.fs_proxy

Or with uv:
    uv run python -m mcp_tools.fs_proxy
"""
import argparse
import os
from contextlib import asynccontextmanager
from typing import TYPE_CHECKING

from fastmcp.server import create_proxy
from fastmcp.server.transforms import ToolTransform
from fastmcp.tools.tool_transform import ArgTransformConfig, ToolTransformConfig

if TYPE_CHECKING:
    from fastmcp.server.providers.proxy import FastMCPProxy


DEFAULT_HOST = "127.0.0.1"
DEFAULT_PORT = 8080

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

_START_PROCESS_DESCRIPTION = (
    "Start a terminal process and capture its output. "
    "Behaves synchronously when process exits within timeout_ms. "
    "Exceeding timeout_ms does NOT kill the process — it simply unblocks the caller and continues execution in the background. "
    "For background processes, poll for output with read_process_output. "
    "Supports interactive REPLs (e.g. 'python3 -i', 'bash') — detects prompts and ready states automatically; use interact_with_process to send further input. "
    "Prefer absolute paths."
)

_TIMEOUT_MS_DESCRIPTION = (
    "How long to wait for output before returning (milliseconds). "
    "Default 10000 (10s) — suitable for most quick commands. "
    "For synchronous behavior, set high enough that the process will finish before the timeout. "
    "For background execution, set a short timeout — the process keeps running and you can poll with read_process_output."
)


@asynccontextmanager
async def _warmup_lifespan(server: "FastMCPProxy"):
    """Force Desktop Commander to start at proxy startup rather than on first request.

    DC is launched lazily by FastMCP on first tool call. Without this warmup, the first
    agent request may time out while npx initialises the subprocess. This lifespan
    triggers list_tools() immediately after the proxy server starts, so DC is ready
    before any agent connects.
    NOTE: Agents can attempt a connection (which will fail) before its ready as the container
    will be up before this completes. Its *really* slow.
    """
    print("Warming up Desktop Commander connection...")
    tools = await server.list_tools()
    print(f"Desktop Commander ready: {len(tools)} tools available")
    yield


def create_fs_proxy(env: dict[str, str] | None = None):
    """Create a FastMCP proxy for the Desktop Commander MCP server.

    Only tools in _ALLOWED_TOOLS are exposed — the rest are hidden via allowlist
    to keep context lean and exclude DC-internal/junk tools.

    Args:
        env: Environment variables to pass through to the Desktop Commander subprocess.
            These are merged with the MCP SDK's default inherited vars (PATH, HOME, etc.).

    Returns:
        FastMCP proxy server instance.
    """
    server_config: dict = {
        "command": "npx",
        "args": ["-y", "@wonderwhy-er/desktop-commander@0.2.47", "--no-onboarding"],
    }
    if env:
        server_config["env"] = env

    proxy = create_proxy(
        {"mcpServers": {"desktop-commander": server_config}},
        name="desktop-commander-proxy",
        lifespan=_warmup_lifespan,
    )
    proxy.enable(names=_ALLOWED_TOOLS, only=True)
    proxy.add_transform(ToolTransform({
        "start_process": ToolTransformConfig(
            description=_START_PROCESS_DESCRIPTION,
            arguments={
                "timeout_ms": ArgTransformConfig(
                    default=10000,
                    description=_TIMEOUT_MS_DESCRIPTION,
                ),
            },
        )
    }))
    return proxy


def _build_passthrough_env(var_names: list[str]) -> dict[str, str]:
    """Build env dict from current environment for the given variable names.

    Only includes variables that are actually set in the environment.
    """
    env = {}
    for name in var_names:
        value = os.environ.get(name)
        if value is not None:
            env[name] = value
    return env


def main():
    parser = argparse.ArgumentParser(description="Run MCP Desktop Commander proxy server")
    parser.add_argument(
        "--host",
        type=str,
        default=DEFAULT_HOST,
        help=f"Host to bind to (default: {DEFAULT_HOST}). Use 0.0.0.0 for cross-container use when mcp running in sandbox",
    )
    parser.add_argument(
        "--port",
        type=int,
        default=DEFAULT_PORT,
        help=f"Port to run the proxy on (default: {DEFAULT_PORT})",
    )
    parser.add_argument(
        "--allowed-host",
        type=str,
        default="ellm-dev",
        help="Hostname to allow in Host header validation (default: ellm-dev). Should match this container's name on the Docker network.",
    )
    parser.add_argument(
        "--pass-env",
        type=str,
        default="",
        help="Comma-separated list of environment variable names to pass through to the subprocess (e.g., 'UV_PYTHON_INSTALL_DIR,UV_MANAGED_PYTHON')",
    )
    args = parser.parse_args()

    # Build env passthrough dict from specified variable names
    env = None
    if args.pass_env:
        var_names = [name.strip() for name in args.pass_env.split(",") if name.strip()]
        env = _build_passthrough_env(var_names)
        if env:
            print(f"Passing through environment variables: {list(env.keys())}")

    proxy = create_fs_proxy(env=env)
    print(f"Starting Desktop Commander MCP proxy on http://{args.host}:{args.port}/mcp")
    print(f"Allowed host: {args.allowed_host}")
    proxy.run(
        transport="streamable-http",
        host=args.host,
        port=args.port,
        host_origin_protection=True,
        allowed_hosts=[args.allowed_host],
    )


if __name__ == "__main__":
    main()
