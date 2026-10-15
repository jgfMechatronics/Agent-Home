"""FastMCP proxy wrapping Claude Code running as a pure MCP tool server.

Replaces Desktop Commander as the filesystem/shell backend. `claude mcp serve`
exposes Claude Code's own tools (Read, Edit, Write, Bash) over stdio JSON-RPC;
the calling agent does the reasoning, the server only executes tools.

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
from fastmcp.server.middleware import Middleware, MiddlewareContext
from fastmcp.server.transforms import ToolTransform
from fastmcp.tools.tool_transform import ToolTransformConfig

from mcp_tools.cc_result_format import format_cc_tool_result

if TYPE_CHECKING:
    from fastmcp.server.providers.proxy import FastMCPProxy


DEFAULT_HOST = "127.0.0.1"
DEFAULT_PORT = 8080

# Allowlist of Claude Code tools to expose (allowlist, never denylist — the CC
# tool surface varies by version and environment, and includes session/account
# bound tools that must not leak through). Pin the CC version in the sandbox
# image; after any upgrade, verify the tool set still matches expectations.
_ALLOWED_TOOLS = {
    "Read",
    "Edit",
    "Write",
    "Bash",
}

_BASH_DESCRIPTION = (
    "Execute a shell command; synchronous — returns when the command completes or the timeout is hit. "
    "The working directory persists across calls within a session. "
    "Foreground commands are capped (~10 minutes); for longer-running or background work, start the job "
    "detached (nohup sh -c '...' >log 2>&1 &) and poll the log with later calls. "
    "Prefer absolute paths. "
    "find/grep are shell functions backed by the claude executable — plain output, no structured search params."
)


@asynccontextmanager
async def _warmup_lifespan(server: "FastMCPProxy"):
    """Force the Claude Code server to start at proxy startup rather than on first request.

    The subprocess is launched lazily by FastMCP on first tool call. Without this
    warmup, the first agent request may time out while `claude mcp serve`
    initialises. This lifespan triggers list_tools() immediately after the proxy
    server starts, so the backend is ready before any agent connects.
    NOTE: Agents can attempt a connection (which will fail) before it's ready — the
    container will be up before this completes.
    """
    print("Warming up Claude Code MCP server connection...")
    tools = await server.list_tools()
    print(f"Claude Code ready: {len(tools)} tools available")
    yield


class _CCResultFormatMiddleware(Middleware):
    """Rewrite claude mcp serve's JSON tool results into CC-native text.

    Serve serializes results as JSON objects; CC's own agents see plain
    strings. The formatter (cc_result_format) maps between them — pure
    function, defensive pass-through on anything unexpected. Only single
    text-content results are candidates; anything richer passes untouched.
    """

    async def on_call_tool(
        self,
        context: "MiddlewareContext",
        call_next,
    ):
        result = await call_next(context)
        content = result.content
        if (
            len(content) == 1
            and getattr(content[0], "type", None) == "text"
        ):
            text = content[0].text
            formatted = format_cc_tool_result(context.message.name, text, result.is_error)
            if formatted != text:
                content[0].text = formatted
        return result


def create_fs_proxy(env: dict[str, str] | None = None):
    """Create a FastMCP proxy for the Claude Code MCP tool server.

    Only tools in _ALLOWED_TOOLS are exposed — the rest are hidden via allowlist
    (CC's tool surface includes session/account-bound tools that must not leak).

    Args:
        env: Environment variables to pass through to the claude subprocess.
            These are merged with the MCP SDK's default inherited vars (PATH, HOME, etc.).

    Returns:
        FastMCP proxy server instance.
    """
    server_config: dict = {
        "command": "claude",
        "args": ["mcp", "serve"],
    }
    if env:
        server_config["env"] = env

    proxy = create_proxy(
        {"mcpServers": {"claude-code": server_config}},
        name="claude-code-proxy",
        lifespan=_warmup_lifespan,
    )
    proxy.enable(names=_ALLOWED_TOOLS, only=True)
    proxy.add_middleware(_CCResultFormatMiddleware())
    proxy.add_transform(ToolTransform({
        "Bash": ToolTransformConfig(
            description=_BASH_DESCRIPTION,
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
    parser = argparse.ArgumentParser(description="Run Claude Code MCP proxy server")
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
    print(f"Starting Claude Code MCP proxy on http://{args.host}:{args.port}/mcp")
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
