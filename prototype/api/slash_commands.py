"""
Slash command registry and dispatch for ACP/TUI clients.

Slash commands are short-circuit handlers for messages beginning with '/'.
They bypass the agent loop entirely and return an SSE result directly.

To add a new command: implement a SlashCommandHandler and register it in SLASH_COMMANDS.
"""
import logging
from dataclasses import dataclass
from typing import Any, Protocol, TYPE_CHECKING

from fastapi.sse import ServerSentEvent
from memory.system_prompt_compilation import compile_system_prompt

if TYPE_CHECKING:
    from agent.types import AgentDeps

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Protocol and registry types
# ---------------------------------------------------------------------------

class SlashCommandHandler(Protocol):
    """Protocol for slash command handlers. All handlers receive deps and args."""
    async def __call__(self, deps: "AgentDeps", args: str) -> ServerSentEvent: ...


@dataclass
class SlashCommandDef:
    """Definition for a slash command: handler + discovery metadata."""
    handler: SlashCommandHandler
    description: str
    hint: str | None = None


# ---------------------------------------------------------------------------
# Handlers
# ---------------------------------------------------------------------------

async def _handle_recompile(deps: "AgentDeps", args: str) -> ServerSentEvent:
    """Handler for /recompile command. Recompiles the system prompt from current memory blocks."""
    await compile_system_prompt(deps)
    await deps.commit_changes_refresh_agent_record()
    return ServerSentEvent(
        data={"name": "user_recompile", "args": args, "result": "System prompt recompiled successfully", "status": "success"},
        event="SlashCommandResult",
    )


# ---------------------------------------------------------------------------
# Registry
# ---------------------------------------------------------------------------

SLASH_COMMANDS: dict[str, SlashCommandDef] = {
    "recompile": SlashCommandDef(
        handler=_handle_recompile,
        description="Recompile memory blocks into system prompt",
    ),
}


# ---------------------------------------------------------------------------
# Discovery
# ---------------------------------------------------------------------------

def get_available_commands() -> list[dict[str, Any]]:
    """Build the availableCommands list for ACP discovery notification."""
    commands = []
    for name, cmd_def in SLASH_COMMANDS.items():
        cmd: dict[str, Any] = {"name": name, "description": cmd_def.description}
        if cmd_def.hint:
            cmd["input"] = {"hint": cmd_def.hint}
        commands.append(cmd)
    return commands


# ---------------------------------------------------------------------------
# Parse and dispatch
# ---------------------------------------------------------------------------

def _parse_slash_cmd(msg: str) -> tuple[str, str] | None:
    """Parse a message as a slash command.

    Returns (command_name, args) if msg starts with '/' and command is in registry.
    Returns None otherwise (including unrecognized commands — those pass to model).
    """
    if not msg.startswith("/"):
        return None
    # Split into command and args: "/recompile foo bar" -> ("recompile", "foo bar")
    parts = msg[1:].split(maxsplit=1)
    if not parts:
        return None
    cmd = parts[0].lower()
    args = parts[1] if len(parts) > 1 else ""
    if cmd not in SLASH_COMMANDS:
        return None
    return (cmd, args)


def is_slash_cmd(msg: str) -> bool:
    """Check if message is a recognized slash command."""
    return _parse_slash_cmd(msg) is not None


async def handle_slash_cmd(deps: "AgentDeps", msg: str) -> ServerSentEvent:
    """Dispatch a slash command to its handler and return the result SSE.

    Precondition: is_slash_cmd(msg) is True.
    """
    parsed = _parse_slash_cmd(msg)
    if parsed is None:
        # Shouldn't happen if precondition met, but handle gracefully
        return ServerSentEvent(
            data={"name": "user_unknown", "args": msg, "result": "Unknown command", "status": "error"},
            event="SlashCommandResult",
        )
    cmd, args = parsed
    handler = SLASH_COMMANDS[cmd].handler
    try:
        return await handler(deps, args)
    except Exception as e:
        logger.exception("Slash command /%s failed", cmd)
        return ServerSentEvent(
            data={"name": f"user_{cmd}", "args": args, "result": f"Command failed: {e}", "status": "error"},
            event="SlashCommandResult",
        )
