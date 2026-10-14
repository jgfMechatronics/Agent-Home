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
from agent.compaction import clear_context, compact
from memory.system_prompt_compilation import compile_system_prompt

if TYPE_CHECKING:
    from agent.types import AgentDeps

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Protocol and registry types
# ---------------------------------------------------------------------------

class SlashCommandHandler(Protocol):
    """Protocol for slash command handlers.

    Handlers receive only deps for now — no args are passed because no current command
    uses them. Add args back to this signature (and handle_slash_cmd dispatch) when needed.
    """
    async def __call__(self, deps: "AgentDeps") -> ServerSentEvent: ...


@dataclass
class SlashCommandDef:
    """Definition for a slash command: handler + discovery metadata."""
    handler: SlashCommandHandler
    description: str
    hint: str | None = None


# ---------------------------------------------------------------------------
# Handlers
# ---------------------------------------------------------------------------

async def _handle_recompile(deps: "AgentDeps") -> ServerSentEvent:
    """Handler for /recompile command. Recompiles the system prompt from current memory blocks."""
    await compile_system_prompt(deps)
    await deps.commit_changes_refresh_agent_record()
    return ServerSentEvent(
        data={"name": "user_recompile", "result": "System prompt recompiled successfully", "status": "success"},
        event="SlashCommandResult",
    )


def _bool_result_sse(name: str, did_work: bool, did_work_msg: str, noop_msg: str) -> ServerSentEvent:
    """Construct a success SlashCommandResult SSE whose result string reflects a boolean operation outcome."""
    return ServerSentEvent(
        data={"name": name, "result": did_work_msg if did_work else noop_msg, "status": "success"},
        event="SlashCommandResult",
    )


async def _handle_clear(deps: "AgentDeps") -> ServerSentEvent:
    """Handler for /clear command. Advances context window start to give a fresh session."""
    did_clear = await clear_context(deps)
    return _bool_result_sse("user_clear", did_clear,
        did_work_msg="Context cleared — fresh session started",
        noop_msg="Nothing to clear — no messages in context",
    )


async def _handle_compact(deps: "AgentDeps") -> ServerSentEvent:
    """Handler for /compact command. Runs on-demand compaction."""
    # Pass soft_compaction_limit as total_tokens — treats the context as if the limit were just hit.
    # Result: post-compact context lands at the configured retention fraction of the limit,
    # deterministic regardless of actual current token count.
    did_compact = await compact(deps, total_tokens=deps.config.soft_compaction_limit)
    return _bool_result_sse("user_compact", did_compact,
        did_work_msg="Compaction complete",
        noop_msg="Nothing to compact — context too small",
    )


# ---------------------------------------------------------------------------
# Registry
# ---------------------------------------------------------------------------

SLASH_COMMANDS: dict[str, SlashCommandDef] = {
    "recompile": SlashCommandDef(
        handler=_handle_recompile,
        description="Recompile memory blocks into system prompt",
    ),
    "clear": SlashCommandDef(
        handler=_handle_clear,
        description="Clear context window — fresh session, history preserved in DB",
    ),
    "compact": SlashCommandDef(
        handler=_handle_compact,
        description="Run on-demand compaction to reduce context size",
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

def _parse_slash_cmd(msg: str) -> str | None:
    """Parse a message as a slash command.

    Returns the command name if msg starts with '/' and command is in registry.
    Returns None otherwise (including unrecognized commands — those pass to model).
    """
    if not msg.startswith("/"):
        return None
    parts = msg[1:].split()
    cmd = parts[0].lower() if parts else None
    if not cmd or cmd not in SLASH_COMMANDS:
        return None
    return cmd


def is_slash_cmd(msg: str) -> bool:
    """Check if message is a recognized slash command."""
    return _parse_slash_cmd(msg) is not None


async def handle_slash_cmd(deps: "AgentDeps", msg: str) -> ServerSentEvent:
    """Dispatch a slash command to its handler and return the result SSE.

    Precondition: is_slash_cmd(msg) is True.
    """
    cmd = _parse_slash_cmd(msg)
    if cmd is None:
        # Shouldn't happen if precondition met, but handle gracefully
        return ServerSentEvent(
            data={"name": "user_unknown", "result": "Unknown command", "status": "error"},
            event="SlashCommandResult",
        )
    handler = SLASH_COMMANDS[cmd].handler
    try:
        return await handler(deps)
    except Exception as e:
        logger.exception("Slash command /%s failed", cmd)
        return ServerSentEvent(
            data={"name": f"user_{cmd}", "result": f"Command failed: {e}", "status": "error"},
            event="SlashCommandResult",
        )
