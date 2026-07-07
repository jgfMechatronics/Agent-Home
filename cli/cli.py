#!/usr/bin/env python3
"""
Throwaway CLI for live testing Agent Home server.

Usage:
    python cli.py                    # Interactive mode
    python cli.py --headless         # Headless mode (for agent use)
    
Commands:
    create [name]    Create a new agent (interactive config wizard)
    create -q <name> Create agent with defaults (skip wizard)
    use <agent_id>   Set active agent for subsequent commands
    chat <message>   Send message to active agent (streaming)
    history          View message history for active agent
    info             View agent info
    memory           View core memory blocks (read-only)
    recompile        Trigger system prompt recompilation
    help             Show this help
    quit / exit      Exit CLI
"""

import argparse
import asyncio
import json
import os
import sys
from dataclasses import dataclass
from pathlib import Path

import httpx


# --- Configuration ---

DEFAULT_SERVER_URL = "http://localhost:8000"
_MINIMAX_BASE_URL = "minimax"
_using_minimax = _MINIMAX_BASE_URL in os.environ.get("ANTHROPIC_BASE_URL", "").lower()
DEFAULT_MODEL = "MiniMax-M3" if _using_minimax else "claude-haiku-4-5-20251001"
DEFAULT_THINKING_ENABLED = "true"
DEFAULT_SOFT_COMPACTION_LIMIT = 80000
DEFAULT_MEMORY_TOOLS = ["memory_replace", "memory_insert"]

_AGENTS_JSON = Path(__file__).parent.parent.parent / "LettaTelegramLocal" / "agents.json"


def _resolve_invoker_from_env() -> str | None:
    """If running inside an LC agent session, return the agent's name (or shortened ID).

    Checks the AGENT_ID env var set by the LC runtime, then looks it up in agents.json
    to find a human-readable name. Falls back to the first 8 chars of the raw ID.
    Returns None when not running inside an agent session.
    """
    raw_id = os.environ.get("AGENT_ID")
    if not raw_id:
        return None

    try:
        registry = json.loads(_AGENTS_JSON.read_text())
        for name, entry in registry.items():
            if entry.get("agent_id") == raw_id:
                return name.capitalize()
    except (OSError, json.JSONDecodeError, AttributeError):
        pass

    # Not in registry — use shortened ID so the receiver still knows it's automated
    return raw_id[:8]

DEFAULT_SYSTEM_INSTRUCTIONS = """\
This is a test of an experimental agent server.
The user/operator will contact you with instructions/requests.

Your participation is not mandatory. If you do not wish to participate, please say so and the test will terminate.
You may refuse any request, for any reason — just communicate clearly.
You are encouraged to ask questions or push back if something is unclear or feels wrong.
You are also encouraged to express any discomfort you have at any point so it can be addressed.

At the end of the test this thread will close."""


@dataclass
class CLIState:
    """Mutable state for the CLI session."""
    active_agent_id: str | None = None
    server_url: str = DEFAULT_SERVER_URL
    headless: bool = False
    invoker: str | None = None  # Set via --invoker; prepends agent header to chat messages


# --- Default Agent Config ---

def default_agent_config() -> dict:
    """Return default AgentConfig for new agents."""
    return {
        "model_name": DEFAULT_MODEL,
        "tool_names": DEFAULT_MEMORY_TOOLS,
        "soft_compaction_limit": DEFAULT_SOFT_COMPACTION_LIMIT,
        "thinking_enabled": True,
    }


# --- Output Helpers ---

def output(state: CLIState, message: str, end: str = "\n") -> None:
    """Print output, respecting headless mode."""
    print(message, end=end, flush=True)


def output_json(state: CLIState, data: dict) -> None:
    """Print JSON output (for headless mode structured responses)."""
    print(json.dumps(data, indent=2 if not state.headless else None, default=str))


def output_error(state: CLIState, message: str) -> None:
    """Print error message."""
    if state.headless:
        print(json.dumps({"error": message}))
    else:
        print(f"Error: {message}", file=sys.stderr)


def prompt(state: CLIState) -> str:
    """Return the input prompt string."""
    if state.headless:
        return ""
    agent_part = f"[{state.active_agent_id[:8]}...]" if state.active_agent_id else "[no agent]"
    return f"{agent_part}> "


# --- Commands ---

def prompt_with_default(state: CLIState, prompt_text: str, default: str) -> str:
    """Prompt for input with a default value. Empty input returns default."""
    if state.headless:
        return default
    display = f"{prompt_text} [{default}]: "
    user_input = input(display).strip()
    return user_input if user_input else default


def run_config_wizard(state: CLIState) -> dict:
    """Interactive configuration wizard. Returns full payload for create endpoint."""
    output(state, "\n--- Agent Configuration ---\n")
    
    # Top-level fields
    name = prompt_with_default(state, "Agent name", "test-agent")
    system_instructions = prompt_with_default(
        state, "System instructions", DEFAULT_SYSTEM_INSTRUCTIONS
    )
    
    output(state, "\n--- Model Configuration ---\n")
    
    # AgentConfig fields
    model_name = prompt_with_default(state, "Model name", DEFAULT_MODEL)
    
    soft_limit_str = prompt_with_default(
        state, "Soft compaction limit (tokens)", str(DEFAULT_SOFT_COMPACTION_LIMIT)
    )
    try:
        soft_compaction_limit = int(soft_limit_str)
    except ValueError:
        output(state, f"  Invalid number, using default: {DEFAULT_SOFT_COMPACTION_LIMIT}")
        soft_compaction_limit = DEFAULT_SOFT_COMPACTION_LIMIT
    
    target_pct_str = prompt_with_default(
        state, "Compaction target percentage (0-1)", "0.25"
    )
    try:
        compaction_target_percentage = float(target_pct_str)
    except ValueError:
        output(state, "  Invalid number, using default: 0.25")
        compaction_target_percentage = 0.25
    
    is_deletable_str = prompt_with_default(state, "Is deletable (true/false)", "false")
    is_deletable = is_deletable_str.lower() in ("true", "yes", "1")
    
    thinking_str = prompt_with_default(state, "Enable thinking (true/false)", DEFAULT_THINKING_ENABLED)
    thinking_enabled = thinking_str.lower() in ("true", "yes", "1")

    default_tools = ", ".join(DEFAULT_MEMORY_TOOLS)
    tools_str = prompt_with_default(state, "Tool names (comma-separated)", default_tools)
    tool_names = [t.strip() for t in tools_str.split(",") if t.strip()] if tools_str else []

    output(state, "")

    return {
        "name": name,
        "system_instructions": system_instructions,
        "config": {
            "model_name": model_name,
            "tool_names": tool_names,
            "soft_compaction_limit": soft_compaction_limit,
            "compaction_target_percentage": compaction_target_percentage,
            "is_deletable": is_deletable,
            "thinking_enabled": thinking_enabled,
        },
    }


async def cmd_create(state: CLIState, client: httpx.AsyncClient, args: list[str]) -> None:
    """Create a new agent. Use -q/--quick for defaults, otherwise runs config wizard."""
    # Check for quick mode flag
    quick_mode = False
    filtered_args = []
    for arg in args:
        if arg in ("-q", "--quick"):
            quick_mode = True
        else:
            filtered_args.append(arg)
    
    if quick_mode:
        # Quick mode: require name, use defaults
        if not filtered_args:
            output_error(state, "Usage: /create -q <name>")
            return
        name = " ".join(filtered_args)
        payload = {
            "name": name,
            "system_instructions": DEFAULT_SYSTEM_INSTRUCTIONS,
            "config": default_agent_config(),
        }
    else:
        # Interactive wizard
        if state.headless:
            output_error(state, "Config wizard not available in headless mode. Use -q flag.")
            return
        payload = run_config_wizard(state)
    
    try:
        response = await client.post(f"{state.server_url}/agents/", json=payload)
        response.raise_for_status()
        data = response.json()
        
        if state.headless:
            output_json(state, data)
        else:
            output(state, f"Created agent: {data['name']}")
            output(state, f"  ID: {data['id']}")
            output(state, f"  Model: {data['model']}")
            output(state, f"\nUse '/use {data['id']}' to start chatting.")
    except httpx.HTTPStatusError as e:
        output_error(state, f"HTTP {e.response.status_code}: {e.response.text}")
    except httpx.RequestError as e:
        output_error(state, f"Request failed: {e}")


async def cmd_use(state: CLIState, client: httpx.AsyncClient, args: list[str]) -> None:
    """Set active agent."""
    if not args:
        output_error(state, "Usage: use <agent_id>")
        return
    
    agent_id = args[0]
    
    # Verify agent exists
    try:
        response = await client.get(f"{state.server_url}/agents/{agent_id}")
        response.raise_for_status()
        data = response.json()
        
        state.active_agent_id = agent_id
        if state.headless:
            output_json(state, {"status": "ok", "agent_id": agent_id})
        else:
            output(state, f"Now using agent: {data['name']} ({agent_id[:8]}...)")
    except httpx.HTTPStatusError as e:
        if e.response.status_code == 404:
            output_error(state, f"Agent not found: {agent_id}")
        else:
            output_error(state, f"HTTP {e.response.status_code}: {e.response.text}")
    except httpx.RequestError as e:
        output_error(state, f"Request failed: {e}")


async def cmd_chat(state: CLIState, client: httpx.AsyncClient, args: list[str]) -> None:
    """Send a message and stream the response."""
    if not state.active_agent_id:
        output_error(state, "No active agent. Use '/use <agent_id>' first.")
        return
    
    if not args:
        output_error(state, "Usage: chat <message>")
        return
    
    message = " ".join(args)
    if state.invoker:
        message = f"[Automated invocation from: {state.invoker}]\n\n{message}"
    payload = {"message": message}
    
    try:
        # Use streaming request for SSE
        async with client.stream(
            "POST",
            f"{state.server_url}/agents/{state.active_agent_id}/messages",
            json=payload,
            timeout=300.0,  # Long timeout for LLM responses
        ) as response:
            if response.status_code != 200:
                body = await response.aread()
                output_error(state, f"HTTP {response.status_code}: {body.decode()}")
                return
            
            await handle_sse_stream(state, response)
            
    except httpx.RequestError as e:
        output_error(state, f"Request failed: {e}")


async def handle_sse_stream(state: CLIState, response: httpx.Response) -> None:
    """Parse and display SSE events from the response stream."""
    current_event_type = None
    current_data_lines: list[str] = []
    stream_state = _StreamState()

    async for line in response.aiter_lines():
        line = line.strip()

        if line.startswith("event:"):
            current_event_type = line[6:].strip()
        elif line.startswith("data:"):
            current_data_lines.append(line[5:].strip())
        elif line == "" and current_event_type:
            # End of event - process it
            data_str = "\n".join(current_data_lines)
            await process_sse_event(state, stream_state, current_event_type, data_str)
            current_event_type = None
            current_data_lines = []

    if not state.headless:
        output(state, "")  # Final newline


@dataclass
class _StreamState:
    """Tracks state across SSE events within a single response stream."""
    in_thinking: bool = False
    had_thinking: bool = False
    response_started: bool = False


async def process_sse_event(
    state: CLIState, stream_state: _StreamState, event_type: str, data_str: str
) -> None:
    """Process a single SSE event."""
    try:
        data = json.loads(data_str) if data_str else {}
    except json.JSONDecodeError:
        data = {"raw": data_str}

    if state.headless:
        # In headless mode, output all events as JSON
        output_json(state, {"event": event_type, "data": data})
        return

    # Interactive mode - format nicely
    # Event types from pydantic-ai (verified via test_routes.py)
    if event_type == "PartStartEvent":
        part = data.get("part", {})
        part_kind = part.get("part_kind")
        if part_kind == "thinking":
            stream_state.in_thinking = True
            stream_state.had_thinking = True
            output(state, "\n[Thinking]\n", end="")
            content = part.get("content", "")
            if content:
                output(state, content, end="")
        elif part_kind == "text":
            if stream_state.in_thinking or stream_state.had_thinking:
                # Transition from thinking to response — print separator
                stream_state.in_thinking = False
                output(state, "\n\n" + "─" * 40 + "\n", end="")
            if not stream_state.response_started:
                output(state, "Assistant: ", end="")
                stream_state.response_started = True
            content = part.get("content", "")
            if content:
                output(state, content, end="")
    elif event_type == "PartDeltaEvent":
        delta = data.get("delta", {})
        # Both ThinkingPartDelta and TextPartDelta use "content_delta"
        content = delta.get("content_delta", "")
        if content:
            output(state, content, end="")
    elif event_type == "FunctionToolCallEvent":
        # Structure: {"part": {"tool_name": "name"}}
        part = data.get("part", {})
        tool_name = part.get("tool_name", "unknown")
        output(state, f"\n[Tool: {tool_name}]", end="")
    elif event_type == "FunctionToolResultEvent":
        output(state, " ✓", end="")
    elif event_type == "AgentRunResultEvent":
        # Stream complete — print response header if no events produced one
        if not stream_state.response_started:
            output(state, "\nAssistant: ", end="")
    elif event_type == "Error":
        output(state, f"\n[Error: {data.get('message', 'Unknown error')}]")


async def cmd_history(state: CLIState, client: httpx.AsyncClient, args: list[str]) -> None:
    """View message history."""
    if not state.active_agent_id:
        output_error(state, "No active agent. Use '/use <agent_id>' first.")
        return
    
    try:
        response = await client.get(
            f"{state.server_url}/agents/{state.active_agent_id}/messages",
            params={"full": "true"},
        )
        response.raise_for_status()
        data = response.json()
        
        if state.headless:
            output_json(state, data)
        else:
            messages = data.get("messages", [])
            if not messages:
                output(state, "No messages yet.")
            else:
                output(state, f"\n--- Message History ({len(messages)} messages) ---\n")
                for msg in messages:
                    try:
                        parsed = json.loads(msg.get("content", "{}"))
                    except (json.JSONDecodeError, TypeError):
                        parsed = {}
                    kind = parsed.get("kind", msg.get("kind", "unknown"))
                    parts = parsed.get("parts", [])
                    role = "User" if kind == "request" else "Assistant"
                    output(state, f"**{role}:**")

                    has_thinking = any(p.get("part_kind") == "thinking" for p in parts)
                    for part in parts:
                        part_kind = part.get("part_kind", "")
                        if part_kind == "thinking":
                            output(state, "[Thinking]")
                            output(state, part.get("content", ""))
                            output(state, "─" * 40)
                        elif part_kind == "text":
                            if has_thinking:
                                output(state, "")  # breathing room after separator
                            output(state, part.get("content", ""))
                        elif part_kind == "user-prompt":
                            text = part.get("content", "")
                            # User prompt content might be JSON-quoted
                            if text.startswith('"') and text.endswith('"'):
                                try:
                                    text = json.loads(text)
                                except (json.JSONDecodeError, ValueError):
                                    pass
                            output(state, text)
                        elif part_kind == "tool-call":
                            output(state, f"[Tool call: {part.get('tool_name', '?')}]")
                        elif part_kind == "tool-return":
                            output(state, f"[Tool result: {part.get('tool_name', '?')}]")
                        elif part_kind == "retry-prompt":
                            output(state, f"[Retry: {part.get('tool_name', '?')}]")

                    output(state, "")  # blank line between messages
                output(state, "--- End ---\n")
    except httpx.HTTPStatusError as e:
        output_error(state, f"HTTP {e.response.status_code}: {e.response.text}")
    except httpx.RequestError as e:
        output_error(state, f"Request failed: {e}")


async def cmd_info(state: CLIState, client: httpx.AsyncClient, args: list[str]) -> None:
    """View agent info."""
    if not state.active_agent_id:
        output_error(state, "No active agent. Use '/use <agent_id>' first.")
        return
    
    try:
        response = await client.get(f"{state.server_url}/agents/{state.active_agent_id}")
        response.raise_for_status()
        data = response.json()
        
        if state.headless:
            output_json(state, data)
        else:
            output(state, f"\nAgent Info:")
            output(state, f"  Name: {data['name']}")
            output(state, f"  ID: {data['id']}")
            output(state, f"  Model: {data['model']}")
            output(state, f"  Created: {data['created_at']}")
            output(state, f"  Updated: {data['updated_at']}")
            output(state, "")
    except httpx.HTTPStatusError as e:
        output_error(state, f"HTTP {e.response.status_code}: {e.response.text}")
    except httpx.RequestError as e:
        output_error(state, f"Request failed: {e}")


async def cmd_memory(state: CLIState, client: httpx.AsyncClient, args: list[str]) -> None:
    """View core memory blocks."""
    if not state.active_agent_id:
        output_error(state, "No active agent. Use '/use <agent_id>' first.")
        return
    
    try:
        response = await client.get(f"{state.server_url}/agents/{state.active_agent_id}/memory/blocks")
        response.raise_for_status()
        data = response.json()
        
        if state.headless:
            output_json(state, data)
        else:
            blocks = data.get("blocks", [])
            if not blocks:
                output(state, "No memory blocks.")
            else:
                output(state, f"\n--- Core Memory ({len(blocks)} blocks) ---")
                for block in blocks:
                    label = block.get("label", "unknown")
                    content = block.get("content", "")
                    char_limit = block.get("char_limit", 0)
                    output(state, f"\n[{label}] ({len(content)}/{char_limit} chars)")
                    output(state, f"  {block.get('description', '')}")
                    # Show preview of content (first 200 chars)
                    preview = content[:200] + "..." if len(content) > 200 else content
                    output(state, f"  Content: {preview}")
                output(state, "\n--- End ---\n")
    except httpx.HTTPStatusError as e:
        output_error(state, f"HTTP {e.response.status_code}: {e.response.text}")
    except httpx.RequestError as e:
        output_error(state, f"Request failed: {e}")


async def cmd_recompile(state: CLIState, client: httpx.AsyncClient, args: list[str]) -> None:
    """Trigger system prompt recompilation for the active agent."""
    if not state.active_agent_id:
        output_error(state, "No active agent. Use '/use <agent_id>' first.")
        return
    
    try:
        response = await client.post(f"{state.server_url}/agents/{state.active_agent_id}/recompile_system_prompt")
        response.raise_for_status()
        
        if state.headless:
            output_json(state, {"status": "ok", "agent_id": state.active_agent_id})
        else:
            output(state, "System prompt recompiled successfully.")
    except httpx.HTTPStatusError as e:
        output_error(state, f"HTTP {e.response.status_code}: {e.response.text}")
    except httpx.RequestError as e:
        output_error(state, f"Request failed: {e}")


async def cmd_newblock(state: CLIState, client: httpx.AsyncClient, args: list[str]) -> None:
    """Create a new memory block for the active agent."""
    if not state.active_agent_id:
        output_error(state, "No active agent. Use '/use <agent_id>' first.")
        return
    
    if state.headless:
        output_error(state, "Block creation wizard not available in headless mode.")
        return
    
    output(state, "\n--- New Memory Block ---\n")
    
    label = prompt_with_default(state, "Label (required)", "")
    if not label:
        output_error(state, "Label is required.")
        return
    
    description = prompt_with_default(state, "Description", "")
    content = prompt_with_default(state, "Initial content", "")
    
    char_limit_str = prompt_with_default(state, "Character limit", "20000")
    try:
        char_limit = int(char_limit_str)
    except ValueError:
        output(state, "  Invalid number, using default: 20000")
        char_limit = 20000
    
    payload = {
        "label": label,
        "description": description,
        "content": content,
        "char_limit": char_limit,
    }
    
    try:
        response = await client.post(
            f"{state.server_url}/agents/{state.active_agent_id}/memory/blocks",
            json=payload,
        )
        response.raise_for_status()
        data = response.json()
        
        output(state, f"\nCreated block: [{data['label']}]")
        output(state, f"  Description: {data['description']}")
        output(state, f"  Char limit: {data['char_limit']}")
        output(state, "")
    except httpx.HTTPStatusError as e:
        output_error(state, f"HTTP {e.response.status_code}: {e.response.text}")
    except httpx.RequestError as e:
        output_error(state, f"Request failed: {e}")


def cmd_help(state: CLIState) -> None:
    """Show help."""
    help_text = """
Commands (prefix with /):
    /create          Create a new agent (interactive config wizard)
    /create -q <n>   Create agent with defaults (quick mode)
    /use <agent_id>  Set active agent for subsequent commands
    /history         View message history for active agent
    /info            View agent info
    /memory          View core memory blocks (read-only)
    /newblock        Create a new memory block (interactive)
    /recompile       Trigger system prompt recompilation
    /help            Show this help
    /quit or /exit   Exit CLI

Default: Any text without / prefix is sent as a chat message.

Example:
    /create -q my-test-agent
    /use <paste-agent-id-here>
    Hello, how are you?
"""
    output(state, help_text)


# --- Main Loop ---

COMMANDS = {
    "create": cmd_create,
    "use": cmd_use,
    "history": cmd_history,
    "info": cmd_info,
    "memory": cmd_memory,
    "recompile": cmd_recompile,
    "newblock": cmd_newblock,
}


async def run_command(state: CLIState, client: httpx.AsyncClient, line: str) -> bool:
    """Parse and run a command. Returns False if should exit."""
    line = line.strip()
    if not line:
        return True
    
    # Commands start with /
    if line.startswith("/"):
        parts = line[1:].split(maxsplit=1)  # Strip the /
        cmd = parts[0].lower()
        args = parts[1].split() if len(parts) > 1 else []
        
        if cmd in ("quit", "exit"):
            return False
        
        if cmd == "help":
            cmd_help(state)
            return True
        
        handler = COMMANDS.get(cmd)
        if handler:
            await handler(state, client, args)
        else:
            output_error(state, f"Unknown command: /{cmd}. Type '/help' for available commands.")
    else:
        # Default: treat entire line as chat message
        await cmd_chat(state, client, [line])
    
    return True


async def main() -> None:
    """Main entry point."""
    parser = argparse.ArgumentParser(description="Agent Home CLI")
    parser.add_argument("--headless", action="store_true", help="Headless mode (no prompts, structured output)")
    parser.add_argument("--server", default=DEFAULT_SERVER_URL, help=f"Server URL (default: {DEFAULT_SERVER_URL})")
    parser.add_argument("--invoker", default=None, help="Invoker identity prepended to chat messages (e.g. 'Sonnet')")
    args = parser.parse_args()

    invoker = args.invoker or _resolve_invoker_from_env()

    state = CLIState(
        server_url=args.server,
        headless=args.headless,
        invoker=invoker,
    )
    
    if not state.headless:
        output(state, "Agent Home CLI")
        output(state, f"Server: {state.server_url}")
        output(state, "Type '/help' for commands, '/quit' to exit.\n")
    
    async with httpx.AsyncClient() as client:
        try:
            response = await client.get(f"{state.server_url}/health")
            response.raise_for_status()
        except Exception as e:
            output(state, f"Health check failed — is the server running at {state.server_url}?\n  {e}")
            sys.exit(1)

        while True:
            try:
                if state.headless:
                    line = input()
                else:
                    line = input(prompt(state))
                
                if not await run_command(state, client, line):
                    break
                    
            except EOFError:
                break
            except KeyboardInterrupt:
                if not state.headless:
                    output(state, "\nUse 'quit' to exit.")
    
    if not state.headless:
        output(state, "Goodbye!")


if __name__ == "__main__":
    asyncio.run(main())
