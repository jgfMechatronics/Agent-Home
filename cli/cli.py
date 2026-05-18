#!/usr/bin/env python3
"""
Throwaway CLI for live testing Agent Home server.

Usage:
    python cli.py                    # Interactive mode
    python cli.py --headless         # Headless mode (for agent use)
    
Commands:
    create <name>    Create a new agent
    use <agent_id>   Set active agent for subsequent commands
    chat <message>   Send message to active agent (streaming)
    history          View message history for active agent
    info             View agent info
    memory           View core memory blocks (read-only)
    help             Show this help
    quit / exit      Exit CLI
"""

import argparse
import asyncio
import json
import sys
from dataclasses import dataclass

import httpx


# --- Configuration ---

DEFAULT_SERVER_URL = "http://localhost:8000"
DEFAULT_MODEL = "claude-sonnet-4-20250514"
DEFAULT_SOFT_COMPACTION_LIMIT = 80000


@dataclass
class CLIState:
    """Mutable state for the CLI session."""
    active_agent_id: str | None = None
    server_url: str = DEFAULT_SERVER_URL
    headless: bool = False


# --- Default Agent Config ---

def default_agent_config() -> dict:
    """Return default AgentConfig for new agents."""
    return {
        "model_name": DEFAULT_MODEL,
        "tool_names": [],  # No tools for basic testing
        "soft_compaction_limit": DEFAULT_SOFT_COMPACTION_LIMIT,
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

async def cmd_create(state: CLIState, client: httpx.AsyncClient, args: list[str]) -> None:
    """Create a new agent."""
    if not args:
        output_error(state, "Usage: create <name>")
        return
    
    name = " ".join(args)
    payload = {
        "name": name,
        "system_instructions": "You are a helpful assistant.",
        "config": default_agent_config(),
    }
    
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
            output(state, f"\nUse 'use {data['id']}' to start chatting.")
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
        output_error(state, "No active agent. Use 'use <agent_id>' first.")
        return
    
    if not args:
        output_error(state, "Usage: chat <message>")
        return
    
    message = " ".join(args)
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
    
    if not state.headless:
        output(state, "\nAssistant: ", end="")
    
    async for line in response.aiter_lines():
        line = line.strip()
        
        if line.startswith("event:"):
            current_event_type = line[6:].strip()
        elif line.startswith("data:"):
            current_data_lines.append(line[5:].strip())
        elif line == "" and current_event_type:
            # End of event - process it
            data_str = "\n".join(current_data_lines)
            await process_sse_event(state, current_event_type, data_str)
            current_event_type = None
            current_data_lines = []
    
    if not state.headless:
        output(state, "")  # Final newline


async def process_sse_event(state: CLIState, event_type: str, data_str: str) -> None:
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
    if event_type == "PartDeltaEvent":
        # Typewriter effect - print text as it arrives
        # Structure: {"delta": {"content_delta": "text"}}
        delta = data.get("delta", {})
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
        # Stream complete
        pass
    elif event_type == "Error":
        output(state, f"\n[Error: {data.get('message', 'Unknown error')}]")


async def cmd_history(state: CLIState, client: httpx.AsyncClient, args: list[str]) -> None:
    """View message history."""
    if not state.active_agent_id:
        output_error(state, "No active agent. Use 'use <agent_id>' first.")
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
                    kind = msg.get("kind", "unknown")
                    parts = msg.get("parts", [])
                    
                    # Extract text content from parts
                    content = ""
                    for part in parts:
                        part_kind = part.get("part_kind", "")
                        if part_kind in ("user-prompt", "text"):
                            text = part.get("content", "")
                            # User prompt content might be JSON-quoted
                            if text.startswith('"') and text.endswith('"'):
                                try:
                                    import json
                                    text = json.loads(text)
                                except:
                                    pass
                            content += text
                    
                    # Format nicely
                    role = "User" if kind == "request" else "Assistant"
                    output(state, f"**{role}:**")
                    output(state, content)
                    output(state, "")  # blank line between messages
                output(state, "--- End ---\n")
    except httpx.HTTPStatusError as e:
        output_error(state, f"HTTP {e.response.status_code}: {e.response.text}")
    except httpx.RequestError as e:
        output_error(state, f"Request failed: {e}")


async def cmd_info(state: CLIState, client: httpx.AsyncClient, args: list[str]) -> None:
    """View agent info."""
    if not state.active_agent_id:
        output_error(state, "No active agent. Use 'use <agent_id>' first.")
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
        output_error(state, "No active agent. Use 'use <agent_id>' first.")
        return
    
    try:
        response = await client.get(f"{state.server_url}/agents/{state.active_agent_id}/core_memory")
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


def cmd_help(state: CLIState) -> None:
    """Show help."""
    help_text = """
Commands:
    create <name>    Create a new agent
    use <agent_id>   Set active agent for subsequent commands
    chat <message>   Send message to active agent (streaming)
    history          View message history for active agent
    info             View agent info
    memory           View core memory blocks (read-only)
    help             Show this help
    quit / exit      Exit CLI

Example:
    create my-test-agent
    use <paste-agent-id-here>
    chat Hello, how are you?
"""
    output(state, help_text)


# --- Main Loop ---

COMMANDS = {
    "create": cmd_create,
    "use": cmd_use,
    "chat": cmd_chat,
    "history": cmd_history,
    "info": cmd_info,
    "memory": cmd_memory,
}


async def run_command(state: CLIState, client: httpx.AsyncClient, line: str) -> bool:
    """Parse and run a command. Returns False if should exit."""
    line = line.strip()
    if not line:
        return True
    
    parts = line.split(maxsplit=1)
    cmd = parts[0].lower()
    args = parts[1].split() if len(parts) > 1 else []
    # For chat, preserve the full message
    if cmd == "chat" and len(parts) > 1:
        args = [parts[1]]
    
    if cmd in ("quit", "exit"):
        return False
    
    if cmd == "help":
        cmd_help(state)
        return True
    
    handler = COMMANDS.get(cmd)
    if handler:
        await handler(state, client, args)
    else:
        output_error(state, f"Unknown command: {cmd}. Type 'help' for available commands.")
    
    return True


async def main() -> None:
    """Main entry point."""
    parser = argparse.ArgumentParser(description="Agent Home CLI")
    parser.add_argument("--headless", action="store_true", help="Headless mode (no prompts, structured output)")
    parser.add_argument("--server", default=DEFAULT_SERVER_URL, help=f"Server URL (default: {DEFAULT_SERVER_URL})")
    args = parser.parse_args()
    
    state = CLIState(
        server_url=args.server,
        headless=args.headless,
    )
    
    if not state.headless:
        output(state, "Agent Home CLI")
        output(state, f"Server: {state.server_url}")
        output(state, "Type 'help' for commands, 'quit' to exit.\n")
    
    async with httpx.AsyncClient() as client:
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
