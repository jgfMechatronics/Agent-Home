#!/usr/bin/env python3
"""
Quick live test script for context reconstructor.
Outputs serialized ReconstructedContext for midpoint and end messages.

Usage: uv run python scripts/live_test_reconstructor.py [db_path]
Default db_path: ~/.agent-home/db.sqlite
"""
import asyncio
import dataclasses
import json
import os
import sys
from pathlib import Path

# Add project root to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent))

from db.connection import create_sqlite_engine, get_session
from db.models import AgentRecord, MessageRecord
from messages.messages import load_messages, deserialize_messages
from utils.ctx_reconstructor import reconstruct_context, ReconstructedContext
from sqlalchemy import select


def serialize_context(ctx: ReconstructedContext) -> str:
    """Serialize ReconstructedContext to readable text."""
    lines = [
        "=" * 80,
        f"RECONSTRUCTED CONTEXT FOR MESSAGE: {ctx.target_message.id}",
        f"Agent ID: {ctx.agent_id}",
        "=" * 80,
        "",
        "--- AGENT CONFIG ---",
        f"model_name: {ctx.agent_config.model_name}",
        f"thinking_enabled: {ctx.agent_config.thinking_enabled}",
        f"tool_names: {ctx.agent_config.tool_names}",
        f"soft_compaction_limit: {ctx.agent_config.soft_compaction_limit}",
        "",
        "--- SYSTEM PROMPT ---",
        ctx.system_prompt,
        "",
        "--- TOOL DEFINITIONS ---",
        f"Count: {len(ctx.tool_definitions)}",
    ]
    for td in ctx.tool_definitions:
        lines.append(f"\n  {td.name}:")
        lines.append(f"    description: {td.description}")
        lines.append(f"    parameters: {json.dumps(td.parameters_json_schema, indent=6)}")
    
    lines.extend([
        "",
        "--- MESSAGE HISTORY (prior to target) ---",
        f"Count: {len(ctx.messages)}",
    ])
    
    for msg in ctx.messages:
        content = json.loads(msg.content)
        lines.append(f"\n[seq={msg.seq_id}] {content.get('kind', 'unknown')}:")
        parts = content.get("parts", [])
        for part in parts:
            part_kind = part.get("part_kind", "unknown")
            if part_kind == "thinking":
                lines.append(f"    <thinking>{part.get('content', '')}</thinking>")
            elif part_kind in ("text", "user-prompt"):
                lines.append(f"    [{part_kind}] {part.get('content', '')}")
            elif part_kind == "tool-call":
                lines.append(f"    [tool-call: {part.get('tool_name', '?')}] args={part.get('args', {})}")
            elif part_kind == "tool-return":
                lines.append(f"    [tool-return: {part.get('tool_call_id', '')[:20]}] {part.get('content', '')[:500]}")
            else:
                lines.append(f"    [{part_kind}] {part}")
    
    lines.extend([
        "",
        "--- TARGET MESSAGE ---",
        f"seq_id: {ctx.target_message.seq_id}",
    ])
    target_content = json.loads(ctx.target_message.content)
    lines.append(f"kind: {target_content.get('kind', 'unknown')}")
    parts = target_content.get("parts", [])
    for part in parts:
        part_kind = part.get("part_kind", "unknown")
        if part_kind == "thinking":
            lines.append(f"<thinking>{part.get('content', '')}</thinking>")
        elif part_kind in ("text", "user-prompt"):
            lines.append(f"[{part_kind}] {part.get('content', '')}")
        elif part_kind == "tool-call":
            lines.append(f"[tool-call: {part.get('tool_name', '?')}] args={part.get('args', {})}")
        elif part_kind == "tool-return":
            lines.append(f"[tool-return: {part.get('tool_call_id', '')[:20]}] {part.get('content', '')}")
        else:
            lines.append(f"[{part_kind}] {part}")
    
    return "\n".join(lines)


async def main():
    db_path = sys.argv[1] if len(sys.argv) > 1 else os.path.expanduser("~/.agent-home/db.sqlite")
    output_dir = Path("/workspace/git/Agent-Home/scripts/reconstructor_output")
    output_dir.mkdir(exist_ok=True)
    
    print(f"Using DB: {db_path}")
    
    if not os.path.exists(db_path):
        print(f"ERROR: DB not found at {db_path}")
        sys.exit(1)
    
    engine = create_sqlite_engine(db_path)
    
    async with get_session(engine) as session:
        # Find all agents
        result = await session.execute(select(AgentRecord))
        agents = list(result.scalars().all())
        
        if not agents:
            print("ERROR: No agents found in DB")
            sys.exit(1)
        
        if len(agents) > 1:
            print(f"WARNING: Found {len(agents)} agents, using first one")
        
        agent = agents[0]
        print(f"Agent: {agent.name} ({agent.id})")
        
        # Load all messages
        messages = await load_messages(session, agent.id)
        print(f"Total messages: {len(messages)}")
        
        if len(messages) < 2:
            print("ERROR: Need at least 2 messages for midpoint/end test")
            sys.exit(1)
        
        # Get midpoint and end message IDs
        midpoint_idx = len(messages) // 2
        midpoint_msg = messages[midpoint_idx]
        end_msg = messages[-1]
        
        print(f"Midpoint message: seq_id={midpoint_msg.seq_id}, id={midpoint_msg.id}")
        print(f"End message: seq_id={end_msg.seq_id}, id={end_msg.id}")
        
        # Reconstruct for midpoint
        print("\nReconstructing midpoint context...")
        midpoint_ctx = await reconstruct_context(session, midpoint_msg.id)
        midpoint_output = serialize_context(midpoint_ctx)
        
        midpoint_file = output_dir / "midpoint_context.txt"
        midpoint_file.write_text(midpoint_output)
        print(f"Wrote: {midpoint_file}")
        
        # Reconstruct for end
        print("Reconstructing end context...")
        end_ctx = await reconstruct_context(session, end_msg.id)
        end_output = serialize_context(end_ctx)
        
        end_file = output_dir / "end_context.txt"
        end_file.write_text(end_output)
        print(f"Wrote: {end_file}")
        
        print("\nDone! Compare these outputs against the logged raw requests.")


if __name__ == "__main__":
    asyncio.run(main())
