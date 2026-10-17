"""Context reconstructor runner — spot-check context reconstruction from production DBs.

Usage (from Agent-Home root):
    # By message UUID
    uv run python utils/run_ctx_reconstructor.py /path/to/agent_home.sqlite --message-id <uuid>

    # By agent name + most recent message
    uv run python utils/run_ctx_reconstructor.py /path/to/agent_home.sqlite --agent Opus --last

    # By agent name + seq_id
    uv run python utils/run_ctx_reconstructor.py /path/to/agent_home.sqlite --agent Opus --seq-id 500

    # By agent name + offset from end (e.g., 5 messages back)
    uv run python utils/run_ctx_reconstructor.py /path/to/agent_home.sqlite --agent Opus --offset 5

Reconstructs the context that existed when a specific message was processed, using
the same snapshots that were active at that time (system prompt, tool definitions,
agent config).

Exit codes:
    0 — reconstruction successful
    1 — error (message not found, agent not found, etc.)
"""

import argparse
import asyncio
import sys
from pathlib import Path

from sqlalchemy import select, func

from agent.crud import get_agent_by_name
from db.connection import create_sqlite_engine, get_session
from db.models import AgentRecord, MessageRecord
from utils.ctx_reconstructor import reconstruct_context, ReconstructedContext


def _format_tool_definitions(ctx: ReconstructedContext) -> str:
    """Format tool definitions for display."""
    if not ctx.tool_definitions:
        return "  (none)"
    lines = []
    for td in ctx.tool_definitions:
        lines.append(f"  - {td.name}: {td.description[:80]}..." if len(td.description) > 80 else f"  - {td.name}: {td.description}")
    return "\n".join(lines)


def _format_messages_summary(ctx: ReconstructedContext) -> str:
    """Format message history summary."""
    if not ctx.messages:
        return "  (empty — target is context window start)"
    
    lines = []
    for msg in ctx.messages:
        # Truncate content for display
        content_preview = msg.content[:100].replace("\n", " ")
        if len(msg.content) > 100:
            content_preview += "..."
        lines.append(f"  [{msg.seq_id}] {msg.type}: {content_preview}")
    return "\n".join(lines)


def _print_reconstruction(ctx: ReconstructedContext) -> None:
    """Print reconstructed context to stdout."""
    print("=" * 70)
    print("RECONSTRUCTED CONTEXT")
    print("=" * 70)
    
    print(f"\nAgent ID: {ctx.agent_id}")
    print(f"Target Message ID: {ctx.target_message.id}")
    print(f"Target Message seq_id: {ctx.target_message.seq_id}")
    print(f"Target Message type: {ctx.target_message.type}")
    
    print(f"\n--- Agent Config ---")
    print(f"  model_name: {ctx.agent_config.model_name}")
    print(f"  thinking_mode: {ctx.agent_config.thinking_mode}")
    print(f"  tool_names: {ctx.agent_config.tool_names}")
    print(f"  soft_compaction_limit: {ctx.agent_config.soft_compaction_limit}")
    
    print(f"\n--- System Prompt ({len(ctx.system_prompt)} chars) ---")
    # Show first 500 chars
    if len(ctx.system_prompt) > 500:
        print(ctx.system_prompt[:500])
        print(f"  ... ({len(ctx.system_prompt) - 500} more chars)")
    else:
        print(ctx.system_prompt)
    
    print(f"\n--- Tool Definitions ({len(ctx.tool_definitions)} tools) ---")
    print(_format_tool_definitions(ctx))
    
    print(f"\n--- Message History ({len(ctx.messages)} messages before target) ---")
    print(_format_messages_summary(ctx))
    
    print(f"\n--- Target Message Content ---")
    print(ctx.target_message.content)
    
    print("\n" + "=" * 70)


async def _resolve_message_id(
    session,
    message_id: str | None,
    agent: str | None,
    seq_id: int | None,
    last: bool,
    offset: int | None,
) -> str:
    """Resolve CLI args to a message UUID."""
    
    # Direct message ID — just validate it exists
    if message_id:
        result = await session.execute(
            select(MessageRecord.id).where(MessageRecord.id == message_id)
        )
        if result.scalar_one_or_none() is None:
            raise ValueError(f"Message not found: {message_id}")
        return message_id
    
    # Agent-based lookup
    if not agent:
        raise ValueError("Must specify either --message-id or --agent")
    
    # Try to find agent by name first, then by ID
    agent_record = await get_agent_by_name(session, agent)
    if agent_record is None:
        # Try as UUID
        result = await session.execute(
            select(AgentRecord).where(AgentRecord.id == agent)
        )
        agent_record = result.scalar_one_or_none()
    
    if agent_record is None:
        raise ValueError(f"Agent not found: {agent}")
    
    agent_id = agent_record.id
    
    # Get max seq_id for this agent
    result = await session.execute(
        select(func.max(MessageRecord.seq_id)).where(MessageRecord.agent_id == agent_id)
    )
    max_seq = result.scalar()
    
    if max_seq is None:
        raise ValueError(f"No messages found for agent: {agent}")
    
    # Determine target seq_id
    if seq_id is not None:
        target_seq = seq_id
    elif offset is not None:
        target_seq = max_seq - offset
        if target_seq < 0:
            raise ValueError(f"Offset {offset} exceeds message count (max seq_id: {max_seq})")
    else:
        # Default to last (--last flag or implicit)
        target_seq = max_seq
    
    # Look up message by seq_id
    result = await session.execute(
        select(MessageRecord.id)
        .where(MessageRecord.agent_id == agent_id)
        .where(MessageRecord.seq_id == target_seq)
    )
    msg_id = result.scalar_one_or_none()
    
    if msg_id is None:
        raise ValueError(f"No message found at seq_id {target_seq} for agent {agent}")
    
    return msg_id


async def run(
    db_path: Path,
    message_id: str | None,
    agent: str | None,
    seq_id: int | None,
    last: bool,
    offset: int | None,
) -> int:
    """Run context reconstruction. Returns 0 on success, 1 on error."""
    
    engine = create_sqlite_engine(str(db_path.absolute()), readonly=True)
    
    try:
        async with get_session(engine) as session:
            try:
                target_msg_id = await _resolve_message_id(
                    session, message_id, agent, seq_id, last, offset
                )
            except ValueError as e:
                print(f"Error: {e}", file=sys.stderr)
                return 1
            
            ctx = await reconstruct_context(session, target_msg_id)
            _print_reconstruction(ctx)
            return 0
    finally:
        await engine.dispose()


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Reconstruct context from a production database.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    parser.add_argument("db_path", type=Path, help="Path to agent_home.sqlite")
    
    # Message selection — either direct ID or agent-based
    msg_group = parser.add_argument_group("message selection")
    msg_group.add_argument(
        "--message-id", "-m",
        help="Direct message UUID to reconstruct",
    )
    msg_group.add_argument(
        "--agent", "-a",
        help="Agent name or ID (use with --seq-id, --last, or --offset)",
    )
    msg_group.add_argument(
        "--seq-id", "-s",
        type=int,
        help="Message seq_id (requires --agent)",
    )
    msg_group.add_argument(
        "--last", "-l",
        action="store_true",
        help="Use most recent message (requires --agent, this is the default)",
    )
    msg_group.add_argument(
        "--offset", "-o",
        type=int,
        help="Offset from most recent message, e.g. -o 5 for 5 messages back (requires --agent)",
    )
    
    args = parser.parse_args()
    
    # Validation
    if not args.db_path.exists():
        print(f"Error: Database not found: {args.db_path}", file=sys.stderr)
        return 1
    
    if not args.message_id and not args.agent:
        print("Error: Must specify either --message-id or --agent", file=sys.stderr)
        return 1
    
    if args.message_id and (args.agent or args.seq_id or args.last or args.offset):
        print("Error: --message-id cannot be combined with agent-based options", file=sys.stderr)
        return 1
    
    if (args.seq_id or args.offset) and not args.agent:
        print("Error: --seq-id and --offset require --agent", file=sys.stderr)
        return 1
    
    if args.seq_id is not None and args.offset is not None:
        print("Error: Cannot specify both --seq-id and --offset", file=sys.stderr)
        return 1
    
    return asyncio.run(run(
        args.db_path,
        args.message_id,
        args.agent,
        args.seq_id,
        args.last,
        args.offset,
    ))


if __name__ == "__main__":
    sys.exit(main())
