"""Context reconstructor runner — spot-check context reconstruction from production DBs.

Usage (from Agent-Home root):
    uv run python utils/run_ctx_reconstructor.py /path/to/agent_home.sqlite <message_uuid> --output /path/to/output.txt

Reconstructs the context that existed when a specific message was processed, using
the same snapshots that were active at that time (system prompt, tool definitions,
agent config). Output is written to the specified file.

Exit codes:
    0 — reconstruction successful
    1 — error (message not found, etc.)
"""

import argparse
import asyncio
import sys
from pathlib import Path

from db.connection import create_readonly_sqlite_engine, get_session
from utils.ctx_reconstructor import reconstruct_context, ReconstructedContext


def _format_reconstruction(ctx: ReconstructedContext) -> str:
    """Format reconstructed context for file output."""
    lines = []
    lines.append("=" * 70)
    lines.append("RECONSTRUCTED CONTEXT")
    lines.append("=" * 70)
    
    lines.append(f"\nAgent ID: {ctx.agent_id}")
    lines.append(f"\nTarget Message:")
    lines.append(repr(ctx.target_message))
    
    lines.append(f"\n\n{'=' * 70}")
    lines.append("AGENT CONFIG")
    lines.append("=" * 70)
    lines.append(repr(ctx.agent_config))
    
    lines.append(f"\n\n{'=' * 70}")
    lines.append(f"SYSTEM PROMPT ({len(ctx.system_prompt)} chars)")
    lines.append("=" * 70)
    lines.append(ctx.system_prompt)
    
    lines.append(f"\n\n{'=' * 70}")
    lines.append(f"TOOL DEFINITIONS ({len(ctx.tool_definitions)} tools)")
    lines.append("=" * 70)
    for td in ctx.tool_definitions:
        lines.append(repr(td))
    
    lines.append(f"\n\n{'=' * 70}")
    lines.append(f"MESSAGE HISTORY ({len(ctx.messages)} messages before target)")
    lines.append("=" * 70)
    for msg in ctx.messages:
        lines.append(f"\n--- seq_id {msg.seq_id} ({msg.type}) ---")
        lines.append(msg.content)
    
    lines.append(f"\n\n{'=' * 70}")
    lines.append("TARGET MESSAGE CONTENT")
    lines.append("=" * 70)
    lines.append(ctx.target_message.content)
    
    lines.append("\n" + "=" * 70)
    return "\n".join(lines)


async def run(db_path: Path, message_id: str, output_path: Path) -> int:
    """Run context reconstruction. Returns 0 on success, 1 on error."""
    
    engine = create_readonly_sqlite_engine(str(db_path.absolute()))
    
    try:
        async with get_session(engine) as session:
            try:
                ctx = await reconstruct_context(session, message_id)
            except ValueError as e:
                print(f"Error: {e}", file=sys.stderr)
                return 1
            
            output = _format_reconstruction(ctx)
            output_path.write_text(output)
            print(f"Reconstruction written to: {output_path}")
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
    parser.add_argument("message_id", help="Message UUID to reconstruct context for")
    parser.add_argument(
        "--output", "-o",
        type=Path,
        required=True,
        help="Output file path",
    )
    
    args = parser.parse_args()
    
    # Validation
    if not args.db_path.exists():
        print(f"Error: Database not found: {args.db_path}", file=sys.stderr)
        return 1
    
    return asyncio.run(run(args.db_path, args.message_id, args.output))


if __name__ == "__main__":
    sys.exit(main())
