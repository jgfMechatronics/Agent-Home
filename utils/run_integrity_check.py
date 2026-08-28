"""Production integrity checker runner.

Usage:
    python utils/run_integrity_check.py /path/to/agent_home.sqlite

Loads all agents from the database, runs the integrity checker on each,
and writes results to integrity_checker_results.txt in the current directory.

Exit codes:
    0 — no errors or warnings found
    1 — one or more ERROR or CRITICAL issues found across any agent
"""

import argparse
import asyncio
import sys
from datetime import datetime, timezone
from pathlib import Path

from sqlalchemy.ext.asyncio import AsyncSession, create_async_engine
from sqlalchemy.orm import sessionmaker

from agent.crud import get_all_agents
from utils.integrity_checker import check_agent_integrity, Severity

_RESULTS_FILE = Path("integrity_checker_results.txt")
_ERROR_SEVERITIES = {Severity.ERROR, Severity.CRITICAL}


def _build_readonly_engine(db_path: Path):
    uri = f"sqlite+aiosqlite:///file:{db_path}?mode=ro&uri=true"
    return create_async_engine(uri, echo=False)


def _format_issue(issue) -> str:
    seq_id_str = f"seq_ids={issue.seq_ids}" if issue.seq_ids else ""
    parts = [f"  [{issue.severity.value.upper()}] {issue.check_type}"]
    if seq_id_str:
        parts.append(f"    {seq_id_str}")
    if issue.details:
        parts.append(f"    {issue.details}")
    return "\n".join(parts)

async def run(db_path: Path) -> int:
    """Run integrity checks on all agents. Returns 1 if any errors found, else 0."""
    engine = _build_readonly_engine(db_path)
    async_session = sessionmaker(engine, class_=AsyncSession, expire_on_commit=False)

    lines: list[str] = []
    timestamp = datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M:%S UTC")
    lines.append(f"Integrity Check Results — {timestamp}")
    lines.append(f"Database: {db_path}")
    lines.append("=" * 60)

    found_errors = False

    try:
        async with async_session() as session:
            agents = await get_all_agents(session)

            if not agents:
                lines.append("\nNo agents found in database.")
            else:
                for agent in agents:
                    lines.append(f"\nAgent: {agent.name} ({agent.id})")
                    lines.append("-" * 40)

                    issues = await check_agent_integrity(session, agent.id)

                    if not issues:
                        lines.append("  ✓ No issues found.")
                        continue

                    for issue in issues:
                        lines.append(_format_issue(issue))
                        if issue.severity in _ERROR_SEVERITIES:
                            found_errors = True

    finally:
        await engine.dispose()

    lines.append("\n" + "=" * 60)
    lines.append("RESULT: ERRORS FOUND — inspect above." if found_errors else "RESULT: Clean.")

    output = "\n".join(lines) + "\n"
    print(output)
    _RESULTS_FILE.write_text(output)
    print(f"Results written to {_RESULTS_FILE.resolve()}")

    return 1 if found_errors else 0


def main():
    parser = argparse.ArgumentParser(description="Run Agent Home integrity checker on a SQLite database.")
    parser.add_argument("db_path", type=Path, help="Path to the SQLite database file.")
    args = parser.parse_args()

    if not args.db_path.exists():
        print(f"Error: database file not found: {args.db_path}", file=sys.stderr)
        sys.exit(1)

    exit_code = asyncio.run(run(args.db_path))
    sys.exit(exit_code)


if __name__ == "__main__":
    main()
