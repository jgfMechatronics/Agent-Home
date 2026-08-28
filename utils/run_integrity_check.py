"""Production integrity checker runner.

Usage (from Agent-Home root):
    uv run python utils/run_integrity_check.py /path/to/agent_home.sqlite

Loads all agents from the database, runs the integrity checker on each,
and writes results to integrity_checker_results.txt in the same directory
as the database file. Overwrites any existing results file.

Dismissals are loaded from integrity_issue_dismissals.json in the same
directory as the database file, if it exists.

Exit codes:
    0 — no issues found (or all issues dismissed)
    1 — one or more issues found for any agent
"""

import argparse
import asyncio
import sys
from datetime import datetime, timezone
from pathlib import Path

from agent.crud import get_all_agents
from db.connection import create_sqlite_engine, get_session
from utils.integrity_checker import check_agent_integrity, load_dismissals, filter_dismissed_issues

_RESULTS_FILENAME = "integrity_checker_results.txt"
_DISMISSALS_FILENAME = "integrity_issue_dismissals.json"


async def run(db_path: Path) -> int:
    """Run integrity checks on all agents. Returns 1 if any issues found, else 0."""
    dismissals_path = db_path.parent / _DISMISSALS_FILENAME
    results_path = db_path.parent / _RESULTS_FILENAME

    engine = create_sqlite_engine(str(db_path.absolute()), readonly=True)

    try:
        lines: list[str] = []
        timestamp = datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M:%S UTC")
        lines.append(f"Integrity Check Results — {timestamp}")
        lines.append(f"Database: {db_path}")
        lines.append("=" * 60)

        found_issues = False
        
        async with get_session(engine) as session:
            agents = await get_all_agents(session)

            if not agents:
                lines.append("\nNo agents found in database.")
            else:
                for agent in agents:
                    lines.append(f"\nAgent: {agent.name} ({agent.id})")
                    lines.append("-" * 40)

                    issues = await check_agent_integrity(session, agent.id)
                    dismissals = load_dismissals(dismissals_path, agent.id)
                    issues = filter_dismissed_issues(issues, dismissals)

                    if not issues:
                        lines.append("  ✓ No issues found.")
                        continue

                    found_issues = True
                    for issue in issues:
                        lines.append(f"  {repr(issue)}")

    finally:
        await engine.dispose()

    lines.append("\n" + "=" * 60)
    lines.append("RESULT: ISSUES FOUND — inspect above." if found_issues else "RESULT: Clean.")

    output = "\n".join(lines) + "\n"
    print(output)
    results_path.write_text(output)
    print(f"Results written to {results_path}")

    return 1 if found_issues else 0


def main():
    parser = argparse.ArgumentParser(description="Run Agent Home integrity checker on a SQLite database.")
    parser.add_argument("db_path", type=Path, help="Path to the SQLite database file.")
    args = parser.parse_args()

    if not args.db_path.exists():
        print(f"Error: database file not found: {args.db_path}", file=sys.stderr)
        sys.exit(1)

    sys.exit(asyncio.run(run(args.db_path)))


if __name__ == "__main__":
    main()
