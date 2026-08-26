"""Live integrity checker test — run against real database files and write results.

Usage (from Agent-Home root):
    uv run python tests/utils/integrity_checker_live_test/run.py
"""

import asyncio
import sys
from pathlib import Path

# Ensure Agent-Home root is on the path for module imports
sys.path.insert(0, str(Path(__file__).parent.parent.parent.parent))

from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession, create_async_engine
from sqlalchemy.orm import load_only, sessionmaker

from db.models import AgentRecord
from utils.integrity_checker import check_agent_integrity

_DIR = Path(__file__).parent
_DB_FILES = [
    _DIR / "db.sqlite.good",
    _DIR / "db.sqlite.verybad",
]
_OUTPUT_FILE = _DIR / "results.txt"


async def _check_db(db_path: Path) -> str:
    # Read-only engine — diagnostic tool, never mutates the DB
    engine = create_async_engine(f"sqlite+aiosqlite:///file:{db_path.absolute()}?mode=ro&uri=true")
    session_factory = sessionmaker(engine, class_=AsyncSession, expire_on_commit=False)

    lines = [f"{'=' * 60}", f"  {db_path.name}", f"{'=' * 60}"]

    async with session_factory() as session:
        # load_only fetches just id + name, skipping the config column's TypeDecorator
        # which can fail on DBs with older schema versions
        stmt = select(AgentRecord).options(load_only(AgentRecord.id, AgentRecord.name))
        records = (await session.execute(stmt)).scalars().all()

        if not records:
            lines.append("  (no agents found)")
        else:
            for record in records:
                lines.append(f"\n  Agent: {record.name}  ({record.id})")
                issues = await check_agent_integrity(session, record.id)
                if not issues:
                    lines.append("  ✓  No issues found")
                else:
                    for issue in issues:
                        lines.append(
                            f"  [{issue.severity.value.upper():8s}]  "
                            f"{issue.check_type}  seq_ids={issue.seq_ids}"
                        )
                        lines.append(f"             {issue.details}")

    await engine.dispose()
    return "\n".join(lines)


async def main() -> None:
    sections = []
    for db_path in _DB_FILES:
        print(f"Checking {db_path.name} ...")
        section = await _check_db(db_path)
        sections.append(section)

    output = "\n\n".join(sections) + "\n"
    print("\n" + output)

    _OUTPUT_FILE.write_text(output)
    print(f"Results written to {_OUTPUT_FILE.relative_to(_DIR.parent.parent.parent)}")


if __name__ == "__main__":
    asyncio.run(main())
