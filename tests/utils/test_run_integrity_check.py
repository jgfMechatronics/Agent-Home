"""Tests for utils/run_integrity_check.py — lockfile creation behavior."""

from unittest.mock import MagicMock, create_autospec, patch

import pytest
from sqlalchemy.ext.asyncio import AsyncEngine

from utils.integrity_checker import INTEGRITY_LOCKFILE_NAME, IntegrityIssue, Severity
from utils.run_integrity_check import run

_MOCK_AGENT_ID = "agent-123"
_MOCK_AGENT_NAME = "test-agent"

_AN_ISSUE = IntegrityIssue(
    check_type="seq_id_gap", severity=Severity.ERROR, seq_ids=[5], details="Gap detected"
)


@pytest.fixture
def mock_agent(tmp_path):
    agent = MagicMock()
    agent.id = _MOCK_AGENT_ID
    agent.name = _MOCK_AGENT_NAME
    return agent


@pytest.fixture
def db_path(tmp_path):
    return tmp_path / "agent_home.sqlite"


@pytest.fixture(autouse=True)
def mock_integrity_runner_deps(db_path, mock_agent):
    """Mock DB layer so no real SQLite engine is needed.

    Every patch uses autospec=True so a call that doesn't match the real signature (e.g. a
    removed kwarg like readonly=) raises TypeError here instead of passing silently.
    """
    with (
        patch("utils.run_integrity_check.create_readonly_sqlite_engine", autospec=True,
              return_value=create_autospec(AsyncEngine, instance=True)),
        patch("utils.run_integrity_check.get_session", autospec=True),
        patch("utils.run_integrity_check.get_all_agents", autospec=True, return_value=[mock_agent]),
        patch("utils.run_integrity_check.load_dismissals", autospec=True, return_value=[]),
    ):
        yield


@pytest.mark.parametrize("issues,expect_lockfile,expected_exit_code", [
    ([_AN_ISSUE], True,  1),
    ([],          False, 0),
])
async def test_lockfile_behavior(db_path, issues, expect_lockfile, expected_exit_code):
    """Lockfile is created when issues found, absent when clean.

    check_agent_integrity and filter_dismissed_issues are patched here rather than in the
    fixture because the return value depends on the parametrized `issues` argument.
    """
    with patch("utils.run_integrity_check.check_agent_integrity", autospec=True, return_value=issues):
        with patch("utils.run_integrity_check.filter_dismissed_issues", autospec=True, side_effect=lambda found_issues, _dismissals: found_issues):
            exit_code = await run(db_path)

    lockfile = db_path.parent / INTEGRITY_LOCKFILE_NAME
    assert lockfile.exists() == expect_lockfile
    assert exit_code == expected_exit_code
