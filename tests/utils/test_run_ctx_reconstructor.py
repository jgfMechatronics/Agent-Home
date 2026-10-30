"""Tests for utils/run_ctx_reconstructor.py — runner wiring around reconstruct_context."""

from unittest.mock import create_autospec, patch

import pytest
from sqlalchemy.ext.asyncio import AsyncEngine

from db.models import MessageRecord
from utils.ctx_reconstructor import ReconstructedContext
from utils.run_ctx_reconstructor import run
from conftest import SAMPLE_AGENT_CONFIG

_MESSAGE_ID = "msg-123"
_TARGET_CONTENT = "target message content"


@pytest.fixture
def db_path(tmp_path):
    return tmp_path / "agent_home.sqlite"


@pytest.fixture
def output_path(tmp_path):
    return tmp_path / "reconstruction.txt"


@pytest.fixture
def mock_engine():
    return create_autospec(AsyncEngine, instance=True)


@pytest.fixture(autouse=True)
def mock_get_session(mock_engine):
    """Mock DB layer so no real SQLite engine is needed.

    Every patch uses autospec=True so a call that doesn't match the real signature (e.g. a
    removed kwarg like readonly=) raises TypeError here instead of passing silently.
    """
    with (
        patch("utils.run_ctx_reconstructor.create_readonly_sqlite_engine", autospec=True, return_value=mock_engine),
        patch("utils.run_ctx_reconstructor.get_session", autospec=True) as mock_get_session,
    ):
        yield mock_get_session


def _reconstructed_context() -> ReconstructedContext:
    """Real (unpersisted) context, so _format_reconstruction runs against the actual dataclass."""
    return ReconstructedContext(
        system_prompt="system prompt",
        tool_definitions=[],
        agent_config=SAMPLE_AGENT_CONFIG,
        messages=[MessageRecord(seq_id=1, type="request", content="earlier message")],
        target_message=MessageRecord(id=_MESSAGE_ID, seq_id=2, type="response", content=_TARGET_CONTENT),
        agent_id="agent-123",
    )


async def test_writes_reconstruction(db_path, output_path, mock_engine, mock_get_session):
    """Success: reconstructs from the session get_session yields, writes the report, disposes the engine."""
    session = mock_get_session.return_value.__aenter__.return_value
    with patch("utils.run_ctx_reconstructor.reconstruct_context", autospec=True,
               return_value=_reconstructed_context()) as mock_reconstruct:
        exit_code = await run(db_path, _MESSAGE_ID, output_path)

    assert exit_code == 0
    mock_get_session.assert_called_once_with(mock_engine)
    mock_reconstruct.assert_awaited_once_with(session, _MESSAGE_ID)
    output = output_path.read_text()
    assert "earlier message" in output
    assert _TARGET_CONTENT in output
    mock_engine.dispose.assert_awaited_once()


async def test_message_not_found(db_path, output_path, mock_engine):
    """reconstruct_context's ValueError maps to exit code 1, no output file, engine still disposed."""
    with patch("utils.run_ctx_reconstructor.reconstruct_context", autospec=True,
               side_effect=ValueError("Message not found")):
        exit_code = await run(db_path, _MESSAGE_ID, output_path)

    assert exit_code == 1
    assert not output_path.exists()
    mock_engine.dispose.assert_awaited_once()
