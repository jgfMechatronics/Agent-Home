"""Tests for utils/run_ctx_reconstructor.py — runner end to end against a real DB file.

No mocks: run() gets the test_engine's DB path and builds its own mode=ro engine and session,
exactly as in prod. Seeded rows must be committed (not just flushed) to be visible to it.
"""

from pathlib import Path

import pytest

from db.models import MessageRecord, utcnow
from utils.run_ctx_reconstructor import run
from conftest import PARTIAL_MESSAGE_FIELDS, STUB_CTX_MSG_ID

_TARGET_ID = "00000000-0000-0000-0000-000000000002"
_EARLIER_CONTENT = "earlier message content"
_TARGET_CONTENT = "target message content"


@pytest.fixture
def db_path(test_engine) -> Path:
    return Path(test_engine.url.database)


@pytest.fixture
def output_path(tmp_path) -> Path:
    return tmp_path / "reconstruction.txt"


async def test_writes_reconstruction(db_path, output_path, session, agent_record, seed_stub_snapshots):
    """Success: the report contains the history before the target and the target itself."""
    session.add_all([
        MessageRecord(**{**PARTIAL_MESSAGE_FIELDS, "content": _EARLIER_CONTENT},
                      id=STUB_CTX_MSG_ID, agent_id=agent_record.id, seq_id=0, timestamp=utcnow()),
        MessageRecord(**{**PARTIAL_MESSAGE_FIELDS, "type": "ModelResponse", "content": _TARGET_CONTENT},
                      id=_TARGET_ID, agent_id=agent_record.id, seq_id=1, timestamp=utcnow()),
    ])
    await session.commit()

    exit_code = await run(db_path, _TARGET_ID, output_path)

    assert exit_code == 0
    output = output_path.read_text()
    assert _EARLIER_CONTENT in output
    assert _TARGET_CONTENT in output


async def test_message_not_found(db_path, output_path):
    """Unknown message id: exit code 1 and no output file."""
    exit_code = await run(db_path, _TARGET_ID, output_path)

    assert exit_code == 1
    assert not output_path.exists()
