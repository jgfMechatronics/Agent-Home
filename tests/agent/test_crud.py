"""
Tests for agent CRUD operations (agent/crud.py)
"""
import uuid

import pytest_asyncio
from sqlalchemy.ext.asyncio import AsyncSession

from agent.crud import create_agent_record, get_agent_record, replace_agent_config, replace_system_instructions
from conftest import SAMPLE_AGENT_CONFIG
from messages.messages import load_messages


# --- create_agent_record tests ---

class TestCreateAgentRecord:
    """Tests share the same setup: one created record per test."""

    @pytest_asyncio.fixture(autouse=True)
    async def created_record(self, session: AsyncSession):
        self.session = session
        self.record = await create_agent_record(
            session, "test-agent", "You are helpful.", SAMPLE_AGENT_CONFIG
        )

    async def assert_record_as_expected(self, record_to_check):
        assert uuid.UUID(record_to_check.id)  # raises ValueError if not a valid UUID
        assert record_to_check.name == "test-agent"
        assert record_to_check.system_instructions == "You are helpful."
        assert record_to_check.agent_config == SAMPLE_AGENT_CONFIG
        assert record_to_check.context_window_start is None
        assert await load_messages(self.session, self.record.id) == []
        assert record_to_check.compiled_system_prompt is not None
        assert "You are helpful." in record_to_check.compiled_system_prompt

    async def test_returns_agent_record_with_correct_fields(self):
        """Returned record has expected values — UUID valid, fields match inputs, no history, expected config"""
        await self.assert_record_as_expected(self.record)

    async def test_config_and_data_survive_db_round_trip(self):
        """Record is actually flushed to DB and config deserializes correctly on re-fetch."""
        record_id = self.record.id  # capture before expiry — accessing after expire() triggers sync lazy load
        self.session.expire(self.record)  # invalidate identity map, force real DB read
        fetched = await get_agent_record(self.session, record_id)
        await self.assert_record_as_expected(fetched)


# --- get_agent_record tests ---

class TestGetAgentRecord:
    """Tests for get_agent_record."""

    async def test_returns_record_for_known_id(self, session: AsyncSession, agent_record):
        result = await get_agent_record(session, agent_record.id)
        assert result == agent_record

    async def test_returns_none_for_unknown_id(self, session: AsyncSession):
        result = await get_agent_record(session, "nonexistent-id")
        assert result is None
