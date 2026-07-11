"""
Tests for agent CRUD operations (agent/crud.py)
"""
import uuid

import pytest
import pytest_asyncio
from sqlalchemy.ext.asyncio import AsyncSession

from agent.crud import create_agent_record, get_agent_record, replace_agent_config, replace_system_instructions
from agent.types import AgentNotFoundError
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


# --- replace function tests ---
#
# Common behaviors (not_found, commits) are tested via parametrization.
# Function-specific behaviors have their own test classes.

# Parametrization data for common replace-function behaviors
_REPLACE_FUNCTIONS = [
    pytest.param(
        replace_agent_config,
        SAMPLE_AGENT_CONFIG.model_copy(update={"soft_compaction_limit": 7777}),
        "agent_config",
        id="replace_agent_config",
    ),
    pytest.param(
        replace_system_instructions,
        "New instructions for test.",
        "system_instructions",
        id="replace_system_instructions",
    ),
]


@pytest.mark.parametrize("replace_fn,new_value,attr_name", _REPLACE_FUNCTIONS)
class TestReplaceFunctionCommonBehaviors:
    """Common behaviors shared by all replace_* functions."""

    async def test_raises_not_found_for_unknown_agent(self, session: AsyncSession, replace_fn, new_value, attr_name):
        """Replace functions raise AgentNotFoundError for unknown agent_id."""
        with pytest.raises(AgentNotFoundError):
            await replace_fn(session, "nonexistent-id", new_value)

    async def test_updates_db_and_returns_new_value(self, session: AsyncSession, agent_record, replace_fn, new_value, attr_name):
        """Replace functions update DB and return the new value."""
        agent_id = agent_record.id
        result = await replace_fn(session, agent_id, new_value)

        assert result == new_value
        session.expire(agent_record)
        refreshed = await get_agent_record(session, agent_id)
        assert getattr(refreshed, attr_name) == new_value

    async def test_commits_on_success(self, session: AsyncSession, agent_record, replace_fn, new_value, attr_name):
        """Replace functions commit changes (persist across new session)."""
        agent_id = agent_record.id
        await replace_fn(session, agent_id, new_value)

        async with AsyncSession(session.bind) as fresh_session:
            refreshed = await get_agent_record(fresh_session, agent_id)
            assert getattr(refreshed, attr_name) == new_value


class TestReplaceSystemInstructions:
    """Function-specific tests for replace_system_instructions."""

    async def test_triggers_recompilation(self, session: AsyncSession, agent_record):
        """System prompt is recompiled after updating instructions."""
        agent_id = agent_record.id
        new_instructions = "You are a helpful assistant who loves cats."

        await replace_system_instructions(session, agent_id, new_instructions)

        session.expire(agent_record)
        refreshed = await get_agent_record(session, agent_id)
        assert new_instructions in refreshed.compiled_system_prompt
