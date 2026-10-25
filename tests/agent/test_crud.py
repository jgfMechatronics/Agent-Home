"""
Tests for agent CRUD operations (agent/crud.py)
"""
import uuid

import pytest
import pytest_asyncio
from sqlalchemy.ext.asyncio import AsyncSession

from agent.crud import create_agent_record, get_agent_record, get_agent_record_by_name, replace_agent_config, replace_system_instructions
from agent.deps import AgentDeps
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
        assert record_to_check.context_window_start == 0
        assert await load_messages(self.session, self.record.id) == []

    async def test_returns_agent_record_with_correct_fields(self):
        """Returned record has expected values — UUID valid, fields match inputs, no history, expected config"""
        await self.assert_record_as_expected(self.record)

    async def test_config_and_data_survive_db_round_trip(self):
        """Record is actually flushed to DB and config deserializes correctly on re-fetch."""
        record_id = self.record.id  # capture before expiry — accessing after expire() triggers sync lazy load
        self.session.expire(self.record)  # invalidate identity map, force real DB read
        fetched = await get_agent_record(self.session, record_id)
        await self.assert_record_as_expected(fetched)


# --- get_agent_record / get_agent_record_by_name tests ---
#
# Both functions return AgentRecord | None based on a lookup key.
# Parametrized to test both via the same test logic.

_GET_AGENT_FUNCTIONS = [
    pytest.param(
        get_agent_record,
        lambda r: r.id,
        "nonexistent-id",
        id="by_id",
    ),
    pytest.param(
        get_agent_record_by_name,
        lambda r: r.name,
        "nonexistent-name",
        id="by_name",
    ),
]


@pytest.mark.parametrize("get_fn,key_extractor,unknown_key", _GET_AGENT_FUNCTIONS)
class TestGetAgentRecordFunctions:
    """Tests for get_agent_record and get_agent_record_by_name."""

    async def test_returns_record_for_known_key(self, session: AsyncSession, agent_record, get_fn, key_extractor, unknown_key):
        result = await get_fn(session, key_extractor(agent_record))
        assert result.id == agent_record.id

    async def test_returns_none_for_unknown_key(self, session: AsyncSession, get_fn, key_extractor, unknown_key):
        result = await get_fn(session, unknown_key)
        assert result is None


# --- replace function tests ---
#
# Common behaviors (commits, round-trip) are tested via parametrization.
# Function-specific behaviors have their own test classes.
#
# Not-found guarantee: replace_* functions take AgentDeps, which proves the agent exists.
# AgentNotFoundError is raised by AgentFactory.build_deps before deps are ever constructed —
# tested in tests/agent/test_factory.py, not here.


# Parametrization data for common replace-function behaviors
# param args: "replace_fn,new_value,attr_name"
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

    async def test_updates_db_and_returns_new_value(
        self, session: AsyncSession, agent_record, agent_deps: AgentDeps, replace_fn, new_value, attr_name
    ):
        """Replace functions update DB and return the new value."""
        agent_id = agent_record.id  # capture before expire — post-expire attr access triggers async lazy load
        result = await replace_fn(agent_deps, new_value)

        assert result == new_value
        # agent_record is detached (conftest expunges) — get_agent_record fetches fresh
        refreshed = await get_agent_record(session, agent_id)
        assert getattr(refreshed, attr_name) == new_value


class TestReplaceAgentConfig:
    """Function-specific tests for replace_agent_config."""

    @pytest.mark.parametrize("limit_delta,reset_expected", [
        (1,  True),   # limit raised  → fresh warning needed for new uncovered zone
        (0,  False),  # limit unchanged → no new uncovered zone
        (-1, False),  # limit lowered  → compaction fires sooner, consolidation already covers it
    ])
    async def test_compaction_warning_reset_depends_on_limit_direction(
        self,
        session: AsyncSession,
        agent_record,
        agent_deps: AgentDeps,
        limit_delta: int,
        reset_expected: bool,
    ):
        """compaction_warning_fired is reset only when soft_compaction_limit is raised."""
        async with agent_deps.write_scope():
            agent_deps.compaction_warning_fired = True
        new_limit = SAMPLE_AGENT_CONFIG.soft_compaction_limit + limit_delta
        new_config = SAMPLE_AGENT_CONFIG.model_copy(update={"soft_compaction_limit": new_limit})

        await replace_agent_config(agent_deps, new_config)

        assert agent_deps.compaction_warning_fired is not reset_expected


class TestReplaceSystemInstructions:
    """Function-specific tests for replace_system_instructions."""

    async def test_triggers_recompilation(self, session: AsyncSession, agent_record, agent_deps: AgentDeps):
        """System prompt is recompiled after updating instructions."""
        agent_id = agent_record.id  # capture before expire — post-expire attr access triggers async lazy load
        new_instructions = "You are a helpful assistant who loves cats."

        await replace_system_instructions(agent_deps, new_instructions)

        # agent_record is detached (conftest expunges) — get_agent_record fetches fresh
        refreshed = await get_agent_record(session, agent_id)
        assert new_instructions in refreshed.compiled_system_prompt
