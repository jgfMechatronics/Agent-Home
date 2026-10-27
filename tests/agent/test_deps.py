"""Tests for AgentDeps construction and write_scope/read_scope lifecycle behavior."""
from datetime import datetime
from unittest.mock import MagicMock
import pytest

from conftest import make_deps
from agent.deps import AgentDeps
from agent.types import BlockSettings
from memory.block_crud import create_block, update_block, ContentExceedsLimitError


@pytest.mark.asyncio
class TestWriteScopeExceptionRecovery:
    """A write_scope that exits via exception must leave the record in a known-good,
    readable state. Routine validation errors (e.g. ContentExceedsLimitError on the
    ModelRetry path) are normal operation — they must not poison subsequent property
    reads with DetachedInstanceError."""

    async def test_record_readable_after_body_exception(self, test_engine, agent_record):
        deps = make_deps(test_engine, agent_record)
        await create_block(deps, BlockSettings(label="probe", char_limit=10), content="12345")

        with pytest.raises(ContentExceedsLimitError):
            await update_block(deps, "probe", "this content is definitely too long")

        assert deps.compaction_warning_fired is False

    async def test_uncommitted_record_mutation_discarded_after_body_exception(
        self, test_engine, agent_record
    ):
        deps = make_deps(test_engine, agent_record)
        async with deps.write_scope():
            deps.compiled_system_prompt = "committed value"

        with pytest.raises(RuntimeError, match="deliberate"):
            async with deps.write_scope():
                deps.compiled_system_prompt = "uncommitted value"
                raise RuntimeError("deliberate")

        assert deps.compiled_system_prompt == "committed value"


# --- AgentDeps construction ---

@pytest.mark.parametrize("missing_field", ["rw_engine", "ro_engine", "agent_record"])
def test_agentdeps_requires_field(missing_field: str):
    """AgentDeps should raise TypeError when a required constructor argument is missing."""
    all_fields = {
        "rw_engine": object(),
        "ro_engine": object(),
        "agent_record": MagicMock(),
        "bypass_factory_guard": True,
    }
    del all_fields[missing_field]

    with pytest.raises(TypeError):
        AgentDeps(**all_fields)


def test_agentdeps_factory_guard_blocks_direct_construction():
    """AgentDeps must be constructed via build_deps or with explicit bypass."""
    with pytest.raises(RuntimeError, match="must be constructed via build_deps"):
        AgentDeps(object(), object(), MagicMock())


async def test_agentdeps_holds_expected_fields(agent_deps, agent_record):
    """AgentDeps properties should delegate to the underlying AgentRecord."""
    assert agent_deps.agent_id == agent_record.id
    assert agent_deps.config is agent_record.agent_config
    assert agent_deps.name == agent_record.name
    assert agent_deps.system_instructions == agent_record.system_instructions


async def test_agentdeps_setters_all_require_write_scope(agent_with_blocks: dict, test_engine):
    """All AgentDeps setters require write_scope context."""
    agent_record = agent_with_blocks["agent"]  # already detached (conftest expunges)

    deps = make_deps(test_engine, agent_record)

    setters_to_test = [
        ("config", agent_record.agent_config),
        ("system_instructions", "new instructions"),
        ("compiled_system_prompt", "new prompt"),
        ("sys_prompt_compiled_at", datetime.now()),
        ("context_window_start", 100),
        ("compaction_warning_fired", True),
    ]

    for prop_name, test_value in setters_to_test:
        with pytest.raises(RuntimeError, match="write_scope"):
            setattr(deps, prop_name, test_value)
