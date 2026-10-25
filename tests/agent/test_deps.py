"""Tests for AgentDeps write_scope/read_scope lifecycle behavior."""
import pytest

from conftest import make_deps
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
