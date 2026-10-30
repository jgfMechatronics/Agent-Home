"""
Tests for system prompt building (memory/system_prompt_compilation.py)

compile_system_prompt(deps) — assembles blocks into prompt, stores result
get_system_prompt(ctx) — returns cached prompt for Pydantic AI instructions param
"""
from datetime import datetime

import pytest_asyncio
from sqlalchemy.ext.asyncio import AsyncSession

from conftest import (
    mock_run_context, SAMPLE_AGENT_CONFIG, make_deps, commit_and_detach_fixture_record
)
from agent.deps import AgentDeps
from db.models import AgentRecord, MemoryBlockRecord, utcnow


from memory.system_prompt_compilation import compile_system_prompt, get_system_prompt


# --- Fixtures ---
# Note: agent_with_blocks comes from conftest.py

@pytest_asyncio.fixture
async def agent_with_blocks_and_deps(session: AsyncSession, test_engine, agent_with_blocks: dict):
    """Extends agent_with_blocks with AgentDeps for write operations."""
    return {**agent_with_blocks, "deps": make_deps(test_engine, agent_with_blocks["agent"]), "session": session}


@pytest_asyncio.fixture
async def agent_no_blocks_with_deps(agent_deps: AgentDeps, agent_record: AgentRecord):
    """Agent with no memory blocks. Uses shared agent_record from conftest."""
    return {"agent": agent_record, "deps": agent_deps}


@pytest_asyncio.fixture
async def agent_with_precompiled_prompt(session: AsyncSession, test_engine):
    """Agent with a pre-populated compiled_system_prompt for get_system_prompt tests."""
    agent = AgentRecord(
        name="precompiled-agent",
        agent_config=SAMPLE_AGENT_CONFIG,
        system_instructions="Base instructions.",
        compiled_system_prompt="<cached>This is the cached prompt.</cached>",
        sys_prompt_compiled_at=datetime(2026, 1, 1, 12, 0, 0),
    )
    session.add(agent)
    await session.flush()
    return {"agent": agent, "deps": make_deps(test_engine, agent)}


# --- Helper ---

def _extract_tag(text: str, tag: str) -> str | None:
    """Extract content between <tag> and </tag>. Returns None if not found."""
    start_tag = f"<{tag}>\n"
    end_tag = f"\n</{tag}>"
    start = text.find(start_tag)
    end = text.find(end_tag)
    if start == -1 or end == -1:
        return None
    return text[start + len(start_tag):end]


# --- compile_system_prompt tests ---

class TestCompileSystemPrompt:
    """
    Tests for compile_system_prompt() that use the standard agent_with_blocks fixture.

    pytest creates a fresh instance per test method (default behavior).
    _autouse_setup unpacks fixture, compiles, and stores result on self.
    """

    @pytest_asyncio.fixture(autouse=True)
    async def _autouse_setup(self, agent_with_blocks_and_deps: dict):
        self.deps = agent_with_blocks_and_deps["deps"]
        self.agent = agent_with_blocks_and_deps["agent"]
        self.blocks = agent_with_blocks_and_deps["blocks"]
        self.session = agent_with_blocks_and_deps["session"]
        async with self.deps.write_scope() as session:
            await compile_system_prompt(self.deps, session)
        self.compiled = self.agent.compiled_system_prompt

    async def test_assembles_blocks_in_position_order(self):
        """Blocks should appear in the compiled prompt ordered by position, not insertion order."""
        pos_0 = self.compiled.find(f"<{self.blocks[0].label}>")
        pos_1 = self.compiled.find(f"<{self.blocks[1].label}>")
        pos_2 = self.compiled.find(f"<{self.blocks[2].label}>")

        assert pos_0 < pos_1 < pos_2, (
            f"Blocks not in position order: "
            f"{self.blocks[0].label}@{pos_0}, {self.blocks[1].label}@{pos_1}, {self.blocks[2].label}@{pos_2}"
        )

    async def test_includes_system_instructions_first(self):
        """system_instructions should appear before any memory blocks wrapped in expected XML."""
        # check for proper XML format
        assert _extract_tag(self.compiled, "system_instructions") == self.agent.system_instructions

        instructions_pos = self.compiled.find(self.agent.system_instructions)
        first_block_pos = self.compiled.find(self.blocks[0].label)

        assert instructions_pos != -1, "system_instructions not found in compiled prompt"
        assert instructions_pos < first_block_pos, "system_instructions should appear before blocks"

    async def test_formats_blocks_with_xml_structure(self):
        """
        Each block should follow the prescribed XML format:
        <label>
        <description>...</description>
        <metadata>...</metadata>
        <content>...</content>
        </label>
        """
        for block in self.blocks:
            block_section = _extract_tag(self.compiled, block.label)
            assert block_section is not None, f"Missing block tags for {block.label}"

            subsections = {}
            for tag in ("description", "metadata", "content"):
                subsections[tag] = _extract_tag(block_section, tag)
                assert subsections[tag] is not None, f"Missing <{tag}> for {block.label}"

            if block.description:
                assert block.description in subsections["description"], f"Description not in <description> for {block.label}"
            assert block.content in subsections["content"], f"Content not in <content> for {block.label}"

            desc_pos = block_section.find("<description>")
            meta_pos = block_section.find("<metadata>")
            content_pos = block_section.find("<content>")

            assert desc_pos < meta_pos < content_pos, (
                f"Sections out of order for {block.label}: "
                f"description@{desc_pos}, metadata@{meta_pos}, content@{content_pos}"
            )

    async def test_metadata_section_accuracy(self):
        """Metadata section should contain accurate chars_current, chars_limit, and lines_current values."""
        for block in self.blocks:
            block_section = _extract_tag(self.compiled, block.label)
            meta_section = _extract_tag(block_section, "metadata")

            expected_chars = len(block.content)
            assert f"chars_current: {expected_chars}" in meta_section, (
                f"chars_current mismatch for {block.label}: expected {expected_chars}"
            )
            assert f"chars_limit: {block.char_limit}" in meta_section, (
                f"chars_limit mismatch for {block.label}: expected {block.char_limit}"
            )
            expected_lines = len(block.content.splitlines())
            assert f"lines_current: {expected_lines}" in meta_section, (
                f"lines_current mismatch for {block.label}: expected {expected_lines}"
            )

    async def test_is_deterministic(self):
        """Compiling the same blocks twice should produce identical output."""
        first_compile = self.compiled

        async with self.deps.write_scope() as session:
            await compile_system_prompt(self.deps, session)
        second_compile = self.agent.compiled_system_prompt

        assert first_compile == second_compile, "Compilation should be deterministic"

    async def test_reflects_updated_blocks_on_recompile(self):
        """Recompiling after a block edit should produce output reflecting the new content."""
        self.blocks[0].content = "Updated persona content after first compile."
        await self.session.merge(self.blocks[0])
        await self.session.commit()

        async with self.deps.write_scope() as session:
            await compile_system_prompt(self.deps, session)

        assert "Updated persona content after first compile." in self.agent.compiled_system_prompt


# --- compile_system_prompt formatting test (standalone, different fixtures). Was written to enable easy inspection of format ---

async def test_exact_compiled_format(session: AsyncSession, agent_deps: AgentDeps, agent_record: AgentRecord):
    """
    Compiled prompt exact XML format — newlines between every section and block.
    Bespoke simple memory structure to enable easy format inspection
    """
    _BLOCK_A_LABEL = "persona"
    _BLOCK_A_DESC = "The agent's identity."
    _BLOCK_A_CONTENT = "I am helpful."
    _BLOCK_A_LIMIT = 100

    _BLOCK_B_LABEL = "notes"
    _BLOCK_B_DESC = "Scratch space."
    _BLOCK_B_CONTENT = "Remember things."
    _BLOCK_B_LIMIT = 500

    block_a = MemoryBlockRecord(
        agent_id=agent_record.id, label=_BLOCK_A_LABEL, description=_BLOCK_A_DESC,
        content=_BLOCK_A_CONTENT, char_limit=_BLOCK_A_LIMIT, position=0,
    )
    block_b = MemoryBlockRecord(
        agent_id=agent_record.id, label=_BLOCK_B_LABEL, description=_BLOCK_B_DESC,
        content=_BLOCK_B_CONTENT, char_limit=_BLOCK_B_LIMIT, position=1,
    )
    session.add_all([block_a, block_b])
    await session.commit()

    async with agent_deps.write_scope() as ws_session:
        await compile_system_prompt(agent_deps, ws_session)

    expected = (
        f"<system_instructions>\n"
        f"{agent_record.system_instructions}\n"
        f"</system_instructions>\n"
        f"<{_BLOCK_A_LABEL}>\n"
        f"<description>\n{_BLOCK_A_DESC}\n</description>\n"
        f"<metadata>\nchars_current: {len(_BLOCK_A_CONTENT)}\nchars_limit: {_BLOCK_A_LIMIT}\nlines_current: {len(_BLOCK_A_CONTENT.splitlines())}\n</metadata>\n"
        f"<content>\n{_BLOCK_A_CONTENT}\n</content>\n"
        f"</{_BLOCK_A_LABEL}>\n"
        f"<{_BLOCK_B_LABEL}>\n"
        f"<description>\n{_BLOCK_B_DESC}\n</description>\n"
        f"<metadata>\nchars_current: {len(_BLOCK_B_CONTENT)}\nchars_limit: {_BLOCK_B_LIMIT}\nlines_current: {len(_BLOCK_B_CONTENT.splitlines())}\n</metadata>\n"
        f"<content>\n{_BLOCK_B_CONTENT}\n</content>\n"
        f"</{_BLOCK_B_LABEL}>"
    )
    assert agent_record.compiled_system_prompt == expected


async def test_compile_handles_agent_with_no_blocks(agent_no_blocks_with_deps: dict):
    """Agent with no blocks should compile to just system_instructions."""
    deps = agent_no_blocks_with_deps["deps"]
    agent = agent_no_blocks_with_deps["agent"]

    async with deps.write_scope() as session:
        await compile_system_prompt(deps, session)
    compiled = agent.compiled_system_prompt

    assert f"<system_instructions>\n{agent.system_instructions}\n</system_instructions>" == compiled


async def test_compile_updates_sys_prompt_compiled_at(agent_with_blocks_and_deps: dict):
    """compile_system_prompt should update the sys_prompt_compiled_at timestamp."""
    deps = agent_with_blocks_and_deps["deps"]
    agent = agent_with_blocks_and_deps["agent"]

    assert agent.sys_prompt_compiled_at is None

    before = utcnow()
    async with deps.write_scope() as session:
        await compile_system_prompt(deps, session)
        # Naive UTC like every other timestamp column (see db.models.utcnow), even before the
        # post-commit refresh reloads it from the DB.
        assert deps.sys_prompt_compiled_at.tzinfo is None
    after = utcnow()

    assert agent.sys_prompt_compiled_at is not None
    assert before < agent.sys_prompt_compiled_at < after


async def test_compile_only_includes_correct_agents_blocks(session: AsyncSession, test_engine):
    """Compilation should only include blocks belonging to the target agent."""
    agent_a = await commit_and_detach_fixture_record(session, AgentRecord(
        name="agent-a",
        agent_config=SAMPLE_AGENT_CONFIG,
        system_instructions="Agent A instructions",
    ))
    agent_b = await commit_and_detach_fixture_record(session, AgentRecord(
        name="agent-b",
        agent_config=SAMPLE_AGENT_CONFIG,
        system_instructions="Agent B instructions",
    ))

    session.add_all([
        MemoryBlockRecord(
            agent_id=agent_a.id, label="a_block", description="",
            content="A's secret content", char_limit=1000, position=0
        ),
        MemoryBlockRecord(
            agent_id=agent_b.id, label="b_block", description="",
            content="B's secret content", char_limit=1000, position=0
        ),
    ])
    await session.commit()

    deps_a = make_deps(test_engine, agent_a)
    async with deps_a.write_scope() as write_session:
        await compile_system_prompt(deps_a, write_session)

    # write_scope refreshes the deps record (agent_a) on exit — readable directly
    compiled_a = agent_a.compiled_system_prompt

    assert "A's secret content" in compiled_a
    assert "B's secret content" not in compiled_a
    assert "a_block" in compiled_a
    assert "b_block" not in compiled_a


# --- get_system_prompt tests ---


class TestGetSystemPrompt:
    """
    Tests for get_system_prompt() using an agent with pre-compiled prompt.
    """

    @pytest_asyncio.fixture(autouse=True)
    async def _autouse_setup(self, agent_with_precompiled_prompt: dict):
        self.agent = agent_with_precompiled_prompt["agent"]
        self.deps = agent_with_precompiled_prompt["deps"]
        self.ctx = mock_run_context(self.deps)

    async def test_returns_cached_compiled_prompt(self):
        """get_system_prompt should return the cached compiled_system_prompt."""
        result = await get_system_prompt(self.ctx)
        assert result == self.agent.compiled_system_prompt

    async def test_does_not_mutate_stored_prompt(self):
        """get_system_prompt should not modify the stored prompt."""
        original_prompt = self.agent.compiled_system_prompt
        original_timestamp = self.agent.sys_prompt_compiled_at

        await get_system_prompt(self.ctx)
        await get_system_prompt(self.ctx)  # for good measure

        assert self.agent.compiled_system_prompt == original_prompt
        assert self.agent.sys_prompt_compiled_at == original_timestamp


# --- get_system_prompt standalone tests (different fixtures) ---

async def test_get_returns_empty_str_when_compiled_is_null(session: AsyncSession, test_engine):
    """get_system_prompt should return empty string when compiled_system_prompt is empty/unset."""
    agent = AgentRecord(
        name="null-prompt-agent",
        agent_config=SAMPLE_AGENT_CONFIG,
        system_instructions="Base instructions.",
    )
    session.add(agent)
    await session.flush()
    assert agent.compiled_system_prompt == ""  # model defaults to ''

    ctx = mock_run_context(make_deps(test_engine, agent))
    result = await get_system_prompt(ctx)
    assert result == ""


async def test_get_returns_stale_prompt_after_block_edit(agent_with_blocks_and_deps: dict):
    """
    Deferred compilation: editing blocks should NOT trigger recompilation.
    get_system_prompt should return the old cached prompt even after blocks change.
    """
    deps = agent_with_blocks_and_deps["deps"]
    agent = agent_with_blocks_and_deps["agent"]
    ctx = mock_run_context(deps)

    # Compile initial prompt
    async with deps.write_scope() as session:
        await compile_system_prompt(deps, session)
    original_compiled = agent.compiled_system_prompt

    # Modify a block directly (simulating what memory tools would do)
    blocks = agent_with_blocks_and_deps["blocks"]
    blocks[0].content = "COMPLETELY NEW CONTENT THAT SHOULD NOT APPEAR"
    await agent_with_blocks_and_deps["session"].merge(blocks[0])
    await agent_with_blocks_and_deps["session"].commit()

    # get_system_prompt should return the OLD compiled prompt
    result = await get_system_prompt(ctx)

    assert result == original_compiled
    assert "COMPLETELY NEW CONTENT" not in result
