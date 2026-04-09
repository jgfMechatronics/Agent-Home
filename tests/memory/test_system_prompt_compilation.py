"""
Tests for system prompt building (memory/system_prompt_compilation.py)

compile_system_prompt(deps) — assembles blocks into prompt, stores result
get_system_prompt(ctx) — returns cached prompt for Pydantic AI instructions param
"""
from datetime import datetime, UTC
from unittest.mock import Mock

import pytest_asyncio
from sqlalchemy.ext.asyncio import AsyncSession

from conftest import make_deps, SAMPLE_AGENT_CONFIG
from db.models import AgentRecord, MemoryBlockRecord
from agent.types import AgentDeps

from memory.system_prompt_compilation import compile_system_prompt, get_system_prompt


# --- Fixtures ---
# Note: agent_with_blocks comes from conftest.py

@pytest_asyncio.fixture
async def agent_with_blocks_and_deps(session: AsyncSession, agent_with_blocks: dict):
    """Extends agent_with_blocks with AgentDeps for write operations."""
    return {**agent_with_blocks, "deps": make_deps(session, agent_with_blocks["agent"])}


@pytest_asyncio.fixture
async def agent_no_blocks_with_deps(session: AsyncSession, agent_record: AgentRecord):
    """Agent with no memory blocks. Uses shared agent_record from conftest."""
    return {"agent": agent_record, "deps": make_deps(session, agent_record)}


@pytest_asyncio.fixture
async def agent_with_precompiled_prompt(session: AsyncSession):
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
    return {"agent": agent, "deps": make_deps(session, agent)}


# --- compile_system_prompt tests ---

async def test_compile_assembles_blocks_in_position_order(agent_with_blocks_and_deps: dict):
    """Blocks should appear in the compiled prompt ordered by position, not insertion order."""
    deps = agent_with_blocks_and_deps["deps"]
    agent = agent_with_blocks_and_deps["agent"]
    blocks = agent_with_blocks_and_deps["blocks"]
    
    await compile_system_prompt(deps)
    
    compiled = agent.compiled_system_prompt
    
    # Verify blocks appear in position order
    pos_0 = compiled.find(blocks[0].label)
    pos_1 = compiled.find(blocks[1].label)
    pos_2 = compiled.find(blocks[2].label)
    
    assert pos_0 < pos_1 < pos_2, (
        f"Blocks not in position order: {blocks[0].label}@{pos_0}, {blocks[1].label}@{pos_1}, {blocks[2].label}@{pos_2}"
    )


async def test_compile_includes_system_instructions_first(agent_with_blocks_and_deps: dict):
    """system_instructions should appear before any memory blocks."""
    deps = agent_with_blocks_and_deps["deps"]
    agent = agent_with_blocks_and_deps["agent"]
    blocks = agent_with_blocks_and_deps["blocks"]
    
    await compile_system_prompt(deps)
    
    compiled = agent.compiled_system_prompt
    
    # system_instructions content should come before first block
    instructions_pos = compiled.find(agent.system_instructions)
    first_block_pos = compiled.find(blocks[0].label)
    
    assert instructions_pos != -1, "system_instructions not found in compiled prompt"
    assert instructions_pos < first_block_pos, "system_instructions should appear before blocks"


async def test_compile_formats_blocks_with_xml_wrappers(agent_with_blocks_and_deps: dict):
    """Each block should be wrapped in XML with label, description, and content."""
    deps = agent_with_blocks_and_deps["deps"]
    agent = agent_with_blocks_and_deps["agent"]
    
    await compile_system_prompt(deps)
    
    compiled = agent.compiled_system_prompt
    
    # Verify XML structure for each block
    for block in agent_with_blocks_and_deps["blocks"]:
        # Block should have opening and closing tags with its label
        assert f"<{block.label}>" in compiled, f"Missing opening tag for {block.label}"
        assert f"</{block.label}>" in compiled, f"Missing closing tag for {block.label}"
        
        # Content should be present
        assert block.content in compiled, f"Missing content for {block.label}"
        
        # Description should be present (if non-empty)
        if block.description:
            assert block.description in compiled, f"Missing description for {block.label}"


async def test_compile_handles_agent_with_no_blocks(agent_no_blocks_with_deps: dict):
    """Agent with no blocks should compile to just system_instructions."""
    deps = agent_no_blocks_with_deps["deps"]
    agent = agent_no_blocks_with_deps["agent"]
    
    await compile_system_prompt(deps)
    
    compiled = agent.compiled_system_prompt
    
    assert agent.system_instructions in compiled
    # Should not have any memory block XML structure
    assert "</" not in compiled  # No closing tags = no blocks


async def test_compile_updates_sys_prompt_compiled_at(agent_with_blocks_and_deps: dict):
    """compile_system_prompt should update the sys_prompt_compiled_at timestamp."""
    deps = agent_with_blocks_and_deps["deps"]
    agent = agent_with_blocks_and_deps["agent"]
    
    # Initially None
    assert agent.sys_prompt_compiled_at is None
    
    before = datetime.now(UTC)
    await compile_system_prompt(deps)
    after = datetime.now(UTC)
    
    assert agent.sys_prompt_compiled_at is not None
    assert before <= agent.sys_prompt_compiled_at <= after


async def test_compile_only_includes_correct_agents_blocks(session: AsyncSession):
    """Compilation should only include blocks belonging to the target agent."""
    # Create two agents with different blocks
    agent_a = AgentRecord(
        name="agent-a",
        agent_config=SAMPLE_AGENT_CONFIG,
        system_instructions="Agent A instructions",
    )
    agent_b = AgentRecord(
        name="agent-b",
        agent_config=SAMPLE_AGENT_CONFIG,
        system_instructions="Agent B instructions",
    )
    session.add_all([agent_a, agent_b])
    await session.flush()
    
    block_a = MemoryBlockRecord(
        agent_id=agent_a.id, label="a_block", description="", 
        content="A's secret content", char_limit=1000, position=0
    )
    block_b = MemoryBlockRecord(
        agent_id=agent_b.id, label="b_block", description="",
        content="B's secret content", char_limit=1000, position=0
    )
    session.add_all([block_a, block_b])
    await session.flush()
    
    # Compile for agent A only
    deps_a = make_deps(session, agent_a)
    await compile_system_prompt(deps_a)
    
    compiled_a = agent_a.compiled_system_prompt
    
    assert "A's secret content" in compiled_a
    assert "B's secret content" not in compiled_a
    assert "b_block" not in compiled_a


async def test_compile_is_deterministic(agent_with_blocks_and_deps: dict):
    """Compiling the same blocks twice should produce identical output."""
    deps = agent_with_blocks_and_deps["deps"]
    agent = agent_with_blocks_and_deps["agent"]
    
    await compile_system_prompt(deps)
    first_compile = agent.compiled_system_prompt
    
    await compile_system_prompt(deps)
    second_compile = agent.compiled_system_prompt
    
    assert first_compile == second_compile, "Compilation should be deterministic"


async def test_compile_reflects_updated_blocks_on_recompile(agent_with_blocks_and_deps: dict):
    """Recompiling after a block edit should produce output reflecting the new content."""
    deps = agent_with_blocks_and_deps["deps"]
    agent = agent_with_blocks_and_deps["agent"]

    await compile_system_prompt(deps)

    # Modify a block and recompile
    blocks = agent_with_blocks_and_deps["blocks"]
    blocks[0].content = "Updated persona content after first compile."
    await deps.session.flush()

    await compile_system_prompt(deps)

    assert "Updated persona content after first compile." in agent.compiled_system_prompt


# --- get_system_prompt tests ---

def _mock_run_context(deps: AgentDeps):
    """Create a mock RunContext with deps attached."""
    ctx = Mock()
    ctx.deps = deps
    return ctx


async def test_get_returns_cached_compiled_prompt(agent_with_precompiled_prompt: dict):
    """get_system_prompt should return the cached compiled_system_prompt."""
    agent = agent_with_precompiled_prompt["agent"]
    deps = agent_with_precompiled_prompt["deps"]
    ctx = _mock_run_context(deps)
    
    result = await get_system_prompt(ctx)
    
    assert result == agent.compiled_system_prompt


async def test_get_returns_empty_string_when_null(session: AsyncSession):
    """get_system_prompt should return empty string when compiled_system_prompt is NULL."""
    agent = AgentRecord(
        name="null-prompt-agent",
        agent_config=SAMPLE_AGENT_CONFIG,
        system_instructions="Base instructions.",
        # compiled_system_prompt defaults to '' but let's be explicit
    )
    session.add(agent)
    await session.flush()
    
    ctx = _mock_run_context(make_deps(session, agent))
    result = await get_system_prompt(ctx)
    assert result == ""


async def test_get_does_not_mutate_stored_prompt(agent_with_precompiled_prompt: dict):
    """get_system_prompt should not modify the stored prompt."""
    deps = agent_with_precompiled_prompt["deps"]
    agent = agent_with_precompiled_prompt["agent"]
    ctx = _mock_run_context(deps)
    
    original_prompt = agent.compiled_system_prompt
    original_timestamp = agent.sys_prompt_compiled_at
    
    await get_system_prompt(ctx)
    await get_system_prompt(ctx)  # Call multiple times
    
    assert agent.compiled_system_prompt == original_prompt
    assert agent.sys_prompt_compiled_at == original_timestamp


async def test_get_returns_stale_prompt_after_block_edit(agent_with_blocks_and_deps: dict):
    """
    Deferred compilation: editing blocks should NOT trigger recompilation.
    get_system_prompt should return the old cached prompt even after blocks change.
    """
    deps = agent_with_blocks_and_deps["deps"]
    agent = agent_with_blocks_and_deps["agent"]
    ctx = _mock_run_context(deps)
    
    # Compile initial prompt
    await compile_system_prompt(deps)
    original_compiled = agent.compiled_system_prompt
    
    # Modify a block directly (simulating what memory tools would do)
    blocks = agent_with_blocks_and_deps["blocks"]
    blocks[0].content = "COMPLETELY NEW CONTENT THAT SHOULD NOT APPEAR"
    await deps.session.flush()
    
    # get_system_prompt should return the OLD compiled prompt
    result = await get_system_prompt(ctx)
    
    assert result == original_compiled
    assert "COMPLETELY NEW CONTENT" not in result
