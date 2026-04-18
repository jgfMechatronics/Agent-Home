"""Unit tests for agent tools — Section 3.2

Tests the tool registry, lookup, and memory editing tools (memory_replace, memory_insert).
"""
from unittest.mock import Mock

import pytest
import pytest_asyncio
from sqlalchemy.ext.asyncio import AsyncSession

from agent.tools import (
    TOOL_REGISTRY,
    _compute_snippet,
    get_tools_for_agent,
    memory_insert,
    memory_replace,
)
from agent.types import AgentDeps
from conftest import SAMPLE_AGENT_CONFIG, make_deps
from db.models import AgentRecord, MemoryBlockRecord


# =============================================================================
# Fixtures
# =============================================================================

# Simple 10-line fixture: letters A through J (one per line)
ALPHABET_CONTENT = "\n".join("ABCDEFGHIJ")


def _mock_run_context(deps: AgentDeps):
    """Create a mock RunContext with the given deps for tool testing."""
    ctx = Mock()
    ctx.deps = deps
    return ctx


@pytest_asyncio.fixture
async def agent_with_editable_block(session: AsyncSession):
    """Agent with a single block suitable for editing tests.
    
    Block has enough content to test replacements/insertions and
    a reasonable char_limit for overflow tests.
    """
    agent = AgentRecord(
        name="edit-test-agent",
        agent_config=SAMPLE_AGENT_CONFIG,
        system_instructions="Test agent",
    )
    session.add(agent)
    await session.flush()
    
    block = MemoryBlockRecord(
        agent_id=agent.id,
        label="notes",
        description="Scratch space",
        content="Line one.\nLine two.\nLine three.",
        char_limit=100,
        position=0,
    )
    session.add(block)
    await session.flush()
    
    deps = make_deps(session, agent)
    ctx = _mock_run_context(deps)
    
    return {"agent": agent, "block": block, "deps": deps, "ctx": ctx}


@pytest_asyncio.fixture
async def agent_with_repeated_content(session: AsyncSession):
    """Agent with a block containing repeated strings for occurrence tests."""
    agent = AgentRecord(
        name="repeat-test-agent",
        agent_config=SAMPLE_AGENT_CONFIG,
        system_instructions="Test agent",
    )
    session.add(agent)
    await session.flush()
    
    block = MemoryBlockRecord(
        agent_id=agent.id,
        label="notes",
        description="Scratch space",
        content="foo bar foo baz foo",  # "foo" appears 3 times
        char_limit=200,
        position=0,
    )
    session.add(block)
    await session.flush()
    
    deps = make_deps(session, agent)
    ctx = _mock_run_context(deps)
    
    return {"agent": agent, "block": block, "deps": deps, "ctx": ctx}


# =============================================================================
# TestComputeSnippet
# =============================================================================


class TestComputeSnippet:
    """Tests for the _compute_snippet helper function.
    
    This helper extracts a window of lines around an edit for returning
    to the model (token optimization vs returning full block content).
    """

    def test_edit_in_middle_returns_context_window(self):
        """Edit at line 5 (F) with 3 context lines returns C-I (lines 2-8)."""
        snippet = _compute_snippet(
            ALPHABET_CONTENT, edit_start_line=5, edit_line_count=1, context_lines=3
        )
        expected = "\n".join("CDEFGHI")
        assert snippet == expected

    def test_edit_at_start_clips_to_beginning(self):
        """Edit at line 0 (A) doesn't go negative — returns A-D."""
        snippet = _compute_snippet(
            ALPHABET_CONTENT, edit_start_line=0, edit_line_count=1, context_lines=3
        )
        expected = "\n".join("ABCD")
        assert snippet == expected

    def test_edit_at_end_clips_to_end(self):
        """Edit at line 9 (J) doesn't exceed bounds — returns G-J."""
        snippet = _compute_snippet(
            ALPHABET_CONTENT, edit_start_line=9, edit_line_count=1, context_lines=3
        )
        expected = "\n".join("GHIJ")
        assert snippet == expected

    def test_multiline_edit_includes_full_edit_region(self):
        """Edit spanning lines 4-6 (E-G) with 2 context returns C-I."""
        snippet = _compute_snippet(
            ALPHABET_CONTENT, edit_start_line=4, edit_line_count=3, context_lines=2
        )
        expected = "\n".join("CDEFGHI")
        assert snippet == expected

    def test_empty_content_returns_empty(self):
        """Empty content returns empty string."""
        assert _compute_snippet("", edit_start_line=0, edit_line_count=0) == ""

    def test_single_line_content(self):
        """Single line content returns that line."""
        assert _compute_snippet("only", edit_start_line=0, edit_line_count=1) == "only"

    def test_default_context_is_three(self):
        """Default context_lines is 3."""
        default = _compute_snippet(ALPHABET_CONTENT, edit_start_line=5, edit_line_count=1)
        explicit = _compute_snippet(
            ALPHABET_CONTENT, edit_start_line=5, edit_line_count=1, context_lines=3
        )
        assert default == explicit


# =============================================================================
# TestToolRegistry
# =============================================================================


class TestToolRegistry:
    """Tests for TOOL_REGISTRY and get_tools_for_agent."""

    def test_registry_contains_memory_tools(self):
        """TOOL_REGISTRY contains memory_replace and memory_insert keyed by name."""
        assert "memory_replace" in TOOL_REGISTRY
        assert "memory_insert" in TOOL_REGISTRY
        assert TOOL_REGISTRY["memory_replace"] is memory_replace
        assert TOOL_REGISTRY["memory_insert"] is memory_insert

    def test_get_tools_returns_callables_for_valid_names(self):
        """get_tools_for_agent returns list of callables for valid tool names."""
        tools = get_tools_for_agent(["memory_replace", "memory_insert"])
        assert len(tools) == 2
        assert memory_replace in tools
        assert memory_insert in tools

    def test_get_tools_raises_keyerror_for_unknown(self):
        """get_tools_for_agent raises KeyError for unknown tool name."""
        with pytest.raises(KeyError, match="nonexistent_tool"):
            get_tools_for_agent(["memory_replace", "nonexistent_tool"])


# =============================================================================
# Shared Memory Tool Behaviors (parametrized)
# =============================================================================

# Valid args for each tool (excluding label, which tests vary)
MEMORY_REPLACE_ARGS = {"old_string": "Line one.", "new_string": "New line."}
MEMORY_INSERT_ARGS = {"content": "Inserted.", "after": "<end>"}


class TestMemoryToolsShared:
    """Shared behaviors for memory_replace and memory_insert, parametrized."""

    @pytest.mark.parametrize("tool_fn,valid_args", [
        pytest.param(memory_replace, MEMORY_REPLACE_ARGS, id="memory_replace"),
        pytest.param(memory_insert, MEMORY_INSERT_ARGS, id="memory_insert"),
    ])
    async def test_returns_error_if_label_not_found(
        self, agent_with_editable_block, tool_fn, valid_args
    ):
        """Tool returns error string when label doesn't exist for this agent."""
        ctx = agent_with_editable_block["ctx"]
        result = await tool_fn(ctx, label="nonexistent", **valid_args)
        assert "not found" in result.lower()

    @pytest.mark.parametrize("tool_fn,valid_args,expected_content", [
        pytest.param(
            memory_replace,
            {"old_string": "Line one.", "new_string": "REPLACED."},
            "REPLACED.\nLine two.\nLine three.",
            id="memory_replace",
        ),
        pytest.param(
            memory_insert,
            {"content": " INSERTED", "after": "Line three."},
            "Line one.\nLine two.\nLine three. INSERTED",
            id="memory_insert",
        ),
    ])
    async def test_updates_correct_block(
        self, agent_with_editable_block, tool_fn, valid_args, expected_content
    ):
        """Tool updates the block content correctly."""
        ctx = agent_with_editable_block["ctx"]
        block = agent_with_editable_block["block"]
        
        await tool_fn(ctx, label="notes", **valid_args)
        
        # Refresh to see persisted changes
        await ctx.deps.session.refresh(block)
        assert block.content == expected_content

    @pytest.mark.parametrize("tool_fn,overflow_args", [
        pytest.param(
            memory_replace,
            {"old_string": "Line one.", "new_string": "X" * 200},
            id="memory_replace",
        ),
        pytest.param(
            memory_insert,
            {"content": "X" * 200, "after": "<end>"},
            id="memory_insert",
        ),
    ])
    async def test_returns_error_if_exceeds_char_limit(
        self, agent_with_editable_block, tool_fn, overflow_args
    ):
        """Tool returns error string when result would exceed char_limit."""
        ctx = agent_with_editable_block["ctx"]
        result = await tool_fn(ctx, label="notes", **overflow_args)
        assert "limit" in result.lower() or "exceed" in result.lower()

    @pytest.mark.parametrize("tool_fn,valid_args", [
        pytest.param(memory_replace, MEMORY_REPLACE_ARGS, id="memory_replace"),
        pytest.param(memory_insert, MEMORY_INSERT_ARGS, id="memory_insert"),
    ])
    async def test_persists_change_immediately(
        self, agent_with_editable_block, tool_fn, valid_args
    ):
        """Tool persists change to DB immediately (flush), not deferred."""
        ctx = agent_with_editable_block["ctx"]
        block = agent_with_editable_block["block"]
        original_content = block.content
        
        await tool_fn(ctx, label="notes", **valid_args)
        
        # Block should be flushed (not in session.new or session.dirty)
        assert block not in ctx.deps.session.new
        assert block not in ctx.deps.session.dirty
        # And content should have changed
        await ctx.deps.session.refresh(block)
        assert block.content != original_content

    @pytest.mark.parametrize("tool_fn,valid_args", [
        pytest.param(memory_replace, MEMORY_REPLACE_ARGS, id="memory_replace"),
        pytest.param(memory_insert, MEMORY_INSERT_ARGS, id="memory_insert"),
    ])
    async def test_returns_snippet_on_success(
        self, agent_with_editable_block, tool_fn, valid_args
    ):
        """Tool returns snippet of updated content, not full block."""
        ctx = agent_with_editable_block["ctx"]
        result = await tool_fn(ctx, label="notes", **valid_args)
        
        # Result should be a non-empty string (the snippet)
        assert isinstance(result, str)
        assert len(result) > 0
        # Should contain part of the edited content
        # (exact snippet content tested in tool-specific tests)

    @pytest.mark.parametrize("tool_fn,ambiguous_args", [
        pytest.param(
            memory_replace,
            {"old_string": "foo", "new_string": "REPLACED"},
            id="memory_replace",
        ),
        pytest.param(
            memory_insert,
            {"content": "INSERTED", "after": "foo"},
            id="memory_insert",
        ),
    ])
    async def test_returns_error_if_multiple_matches_without_occurrence(
        self, agent_with_repeated_content, tool_fn, ambiguous_args
    ):
        """Tool returns error when target appears multiple times and occurrence not specified."""
        ctx = agent_with_repeated_content["ctx"]
        result = await tool_fn(ctx, label="notes", **ambiguous_args)
        assert "multiple" in result.lower() or "ambiguous" in result.lower() or "occurrence" in result.lower()

    @pytest.mark.parametrize("tool_fn,not_found_args", [
        pytest.param(
            memory_replace,
            {"old_string": "DOES_NOT_EXIST", "new_string": "new"},
            id="memory_replace",
        ),
        pytest.param(
            memory_insert,
            {"content": "new", "after": "DOES_NOT_EXIST"},
            id="memory_insert",
        ),
    ])
    async def test_returns_error_if_target_not_found(
        self, agent_with_editable_block, tool_fn, not_found_args
    ):
        """Tool returns error when old_string/after not found in block."""
        ctx = agent_with_editable_block["ctx"]
        result = await tool_fn(ctx, label="notes", **not_found_args)
        assert "not found" in result.lower()

    @pytest.mark.parametrize("tool_fn,occurrence_args", [
        pytest.param(
            memory_replace,
            {"old_string": "foo", "new_string": "REPLACED", "occurrence": 5},
            id="memory_replace",
        ),
        pytest.param(
            memory_insert,
            {"content": "INSERTED", "after": "foo", "occurrence": 5},
            id="memory_insert",
        ),
    ])
    async def test_returns_error_if_occurrence_exceeds_count(
        self, agent_with_repeated_content, tool_fn, occurrence_args
    ):
        """Tool returns error when occurrence=N but fewer than N occurrences exist."""
        ctx = agent_with_repeated_content["ctx"]
        # "foo" appears 3 times, requesting 5th
        result = await tool_fn(ctx, label="notes", **occurrence_args)
        assert "occurrence" in result.lower() or "not found" in result.lower()

    @pytest.mark.parametrize("tool_fn,empty_args", [
        pytest.param(
            memory_replace,
            {"old_string": "", "new_string": "new"},
            id="memory_replace_empty_old",
        ),
        pytest.param(
            memory_insert,
            {"content": "new", "after": ""},
            id="memory_insert_empty_after",
        ),
    ])
    async def test_returns_error_if_target_empty(
        self, agent_with_editable_block, tool_fn, empty_args
    ):
        """Tool returns error when old_string/after is empty."""
        ctx = agent_with_editable_block["ctx"]
        result = await tool_fn(ctx, label="notes", **empty_args)
        assert "empty" in result.lower() or "required" in result.lower()


# =============================================================================
# TestMemoryReplace — tool-specific behaviors
# =============================================================================


class TestMemoryReplace:
    """Tests specific to memory_replace behavior."""

    async def test_replaces_target_and_returns_snippet_with_edit(self, agent_with_editable_block):
        """memory_replace returns snippet containing the replaced text."""
        ctx = agent_with_editable_block["ctx"]
        block = agent_with_editable_block["block"]
        
        result = await memory_replace(ctx, label="notes", old_string="Line two.", new_string="REPLACED.")
        
        await ctx.deps.session.refresh(block)
        assert block.content == "Line one.\nREPLACED.\nLine three."
        # Snippet should contain the new text
        assert "REPLACED." in result

    async def test_occurrence_zero_returns_error(self, agent_with_repeated_content):
        """occurrence=0 returns clear error about 1-indexing."""
        ctx = agent_with_repeated_content["ctx"]
        result = await memory_replace(
            ctx, label="notes", old_string="foo", new_string="X", occurrence=0
        )
        assert "must be >= 1" in result or "1-indexed" in result

    async def test_occurrence_targets_nth_match(self, agent_with_repeated_content):
        """occurrence=N replaces the Nth occurrence (1-indexed)."""
        ctx = agent_with_repeated_content["ctx"]
        block = agent_with_repeated_content["block"]
        # Content: "foo bar foo baz foo"
        
        await memory_replace(
            ctx, label="notes", old_string="foo", new_string="SECOND", occurrence=2
        )
        
        await ctx.deps.session.refresh(block)
        assert block.content == "foo bar SECOND baz foo"

    async def test_only_replaces_target_occurrence(self, agent_with_repeated_content):
        """Only the target occurrence is replaced, others unchanged."""
        ctx = agent_with_repeated_content["ctx"]
        block = agent_with_repeated_content["block"]
        # Content: "foo bar foo baz foo"
        
        await memory_replace(
            ctx, label="notes", old_string="foo", new_string="X", occurrence=1
        )
        
        await ctx.deps.session.refresh(block)
        # Only first "foo" replaced
        assert block.content == "X bar foo baz foo"
        assert block.content.count("foo") == 2  # Other two remain


# =============================================================================
# TestMemoryInsert — tool-specific behaviors
# =============================================================================


class TestMemoryInsert:
    """Tests specific to memory_insert behavior."""

    async def test_after_start_inserts_at_beginning(self, agent_with_editable_block):
        """after='<start>' inserts content at the start of the block."""
        ctx = agent_with_editable_block["ctx"]
        block = agent_with_editable_block["block"]
        
        result = await memory_insert(ctx, label="notes", content="PREPENDED\n", after="<start>")
        
        await ctx.deps.session.refresh(block)
        assert block.content.startswith("PREPENDED\n")
        assert block.content == "PREPENDED\nLine one.\nLine two.\nLine three."
        # Snippet should contain the inserted text
        assert "PREPENDED" in result

    async def test_occurrence_with_start_returns_error(self, agent_with_editable_block):
        """occurrence cannot be used with '<start>'."""
        ctx = agent_with_editable_block["ctx"]
        result = await memory_insert(
            ctx, label="notes", content="X", after="<start>", occurrence=1
        )
        assert "cannot be used" in result.lower()

    async def test_occurrence_with_end_returns_error(self, agent_with_editable_block):
        """occurrence cannot be used with '<end>'."""
        ctx = agent_with_editable_block["ctx"]
        result = await memory_insert(
            ctx, label="notes", content="X", after="<end>", occurrence=1
        )
        assert "cannot be used" in result.lower()

    async def test_after_end_inserts_at_end(self, agent_with_editable_block):
        """after='<end>' inserts content at the end of the block."""
        ctx = agent_with_editable_block["ctx"]
        block = agent_with_editable_block["block"]
        
        await memory_insert(ctx, label="notes", content="\nAPPENDED", after="<end>")
        
        await ctx.deps.session.refresh(block)
        assert block.content.endswith("\nAPPENDED")
        assert block.content == "Line one.\nLine two.\nLine three.\nAPPENDED"

    async def test_after_anchor_inserts_after_match(self, agent_with_editable_block):
        """after='anchor' inserts content immediately after the anchor string."""
        ctx = agent_with_editable_block["ctx"]
        block = agent_with_editable_block["block"]
        
        await memory_insert(ctx, label="notes", content=" [INSERTED]", after="Line two.")
        
        await ctx.deps.session.refresh(block)
        assert block.content == "Line one.\nLine two. [INSERTED]\nLine three."

    async def test_occurrence_targets_nth_anchor(self, agent_with_repeated_content):
        """occurrence=N inserts after the Nth occurrence of anchor (1-indexed)."""
        ctx = agent_with_repeated_content["ctx"]
        block = agent_with_repeated_content["block"]
        # Content: "foo bar foo baz foo"
        
        await memory_insert(
            ctx, label="notes", content="[2]", after="foo", occurrence=2
        )
        
        await ctx.deps.session.refresh(block)
        assert block.content == "foo bar foo[2] baz foo"

    async def test_insert_does_not_overwrite(self, agent_with_editable_block):
        """Insert adds content without removing existing content."""
        ctx = agent_with_editable_block["ctx"]
        block = agent_with_editable_block["block"]
        original_content = block.content
        
        await memory_insert(ctx, label="notes", content="NEW", after="<end>")
        
        await ctx.deps.session.refresh(block)
        # Original content should still be present
        assert "Line one." in block.content
        assert "Line two." in block.content
        assert "Line three." in block.content
        # And new content added
        assert "NEW" in block.content
