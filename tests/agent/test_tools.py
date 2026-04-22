"""Unit tests for agent tools — Section 3.2

Tests the tool registry, lookup, and memory editing tools (memory_replace, memory_insert).
"""
import pytest
import pytest_asyncio
from pydantic_ai.exceptions import ModelRetry
from sqlalchemy.ext.asyncio import AsyncSession

from agent.tools import (
    TOOL_REGISTRY,
    _compute_snippet,
    get_tools_for_agent,
    memory_insert,
    memory_replace,
)
from conftest import SAMPLE_AGENT_CONFIG, make_deps, mock_run_context
from db.models import AgentRecord, MemoryBlockRecord


# --- Fixtures ---

# Simple 10-line fixture: letters A through J (one per line)
ALPHABET_CONTENT = "\n".join("ABCDEFGHIJ")


async def _make_agent_with_block(
    session: AsyncSession,
    content: str,
    char_limit: int = 100,
    agent_name: str = "test-agent",
) -> dict:
    """Factory for creating an agent with a single editable block.
    
    Returns dict with agent, block, deps, ctx for test access.
    """
    agent = AgentRecord(
        name=agent_name,
        agent_config=SAMPLE_AGENT_CONFIG,
        system_instructions="Test agent",
    )
    session.add(agent)
    await session.flush()
    
    block = MemoryBlockRecord(
        agent_id=agent.id,
        label="notes",
        description="Scratch space",
        content=content,
        char_limit=char_limit,
        position=0,
    )
    session.add(block)
    await session.flush()
    
    deps = make_deps(session, agent)
    ctx = mock_run_context(deps)
    
    return {"agent": agent, "block": block, "deps": deps, "ctx": ctx}


@pytest_asyncio.fixture
async def agent_with_editable_block(session: AsyncSession):
    """Agent with a single block suitable for editing tests.
    
    WARNING: TestMemoryToolsShared params reference these exact strings.
    If you change the content, update the test params to match.
    """
    return await _make_agent_with_block(
        session,
        content="Line one.\nLine two.\nLine three.",
        char_limit=100,
    )


@pytest_asyncio.fixture
async def agent_with_repeated_content(session: AsyncSession):
    """Agent with a block containing repeated strings for occurrence tests.
    
    WARNING: TestMemoryToolsShared params reference "foo" as the repeated target.
    If you change the content, update the test params to match.
    """
    return await _make_agent_with_block(
        session,
        content="foo bar foo baz foo",  # "foo" appears 3 times
        char_limit=200,
    )


# --- TestComputeSnippet ---


class TestComputeSnippet:
    """Tests for the _compute_snippet helper function.
    
    This helper extracts a window of lines around an edit for returning
    to the model (token optimization vs returning full block content).
    """


    def test_edit_in_middle_returns_surrounding_window(self):
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


# --- TestToolRegistry ---


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


# --- Shared Memory Tool Behaviors (parametrized) ---

# Valid args for each tool (excluding label, which tests vary)
MEMORY_REPLACE_ARGS = {"old_string": "Line one.", "new_string": "New line."}
MEMORY_INSERT_ARGS = {"content": "Inserted.", "after": "<end>"}


class TestMemoryToolsShared:
    """Shared behaviors for memory_replace and memory_insert, parametrized."""

    @pytest.mark.parametrize("tool_fn,valid_args", [
        pytest.param(memory_replace, MEMORY_REPLACE_ARGS, id="memory_replace"),
        pytest.param(memory_insert, MEMORY_INSERT_ARGS, id="memory_insert"),
    ])
    async def test_raises_if_label_not_found(
        self, agent_with_editable_block, tool_fn, valid_args
    ):
        """Tool raises ModelRetry when label doesn't exist for this agent."""
        ctx = agent_with_editable_block["ctx"]
        with pytest.raises(ModelRetry, match="not found"):
            await tool_fn(ctx, label="nonexistent", **valid_args)


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
        
        await tool_fn(ctx, label=block.label, **valid_args)
        
        # Refresh to see persisted changes
        await ctx.deps.session.refresh(block)
        assert block.content == expected_content


    async def test_edit_does_not_affect_other_blocks(self, session: AsyncSession):
        """Editing one block doesn't affect other blocks on the same agent."""
        agent_data = await _make_agent_with_block(
            session, content="Block one content.", agent_name="multi-block-agent"
        )
        ctx = agent_data["ctx"]
        block_one = agent_data["block"]
        
        # Add a second block to the same agent
        block_two = MemoryBlockRecord(
            agent_id=agent_data["agent"].id,
            label="other",
            description="Another block",
            content="Block two content.",
            char_limit=100,
            position=1,
        )
        session.add(block_two)
        await session.flush()
        
        # Edit block one
        await memory_replace(
            ctx, label="notes", old_string="Block one content.", new_string="MODIFIED"
        )
        
        # Block one should be modified
        await session.refresh(block_one)
        assert block_one.content == "MODIFIED"
        
        # Block two should be untouched
        await session.refresh(block_two)
        assert block_two.content == "Block two content."


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
    async def test_raises_if_exceeds_char_limit(
        self, agent_with_editable_block, tool_fn, overflow_args
    ):
        """Tool raises ModelRetry when result would exceed char_limit."""
        ctx = agent_with_editable_block["ctx"]
        block = agent_with_editable_block["block"]
        with pytest.raises(ModelRetry, match="char_limit"):
            await tool_fn(ctx, label=block.label, **overflow_args)


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
        
        await tool_fn(ctx, label=block.label, **valid_args)
        
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
        block = agent_with_editable_block["block"]
        result = await tool_fn(ctx, label=block.label, **valid_args)
        
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
    async def test_raises_if_multiple_matches_without_occurrence(
        self, agent_with_repeated_content, tool_fn, ambiguous_args
    ):
        """Tool raises ModelRetry when target appears multiple times and occurrence not specified."""
        ctx = agent_with_repeated_content["ctx"]
        block = agent_with_repeated_content["block"]
        with pytest.raises(ModelRetry, match="appears.*times"):
            await tool_fn(ctx, label=block.label, **ambiguous_args)


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
    async def test_raises_if_target_not_found(
        self, agent_with_editable_block, tool_fn, not_found_args
    ):
        """Tool raises ModelRetry when old_string/after not found in block."""
        ctx = agent_with_editable_block["ctx"]
        block = agent_with_editable_block["block"]
        with pytest.raises(ModelRetry, match="not found"):
            await tool_fn(ctx, label=block.label, **not_found_args)


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
    async def test_raises_if_occurrence_exceeds_count(
        self, agent_with_repeated_content, tool_fn, occurrence_args
    ):
        """Tool raises ModelRetry when occurrence=N but fewer than N occurrences exist."""
        ctx = agent_with_repeated_content["ctx"]
        block = agent_with_repeated_content["block"]
        # "foo" appears 3 times, requesting 5th
        with pytest.raises(ModelRetry, match="occurrence.*not found"):
            await tool_fn(ctx, label=block.label, **occurrence_args)


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
    async def test_raises_if_target_empty(
        self, agent_with_editable_block, tool_fn, empty_args
    ):
        """Tool raises ModelRetry when old_string/after is empty."""
        ctx = agent_with_editable_block["ctx"]
        block = agent_with_editable_block["block"]
        with pytest.raises(ModelRetry, match="empty"):
            await tool_fn(ctx, label=block.label, **empty_args)


    @pytest.mark.parametrize("tool_fn,zero_occurrence_args", [
        pytest.param(
            memory_replace,
            {"old_string": "foo", "new_string": "X", "occurrence": 0},
            id="memory_replace",
        ),
        pytest.param(
            memory_insert,
            {"content": "X", "after": "foo", "occurrence": 0},
            id="memory_insert",
        ),
    ])
    async def test_raises_if_occurrence_zero(
        self, agent_with_repeated_content, tool_fn, zero_occurrence_args
    ):
        """Tool raises ModelRetry when occurrence=0 (must be 1-indexed)."""
        ctx = agent_with_repeated_content["ctx"]
        block = agent_with_repeated_content["block"]
        with pytest.raises(ModelRetry, match="must be >= 1"):
            await tool_fn(ctx, label=block.label, **zero_occurrence_args)


    async def test_cannot_edit_other_agents_block(self, session: AsyncSession):
        """Tool edits are scoped to the calling agent — can't affect other agents' blocks."""
        # Create two agents, both with a block labeled "notes"
        agent_a = await _make_agent_with_block(
            session, content="Agent A: Line one.", agent_name="agent-a"
        )
        agent_b = await _make_agent_with_block(
            session, content="Agent B: Line one.", agent_name="agent-b"
        )
        
        # Agent A edits their "notes" block
        ctx_a = agent_a["ctx"]
        await memory_replace(
            ctx_a, label=agent_a["block"].label, old_string="Agent A: Line one.", new_string="MODIFIED"
        )
        
        # Agent A's block should be modified
        await session.refresh(agent_a["block"])
        assert agent_a["block"].content == "MODIFIED"
        
        # Agent B's block should be untouched
        await session.refresh(agent_b["block"])
        assert agent_b["block"].content == "Agent B: Line one."


# --- TestMemoryReplace (tool-specific) ---

class TestMemoryReplace:
    """Tests specific to memory_replace behavior."""

    async def test_replaces_target_and_returns_snippet_with_edit(self, agent_with_editable_block):
        """memory_replace returns snippet containing the replaced text."""
        ctx = agent_with_editable_block["ctx"]
        block = agent_with_editable_block["block"]
        
        result = await memory_replace(ctx, label=block.label, old_string="Line two.", new_string="REPLACED.")
        
        await ctx.deps.session.refresh(block)
        assert block.content == "Line one.\nREPLACED.\nLine three."
        # Snippet should contain the new text
        assert "REPLACED." in result


    async def test_occurrence_targets_nth_match(self, agent_with_repeated_content):
        """occurrence=N replaces the Nth occurrence (1-indexed)."""
        ctx = agent_with_repeated_content["ctx"]
        block = agent_with_repeated_content["block"]
        # Content: "foo bar foo baz foo"
        
        await memory_replace(
            ctx, label=block.label, old_string="foo", new_string="SECOND", occurrence=2
        )
        
        await ctx.deps.session.refresh(block)
        assert block.content == "foo bar SECOND baz foo"


    async def test_only_replaces_target_occurrence(self, agent_with_repeated_content):
        """Only the target occurrence is replaced, others unchanged."""
        ctx = agent_with_repeated_content["ctx"]
        block = agent_with_repeated_content["block"]
        # Content: "foo bar foo baz foo"
        
        await memory_replace(
            ctx, label=block.label, old_string="foo", new_string="X", occurrence=1
        )
        
        await ctx.deps.session.refresh(block)
        # Only first "foo" replaced
        assert block.content == "X bar foo baz foo"


    async def test_empty_new_string_deletes_target(self, agent_with_editable_block):
        """new_string='' effectively deletes the old_string."""
        ctx = agent_with_editable_block["ctx"]
        block = agent_with_editable_block["block"]
        # Content: "Line one.\nLine two.\nLine three."
        
        await memory_replace(ctx, label=block.label, old_string="Line two.\n", new_string="")
        
        await ctx.deps.session.refresh(block)
        assert block.content == "Line one.\nLine three."


# --- TestMemoryInsert (tool-specific) ---


class TestMemoryInsert:
    """Tests specific to memory_insert behavior."""


    @pytest_asyncio.fixture(autouse=True)
    async def setup(self, agent_with_editable_block):
        """Most tests use agent_with_editable_block; pull ctx/block into self."""
        self.ctx = agent_with_editable_block["ctx"]
        self.block = agent_with_editable_block["block"]


    async def test_after_start_inserts_at_beginning(self):
        """after='<start>' inserts content at the start of the block."""
        result = await memory_insert(self.ctx, label=self.block.label, content="PREPENDED\n", after="<start>")
        
        await self.ctx.deps.session.refresh(self.block)
        assert self.block.content.startswith("PREPENDED\n")
        assert self.block.content == "PREPENDED\nLine one.\nLine two.\nLine three."
        # Snippet should contain the inserted text
        assert "PREPENDED" in result


    async def test_occurrence_with_start_raises(self):
        """occurrence cannot be used with '<start>'."""
        with pytest.raises(ModelRetry, match="cannot be used"):
            await memory_insert(
                self.ctx, label=self.block.label, content="X", after="<start>", occurrence=1
            )


    async def test_occurrence_with_end_raises(self):
        """occurrence cannot be used with '<end>'."""
        with pytest.raises(ModelRetry, match="cannot be used"):
            await memory_insert(
                self.ctx, label=self.block.label, content="X", after="<end>", occurrence=1
            )


    async def test_after_end_inserts_at_end(self):
        """after='<end>' inserts content at the end of the block."""
        await memory_insert(self.ctx, label=self.block.label, content="\nAPPENDED", after="<end>")
        
        await self.ctx.deps.session.refresh(self.block)
        assert self.block.content.endswith("\nAPPENDED")
        assert self.block.content == "Line one.\nLine two.\nLine three.\nAPPENDED"


    async def test_after_anchor_inserts_after_match(self):
        """after='anchor' inserts content immediately after the anchor string."""
        await memory_insert(self.ctx, label=self.block.label, content=" [INSERTED]", after="Line two.")
        
        await self.ctx.deps.session.refresh(self.block)
        assert self.block.content == "Line one.\nLine two. [INSERTED]\nLine three."


    async def test_occurrence_targets_nth_anchor(self, agent_with_repeated_content):
        """occurrence=N inserts after the Nth occurrence of anchor (1-indexed)."""
        ctx = agent_with_repeated_content["ctx"]
        block = agent_with_repeated_content["block"]
        # Content: "foo bar foo baz foo"
        
        await memory_insert(
            ctx, label=block.label, content="[2]", after="foo", occurrence=2
        )
        
        await ctx.deps.session.refresh(block)
        assert block.content == "foo bar foo[2] baz foo"


    async def test_insert_does_not_overwrite(self):
        """Insert adds content without removing existing content."""
        await memory_insert(self.ctx, label=self.block.label, content="NEW", after="<end>")
        
        await self.ctx.deps.session.refresh(self.block)
        # Original content should still be present
        assert "Line one." in self.block.content
        assert "Line two." in self.block.content
        assert "Line three." in self.block.content
        # And new content added
        assert "NEW" in self.block.content
