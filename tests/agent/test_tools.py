"""Unit tests for agent tools — Section 3.2

Tests the tool registry, lookup, and memory editing tools (memory_replace, memory_insert).
"""
import asyncio
import json

import pytest
import pytest_asyncio
from pydantic_ai import Agent, AgentRunResultEvent
from pydantic_ai.exceptions import ModelRetry
from pydantic_ai.messages import ModelRequest, ModelResponse, TextPart, UserPromptPart
from pydantic_ai.models.function import DeltaToolCall, DeltaToolCalls, FunctionModel
from sqlalchemy.ext.asyncio import AsyncSession

from agent.runner import COMPACTION_RESUME_NOTICE, run_stateful_agent, is_compaction_needed as _real_is_compaction_needed
from agent.tools import (
    TOOL_REGISTRY,
    _compute_snippet,
    _format_inter_agent_message,
    get_tools_for_agent,
    memory_insert,
    memory_replace,
    send_message,
)
from agent.types import AgentAppState, AgentDeps
from conftest import SAMPLE_AGENT_CONFIG, _make_mock_session, make_alternating_messages, make_deps, mock_run_context
from db.models import AgentRecord, MemoryBlockRecord

# TODO: move to conftest once on a proper PR branch
from tests.agent.test_runner import FunctionModelTestAgent, _BaseRouteTest, _PersistenceAndCancellationTestBase


# --- Fixtures ---

# Simple 10-line fixture: letters A through J (one per line)
ALPHABET_CONTENT = "\n".join("ABCDEFGHIJ")

DEFAULT_BLOCK_CHAR_LIM = 100

async def _make_agent_with_block(
    session: AsyncSession,
    content: str,
    char_limit: int = DEFAULT_BLOCK_CHAR_LIM,
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
    """Agent with a single block used for all tool editing tests.

    Content is designed to satisfy multiple test requirements simultaneously:
    - 3 lines (supports line-based editing tests like insert-at-anchor)
    - "foo" repeated 3x at the start of each line (supports occurrence parameter tests)
    - Unique anchors "one.", "two.", "three." (supports unambiguous single-target operations)

    WARNING: Test params reference these exact strings
    If you change the content, update all test params to match.
    """
    return await _make_agent_with_block(
        session,
        content="foo one.\nfoo two.\nfoo three.",
        char_limit=DEFAULT_BLOCK_CHAR_LIM,
    )


# --- TestComputeSnippet ---


class TestComputeSnippet:
    """
    Tests for the _compute_snippet helper function.

    This helper extracts a window of lines around an edit for returning
    to the model (token optimization vs returning full block content).
    """

    def test_edit_in_middle_returns_surrounding_window(self):
        """Edit at line 5 (F, char idx 10) with 3 context lines returns C-I."""
        snippet = _compute_snippet(ALPHABET_CONTENT, edit_start_idx=10, new_text="F", context_lines=3)
        expected = "\n".join("CDEFGHI")
        assert snippet == expected


    def test_edit_at_start_clips_to_beginning(self):
        """Edit at line 0 (A, char idx 0) doesn't go negative — returns A-D."""
        snippet = _compute_snippet(ALPHABET_CONTENT, edit_start_idx=0, new_text="A", context_lines=3)
        expected = "\n".join("ABCD")
        assert snippet == expected


    def test_edit_at_end_clips_to_end(self):
        """Edit at line 9 (J, char idx 18) doesn't exceed bounds — returns G-J."""
        snippet = _compute_snippet(ALPHABET_CONTENT, edit_start_idx=18, new_text="J", context_lines=3)
        expected = "\n".join("GHIJ")
        assert snippet == expected


    def test_multiline_edit_includes_full_edit_region(self):
        """Edit spanning lines 4-6 (E-G, char idx 8) with 2 context lines returns C-I."""
        snippet = _compute_snippet(ALPHABET_CONTENT, edit_start_idx=8, new_text="E\nF\nG", context_lines=2)
        expected = "\n".join("CDEFGHI")
        assert snippet == expected


    def test_empty_content_returns_empty(self):
        """Empty content returns empty string."""
        assert _compute_snippet("", edit_start_idx=0, new_text="") == ""


    def test_single_line_content(self):
        """Single line content returns that line."""
        assert _compute_snippet("only", edit_start_idx=0, new_text="only") == "only"


    def test_default_context_is_three(self):
        """Default context_lines is 3."""
        default = _compute_snippet(ALPHABET_CONTENT, edit_start_idx=10, new_text="F")
        explicit = _compute_snippet(ALPHABET_CONTENT, edit_start_idx=10, new_text="F", context_lines=3)
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

# Valid args for each tool (excluding label, which tests vary).
# WARNING: MEMORY_REPLACE_ARGS targets "foo one." which must exist in agent_with_editable_block.
# If you change the fixture content, update these args to match.
MEMORY_REPLACE_ARGS = {"old_string": "foo one.", "new_string": "NEW one."}
MEMORY_INSERT_ARGS = {"content": "Inserted.", "after": "<end>"}


class TestMemoryToolsShared:
    """
    Shared behaviors for memory_replace and memory_insert, parametrized.
    The use of mock_run_context in the particular position it is used in the fcn call enforces
    a function signature required for pydantic AI compatibility
    """

    @pytest_asyncio.fixture(autouse=True)
    async def setup(self, agent_with_editable_block):
        """Pull ctx/block/agent into self for all tests in this class."""
        self.ctx = agent_with_editable_block["ctx"]
        self.block = agent_with_editable_block["block"]
        self.agent = agent_with_editable_block["agent"]


    @pytest.mark.parametrize("tool_fn,valid_args", [
        pytest.param(memory_replace, MEMORY_REPLACE_ARGS, id="memory_replace"),
        pytest.param(memory_insert, MEMORY_INSERT_ARGS, id="memory_insert"),
    ])
    async def test_raises_if_label_not_found(self, tool_fn, valid_args):
        """Tool raises ModelRetry when label doesn't exist for this agent."""
        with pytest.raises(ModelRetry, match="not found"):
            await tool_fn(self.ctx, label="nonexistent", **valid_args)


    @pytest.mark.parametrize("tool_fn,valid_args,expected_content", [
        pytest.param(
            memory_replace,
            {"old_string": "foo one.", "new_string": "REPLACED."},
            "REPLACED.\nfoo two.\nfoo three.",
            id="memory_replace",
        ),
        pytest.param(
            memory_insert,
            {"content": " INSERTED", "after": "foo three."},
            "foo one.\nfoo two.\nfoo three. INSERTED",
            id="memory_insert",
        ),
    ])
    async def test_updates_correct_block(self, tool_fn, valid_args, expected_content):
        """Tool updates the block content correctly and doesn't affect other blocks."""
        session = self.ctx.deps.session

        # Add a second block to verify it's unaffected
        other_block = MemoryBlockRecord(
            agent_id=self.agent.id,
            label="other",
            description="Should be untouched",
            content="Original content.",
            char_limit=DEFAULT_BLOCK_CHAR_LIM,
            position=1,
        )
        session.add(other_block)
        await session.flush()

        await tool_fn(self.ctx, label=self.block.label, **valid_args)

        # Target block should be updated
        await session.refresh(self.block)
        assert self.block.content == expected_content

        # Other block should be untouched
        await session.refresh(other_block)
        assert other_block.content == "Original content."


    @pytest.mark.parametrize("tool_fn,overflow_args", [
        pytest.param(
            memory_replace,
            {"old_string": "foo one.", "new_string": "X" * (DEFAULT_BLOCK_CHAR_LIM + 1)},
            id="memory_replace",
        ),
        pytest.param(
            memory_insert,
            {"content": "X" * (DEFAULT_BLOCK_CHAR_LIM + 1), "after": "<end>"},
            id="memory_insert",
        ),
    ])
    async def test_raises_if_exceeds_char_limit(self, tool_fn, overflow_args):
        """Tool raises ModelRetry when result would exceed char_limit."""
        with pytest.raises(ModelRetry, match="exceeds char limit"):
            await tool_fn(self.ctx, label=self.block.label, **overflow_args)


    @pytest.mark.parametrize("tool_fn,valid_args", [
        pytest.param(memory_replace, MEMORY_REPLACE_ARGS, id="memory_replace"),
        pytest.param(memory_insert, MEMORY_INSERT_ARGS, id="memory_insert"),
    ])
    async def test_persists_change_immediately(self, tool_fn, valid_args):
        """Tool persists change to DB immediately (flush), not deferred."""
        original_content = self.block.content

        await tool_fn(self.ctx, label=self.block.label, **valid_args)

        # Block should be flushed (not in session.new or session.dirty)
        assert self.block not in self.ctx.deps.session.new
        assert self.block not in self.ctx.deps.session.dirty
        # And content should have changed
        await self.ctx.deps.session.refresh(self.block)
        assert self.block.content != original_content


    @pytest.mark.parametrize("tool_fn,tool_args,expected_new_content,edit_start_idx,new_text", [
        pytest.param(
            memory_replace,
            {"old_string": "E", "new_string": "EDITED"},
            "A\nB\nC\nD\nEDITED\nF\nG\nH\nI\nJ",
            8,      # "E" is at char idx 8 in ALPHABET_CONTENT
            "EDITED",
            id="memory_replace",
        ),
        pytest.param(
            memory_insert,
            {"content": "[INS]", "after": "E"},
            "A\nB\nC\nD\nE[INS]\nF\nG\nH\nI\nJ",
            9,      # insert_pos = idx("E") + len("E") = 8 + 1 = 9
            "[INS]",
            id="memory_insert",
        ),
    ])
    async def test_returns_snippet_on_success(
        self, session: AsyncSession, tool_fn, tool_args, expected_new_content, edit_start_idx, new_text
    ):
        """Tool returns snippet matching _compute_snippet output."""
        # Use 10-line content so snippeting actually happens
        agent_data = await _make_agent_with_block(
            session, content=ALPHABET_CONTENT, char_limit=500, agent_name="snippet-test-agent"
        )
        ctx = agent_data["ctx"]
        block = agent_data["block"]

        result = await tool_fn(ctx, label=block.label, **tool_args)

        # Result should match what _compute_snippet produces
        expected_snippet = _compute_snippet(expected_new_content, edit_start_idx, new_text)
        assert result == expected_snippet


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
    async def test_raises_if_multiple_matches_without_specify_occurrence(
        self, tool_fn, ambiguous_args
    ):
        """Tool raises ModelRetry when target appears multiple times and occurrence not specified."""
        with pytest.raises(ModelRetry, match="appears.*times"):
            await tool_fn(self.ctx, label=self.block.label, **ambiguous_args)


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
    async def test_raises_if_target_not_found(self, tool_fn, not_found_args):
        """Tool raises ModelRetry when old_string/after not found in block."""
        with pytest.raises(ModelRetry, match="not found"):
            await tool_fn(self.ctx, label=self.block.label, **not_found_args)


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
    async def test_raises_if_occurrence_exceeds_count(self, tool_fn, occurrence_args):
        """Tool raises ModelRetry when occurrence=N but fewer than N occurrences exist."""
        # "foo" appears 3 times, requesting 5th
        with pytest.raises(ModelRetry, match="occurrence.*not found"):
            await tool_fn(self.ctx, label=self.block.label, **occurrence_args)


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
    async def test_raises_if_target_empty(self, tool_fn, empty_args):
        """Tool raises ModelRetry when old_string/after is empty."""
        with pytest.raises(ModelRetry, match="empty"):
            await tool_fn(self.ctx, label=self.block.label, **empty_args)


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
    async def test_raises_if_occurrence_zero(self, tool_fn, zero_occurrence_args):
        """Tool raises ModelRetry when occurrence=0 (must be 1-indexed)."""
        with pytest.raises(ModelRetry, match="must be >= 1"):
            await tool_fn(self.ctx, label=self.block.label, **zero_occurrence_args)


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


    async def test_tools_module_cannot_trigger_recompilation(self):
        """Tools module has no access to compile_system_prompt — deferred by architecture."""
        import agent.tools as tools_module
        assert not hasattr(tools_module, "compile_system_prompt")


# --- TestMemoryReplace (tool-specific) ---

class TestMemoryReplace:
    """Tests specific to memory_replace behavior."""

    @pytest_asyncio.fixture(autouse=True)
    async def setup(self, agent_with_editable_block):
        """Pull ctx/block into self for all tests in this class."""
        self.ctx = agent_with_editable_block["ctx"]
        self.block = agent_with_editable_block["block"]


    async def test_replaces_target_and_returns_snippet_with_edit(self):
        """memory_replace returns snippet containing the replaced text."""
        result = await memory_replace(self.ctx, label=self.block.label, old_string="foo two.", new_string="REPLACED.")

        await self.ctx.deps.session.refresh(self.block)
        assert self.block.content == "foo one.\nREPLACED.\nfoo three."
        # Snippet should contain the new text
        assert "REPLACED." in result


    async def test_occurrence_targets_nth_match(self):
        """occurrence=N replaces the Nth occurrence (1-indexed)."""
        # Content: "foo one.\nfoo two.\nfoo three."
        await memory_replace(
            self.ctx, label=self.block.label, old_string="foo", new_string="SECOND", occurrence=2
        )

        await self.ctx.deps.session.refresh(self.block)
        assert self.block.content == "foo one.\nSECOND two.\nfoo three."


    async def test_only_replaces_target_occurrence(self):
        """Only the target occurrence is replaced, others unchanged."""
        # Content: "foo one.\nfoo two.\nfoo three."
        await memory_replace(
            self.ctx, label=self.block.label, old_string="foo", new_string="X", occurrence=1
        )

        await self.ctx.deps.session.refresh(self.block)
        # Only first "foo" replaced
        assert self.block.content == "X one.\nfoo two.\nfoo three."


    async def test_empty_new_string_deletes_target(self):
        """new_string='' effectively deletes the old_string."""
        # Content: "foo one.\nfoo two.\nfoo three."
        await memory_replace(self.ctx, label=self.block.label, old_string="foo two.\n", new_string="")

        await self.ctx.deps.session.refresh(self.block)
        assert self.block.content == "foo one.\nfoo three."


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
        assert self.block.content == "PREPENDED\nfoo one.\nfoo two.\nfoo three."
        # Snippet should contain the inserted text
        assert "PREPENDED" in result


    async def test_after_end_inserts_at_end(self):
        """after='<end>' inserts content at the end of the block."""
        await memory_insert(self.ctx, label=self.block.label, content="\nAPPENDED", after="<end>")

        await self.ctx.deps.session.refresh(self.block)
        assert self.block.content.endswith("\nAPPENDED")
        assert self.block.content == "foo one.\nfoo two.\nfoo three.\nAPPENDED"


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


    async def test_after_anchor_inserts_after_match(self):
        """after='anchor' inserts content immediately after the anchor string."""
        await memory_insert(self.ctx, label=self.block.label, content=" [INSERTED]", after="foo two.")

        await self.ctx.deps.session.refresh(self.block)
        assert self.block.content == "foo one.\nfoo two. [INSERTED]\nfoo three."


    async def test_occurrence_targets_nth_anchor(self):
        """occurrence=N inserts after the Nth occurrence of anchor (1-indexed)."""
        # Content: "foo one.\nfoo two.\nfoo three."

        await memory_insert(
            self.ctx, label=self.block.label, content="[2]", after="foo", occurrence=2
        )

        await self.ctx.deps.session.refresh(self.block)
        assert self.block.content == "foo one.\nfoo[2] two.\nfoo three."


    async def test_insert_does_not_overwrite(self):
        """Insert adds content without removing existing content."""
        await memory_insert(self.ctx, label=self.block.label, content="NEW", after="<end>")

        await self.ctx.deps.session.refresh(self.block)
        # Original content should still be present
        assert "foo one." in self.block.content
        assert "foo two." in self.block.content
        assert "foo three." in self.block.content
        # And new content added
        assert "NEW" in self.block.content


# =============================================================================
# send_message tests
# =============================================================================

class TestSendMessage:
    """send_message tool: inter-agent communication."""

    @pytest_asyncio.fixture(autouse=True)
    async def setup(self, session: AsyncSession):
        """Create sender agent and optionally a target for delivery tests."""
        sender = AgentRecord(
            name="sender-agent",
            agent_config=SAMPLE_AGENT_CONFIG,
            system_instructions="Sender",
        )
        session.add(sender)
        await session.flush()
        self.sender = sender
        self.session = session
        self.deps = make_deps(session, sender)
        self.ctx = mock_run_context(self.deps)

    async def _create_target(self):
        """Helper: create target agent in DB."""
        target = AgentRecord(
            name="target-agent",
            agent_config=SAMPLE_AGENT_CONFIG,
            system_instructions="Target",
        )
        self.session.add(target)
        await self.session.flush()
        return target

    def _ctx_with_registry(self, mocker):
        """Helper: context with registry configured (enables send_message)."""
        deps = AgentDeps(
            session=self.session,
            agent_record=self.sender,
            agent_app_state_reg={self.sender.id: mocker.MagicMock()},
        )
        return mock_run_context(deps)

    async def test_target_not_found_raises_model_retry(self):
        """Raises ModelRetry when no agent with that name exists."""
        with pytest.raises(ModelRetry, match="ghost"):
            await send_message(self.ctx, target_name="ghost", content="hello")

    async def test_target_not_found_is_case_sensitive(self):
        """Name lookup is case-sensitive — mismatched case raises ModelRetry."""
        with pytest.raises(ModelRetry, match="Sender-Agent"):
            await send_message(self.ctx, target_name="Sender-Agent", content="hi")

    async def test_rejects_self_message(self):
        """Raises ModelRetry when agent tries to message itself."""
        with pytest.raises(ModelRetry, match="cannot send.*yourself|self"):
            await send_message(self.ctx, target_name="sender-agent", content="talking to myself")

    async def test_raises_model_retry_when_registry_not_configured(self):
        """Raises ModelRetry when deps lacks agent_app_state_reg."""
        await self._create_target()
        with pytest.raises(ModelRetry, match="not configured"):
            await send_message(self.ctx, target_name="target-agent", content="hello")

    @pytest.mark.parametrize("delivery_succeeds,expect_error", [
        pytest.param(True, False, id="success"),
        pytest.param(False, True, id="busy"),
    ])
    async def test_delivery_outcome(self, mocker, delivery_succeeds, expect_error):
        """Delivery success returns message; failure raises ModelRetry."""
        await self._create_target()
        ctx = self._ctx_with_registry(mocker)

        async def mock_deliver(*args, **kwargs):
            kwargs["delivery_future"].set_result(delivery_succeeds)
        mocker.patch("agent.tools._deliver_message", side_effect=mock_deliver)

        if expect_error:
            with pytest.raises(ModelRetry, match="target-agent"):
                await send_message(ctx, target_name="target-agent", content="hello")
        else:
            result = await send_message(ctx, target_name="target-agent", content="hello")
            assert "delivered" in result.lower() and "target-agent" in result


class TestDeliverMessage:
    """_deliver_message: background task for inter-agent delivery."""

    @pytest.fixture
    def mock_session(self, mocker):
        """Mock get_session context manager (patch at source — imports are deferred)."""
        mock_cm = mocker.AsyncMock()
        mock_cm.__aenter__.return_value = mocker.MagicMock()
        mock_cm.__aexit__.return_value = None
        mocker.patch("db.connection.get_session", return_value=mock_cm)
        return mock_cm

    @pytest.mark.parametrize("lock_acquired", [
        pytest.param(True, id="lock_acquired"),
        pytest.param(False, id="lock_unavailable"),
    ])
    async def test_signals_future_based_on_lock_outcome(self, mocker, mock_session, lock_acquired):
        """Future receives True when lock acquired, False when AgentLockedError."""
        import asyncio
        from agent.tools import _deliver_message
        from agent.factory import AgentLockedError

        # Configure factory mock based on test case
        mock_factory_cm = mocker.AsyncMock()
        if lock_acquired:
            mock_factory_cm.__aenter__.return_value = (mocker.MagicMock(), mocker.MagicMock())
            mock_factory_cm.__aexit__.return_value = None
            async def mock_run(*args, **kwargs):
                return
                yield
            mocker.patch("agent.runner.run_stateful_agent", side_effect=mock_run)
        else:
            mock_factory_cm.__aenter__.side_effect = AgentLockedError("test-agent-id")

        mock_factory = mocker.MagicMock()
        mock_factory.build_agent_and_deps.return_value = mock_factory_cm
        mocker.patch("agent.factory.AgentFactory", return_value=mock_factory)

        future: asyncio.Future[bool] = asyncio.get_running_loop().create_future()
        await _deliver_message(
            agent_id="test-agent-id",
            user_prompt="hello",
            engine=mocker.MagicMock(),
            agent_app_state_reg={"test-agent-id": mocker.MagicMock()},
            delivery_future=future,
        )

        assert future.done() and future.result() is lock_acquired

    @pytest.mark.xfail(reason="TODO: Integration test for final architecture — verify A→B delivery works and histories stay isolated")
    async def test_cross_agent_delivery_and_history_isolation(self):
        """Integration test: A sends to B, both histories are correct and isolated.
        
        Should verify:
        - Message actually delivered to B
        - A's history contains only A's messages
        - B's history contains only B's messages (+ inter-agent message)
        - No cross-contamination from contextvars or other shared state
        """
        pytest.fail("Not implemented — waiting for stable architecture")


class _SenderTestAgent(FunctionModelTestAgent):
    """FunctionModelTestAgent subclass: emits a send_message tool call, then completes."""

    RECIPIENT_NAME = "recipient-agent"
    SEND_MSG_ARGS = json.dumps({"target_name": RECIPIENT_NAME, "content": "hello from sender"})
    SEND_MSG_CALL = DeltaToolCalls({
        0: DeltaToolCall(name="send_message", json_args=SEND_MSG_ARGS, tool_call_id="sm-tc-1")
    })

    def __init__(self):
        super().__init__()
        self.set_steps([[self.SEND_MSG_CALL], self.COMPLETION_TEXT])

    def _build(self) -> Agent:
        """Agent with send_message as its only tool."""
        agent = Agent(FunctionModel(stream_function=self._stream), deps_type=AgentDeps)
        agent.tool(send_message)
        return agent


class TestSendMessageContextIsolation(_PersistenceAndCancellationTestBase):
    """
    Integration: send_message spawns a background agent (B) that runs through
    compaction+resume (2 iterations). Verifies B's messages from both iterations
    are correctly persisted.

    Fails without the asyncio.create_task contextvar isolation fix: B's iter2
    messages are invisible to the runner (it watches the stale iter1 _RunMessages
    list while pydantic-ai writes to a fresh one).

    TODO: Move fixture dependencies to conftest once on a proper PR branch
    TODO: This test got very painful. Will need to rework once we are targeting the proper send messages impl,
    and really we probably want better integration test infrastructure in general.
    """

    @pytest_asyncio.fixture(autouse=True)
    async def setup(self, session: AsyncSession, mocker):
        # Real DB records — only needed for flush (gives us real UUIDs) and
        # as registry keys. NOT used inside the agent-run path to avoid
        # ORM attribute access inside pydantic-ai's async context (which
        # uses anyio greenlets that are not SQLAlchemy greenlets, causing
        # MissingGreenlet on lazy-load of expired ORM objects).
        sender_record = AgentRecord(
            name="sender-agent",
            agent_config=SAMPLE_AGENT_CONFIG,
            system_instructions="You are the sender.",
        )
        recipient_record = AgentRecord(
            name=_SenderTestAgent.RECIPIENT_NAME,
            agent_config=SAMPLE_AGENT_CONFIG,
            system_instructions="You are the recipient.",
        )
        session.add_all([sender_record, recipient_record])
        await session.flush()

        # Cache primitive IDs now while session is live (before any potential expiry)
        sender_id = sender_record.id
        recipient_id = recipient_record.id

        # -----------------------------------------------------------------
        # Mock records for deps — plain MagicMocks with pre-baked values.
        # deps properties (agent_id, name, config …) read through _agent_record,
        # so we replace it with a mock that never touches the ORM layer.
        # -----------------------------------------------------------------
        def _mock_record(rec_id, name, instructions):
            r = mocker.MagicMock(spec=AgentRecord)
            r.id = rec_id
            r.name = name
            r.agent_config = SAMPLE_AGENT_CONFIG
            r.system_instructions = instructions
            r.compiled_system_prompt = None
            r.sys_prompt_compiled_at = None
            r.context_window_start = None
            r.compaction_warning_fired = False
            return r

        mock_sender_record = _mock_record(sender_id, "sender-agent", "You are the sender.")
        self.mock_recipient_record = _mock_record(
            recipient_id, _SenderTestAgent.RECIPIENT_NAME, "You are the recipient."
        )

        # Shared registry — keyed by real recipient UUID
        self.app_state_reg = {recipient_id: AgentAppState()}

        # Sender deps: mock session (DB access in send_message is bypassed via
        # mocked get_all_agents; session.bind still needed for engine extraction)
        self.sender_deps = AgentDeps(
            session=_make_mock_session(),
            agent_record=mock_sender_record,
            agent_app_state_reg=self.app_state_reg,
        )
        self.sender_app_state = AgentAppState()
        self.sender_agent = _SenderTestAgent()

        # Recipient deps: mock session + mock record (no ORM access needed during B's run)
        self.recipient_deps = AgentDeps(
            session=_make_mock_session(),
            agent_record=self.mock_recipient_record,
        )
        # B runs two loop iterations: iter1 ends at ToolResultEvent (compaction fires),
        # iter2 produces only the completion text.  With the fix, all 5 expected messages
        # (UserPrompt, ToolCall+Return, ResumeNotice, Completion) are persisted correctly.
        # Without the fix, iter2's stale _RunMessages list ends in ToolCallPart which gates
        # out the AgentRunResultEvent elif branch — items 4+5 are never persisted.
        recipient_test_agent = FunctionModelTestAgent()
        # THREE_TOOL_CALL_STEPS: compaction fires after step 1 (iter1), then iter2 also has
        # tool calls. Critical: without the fix, iter2's ToolResultEvent fires with stale
        # L0 last_part=ToolCallPart, triggering re-persist of stale history via the if-branch.
        recipient_test_agent.set_steps(FunctionModelTestAgent.THREE_TOOL_CALL_STEPS)

        # Mock get_all_agents: return plain-mock records so send_message's name
        # lookup (a.name == target_name) never hits the ORM layer.
        lookup_sender = mocker.MagicMock(spec=AgentRecord)
        lookup_sender.id = sender_id
        lookup_sender.name = "sender-agent"
        lookup_recipient = mocker.MagicMock(spec=AgentRecord)
        lookup_recipient.id = recipient_id
        lookup_recipient.name = _SenderTestAgent.RECIPIENT_NAME
        mocker.patch(
            "agent.tools.get_all_agents",
            new_callable=mocker.AsyncMock,
            return_value=[lookup_sender, lookup_recipient],
        )

        # Mock AgentFactory → yields (recipient's agent, recipient's deps)
        mock_cm = mocker.AsyncMock()
        mock_cm.__aenter__.return_value = (recipient_test_agent.agent, self.recipient_deps)
        mock_cm.__aexit__.return_value = None
        mock_factory = mocker.MagicMock()
        mock_factory.build_agent_and_deps.return_value = mock_cm
        mocker.patch("agent.factory.AgentFactory", return_value=mock_factory)

        # Mock get_session in _deliver_message (engine from session.bind isn't real in tests)
        mock_sess_cm = mocker.AsyncMock()
        mock_sess_cm.__aenter__.return_value = mocker.MagicMock()
        mock_sess_cm.__aexit__.return_value = None
        mocker.patch("db.connection.get_session", return_value=mock_sess_cm)

        # Use real is_compaction_needed so token values drive compaction, not call ordering.
        # _real_is_compaction_needed is captured at module import time, before _BaseRouteTest
        # patches agent.runner.is_compaction_needed, so it still points to the real function.
        self.mock_needs_compact.side_effect = _real_is_compaction_needed

        # Return a large token count only for B's first persist call; that value propagates
        # to last_total_tokens_value → real is_compaction_needed returns True → compaction fires.
        # All other calls (A's persists and B's iter2 persists) return None → no compaction.
        b_first_persist_done = [False]

        async def _persist_side_effect(deps, messages, tool_schemas):
            if deps._agent_record is self.mock_recipient_record and not b_first_persist_done[0]:
                b_first_persist_done[0] = True
                return 999_999  # > soft_compaction_limit (10 000) → compaction fires
            return None

        self.mock_persist_messages.side_effect = _persist_side_effect

        # Simulate realistic message history: B has 5 pairs on iter1 (pre-compaction),
        # then 1 pair on iter2 (post-compaction trim). Without the fix, the stale
        # capture list still reflects the longer iter1 history, so the lower
        # new_message_idx from the short iter2 history causes old messages to be
        # re-persisted. Agent A gets empty history (only B's history matters here).
        _B_ITER1 = "__b_iter1__"
        _B_ITER2 = "__b_iter2__"
        b_history_long = make_alternating_messages(10)  # 5 req/resp pairs
        b_history_short = make_alternating_messages(2)  # 1 req/resp pair
        b_load_call = [0]

        async def _load_side_effect(session, agent_id, start_seq_id=0, end_seq_id=None):
            if agent_id == recipient_id:
                b_load_call[0] += 1
                return _B_ITER1 if b_load_call[0] == 1 else _B_ITER2
            return []

        def _deserialize_side_effect(raw):
            if raw == _B_ITER1:
                return b_history_long
            if raw == _B_ITER2:
                return b_history_short
            return []

        self.mock_load_messages.side_effect = _load_side_effect
        self.mock_deserialize_msgs.side_effect = _deserialize_side_effect

    async def test_recipient_messages_persisted_after_compaction(self):
        """
        A calls send_message → B spawned in background task.
        B runs 2 loop iterations using THREE_TOOL_CALL_STEPS (tool, tool, tool, done):
          iter1: step 1 (tool call) → compaction fires after ToolResultEvent
          iter2: steps 2–4 (two more tool calls + completion text)

        With the contextvar fix (context=contextvars.Context()):
          B's capture list is isolated per-iteration → all 9 messages persisted correctly.
        Without the fix (surgical _RunMessages only):
          iter2 reuses the stale iter1 capture list which ends in ToolCallPart;
          the ToolResultEvent if-branch fires with stale history and re-persists it → 15 msgs.

        Expected persisted messages for B (9 total):
          iter1  ① ModelRequest  [UserPromptPart(inter-agent msg)]        ← PartStartEvent elif
          iter1  ② ModelResponse [DUMMY_TOOL_CALL_PART]                   ┐ ToolResultEvent
          iter1  ③ ModelRequest  [DUMMY_TOOL_RETURN_PART]                 ┘  step 1 atomic persist
          iter2  ④ ModelRequest  [UserPromptPart(COMPACTION_RESUME_NOTICE)] ← PartStartEvent elif
          iter2  ⑤ ModelResponse [DUMMY_TOOL_CALL_PART]                   ┐ ToolResultEvent
          iter2  ⑥ ModelRequest  [DUMMY_TOOL_RETURN_PART]                 ┘  step 2 atomic persist
          iter2  ⑦ ModelResponse [DUMMY_TOOL_CALL_PART]                   ┐ ToolResultEvent
          iter2  ⑧ ModelRequest  [DUMMY_TOOL_RETURN_PART]                 ┘  step 3 atomic persist
          iter2  ⑨ ModelResponse [TextPart(COMPLETION_TEXT)]              ← AgentRunResultEvent elif
        """
        a_events = [event async for event in run_stateful_agent(
            self.sender_agent.agent,
            self.sender_deps,
            self.sender_app_state,
            "send a message",
        )]
        assert isinstance(a_events[-1], AgentRunResultEvent), "Sender should complete normally"

        # Await B's background task
        from agent.tools import background_tasks
        await asyncio.gather(*list(background_tasks))

        # Flatten all messages persisted by B across all persist_messages calls.
        # Use _agent_record identity (not .id) to avoid ORM lazy-load outside async context.
        b_persisted = [
            msg
            for call in self.mock_persist_messages.call_args_list
            if call.kwargs["deps"]._agent_record is self.mock_recipient_record
            for msg in call.kwargs["messages"]
        ]

        expected = [
            # iter1: initial inter-agent message + step 1 tool call/return
            ModelRequest(parts=[UserPromptPart(content=_format_inter_agent_message(
                "sender-agent", "hello from sender"
            ))]),
            ModelResponse(parts=[FunctionModelTestAgent.DUMMY_TOOL_CALL_PART]),
            ModelRequest(parts=[FunctionModelTestAgent.DUMMY_TOOL_RETURN_PART]),
            # iter2: compaction resume notice + steps 2 and 3 tool call/return pairs
            ModelRequest(parts=[UserPromptPart(content=COMPACTION_RESUME_NOTICE)]),
            ModelResponse(parts=[FunctionModelTestAgent.DUMMY_TOOL_CALL_PART]),
            ModelRequest(parts=[FunctionModelTestAgent.DUMMY_TOOL_RETURN_PART]),
            ModelResponse(parts=[FunctionModelTestAgent.DUMMY_TOOL_CALL_PART]),
            ModelRequest(parts=[FunctionModelTestAgent.DUMMY_TOOL_RETURN_PART]),
            # iter2: step 4 completion
            ModelResponse(parts=[TextPart(content=FunctionModelTestAgent.COMPLETION_TEXT)]),
        ]
        self._assert_ModelMessage_list_eq(b_persisted, expected)


class TestFormatInterAgentMessage:
    """_format_inter_agent_message: pure helper for origin marker formatting."""

    def test_format_structure(self):
        """Message has header with sender, newline, then content."""
        result = _format_inter_agent_message("alice", "hello world")
        assert result.startswith("[INTER AGENT MESSAGE. From: alice]")
        header, _, body = result.partition("\n")
        assert body == "hello world"
