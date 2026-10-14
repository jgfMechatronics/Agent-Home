"""Unit tests for compaction — Section 3.3

Tests is_compaction_needed and compact functions.

compact(deps) derives total_tokens from the last ModelResponse in context.
It estimates system prompt tokens from char count, calculates message tokens,
and advances context_window_start to retain the target fraction of current message tokens.
"""
import logging
import pytest
import pytest_asyncio
from sqlalchemy import event, func, select
from sqlalchemy.ext.asyncio import AsyncSession

from pydantic_ai.messages import ModelMessage, ToolCallPart, ToolReturnPart, RetryPromptPart

from messages.messages import deserialize_messages, load_messages, persist_messages

from agent.compaction import clear_context, compact, is_compaction_needed
from agent.types import AgentConfig, AgentDeps
from conftest import (
    SAMPLE_AGENT_CONFIG,
    make_alternating_messages,
    make_request,
    make_response,
    make_retry_pair,
    make_tool_pair,
)
from db.models import AgentRecord, MessageRecord


# --- Fixtures ---

def _make_config(
    soft_compaction_limit: int = 10000,
    compaction_target_fraction: float = 0.5,
) -> AgentConfig:
    """Create AgentConfig with specified compaction settings."""
    return AgentConfig(
        model_name="anthropic:claude-sonnet-4-20250514",
        tool_names=["memory_replace"],
        soft_compaction_limit=soft_compaction_limit,
        compaction_target_fraction=compaction_target_fraction,
    )


async def _persist_messages_load_records(
    deps: AgentDeps,
    messages: list[ModelMessage],
) -> list[MessageRecord]:
    """Persist messages via the official persist_messages, then load and return the MessageRecords."""
    await persist_messages(deps, messages, [])
    return await load_messages(deps.session, deps.agent_id)


async def _make_agent_with_messages(
    session: AsyncSession,
    message_count: int,
    *,
    config: AgentConfig | None = None,
    system_prompt_chars: int = 400,
    total_tokens: int | None = None,
) -> dict:
    """Factory for creating an agent with N alternating request/response messages.

    System prompt is set to a string of specified char length (for token estimation testing).
    Pass total_tokens to set usage on the last response, simulating real API data that
    compact() reads when calculating the compaction target.
    Returns dict with agent, messages, deps for test access.
    """
    if config is None:
        config = SAMPLE_AGENT_CONFIG

    agent = AgentRecord(
        name="test-agent",
        agent_config=config,
        system_instructions="Test agent",
        compiled_system_prompt="x" * system_prompt_chars,
    )
    session.add(agent)
    await session.flush()

    deps = AgentDeps(session=session, agent_record=agent)
    pydantic_msgs = make_alternating_messages(message_count)
    if total_tokens is not None:
        # Replace the last message (always a response for even counts) with one carrying usage data
        pydantic_msgs[-1] = make_response(total_tokens=total_tokens)
    messages = await _persist_messages_load_records(deps, pydantic_msgs)

    return {"agent": agent, "messages": messages, "deps": deps}


# --- is_compaction_needed tests ---

class TestIsCompactionNeeded:
    """Tests for is_compaction_needed(total_tokens, config)."""

    @pytest.mark.parametrize("total_tokens,expected", [
        (10001, True),   # over limit
        (10000, False),  # at limit
        (5000, False),   # under limit
    ])
    def test_threshold_behavior(self, total_tokens, expected):
        """Returns True only when total_tokens > soft_compaction_limit."""
        config = _make_config(soft_compaction_limit=10000)
        assert is_compaction_needed(total_tokens, config) is expected

    def test_returns_false_and_warns_when_total_tokens_is_none(self, caplog):
        """None total_tokens means no usage data was available; compaction is skipped with a warning."""
        config = _make_config(soft_compaction_limit=10000)
        with caplog.at_level(logging.WARNING, logger="agent.compaction"):
            result = is_compaction_needed(None, config)
        assert result is False
        assert "total_tokens=None" in caplog.text


# --- compact tests ---

class CompactTestBase:
    """
    Base class for compact tests. Allows easier commonization of setup
    
    Token math for tests:
    - System prompt tokens ≈ len(compiled_system_prompt) / 4
    - Message tokens = total_tokens - system_prompt_tokens
    - Avg tokens per message = message_tokens / message_count
    """

    async def _setup(
        self,
        session: AsyncSession,
        *,
        limit: int,
        target: float,
        msg_count: int,
        total_tokens: int,
    ):
        """Set up test scenario with specified compaction parameters.
        
        total_tokens is stored on the last ModelResponse record so compact() can
        derive it without relying on a caller-supplied value.
        """
        config = _make_config(soft_compaction_limit=limit, compaction_target_fraction=target)
        data = await _make_agent_with_messages(
            session, message_count=msg_count, config=config, system_prompt_chars=400,
            total_tokens=total_tokens,
        )
        self.agent = data["agent"]
        self.messages = data["messages"]
        self.deps = data["deps"]

class TestCompactCommon(CompactTestBase):
    """Tests with standard config: 500 limit, 0.5 target, 10 messages, 1100 tokens."""
    
    MSG_COUNT = 10

    @pytest_asyncio.fixture(autouse=True)
    async def setup(self, session: AsyncSession):
        await self._setup(session, limit=500, target=0.5, msg_count=self.MSG_COUNT, total_tokens=1100)

    async def test_advances_context_window_start(self, session: AsyncSession):
        """compact advances context_window_start pointer in DB."""
        assert self.agent.context_window_start == 0  # Defaults to 0 for fresh agents
        
        await compact(self.deps)

        await session.refresh(self.agent)
        assert self.agent.context_window_start > 0

    async def test_does_not_delete_messages(self, session: AsyncSession):
        """compact does NOT delete any messages — pointer only."""
        await compact(self.deps)

        # Verify all messages still exist in DB via count query
        result = await session.execute(
            select(func.count()).select_from(MessageRecord).where(
                MessageRecord.agent_id == self.agent.id
            )
        )
        db_count = result.scalar()
        assert db_count == self.MSG_COUNT

    async def test_calls_compile_system_prompt(self, mocker):
        """compact calls compile_system_prompt after advancing pointer."""
        from agent import compaction as compaction_module
        
        spy = mocker.spy(compaction_module, "compile_system_prompt")
        
        await compact(self.deps)
        
        spy.assert_called_once_with(self.deps)


class TestCompactReturnValue(CompactTestBase):
    """compact() returns True when it advances the pointer, False when it's a no-op."""

    async def test_returns_true_when_compaction_occurs(self, session: AsyncSession):
        await self._setup(session, limit=500, target=0.5, msg_count=10, total_tokens=1100)
        result = await compact(self.deps)
        assert result is True

    async def test_returns_false_when_context_too_small(self, session: AsyncSession):
        """MIN_MESSAGES_AFTER_COMPACTION guard prevents compaction; compact returns False."""
        await self._setup(session, limit=100, target=0.5, msg_count=2, total_tokens=1100)
        result = await compact(self.deps)
        assert result is False


class TestCompactEdgeCases(CompactTestBase):
    """Tests for edge cases requiring per-test bespoke setup params"""

    async def test_minimum_history_guard_preserves_recent_messages(self, session: AsyncSession):
        """compact never evicts the most recent 4 messages."""
        await self._setup(session, limit=100, target=0.01, msg_count=10, total_tokens=5000)
        
        await compact(self.deps)
        
        await session.refresh(self.agent)
        await session.refresh(self.messages[-4])
        assert self.agent.context_window_start <= self.messages[-4].seq_id

    async def test_no_op_with_four_or_fewer_messages(self, session: AsyncSession):
        """compact is a no-op when agent has 4 or fewer messages in context."""
        await self._setup(session, limit=100, target=0.1, msg_count=4, total_tokens=5000)
        
        await compact(self.deps)
        
        await session.refresh(self.agent)
        # Should remain 0 — nothing to compact
        assert self.agent.context_window_start == 0

    async def test_targets_percentage_of_message_tokens(self, session: AsyncSession):
        """compact retains compaction_target_fraction of current message tokens.

        Setup: 400 char prompt ≈ 100 sys_tokens, 20 messages, total_tokens=2100
        → msg_tokens = 2100 - 100 = 2000, avg = 100 tok/msg

        Target: 50% of 2000 msg_tokens = 1000 msg_tokens → keep 10 messages.
        The soft_compaction_limit is deliberately set high (10000) to confirm the target
        is relative to message tokens, not the limit.
        """
        await self._setup(session, limit=10000, target=0.5, msg_count=20, total_tokens=2100)
        
        await compact(self.deps)
        
        await session.refresh(self.agent)
        for m in self.messages:
            await session.refresh(m)
        
        # Count messages still in context (seq_id >= context_window_start)
        in_context = [m for m in self.messages if m.seq_id >= self.agent.context_window_start]
        
        # Expect exactly 10 (50% of 20); allow ±1 for integer rounding
        assert 9 <= len(in_context) <= 11

    async def test_returns_false_when_no_model_responses_in_context(self, session: AsyncSession, agent_deps: AgentDeps):
        """compact returns False without error when context contains only ModelRequests."""
        requests_only = [make_request(f"msg {i}") for i in range(6)]
        await _persist_messages_load_records(agent_deps, requests_only)

        result = await compact(agent_deps)

        assert result is False
        assert agent_deps.context_window_start == 0

    async def test_crawls_back_past_trailing_request_for_total_tokens(self, agent_deps: AgentDeps):
        """compact() correctly finds total_tokens when the last message in context is a ModelRequest."""
        msgs = make_alternating_messages(10)   # ends with response at index 9
        msgs[-1] = make_response(total_tokens=1100)
        msgs.append(make_request("new user message"))  # trailing request — compact() must skip it
        await _persist_messages_load_records(agent_deps, msgs)

        result = await compact(agent_deps)

        assert result is True  # found total_tokens on the response, compaction proceeded

    async def test_raises_when_model_response_total_tokens_is_none(self, session: AsyncSession):
        """compact raises RuntimeError when the last ModelResponse has no usage data."""
        # make_response() without total_tokens produces a record with total_tokens=None in DB
        await self._setup(session, limit=500, target=0.5, msg_count=10, total_tokens=None)

        with pytest.raises(RuntimeError, match="total_tokens=None"):
            await compact(self.deps)


class TestCompactToolPairAtomicity:
    """compact() must never split a tool call/return pair at context_window_start.

    When the naive trim point lands on a ModelRequest whose parts are ToolReturnPart or
    RetryPromptPart, the fix must walk back one message to include the preceding
    ModelResponse(ToolCallPart).  Failing to do so produces a 400 from Anthropic:
    "tool_result block(s) provided when previous turn did not request tool use."
    This results in an invalid message history being sent to the model on every subsequent turn,
    soft locking the agent.
    """

    TOTAL_TOKENS = 1250

    async def _make_agent_with_tool_sequence(
        self,
        agent_deps: AgentDeps,
        tool_pair_generator,
        *,
        config: AgentConfig,
    ) -> dict:
        agent_deps._agent_record.agent_config = config
        await agent_deps.session.flush()

        tool_call_response, tool_response_request = tool_pair_generator()
        pydantic_msgs = [
            make_request("msg 0"),
            make_response("resp 1"),
            make_request("msg 2"),
            make_response("resp 3"),
            make_request("msg 4"),
            tool_call_response,       # – ModelResponse(ToolCallPart)
            tool_response_request,    # – ModelRequest(ToolReturnPart | RetryPromptPart) — must stay paired with 5
            make_response("resp 7"),
            make_request("msg 8"),
            make_response("resp 9", total_tokens=self.TOTAL_TOKENS),
        ]
        records = await _persist_messages_load_records(agent_deps, pydantic_msgs)
        return {"agent": agent_deps._agent_record, "records": records, "deps": agent_deps}

    @pytest.mark.parametrize("tool_pair_generator", [make_tool_pair, make_retry_pair])
    async def test_does_not_orphan_tool_response(self, session: AsyncSession, agent_deps: AgentDeps, tool_pair_generator):
        """compact() walks back from the naive trim point to preserve tool pair atomicity.

        Token math — engineered to force the naive trim to land at records[6] (tool
        response), which would orphan it from records[5] (tool call):

            no system prompt → sys_tokens = 0
            10 messages, total_tokens = 1250,  avg = 125 tok/msg

            soft_compaction_limit = 1000,  compaction_target_fraction = 0.5
            target_tokens = 0.5 × 1000 = 500
            n_msg_to_keep = max(4, int(500 / 125)) = max(4, 4) = 4

        Naive result: context_window_start = records[-4].seq_id = records[6].seq_id  ← ORPHAN
        Fixed result: context_window_start = records[5].seq_id  ← pair kept intact
        """
        config = _make_config(soft_compaction_limit=1000, compaction_target_fraction=0.5)
        data = await self._make_agent_with_tool_sequence(agent_deps, tool_pair_generator, config=config)

        # sanity check: records[5] and [6] are the tool pair (assumed in ending assertion)
        call_record, return_record = deserialize_messages(data["records"][5:7])
        assert any(isinstance(p, ToolCallPart) for p in call_record.parts)
        assert any(isinstance(p, (ToolReturnPart, RetryPromptPart)) for p in return_record.parts)

        await compact(data["deps"])

        await session.refresh(data["agent"])
        await session.refresh(data["records"][5])
        assert data["agent"].context_window_start == data["records"][5].seq_id


class TestClearContext:
    """clear_context advances context_window_start past all existing messages."""

    N_MESSAGES = 4

    @pytest_asyncio.fixture
    async def persist_some_messages(self, agent_deps: AgentDeps):
        """Fixture that persists N_MESSAGES alternating messages for test setup."""
        msgs = make_alternating_messages(self.N_MESSAGES)
        records = await _persist_messages_load_records(agent_deps, msgs)
        return records

    async def test_old_messages_excluded_new_messages_visible_after_clear(self, session: AsyncSession, agent_deps: AgentDeps, persist_some_messages):
        """After clear_context, windowed load shows no old messages; new messages after clear are visible."""
        result = await clear_context(agent_deps)

        assert result is True

        # Old messages excluded from windowed view
        visible = await load_messages(session, agent_deps.agent_id, start_seq_id=agent_deps.context_window_start)
        assert not visible

        # New message added after clear IS visible in windowed view
        new_msg = make_request("test message after clear")
        await persist_messages(agent_deps, [new_msg], [])
        visible = await load_messages(session, agent_deps.agent_id, start_seq_id=agent_deps.context_window_start)
        assert len(visible) == 1

    async def test_no_messages_is_noop_and_returns_false(self, agent_deps: AgentDeps):
        """clear_context on an agent with no messages is a no-op and returns False."""
        result = await clear_context(agent_deps)

        assert result is False
        assert agent_deps.context_window_start == 0

    async def test_already_cleared_is_noop_and_returns_false(self, agent_deps: AgentDeps, persist_some_messages):
        """Second /clear with no new messages since the first is a no-op."""
        await clear_context(agent_deps)  # first clear advances pointer

        pre_no_op_ctx_start = agent_deps.context_window_start
        result = await clear_context(agent_deps)  # nothing new to clear

        assert result is False
        assert pre_no_op_ctx_start == agent_deps.context_window_start

    async def test_resets_compaction_warning_fired(self, agent_deps: AgentDeps, persist_some_messages):
        """clear_context resets compaction_warning_fired so a fresh warning can fire."""
        agent_deps.compaction_warning_fired = True

        await clear_context(agent_deps)

        assert agent_deps.compaction_warning_fired is False

    async def test_messages_remain_in_db(self, session: AsyncSession, agent_deps: AgentDeps, persist_some_messages):
        """Messages before the clear are preserved in the DB, not deleted."""
        await clear_context(agent_deps)

        all_records = await session.execute(
            select(MessageRecord).where(MessageRecord.agent_id == agent_deps.agent_id)
        )
        assert len(all_records.scalars().all()) == self.N_MESSAGES

    async def test_commits_changes(self, session: AsyncSession, agent_deps: AgentDeps, persist_some_messages):
        """clear_context commits so the new pointer survives a session expiry."""
        committed = False

        @event.listens_for(session.sync_session, "after_commit")
        def on_commit(sess):
            nonlocal committed
            committed = True

        await clear_context(agent_deps)

        assert committed
