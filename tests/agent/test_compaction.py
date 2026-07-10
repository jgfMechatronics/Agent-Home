"""Unit tests for compaction — Section 3.3

Tests is_compaction_needed and compact functions.

compact(deps, total_tokens) receives the total_tokens from the API response.
It estimates system prompt tokens from char count, calculates message tokens,
and advances context_window_start to hit the target percentage.

TODO: We may want to change compaction target calculation to be relative to tokens free for messages
as opposed to relative to total tokens. With the current impl the compaction gets progressively more
aggressive as system prompt grows. It would probably be preferable for compactions to just get more and more
frequent as the system prompt gets problematically large as opposed to more and more aggressive 
where they're basically deleting all messages.
"""
import logging
import pytest
import pytest_asyncio
from sqlalchemy import func, select
from sqlalchemy.ext.asyncio import AsyncSession

from pydantic_ai.messages import ModelMessage, ToolCallPart, ToolReturnPart, RetryPromptPart

from messages.messages import deserialize_messages, load_messages, persist_messages

from agent.compaction import compact, is_compaction_needed
from agent.types import AgentConfig, AgentDeps
from conftest import (
    SAMPLE_AGENT_CONFIG,
    make_deps,
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
        model_name="claude-sonnet-4-20250514",
        tool_names=["memory_replace"],
        soft_compaction_limit=soft_compaction_limit,
        compaction_target_fraction=compaction_target_fraction,
    )


async def _persist_messages_load_records(
    deps: AgentDeps,
    messages: list[ModelMessage],
) -> list[MessageRecord]:
    """Persist messages via the official persist_messages, then load and return the MessageRecords."""
    await persist_messages(deps, messages)
    return await load_messages(deps.session, deps.agent_id)


async def _make_agent_with_messages(
    session: AsyncSession,
    message_count: int,
    *,
    config: AgentConfig | None = None,
    system_prompt_chars: int = 400,
) -> dict:
    """Factory for creating an agent with N alternating request/response messages.

    System prompt is set to a string of specified char length (for token estimation testing).
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

    deps = make_deps(session, agent)
    pydantic_msgs = [
        make_request(f"msg {i}") if i % 2 == 0 else make_response(f"resp {i}")
        for i in range(message_count)
    ]
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
        """Set up test scenario with specified compaction parameters."""
        config = _make_config(soft_compaction_limit=limit, compaction_target_fraction=target)
        data = await _make_agent_with_messages(
            session, message_count=msg_count, config=config, system_prompt_chars=400
        )
        self.agent = data["agent"]
        self.messages = data["messages"]
        self.deps = data["deps"]
        self.total_tokens = total_tokens

class TestCompactCommon(CompactTestBase):
    """Tests with standard config: 500 limit, 0.5 target, 10 messages, 1100 tokens."""
    
    MSG_COUNT = 10

    @pytest_asyncio.fixture(autouse=True)
    async def setup(self, session: AsyncSession):
        await self._setup(session, limit=500, target=0.5, msg_count=self.MSG_COUNT, total_tokens=1100)

    async def test_advances_context_window_start(self, session: AsyncSession):
        """compact advances context_window_start pointer in DB."""
        assert self.agent.context_window_start is None  # Initially null
        
        await compact(self.deps, total_tokens=self.total_tokens)

        await session.refresh(self.agent)
        assert self.agent.context_window_start is not None

    async def test_does_not_delete_messages(self, session: AsyncSession):
        """compact does NOT delete any messages — pointer only."""
        await compact(self.deps, total_tokens=self.total_tokens)

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
        
        await compact(self.deps, total_tokens=self.total_tokens)
        
        spy.assert_called_once_with(self.deps)


class TestCompactEdgeCases(CompactTestBase):
    """Tests for edge cases requiring per-test bespoke setup params"""

    async def test_minimum_history_guard_preserves_recent_messages(self, session: AsyncSession):
        """compact never evicts the most recent 4 messages."""
        await self._setup(session, limit=100, target=0.01, msg_count=10, total_tokens=5000)
        
        await compact(self.deps, total_tokens=self.total_tokens)
        
        await session.refresh(self.agent)
        await session.refresh(self.messages[-4])
        assert self.agent.context_window_start <= self.messages[-4].timestamp

    async def test_no_op_with_four_or_fewer_messages(self, session: AsyncSession):
        """compact is a no-op when agent has 4 or fewer messages in context."""
        await self._setup(session, limit=100, target=0.1, msg_count=4, total_tokens=5000)
        
        await compact(self.deps, total_tokens=self.total_tokens)
        
        await session.refresh(self.agent)
        # Should remain None — nothing to compact
        assert self.agent.context_window_start is None

    async def test_targets_percentage_of_limit(self, session: AsyncSession):
        """compact targets compaction_target_fraction of soft_compaction_limit.
        
        Setup: 400 char prompt ≈ 100 tokens, 20 messages, total_tokens=2100
        → message_tokens = 2100 - 100 = 2000, avg = 100 tok/msg
        
        Target: 50% of 2000 limit = 1000 tokens
        System prompt = 100, so message budget = 900 tokens = ~9 messages
        Should keep ~8-10 messages (well above the 4-message guard)
        """
        await self._setup(session, limit=2000, target=0.5, msg_count=20, total_tokens=2100)
        
        await compact(self.deps, total_tokens=self.total_tokens)
        
        await session.refresh(self.agent)
        for m in self.messages:
            await session.refresh(m)
        
        # Count messages still in context (timestamp >= context_window_start)
        in_context = [m for m in self.messages if m.timestamp >= self.agent.context_window_start]
        
        # Clear of the 4-message guard — tests percentage targeting, not the guard
        assert 8 <= len(in_context) <= 10


class TestCompactToolPairAtomicity:
    """compact() must never split a tool call/return pair at context_window_start.

    When the naive trim point lands on a ModelRequest whose parts are ToolReturnPart or
    RetryPromptPart, the fix must walk back one message to include the preceding
    ModelResponse(ToolCallPart).  Failing to do so produces a 400 from Anthropic:
    "tool_result block(s) provided when previous turn did not request tool use."
    This results in an invalid message history being sent to the model on every subsequent turn,
    soft locking the agent.
    """

    async def _make_agent_with_tool_sequence(
        self,
        session: AsyncSession,
        agent_record: AgentRecord,
        tool_pair_generator,
        *,
        config: AgentConfig,
    ) -> dict:
        agent_record.agent_config = config
        await session.flush()

        deps = make_deps(session, agent_record)
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
            make_response("resp 9"),
        ]
        records = await _persist_messages_load_records(deps, pydantic_msgs)
        return {"agent": agent_record, "records": records, "deps": deps}

    @pytest.mark.parametrize("tool_pair_generator", [make_tool_pair, make_retry_pair])
    async def test_does_not_orphan_tool_response(self, session: AsyncSession, agent_record: AgentRecord, tool_pair_generator):
        """compact() walks back from the naive trim point to preserve tool pair atomicity.

        Token math — engineered to force the naive trim to land at records[6] (tool
        response), which would orphan it from records[5] (tool call):

            no system prompt → sys_tokens = 0
            10 messages, total_tokens = 1250,  avg = 125 tok/msg

            soft_compaction_limit = 1000,  compaction_target_fraction = 0.5
            target_tokens = 0.5 × 1000 = 500
            n_msg_to_keep = max(4, int(500 / 125)) = max(4, 4) = 4

        Naive result: context_window_start = records[-4].timestamp = records[6].timestamp  ← ORPHAN
        Fixed result: context_window_start = records[5].timestamp  ← pair kept intact
        """
        config = _make_config(soft_compaction_limit=1000, compaction_target_fraction=0.5)
        data = await self._make_agent_with_tool_sequence(session, agent_record, tool_pair_generator, config=config)

        # sanity check: records[5] and [6] are the tool pair (assumed in ending assertion)
        call_record, return_record = deserialize_messages(data["records"][5:7])
        assert any(isinstance(p, ToolCallPart) for p in call_record.parts)
        assert any(isinstance(p, (ToolReturnPart, RetryPromptPart)) for p in return_record.parts)

        await compact(data["deps"], total_tokens=1250)

        await session.refresh(data["agent"])
        await session.refresh(data["records"][5])
        assert data["agent"].context_window_start == data["records"][5].timestamp
