"""Unit tests for compaction — Section 3.3

Tests is_compaction_needed and compact functions.

compact(deps, input_tokens) receives the total input_tokens from the API response.
It estimates system prompt tokens from char count, calculates message tokens,
and advances context_window_start to hit the target percentage.
"""
from datetime import datetime, timedelta, timezone

import pytest
import pytest_asyncio
from sqlalchemy import func, select
from sqlalchemy.ext.asyncio import AsyncSession

from agent.compaction import compact, is_compaction_needed
from agent.types import AgentConfig
from conftest import make_deps, SAMPLE_AGENT_CONFIG
from db.models import AgentRecord, MessageRecord


# --- Fixtures ---

def _utcnow() -> datetime:
    """Return current UTC time as naive datetime (matches DB convention)."""
    return datetime.now(timezone.utc).replace(tzinfo=None)


def _make_config(
    soft_compaction_limit: int = 10000,
    compaction_target_percentage: float = 0.5,
) -> AgentConfig:
    """Create AgentConfig with specified compaction settings."""
    return AgentConfig(
        model_name="claude-sonnet-4-20250514",
        tool_names=["memory_replace"],
        soft_compaction_limit=soft_compaction_limit,
        compaction_target_percentage=compaction_target_percentage,
    )


async def _make_agent_with_messages(
    session: AsyncSession,
    message_count: int,
    *,
    config: AgentConfig | None = None,
    system_prompt_chars: int = 400,
) -> dict:
    """
    Factory for creating an agent with N messages and a compiled system prompt.
    
    Messages are created with sequential timestamps (1 second apart).
    System prompt is set to a string of specified char length (for token estimation testing).
    
    Returns dict with agent, messages, deps for test access.
    """
    if config is None:
        config = SAMPLE_AGENT_CONFIG
    
    # Create system prompt of specified length
    compiled_prompt = "x" * system_prompt_chars
        
    agent = AgentRecord(
        name="test-agent",
        agent_config=config,
        system_instructions="Test agent",
        compiled_system_prompt=compiled_prompt,
    )
    session.add(agent)
    await session.flush()
    
    base_time = _utcnow() - timedelta(seconds=message_count)
    messages = []
    for i in range(message_count):
        msg = MessageRecord(
            agent_id=agent.id,
            type="ModelRequest" if i % 2 == 0 else "ModelResponse",
            content=f"Message {i}",
            timestamp=base_time + timedelta(seconds=i),
        )
        messages.append(msg)
    
    session.add_all(messages)
    await session.flush()
    
    deps = make_deps(session, agent)
    
    return {"agent": agent, "messages": messages, "deps": deps}


# --- is_compaction_needed tests ---

class TestIsCompactionNeeded:
    """Tests for is_compaction_needed(input_tokens, config)."""

    @pytest.mark.parametrize("input_tokens,expected", [
        (10001, True),   # over limit
        (10000, False),  # at limit
        (5000, False),   # under limit
    ])
    def test_threshold_behavior(self, input_tokens, expected):
        """Returns True only when input_tokens > soft_compaction_limit."""
        config = _make_config(soft_compaction_limit=10000)
        assert is_compaction_needed(input_tokens, config) is expected


# --- compact tests ---

class CompactTestBase:
    """
    Base class for compact tests. Allows easier commonization of setup
    
    Token math for tests:
    - System prompt tokens ≈ len(compiled_system_prompt) / 4
    - Message tokens = input_tokens - system_prompt_tokens
    - Avg tokens per message = message_tokens / message_count
    """

    async def _setup(
        self,
        session: AsyncSession,
        *,
        limit: int,
        target: float,
        msg_count: int,
        input_tokens: int,
    ):
        """Set up test scenario with specified compaction parameters."""
        config = _make_config(soft_compaction_limit=limit, compaction_target_percentage=target)
        data = await _make_agent_with_messages(
            session, message_count=msg_count, config=config, system_prompt_chars=400
        )
        self.agent = data["agent"]
        self.messages = data["messages"]
        self.deps = data["deps"]
        self.input_tokens = input_tokens

@pytest.mark.xfail(reason="Compaction implementation pending messages.py")
class TestCompactCommon(CompactTestBase):
    """Tests with standard config: 500 limit, 0.5 target, 10 messages, 1100 tokens."""
    
    MSG_COUNT = 10

    @pytest_asyncio.fixture(autouse=True)
    async def setup(self, session: AsyncSession):
        await self._setup(session, limit=500, target=0.5, msg_count=self.MSG_COUNT, input_tokens=1100)

    async def test_advances_context_window_start(self, session: AsyncSession):
        """compact advances context_window_start pointer in DB."""
        assert self.agent.context_window_start is None  # Initially null
        
        await compact(self.deps, input_tokens=self.input_tokens)
        
        await session.refresh(self.agent)
        assert self.agent.context_window_start is not None

    async def test_does_not_delete_messages(self, session: AsyncSession):
        """compact does NOT delete any messages — pointer only."""
        await compact(self.deps, input_tokens=self.input_tokens)
        
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
        
        await compact(self.deps, input_tokens=self.input_tokens)
        
        spy.assert_called_once_with(self.deps)


@pytest.mark.xfail(reason="Compaction implementation pending messages.py")
class TestCompactEdgeCases(CompactTestBase):
    """Tests for edge cases requiring per-test bespoke setup params"""

    async def test_minimum_history_guard_preserves_recent_messages(self, session: AsyncSession):
        """compact never evicts the most recent 4 messages."""
        await self._setup(session, limit=100, target=0.01, msg_count=10, input_tokens=5000)
        
        await compact(self.deps, input_tokens=self.input_tokens)
        
        await session.refresh(self.agent)
        
        # The 4 most recent messages should still be in context
        fourth_from_last = self.messages[-4]
        assert self.agent.context_window_start <= fourth_from_last.timestamp

    async def test_no_op_with_four_or_fewer_messages(self, session: AsyncSession):
        """compact is a no-op when agent has 4 or fewer messages in context."""
        await self._setup(session, limit=100, target=0.1, msg_count=4, input_tokens=5000)
        
        await compact(self.deps, input_tokens=self.input_tokens)
        
        await session.refresh(self.agent)
        # Should remain None — nothing to compact
        assert self.agent.context_window_start is None

    async def test_targets_percentage_of_limit(self, session: AsyncSession):
        """compact targets compaction_target_percentage of soft_compaction_limit.
        
        Setup: 400 char prompt ≈ 100 tokens, 20 messages, input_tokens=2100
        → message_tokens = 2100 - 100 = 2000, avg = 100 tok/msg
        
        Target: 50% of 2000 limit = 1000 tokens
        System prompt = 100, so message budget = 900 tokens = ~9 messages
        Should keep ~8-10 messages (well above the 4-message guard)
        """
        await self._setup(session, limit=2000, target=0.5, msg_count=20, input_tokens=2100)
        
        await compact(self.deps, input_tokens=self.input_tokens)
        
        await session.refresh(self.agent)
        
        # Count messages still in context (timestamp >= context_window_start)
        in_context = [m for m in self.messages if m.timestamp >= self.agent.context_window_start]
        
        # Clear of the 4-message guard — tests percentage targeting, not the guard
        assert 8 <= len(in_context) <= 10
