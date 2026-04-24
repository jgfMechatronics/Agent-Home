"""Unit tests for compaction — Section 3.3

Tests is_compaction_needed and compact functions.

compact(deps, input_tokens) receives the total input_tokens from the API response.
It estimates system prompt tokens from char count, calculates message tokens,
and advances context_window_start to hit the target percentage.
"""
from datetime import datetime, timedelta, timezone

import pytest
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

    def test_returns_true_when_over_limit(self):
        """Returns True when input_tokens > soft_compaction_limit."""
        config = _make_config(soft_compaction_limit=10000)
        assert is_compaction_needed(10001, config) is True

    def test_returns_false_when_at_limit(self):
        """Returns False when input_tokens == soft_compaction_limit."""
        config = _make_config(soft_compaction_limit=10000)
        assert is_compaction_needed(10000, config) is False

    def test_returns_false_when_under_limit(self):
        """Returns False when input_tokens < soft_compaction_limit."""
        config = _make_config(soft_compaction_limit=10000)
        assert is_compaction_needed(5000, config) is False


# --- compact tests ---

class TestCompact:
    """Tests for compact(deps, input_tokens).
    
    Token math for tests:
    - System prompt tokens ≈ len(compiled_system_prompt) / 4
    - Message tokens = input_tokens - system_prompt_tokens
    - Avg tokens per message = message_tokens / message_count
    """

    async def test_advances_context_window_start(self, session: AsyncSession):
        """compact advances context_window_start pointer in DB."""
        # 400 char prompt ≈ 100 tokens, 10 messages, input_tokens=1100 → ~100 tok/msg
        config = _make_config(soft_compaction_limit=500, compaction_target_percentage=0.5)
        data = await _make_agent_with_messages(
            session, message_count=10, config=config, system_prompt_chars=400
        )
        agent = data["agent"]
        deps = data["deps"]
        
        assert agent.context_window_start is None  # Initially null
        
        await compact(deps, input_tokens=1100)
        
        await session.refresh(agent)
        assert agent.context_window_start is not None

    async def test_does_not_delete_messages(self, session: AsyncSession):
        """compact does NOT delete any messages — pointer only."""
        config = _make_config(soft_compaction_limit=500, compaction_target_percentage=0.5)
        data = await _make_agent_with_messages(
            session, message_count=10, config=config, system_prompt_chars=400
        )
        agent = data["agent"]
        deps = data["deps"]
        
        original_count = 10
        
        await compact(deps, input_tokens=1100)
        
        # Verify all messages still exist in DB via count query
        result = await session.execute(
            select(func.count()).select_from(MessageRecord).where(
                MessageRecord.agent_id == agent.id
            )
        )
        db_count = result.scalar()
        assert db_count == original_count

    async def test_minimum_history_guard_preserves_recent_messages(self, session: AsyncSession):
        """compact never evicts the most recent 4 messages."""
        # Even with aggressive settings, last 4 messages must stay
        config = _make_config(soft_compaction_limit=100, compaction_target_percentage=0.1)
        data = await _make_agent_with_messages(
            session, message_count=10, config=config, system_prompt_chars=400
        )
        agent = data["agent"]
        messages = data["messages"]
        deps = data["deps"]
        
        # High input_tokens to trigger aggressive eviction
        await compact(deps, input_tokens=5000)
        
        await session.refresh(agent)
        
        # The 4 most recent messages should still be in context
        # context_window_start should be <= timestamp of 4th-from-last message
        fourth_from_last = messages[-4]
        assert agent.context_window_start <= fourth_from_last.timestamp

    async def test_no_op_with_four_or_fewer_messages(self, session: AsyncSession):
        """compact is a no-op when agent has 4 or fewer messages in context."""
        config = _make_config(soft_compaction_limit=100, compaction_target_percentage=0.1)
        data = await _make_agent_with_messages(
            session, message_count=4, config=config, system_prompt_chars=400
        )
        agent = data["agent"]
        deps = data["deps"]
        
        await compact(deps, input_tokens=5000)
        
        await session.refresh(agent)
        # Should remain None — nothing to compact
        assert agent.context_window_start is None

    async def test_targets_percentage_of_limit(self, session: AsyncSession):
        """compact targets compaction_target_percentage of soft_compaction_limit.
        
        Setup: 400 char prompt ≈ 100 tokens, 10 messages, input_tokens=1100
        → message_tokens = 1100 - 100 = 1000, avg = 100 tok/msg
        
        Target: 50% of 1000 limit = 500 tokens
        System prompt = 100, so message budget = 400 tokens = ~4 messages
        Should keep ~4-5 messages (with tolerance for estimation drift)
        """
        config = _make_config(soft_compaction_limit=1000, compaction_target_percentage=0.5)
        data = await _make_agent_with_messages(
            session, message_count=10, config=config, system_prompt_chars=400
        )
        agent = data["agent"]
        messages = data["messages"]
        deps = data["deps"]
        
        await compact(deps, input_tokens=1100)
        
        await session.refresh(agent)
        
        # Count messages still in context (timestamp >= context_window_start)
        in_context = [m for m in messages if m.timestamp >= agent.context_window_start]
        
        # Allow tolerance for estimation drift (minimum 4 due to guard)
        assert 4 <= len(in_context) <= 6

    async def test_calls_compile_system_prompt(self, session: AsyncSession, mocker):
        """compact calls compile_system_prompt after advancing pointer."""
        from agent import compaction as compaction_module
        
        config = _make_config(soft_compaction_limit=500, compaction_target_percentage=0.5)
        data = await _make_agent_with_messages(
            session, message_count=10, config=config, system_prompt_chars=400
        )
        deps = data["deps"]
        
        spy = mocker.spy(compaction_module, "compile_system_prompt")
        
        await compact(deps, input_tokens=1100)
        
        spy.assert_called_once_with(deps)

    async def test_updates_pointer_and_recompiles_together(self, session: AsyncSession):
        """Both pointer advance and prompt recompilation happen in the same compact call."""
        config = _make_config(soft_compaction_limit=500, compaction_target_percentage=0.5)
        data = await _make_agent_with_messages(
            session, message_count=10, config=config, system_prompt_chars=400
        )
        agent = data["agent"]
        deps = data["deps"]
        
        original_compiled_at = agent.sys_prompt_compiled_at
        
        await compact(deps, input_tokens=1100)
        
        await session.refresh(agent)
        
        # Both should be updated
        assert agent.context_window_start is not None
        assert agent.sys_prompt_compiled_at is not None
        assert agent.sys_prompt_compiled_at != original_compiled_at
