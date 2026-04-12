"""
Tests for agent/factory.py — Section 3.1

Agent factory and dependency management:
- get_agent_lock: Per-agent lock registry management
- get_model: Map model name strings to Pydantic AI model instances
- get_deps: Async context manager yielding AgentDeps with lock acquisition
- build_agent_and_deps: Async context manager yielding (Agent, AgentDeps)
"""
import asyncio
from unittest.mock import MagicMock, patch

import pytest
from pydantic_ai import Agent
from pydantic_ai.models.anthropic import AnthropicModel
from sqlalchemy.ext.asyncio import AsyncSession

from agent.factory import (
    build_agent_and_deps,
    get_agent_lock,
    get_deps,
    get_model,
)
from agent.types import AgentConfig, AgentDeps
from conftest import SAMPLE_AGENT_CONFIG
from db.models import AgentRecord


# --- Fixtures ---


@pytest.fixture
def lock_reg() -> dict[str, asyncio.Lock]:
    """Fresh lock registry for each test."""
    return {}


# --- get_agent_lock tests ---


def test_get_agent_lock_returns_same_lock_for_same_id(lock_reg: dict):
    """get_agent_lock should return the same Lock instance for the same agent_id."""
    lock1 = get_agent_lock("agent-123", lock_reg)
    lock2 = get_agent_lock("agent-123", lock_reg)
    
    assert lock1 is lock2
    assert isinstance(lock1, asyncio.Lock)


def test_get_agent_lock_returns_different_locks_for_different_ids(lock_reg: dict):
    """get_agent_lock should return different Lock instances for different agent_ids."""
    lock_a = get_agent_lock("agent-aaa", lock_reg)
    lock_b = get_agent_lock("agent-bbb", lock_reg)
    
    assert lock_a is not lock_b
    assert isinstance(lock_a, asyncio.Lock)
    assert isinstance(lock_b, asyncio.Lock)


# --- get_model tests ---


def test_get_model_returns_model_for_valid_name():
    """get_model should return an AnthropicModel instance for a valid model name."""
    model = get_model("claude-sonnet-4-20250514")
    
    assert isinstance(model, AnthropicModel)


@pytest.mark.parametrize("invalid_name", [
    "not-a-real-model",
    "gpt-4",  # Wrong provider
    "",
    "claude-unknown-version",
])
def test_get_model_raises_for_invalid_name(invalid_name: str):
    """get_model should raise for unknown/unsupported model names."""
    with pytest.raises((ValueError, KeyError)):
        get_model(invalid_name)


# --- get_deps tests ---


@pytest.mark.asyncio
async def test_get_deps_yields_deps_with_expected_fields(
    session: AsyncSession, 
    agent_record: AgentRecord, 
    lock_reg: dict,
):
    """get_deps should yield AgentDeps with agent_id, session, and config populated."""
    async with get_deps(session, agent_record.id, lock_reg) as deps:
        assert isinstance(deps, AgentDeps)
        assert deps.agent_id == agent_record.id
        assert deps.session is session
        assert isinstance(deps.config, AgentConfig)
        assert deps.config.model_name == agent_record.agent_config.model_name


@pytest.mark.asyncio
async def test_get_deps_acquires_and_releases_lock(
    session: AsyncSession,
    agent_record: AgentRecord,
    lock_reg: dict,
):
    """get_deps should acquire lock before yielding and release after exit."""
    async with get_deps(session, agent_record.id, lock_reg) as deps:
        # Inside the context, the lock should be held
        lock = lock_reg[agent_record.id]
        assert lock.locked()
    
    # After exiting, lock should be released
    assert not lock.locked()


@pytest.mark.asyncio
async def test_get_deps_releases_lock_on_exception(
    session: AsyncSession,
    agent_record: AgentRecord,
    lock_reg: dict,
):
    """get_deps should release the lock even if an exception is raised inside the context."""
    with pytest.raises(RuntimeError):
        async with get_deps(session, agent_record.id, lock_reg) as deps:
            raise RuntimeError("Intentional test error")
    
    # Lock should still be released
    lock = lock_reg[agent_record.id]
    assert not lock.locked()


@pytest.mark.asyncio
async def test_get_deps_raises_for_unknown_agent_id(
    session: AsyncSession,
    lock_reg: dict,
):
    """get_deps should raise NotFound (or equivalent) for an unknown agent_id."""
    # Using a UUID that doesn't exist in the database
    fake_agent_id = "nonexistent-agent-id-12345"
    
    with pytest.raises(Exception) as exc_info:  # TODO: Narrow to specific NotFound exception
        async with get_deps(session, fake_agent_id, lock_reg) as deps:
            pass
    
    # Verify it's some kind of not-found error (exact type TBD)
    assert exc_info.value is not None


@pytest.mark.asyncio
async def test_get_deps_releases_lock_on_fetch_failure(
    session: AsyncSession,
    lock_reg: dict,
):
    """get_deps should release lock even if agent fetch fails (pre-yield exception).
    
    If implementation is lock-then-fetch, a DB failure after acquiring the lock
    must still release it. This tests that the lock is cleaned up properly.
    """
    fake_agent_id = "nonexistent-agent-id-12345"
    
    with pytest.raises(Exception):
        async with get_deps(session, fake_agent_id, lock_reg) as deps:
            pass
    
    # If a lock was created for this agent_id, it should be released
    if fake_agent_id in lock_reg:
        assert not lock_reg[fake_agent_id].locked()


# --- get_deps concurrency tests ---
# These test concurrent lock behavior. Skip if too complex per James's instruction.


@pytest.mark.asyncio
async def test_get_deps_concurrent_same_agent_blocks(
    session: AsyncSession,
    agent_record: AgentRecord,
    lock_reg: dict,
):
    """Second concurrent call on same agent_id should block until first exits."""
    execution_order = []
    
    async def first_caller():
        async with get_deps(session, agent_record.id, lock_reg):
            execution_order.append("first_entered")
            await asyncio.sleep(0.05)  # Hold lock briefly
            execution_order.append("first_exiting")
    
    async def second_caller():
        await asyncio.sleep(0.01)  # Ensure first_caller enters first
        async with get_deps(session, agent_record.id, lock_reg):
            execution_order.append("second_entered")
    
    await asyncio.gather(first_caller(), second_caller())
    
    # Second should only enter after first exits
    assert execution_order == ["first_entered", "first_exiting", "second_entered"]


@pytest.mark.asyncio
async def test_get_deps_concurrent_different_agents_no_block(
    session: AsyncSession,
    lock_reg: dict,
):
    """Concurrent calls on different agent_ids should not block each other."""
    # Create two agents for this test
    agent_a = AgentRecord(
        name="agent-a", 
        agent_config=SAMPLE_AGENT_CONFIG.model_copy(),
        system_instructions="Agent A",
    )
    agent_b = AgentRecord(
        name="agent-b",
        agent_config=SAMPLE_AGENT_CONFIG.model_copy(), 
        system_instructions="Agent B",
    )
    session.add_all([agent_a, agent_b])
    await session.flush()
    
    execution_order = []
    
    async def caller_a():
        async with get_deps(session, agent_a.id, lock_reg):
            execution_order.append("a_entered")
            await asyncio.sleep(0.05)
            execution_order.append("a_exiting")
    
    async def caller_b():
        await asyncio.sleep(0.01)  # Small delay so A enters first
        async with get_deps(session, agent_b.id, lock_reg):
            execution_order.append("b_entered")
            await asyncio.sleep(0.01)
            execution_order.append("b_exiting")
    
    await asyncio.gather(caller_a(), caller_b())
    
    # B should enter while A is still holding its lock (different agents, no blocking)
    # Expected: a_entered, b_entered, b_exiting, a_exiting
    assert execution_order.index("b_entered") < execution_order.index("a_exiting")


# --- build_agent_and_deps tests ---


@pytest.mark.asyncio
async def test_build_agent_and_deps_yields_tuple(
    session: AsyncSession,
    agent_record: AgentRecord,
    lock_reg: dict,
):
    """build_agent_and_deps should yield a valid (agent, deps) tuple."""
    async with build_agent_and_deps(session, agent_record.id, lock_reg) as (agent, deps):
        assert isinstance(agent, Agent)
        assert isinstance(deps, AgentDeps)


@pytest.mark.asyncio
async def test_build_agent_and_deps_holds_lock(
    session: AsyncSession,
    agent_record: AgentRecord,
    lock_reg: dict,
):
    """build_agent_and_deps should hold the lock for the duration of the context."""
    async with build_agent_and_deps(session, agent_record.id, lock_reg) as (agent, deps):
        lock = lock_reg[agent_record.id]
        assert lock.locked()
    
    # Released after exit
    assert not lock_reg[agent_record.id].locked()


@pytest.mark.asyncio
async def test_build_agent_and_deps_uses_correct_model(
    session: AsyncSession,
    agent_record: AgentRecord,
    lock_reg: dict,
):
    """Constructed agent should use the model from agent_config.model_name."""
    async with build_agent_and_deps(session, agent_record.id, lock_reg) as (agent, deps):
        # Agent.model is the model instance used for this agent
        assert isinstance(agent.model, AnthropicModel)
        # TODO: Check AnthropicModel API for model_name attribute instead of fragile str() check
        # For now, verify it's the right type; exact model name check deferred to implementation


@pytest.mark.xfail(reason="Cache settings are run-time (model_settings), not construction-time. Test belongs in run/integration tests.")
@pytest.mark.asyncio 
async def test_build_agent_and_deps_has_cache_settings(
    session: AsyncSession,
    agent_record: AgentRecord,
    lock_reg: dict,
):
    """Constructed agent should have Anthropic cache settings enabled.
    
    Note: anthropic_cache_* settings are passed via model_settings at run time,
    not at Agent construction. This test should verify that when we run the agent,
    we pass the correct settings. Moving to integration tests.
    """
    pytest.fail("Cache settings verified at run time, not construction")


@pytest.mark.asyncio
async def test_build_agent_and_deps_has_correct_tools(
    session: AsyncSession,
    agent_record: AgentRecord,
    lock_reg: dict,
):
    """Constructed agent should have tools matching agent_config.tool_names.
    
    Note: This test verifies get_tools_for_agent is called correctly. Verifying
    the Agent actually received the tools requires checking Agent API during impl.
    """
    # Mock get_tools_for_agent since it's Section 3.2
    mock_tools = [MagicMock(name="memory_replace"), MagicMock(name="memory_insert")]
    
    with patch("agent.factory.get_tools_for_agent", return_value=mock_tools) as mock_get_tools:
        async with build_agent_and_deps(session, agent_record.id, lock_reg) as (agent, deps):
            # Verify get_tools_for_agent was called with the agent's tool_names
            mock_get_tools.assert_called_once_with(agent_record.agent_config.tool_names)
