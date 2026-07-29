"""Tests for CompactionWarner capability.

Integration test confirms factory wiring and end-to-end behavior.
Unit tests cover edge cases for the capability logic itself.
"""
from unittest.mock import patch

import pytest
import pytest_asyncio
from pydantic_ai import ModelSettings
from pydantic_ai.messages import ModelRequest, UserPromptPart
from pydantic_ai.models.test import TestModel
from sqlalchemy.ext.asyncio import AsyncSession

from agent.compaction_warner import (
    COMPACTION_WARNING_TEXT,
    COMPACTION_WARNING_THRESHOLD_FRACTION,
)
from agent.factory import AgentFactory
from agent.runner import run_stateful_agent
from agent.types import AgentAppState, AgentConfig
from db.models import AgentRecord
from messages.messages import deserialize_messages, load_messages


# --- Integration Test ---

@pytest.mark.asyncio
class TestCompactionWarnerIntegration:
    """Integration: Factory installs capability, warning fires during run."""

    @pytest_asyncio.fixture(autouse=True)
    async def setup(self, session: AsyncSession, agent_record: AgentRecord):
        """Setup agent with low compaction threshold."""
        # Low limit so 75% threshold is crossed by TestModel's cumulative usage.
        # TestModel produces ~52 tokens per request. With call_tools='all', we get 2 requests.
        # On the 2nd request, before_model_request sees usage=52 from the 1st request.
        # Threshold = 50 * 0.75 = 37.5 ≈ 37. Since 52 > 37, warning fires on 2nd request.
        agent_record.agent_config = AgentConfig(
            model_name="claude-sonnet-4-20250514",
            tool_names=["duckduckgo_search"],
            soft_compaction_limit=50,
        )
        await session.flush()
        self.session = session
        self.agent_record = agent_record

    async def test_warning_fires_when_threshold_crossed(self):
        """Factory installs capability, warning fires when threshold crossed, persists to history.
        
        Uses ctx.enqueue() which delivers warning on NEXT request and persists to message history.
        Patches is_compaction_needed to prevent actual compaction from preempting the enqueue.
        
        TODO: Test with sequential tool calls to verify warning can interrupt tool chains.
        Currently using call_tools list to attempt multiple calls.
        """
        agent_app_state_reg: dict[str, AgentAppState] = {}
        # Try to get sequential tool calls to test mid-chain warning injection
        test_model = TestModel(
            call_tools=["duckduckgo_search", "duckduckgo_search"],
            custom_output_text="I received your message.",
        )

        with (
            patch("agent.factory.get_model", return_value=test_model),
            patch("agent.runner.is_compaction_needed", return_value=False),
        ):
            factory = AgentFactory(self.agent_record.id, agent_app_state_reg, self.session)
            async with factory.build_agent_and_deps() as (pydantic_agent, deps):
                events = [
                    event async for event in run_stateful_agent(
                        pydantic_agent, deps,
                        agent_app_state_reg[self.agent_record.id],
                        "Hello",
                    )
                ]

        # Verify run completed successfully
        assert len(events) > 0, "Should have received events"

        # Verify warning flag was set
        await self.session.refresh(self.agent_record)
        assert self.agent_record.compaction_warning_fired is True

        # Verify warning appears in persisted message history
        records = await load_messages(self.session, self.agent_record.id)
        messages = deserialize_messages(records)
        
        # Debug: concise message sequence
        for i, msg in enumerate(messages):
            parts_summary = []
            if hasattr(msg, 'parts'):
                for p in msg.parts:
                    ptype = type(p).__name__[:8]
                    if hasattr(p, 'content'):
                        content = str(p.content)[:30].replace('\n', ' ')
                        parts_summary.append(f"{ptype}:{content}")
                    else:
                        parts_summary.append(ptype)
            print(f"{i}: {type(msg).__name__[:7]} | {', '.join(parts_summary)}")
        
        found_warning = False
        for msg in messages:
            if isinstance(msg, ModelRequest):
                for part in msg.parts:
                    if isinstance(part, UserPromptPart) and COMPACTION_WARNING_TEXT in part.content:
                        found_warning = True
                        break
        
        assert found_warning, f"Warning should appear in persisted history. Messages: {[type(m).__name__ for m in messages]}"

    async def test_warning_not_injected_when_below_threshold(self):
        """No warning when token usage stays below threshold."""
        # Set high limit so threshold won't be crossed
        self.agent_record.agent_config = AgentConfig(
            model_name="claude-sonnet-4-20250514",
            tool_names=[],
            soft_compaction_limit=100000,  # threshold = 75000 tokens, won't be crossed
        )
        await self.session.flush()

        agent_app_state_reg: dict[str, AgentAppState] = {}
        test_model = TestModel(custom_output_text="Response")

        with patch("agent.factory.get_model", return_value=test_model):
            factory = AgentFactory(self.agent_record.id, agent_app_state_reg, self.session)
            async with factory.build_agent_and_deps() as (pydantic_agent, deps):
                events = [
                    event async for event in run_stateful_agent(
                        pydantic_agent, deps,
                        agent_app_state_reg[self.agent_record.id],
                        "Hello"
                    )
                ]

        # Verify run completed
        assert len(events) > 0

        # Verify warning flag NOT set
        await self.session.refresh(self.agent_record)
        assert self.agent_record.compaction_warning_fired is False


# --- Unit Tests ---

@pytest.mark.asyncio
class TestCompactionWarnerUnit:
    """Unit tests for CompactionWarner capability logic."""

    async def test_fire_once_behavior(self, session: AsyncSession, agent_record: AgentRecord):
        """Warning only fires once per compaction cycle (flag prevents re-fire).
        
        When flag is already True, capability should skip injection. Run completes normally.
        """
        # Set flag as if warning already fired
        agent_record.compaction_warning_fired = True
        # Use low limit that would otherwise trigger warning
        agent_record.agent_config = AgentConfig(
            model_name="claude-sonnet-4-20250514",
            tool_names=["duckduckgo_search"],
            soft_compaction_limit=50,
        )
        await session.flush()

        agent_app_state_reg: dict[str, AgentAppState] = {}
        # This would trigger warning on 2nd request, but flag is already set
        test_model = TestModel(call_tools="all", custom_output_text="Response", settings=ModelSettings(parallel_tool_calls=False))

        with patch("agent.factory.get_model", return_value=test_model):
            factory = AgentFactory(agent_record.id, agent_app_state_reg, session)
            async with factory.build_agent_and_deps() as (pydantic_agent, deps):
                events = [
                    event async for event in run_stateful_agent(
                        pydantic_agent, deps,
                        agent_app_state_reg[agent_record.id],
                        "Hello"
                    )
                ]

        # Verify run completed
        assert len(events) > 0

        # Flag should still be True (not reset)
        await session.refresh(agent_record)
        assert agent_record.compaction_warning_fired is True

    async def test_threshold_calculation(self):
        """Threshold is 75% of soft_compaction_limit."""
        assert COMPACTION_WARNING_THRESHOLD_FRACTION == 0.75
        
        # Verify math: 100 * 0.75 = 75
        soft_limit = 100
        expected_threshold = 75
        actual_threshold = int(soft_limit * COMPACTION_WARNING_THRESHOLD_FRACTION)
        assert actual_threshold == expected_threshold

    async def test_warning_text_content(self):
        """Warning text contains key instructions."""
        assert "compaction" in COMPACTION_WARNING_TEXT.lower()
        assert "memory" in COMPACTION_WARNING_TEXT.lower()
        assert "oldest" in COMPACTION_WARNING_TEXT.lower()
