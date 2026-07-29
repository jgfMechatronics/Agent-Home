"""Tests for CompactionWarner capability."""
from unittest.mock import patch

import pytest
import pytest_asyncio
from pydantic_ai.messages import (
    ModelMessage,
    ModelRequest,
    ModelResponse,
    TextPart,
    ToolCallPart,
    ToolReturnPart,
    UserPromptPart,
)
from pydantic_ai.models import ModelRequestParameters, ModelSettings
from pydantic_ai.models.test import TestModel
from sqlalchemy.ext.asyncio import AsyncSession

from agent.compaction_warner import COMPACTION_WARNING_TEXT
from agent.factory import AgentFactory
from agent.runner import run_stateful_agent
from agent.types import AgentAppState, AgentConfig
from db.models import AgentRecord
from messages.messages import deserialize_messages, load_messages


class SequentialTestModel(TestModel):
    """TestModel variant that emits tool calls one at a time (sequential, not parallel)."""

    def _request(
        self,
        messages: list[ModelMessage],
        model_settings: ModelSettings | None,
        model_request_parameters: ModelRequestParameters,
    ) -> ModelResponse:
        tool_calls = self._get_tool_calls(model_request_parameters)

        # Count tool returns to determine which tool to call next
        tool_returns = sum(
            1 for msg in messages if isinstance(msg, ModelRequest)
            for part in msg.parts if isinstance(part, ToolReturnPart)
        )

        if tool_calls and tool_returns < len(tool_calls):
            name, args = tool_calls[tool_returns]
            return ModelResponse(
                parts=[ToolCallPart(name, self.gen_tool_args(args), tool_call_id=f"seq_{tool_returns}")],
                model_name=self._model_name,
            )

        return ModelResponse(
            parts=[TextPart(content=self.custom_output_text or "Done")],
            model_name=self._model_name,
        )


@pytest.mark.asyncio
class TestCompactionWarner:
    """Integration tests for CompactionWarner capability."""

    @pytest_asyncio.fixture(autouse=True)
    async def setup(self, session: AsyncSession, agent_record: AgentRecord):
        self.session = session
        self.agent_record = agent_record

    async def _run_and_get_messages(self, model: TestModel, soft_limit: int = 50) -> list[ModelMessage]:
        """Run agent with given model and return persisted messages."""
        self.agent_record.agent_config = AgentConfig(
            model_name="claude-sonnet-4-20250514",
            tool_names=["duckduckgo_search"],
            soft_compaction_limit=soft_limit,
        )
        await self.session.flush()

        agent_app_state_reg: dict[str, AgentAppState] = {}
        with (
            patch("agent.factory.get_model", return_value=model),
            patch("agent.runner.is_compaction_needed", return_value=False),
        ):
            factory = AgentFactory(self.agent_record.id, agent_app_state_reg, self.session)
            async with factory.build_agent_and_deps() as (agent, deps):
                events = [e async for e in run_stateful_agent(
                    agent, deps, agent_app_state_reg[self.agent_record.id], "Hello"
                )]
        assert events, "Should have received events"

        records = await load_messages(self.session, self.agent_record.id)
        return deserialize_messages(records)

    def _find_warning_index(self, messages: list[ModelMessage]) -> int | None:
        """Find index of message containing compaction warning."""
        for i, msg in enumerate(messages):
            if isinstance(msg, ModelRequest):
                for part in msg.parts:
                    if isinstance(part, UserPromptPart) and COMPACTION_WARNING_TEXT in part.content:
                        return i
        return None

    def _find_tool_call_indices(self, messages: list[ModelMessage]) -> list[int]:
        """Find indices of messages containing tool calls."""
        return [
            i for i, msg in enumerate(messages) if isinstance(msg, ModelResponse)
            for part in msg.parts if isinstance(part, ToolCallPart)
        ]

    async def test_warning_fires_mid_chain_and_persists(self):
        """Warning fires when threshold crossed, injects mid-chain, and persists.
        
        Uses SequentialTestModel to prove after_model_request fires between each
        tool call, allowing warnings to interrupt the chain.
        """
        model = SequentialTestModel(
            call_tools=["duckduckgo_search", "duckduckgo_search", "duckduckgo_search"],
            custom_output_text="Done",
        )
        messages = await self._run_and_get_messages(model)

        # Warning should be present and flag set
        warning_idx = self._find_warning_index(messages)
        assert warning_idx is not None, "Warning should appear in persisted history"
        
        await self.session.refresh(self.agent_record)
        assert self.agent_record.compaction_warning_fired is True

        # Warning should appear BEFORE last tool call (proves mid-chain injection)
        tool_call_indices = self._find_tool_call_indices(messages)
        assert len(tool_call_indices) >= 2, f"Should have multiple tool calls, got {len(tool_call_indices)}"
        assert warning_idx < max(tool_call_indices), (
            f"Warning (idx {warning_idx}) should appear before last tool call (idx {max(tool_call_indices)})"
        )

    async def test_no_warning_below_threshold(self):
        """No warning when token usage stays below threshold."""
        model = TestModel(custom_output_text="Response")
        # High limit = threshold won't be crossed
        await self._run_and_get_messages(model, soft_limit=100000)

        await self.session.refresh(self.agent_record)
        assert self.agent_record.compaction_warning_fired is False

    async def test_fire_once_per_cycle(self):
        """Warning only fires once per compaction cycle (flag prevents re-fire)."""
        self.agent_record.compaction_warning_fired = True  # Pre-set flag
        
        model = TestModel(call_tools="all", custom_output_text="Response")
        messages = await self._run_and_get_messages(model)

        # Should complete without injecting another warning
        warning_idx = self._find_warning_index(messages)
        assert warning_idx is None, "Should not inject warning when flag already set"
        
        await self.session.refresh(self.agent_record)
        assert self.agent_record.compaction_warning_fired is True
