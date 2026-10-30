"""Tests for CompactionWarner capability."""
from unittest.mock import AsyncMock, MagicMock, create_autospec, patch

import pytest
import pytest_asyncio
from conftest import make_ro_engine
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

from agent.compaction import compact
from agent.deps import AgentDeps
from agent.compaction_warner import COMPACTION_WARNING_TEXT, COMPACTION_WARNING_THRESHOLD_FRACTION,CompactionWarner
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
class TestCompactionWarnerIntegration:
    """Integration tests for CompactionWarner capability."""

    @pytest.fixture(autouse=True)
    def _fake_provider_keys(self, fake_provider_keys):
        """Fake keys — _run_and_get_messages constructs real provider clients
        via build_agent_and_deps, then overrides the model with TestModel."""

    @pytest_asyncio.fixture(autouse=True)
    async def setup(self, session: AsyncSession, agent_record: AgentRecord, test_engine):
        self.session = session
        self.agent_record = agent_record
        self.test_engine = test_engine
        self.ro_engine = make_ro_engine(test_engine)

    async def _update_record(self, **fields) -> None:
        """Update agent record fields directly in the DB (test setup)."""
        async with AsyncSession(self.test_engine) as update_session:
            record = await update_session.get(AgentRecord, self.agent_record.id)
            for key, value in fields.items():
                setattr(record, key, value)
            await update_session.commit()

    async def _fresh_record(self) -> AgentRecord:
        """Fetch the agent record fresh from the DB (self.agent_record is detached/stale)."""
        async with AsyncSession(self.ro_engine) as check_session:
            return await check_session.get(AgentRecord, self.agent_record.id)

    async def _run_and_get_messages(self, model: TestModel, soft_limit: int = 100) -> list[ModelMessage]:
        """Run agent with given model and return persisted messages."""
        await self._update_record(agent_config=AgentConfig(
            model_name="anthropic:claude-sonnet-4-20250514",
            tool_names=["duckduckgo_search"],
            soft_compaction_limit=soft_limit,
        ))

        agent_app_state_reg: dict[str, AgentAppState] = {}
        with patch("agent.runner.is_compaction_needed", return_value=False):
            factory = AgentFactory(self.agent_record.id, agent_app_state_reg, self.test_engine, self.ro_engine)
            async with factory.build_agent_and_deps() as (agent, deps):
                with agent.override(model=model):
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
        
        # Verify flag is set in the database
        assert (await self._fresh_record()).compaction_warning_fired is True

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

        # Verify flag is not set in the database
        assert (await self._fresh_record()).compaction_warning_fired is False

    async def test_fire_once_per_cycle(self):
        """Warning only fires once per compaction cycle (flag prevents re-fire)."""
        await self._update_record(compaction_warning_fired=True)

        model = TestModel(call_tools="all", custom_output_text="Response")
        messages = await self._run_and_get_messages(model)

        # Should complete without injecting another warning
        warning_idx = self._find_warning_index(messages)
        assert warning_idx is None, "Should not inject warning when flag already set"

        # Verify flag is still set
        assert (await self._fresh_record()).compaction_warning_fired is True


SMALL_COMPACT_LIMIT = 100
LARGE_COMPACT_LIMIT = 1000
SMALL_THRESHOLD = int(SMALL_COMPACT_LIMIT * COMPACTION_WARNING_THRESHOLD_FRACTION)
LARGE_THRESHOLD = int(LARGE_COMPACT_LIMIT * COMPACTION_WARNING_THRESHOLD_FRACTION)


@pytest.mark.asyncio
class TestCompactionWarnerUnit:
    """Unit tests for CompactionWarner threshold logic and compaction flag reset."""

    @pytest.mark.parametrize("tokens,soft_limit,should_warn", [
        (SMALL_THRESHOLD - 1, SMALL_COMPACT_LIMIT, False),    # Below
        (SMALL_THRESHOLD, SMALL_COMPACT_LIMIT, True),         # At
        (SMALL_THRESHOLD + 1, SMALL_COMPACT_LIMIT, True),     # Above
        (LARGE_THRESHOLD - 1, LARGE_COMPACT_LIMIT, False),    # Below (large limit)
        (LARGE_THRESHOLD, LARGE_COMPACT_LIMIT, True),         # At (large limit)
        (LARGE_THRESHOLD + 1, LARGE_COMPACT_LIMIT, True),     # Above (large limit)
        (LARGE_COMPACT_LIMIT + 1, LARGE_COMPACT_LIMIT, True), # Probably shouldn't be possible
        (0, SMALL_COMPACT_LIMIT, False),                      # Zero tokens
    ])
    async def test_threshold_boundary(self, tokens: int, soft_limit: int, should_warn: bool):
        """Verify exact threshold calculation: warn iff tokens >= soft_limit * COMPACTION_WARNING_THRESHOLD_FRACTION."""
        # Mock the minimal context needed by after_model_request
        mock_config = MagicMock()
        mock_config.soft_compaction_limit = soft_limit
        
        mock_deps = create_autospec(AgentDeps, instance=True)
        mock_deps.config = mock_config
        mock_deps.compaction_warning_fired = False
        
        mock_ctx = MagicMock()
        mock_ctx.deps = mock_deps
        mock_ctx.enqueue = MagicMock()
        
        mock_response = MagicMock()
        mock_response.usage.total_tokens = tokens
        mock_request_context = MagicMock()
        
        # Call the capability directly
        warner = CompactionWarner()
        await warner.after_model_request(mock_ctx, request_context=mock_request_context, response=mock_response)
        
        if should_warn:
            mock_ctx.enqueue.assert_called_once()
            assert mock_deps.compaction_warning_fired is True
        else:
            mock_ctx.enqueue.assert_not_called()
            assert mock_deps.compaction_warning_fired is False

    async def test_compact_resets_warning_flag(self):
        """Compaction resets the warning flag for the next cycle."""
        # Constants chosen so compaction actually runs (> 4 messages, high token count)
        n_messages = 10
        soft_limit = 1000
        target_fraction = 0.25
        total_tokens = 5000

        mock_deps = create_autospec(AgentDeps, instance=True)
        mock_deps.context_window_start = None
        mock_deps.compiled_system_prompt = ""
        mock_deps.config = MagicMock()
        mock_deps.config.compaction_target_fraction = target_fraction
        mock_deps.config.soft_compaction_limit = soft_limit
        mock_deps.compaction_warning_fired = True  # Flag is set from previous warning
        
        # Need > 4 messages; type="ModelResponse" avoids deserialization branch.
        # Set total_tokens on the last record so compact() can derive it without a real API call.
        mock_messages = [MagicMock(seq_id=i, type="ModelResponse", total_tokens=None) for i in range(n_messages)]
        mock_messages[-1].total_tokens = total_tokens
        
        with patch("agent.compaction.load_messages", return_value=mock_messages):
            with patch("agent.compaction.compile_system_prompt", new_callable=AsyncMock):
                await compact(mock_deps)
        
        # Verify compaction actually ran (didn't early-return)
        assert mock_deps.context_window_start is not None, "Compaction didn't run — check test constants"
        # Flag should be reset to False
        assert mock_deps.compaction_warning_fired is False
