"""
Tests for utils/ctx_reconstructor.py — context reconstruction from stored snapshots.
"""
import json
from unittest.mock import patch
from uuid import uuid4

import pytest
import pytest_asyncio
from pydantic_ai.models.test import TestModel
from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from agent.factory import AgentFactory
from agent.runner import run_stateful_agent
from agent.tools import TOOL_REGISTRY
from agent.types import AgentAppState, AgentConfig, AgentDeps
from db.models import (
    AgentRecord,
    MessageRecord,
    SystemPromptSnapshot,
    ToolSchemaSnapshot,
    utcnow,
)
from memory.system_prompt_compilation import compile_system_prompt
from messages.messages import _compute_sha256, deserialize_messages
from utils.ctx_reconstructor import reconstruct_context


async def _create_snapshots(session: AsyncSession) -> tuple[str, str, str, list[dict]]:
    """Create and persist snapshots. Returns (sys_hash, tool_hash, sys_content, tool_list)."""
    system_prompt = "You are a helpful assistant."
    system_prompt_hash = _compute_sha256(system_prompt)
    
    tool_list = [{"name": "test_tool", "description": "A test tool"}]
    tool_schemas_json = json.dumps(tool_list, sort_keys=True)
    tool_schema_hash = _compute_sha256(tool_schemas_json)
    
    session.add_all([
        SystemPromptSnapshot(id=system_prompt_hash, content=system_prompt, created_at=utcnow()),
        ToolSchemaSnapshot(id=tool_schema_hash, content=tool_schemas_json, created_at=utcnow()),
    ])
    await session.flush()
    
    return system_prompt_hash, tool_schema_hash, system_prompt, tool_list


async def _create_snapshots_with_noise(session: AsyncSession) -> tuple[str, str, str, list[dict]]:
    """Create target snapshots plus noise snapshots to test correct hash selection."""
    target_snapshots = await _create_snapshots(session)
    
    # Add noise: different system prompt and tool schema that shouldn't be selected
    noise_prompt = "You are a different assistant entirely."
    noise_tools = [{"name": "noise_tool", "description": "Not the tool you want"}]
    noise_tools_json = json.dumps(noise_tools, sort_keys=True)
    
    session.add_all([
        SystemPromptSnapshot(id=_compute_sha256(noise_prompt), content=noise_prompt, created_at=utcnow()),
        ToolSchemaSnapshot(id=_compute_sha256(noise_tools_json), content=noise_tools_json, created_at=utcnow()),
    ])
    await session.flush()
    
    return target_snapshots


@pytest.mark.asyncio
class TestReconstructContext:
    """Tests for reconstruct_context(session, message_id)."""

    @pytest_asyncio.fixture(autouse=True, params=[_create_snapshots, _create_snapshots_with_noise])
    async def setup_snapshots(self, session: AsyncSession, agent_record: AgentRecord, request):
        """Create snapshots once per test, store as member vars. Parametrized to test with/without noise."""
        self.session = session
        self.agent_record = agent_record
        snapshot_factory = request.param
        self.sys_hash, self.tool_hash, self.sys_prompt, self.tool_list = (
            await snapshot_factory(session)
        )

    def _make_message(
        self,
        agent_id: str,
        seq_id: int,
        context_window_start_msg_id: str,
        msg_id: str | None = None,
    ) -> MessageRecord:
        """Create MessageRecord using class snapshot hashes."""
        msg_id = msg_id or str(uuid4())
        is_request = seq_id % 2 == 0
        return MessageRecord(
            id=msg_id,
            agent_id=agent_id,
            type="ModelRequest" if is_request else "ModelResponse",
            content=f'{{"parts": [{{"type": "text", "content": "msg {seq_id}"}}]}}',
            total_tokens=None if is_request else 100,
            seq_id=seq_id,
            timestamp=utcnow(),
            system_prompt_hash=self.sys_hash,
            tool_schema_hash=self.tool_hash,
            context_window_start_msg_id=context_window_start_msg_id,
        )

    def _create_and_add_context_window(
        self, agent_id: str, count: int, seq_id_start: int = 0
    ) -> list[MessageRecord]:
        """Create a context window and add to session."""
        ctx_start_id = str(uuid4())
        msgs = [
            self._make_message(
                agent_id,
                seq_id_start + i,
                ctx_start_id,
                msg_id=ctx_start_id if i == 0 else None,
            )
            for i in range(count)
        ]
        self.session.add_all(msgs)
        return msgs

    async def _assert_reconstruction(
        self,
        target: MessageRecord,
        expected_messages: list[MessageRecord],
        expected_agent_id: str,
    ):
        """Common assertions for reconstruction tests. Uses self.session, self.sys_prompt, self.tool_list."""
        result = await reconstruct_context(self.session, target.id)
        
        assert result.system_prompt == self.sys_prompt
        assert result.tool_schemas == self.tool_list
        assert result.agent_id == expected_agent_id
        assert result.target_message.id == target.id
        assert [m.id for m in result.messages] == [m.id for m in expected_messages]

    async def test_reconstructs_context_clean_environment(self):
        """Basic case: 3 messages, target is last, context_window_start is first."""
        # Create messages: msg0 (ctx start) -> msg1 -> msg2 (target)
        msg0_id = str(uuid4())
        msg0 = self._make_message(self.agent_record.id, 0, msg0_id, msg_id=msg0_id)
        msg1 = self._make_message(self.agent_record.id, 1, msg0_id)
        msg2 = self._make_message(self.agent_record.id, 2, msg0_id)
        
        self.session.add_all([msg0, msg1, msg2])
        await self.session.flush()
        
        await self._assert_reconstruction(
            target=msg2, expected_messages=[msg0, msg1], expected_agent_id=self.agent_record.id,
        )

    async def test_reconstructs_context_noisy_environment(self):
        """
        Same expected result as clean, but with noise:
        - Another agent's messages in the DB
        - Messages before context_window_start (earlier conversation)
        - Messages after target (later in same conversation)
        """
        # --- Noise: other agent's messages ---
        other_agent = AgentRecord(
            name="other-agent",
            agent_config=self.agent_record.agent_config,
            system_instructions="Other agent.",
        )
        self.session.add(other_agent)
        await self.session.flush()
        
        # Add messages for the other agent
        self._create_and_add_context_window(other_agent.id, 3, seq_id_start=0)
        
        # --- Noise: earlier conversation (before context_window_start) ---
        self._create_and_add_context_window(self.agent_record.id, 3, seq_id_start=0)
        
        # --- The actual context window we care about (seq_ids 3, 4, 5) ---
        msg0, msg1, msg2 = self._create_and_add_context_window(self.agent_record.id, 3, seq_id_start=3)
        
        # --- Noise: later messages (after target, different context window) ---
        self._create_and_add_context_window(self.agent_record.id, 2, seq_id_start=6)
        
        await self.session.flush()
        
        # Same assertions as clean environment
        await self._assert_reconstruction(
            target=msg2, expected_messages=[msg0, msg1], expected_agent_id=self.agent_record.id,
        )

    async def test_returns_empty_messages_when_target_is_context_start(self):
        """Edge case: target message IS the context_window_start (points to itself)."""
        msg_id = str(uuid4())
        msg = self._make_message(self.agent_record.id, 0, msg_id, msg_id=msg_id)
        self.session.add(msg)
        await self.session.flush()
        
        await self._assert_reconstruction(
            target=msg, expected_messages=[], expected_agent_id=self.agent_record.id,
        )

    async def test_raises_value_error_for_unknown_message_id(self):
        """Requesting a non-existent message_id raises ValueError."""
        with pytest.raises(ValueError, match="Message not found"):
            await reconstruct_context(self.session, str(uuid4()))


# =============================================================================
# Integration Tests — Full round-trip through run_stateful_agent
# =============================================================================

# All available tool names — used to attach all tools as a compatibility canary
ALL_TOOL_NAMES = list(TOOL_REGISTRY.keys())

# Known system instructions for assertion
INTEGRATION_TEST_SYSTEM_INSTRUCTIONS = "You are an integration test agent."


@pytest.mark.asyncio
class TestReconstructContextIntegration:
    """Integration tests: run_stateful_agent → DB persistence → reconstruct_context."""

    @pytest_asyncio.fixture
    async def integration_agent(self, session: AsyncSession) -> AgentRecord:
        """AgentRecord with all tools attached and known system instructions."""
        config = AgentConfig(
            model_name="claude-sonnet-4-20250514",
            tool_names=ALL_TOOL_NAMES,
            soft_compaction_limit=10000,
        )
        agent = AgentRecord(
            name="integration-test-agent",
            agent_config=config,
            system_instructions=INTEGRATION_TEST_SYSTEM_INSTRUCTIONS,
        )
        session.add(agent)
        await session.flush()
        
        # Compile system prompt (normally done by agent creation route)
        deps = AgentDeps(session, agent)
        await compile_system_prompt(deps)
        
        return agent

    async def _run_agent_turn(
        self,
        session: AsyncSession,
        agent_record: AgentRecord,
        user_prompt: str,
        test_model: TestModel,
    ) -> None:
        """Run one agent turn via AgentFactory + run_stateful_agent, draining all events."""
        agent_app_state_reg: dict[str, AgentAppState] = {}
        
        with patch("agent.factory.get_model", return_value=test_model):
            factory = AgentFactory(agent_record.id, agent_app_state_reg, session)
            async with factory.build_agent_and_deps() as (agent, deps):
                agent_app_state = agent_app_state_reg[agent_record.id]
                async for _ in run_stateful_agent(agent, deps, agent_app_state, user_prompt):
                    pass  # Drain all events; messages auto-persist

    async def _get_last_message_id(self, session: AsyncSession, agent_id: str) -> str:
        """Get the ID of the most recent message for an agent."""
        result = await session.execute(
            select(MessageRecord.id)
            .where(MessageRecord.agent_id == agent_id)
            .order_by(MessageRecord.seq_id.desc())
            .limit(1)
        )
        row = result.scalar_one()
        return row

    async def test_text_only_round_trip(self, session: AsyncSession, integration_agent: AgentRecord):
        """Basic round-trip: text-only agent run, then reconstruct context."""
        test_model = TestModel(custom_output_text="Hello from TestModel!", call_tools=[])
        
        await self._run_agent_turn(session, integration_agent, "Hi there", test_model)
        
        # Get the last message and reconstruct
        last_msg_id = await self._get_last_message_id(session, integration_agent.id)
        result = await reconstruct_context(session, last_msg_id)
        
        # Verify system prompt matches what we set
        assert INTEGRATION_TEST_SYSTEM_INSTRUCTIONS in result.system_prompt
        
        # Verify tool schemas count matches tools we attached
        assert len(result.tool_schemas) == len(ALL_TOOL_NAMES)
        
        # Verify agent_id
        assert result.agent_id == integration_agent.id
        
        # Verify messages can be deserialized back to ModelMessages
        deserialized = deserialize_messages(result.messages)
        assert len(deserialized) >= 1  # At least the user request

    async def test_tool_call_round_trip(self, session: AsyncSession, integration_agent: AgentRecord):
        """Round-trip with tool call: tests richer message types (ToolCallPart, ToolReturnPart)."""
        # TestModel with call_tools will emit a tool call before producing text
        test_model = TestModel(
            custom_output_text="Search complete!",
            call_tools=["duckduckgo_search"],  # Will call this tool with generated args
        )
        
        await self._run_agent_turn(session, integration_agent, "Search for cats", test_model)
        
        # Get the last message and reconstruct
        last_msg_id = await self._get_last_message_id(session, integration_agent.id)
        result = await reconstruct_context(session, last_msg_id)
        
        # Same structural assertions as text-only
        assert INTEGRATION_TEST_SYSTEM_INSTRUCTIONS in result.system_prompt
        assert len(result.tool_schemas) == len(ALL_TOOL_NAMES)
        assert result.agent_id == integration_agent.id
        
        # Verify tool call messages are present in the history
        deserialized = deserialize_messages(result.messages)
        # Should have: user request, model response with tool call, tool return
        assert len(deserialized) >= 3, f"Expected >= 3 messages for tool call flow, got {len(deserialized)}"

    async def test_second_run_context_window_shift(self, session: AsyncSession, integration_agent: AgentRecord):
        """Second run with existing data: tests context_window_start_msg_id shifts correctly."""
        test_model = TestModel(custom_output_text="Response", call_tools=[])
        
        # First run
        await self._run_agent_turn(session, integration_agent, "First message", test_model)
        first_run_last_msg_id = await self._get_last_message_id(session, integration_agent.id)
        
        # Get first run's context for comparison
        first_result = await reconstruct_context(session, first_run_last_msg_id)
        
        # Second run (same agent, more messages)
        await self._run_agent_turn(session, integration_agent, "Second message", test_model)
        second_run_last_msg_id = await self._get_last_message_id(session, integration_agent.id)
        
        # Reconstruct context for second run's last message
        second_result = await reconstruct_context(session, second_run_last_msg_id)
        
        # The second run's context should include messages from the first run
        assert len(second_result.messages) > len(first_result.messages), (
            f"Second run should have more context messages than first run. "
            f"First: {len(first_result.messages)}, Second: {len(second_result.messages)}"
        )
        
        # Both runs should share the same context window start (no spurious shift)
        assert second_result.messages[0].id == first_result.messages[0].id, (
            "Context window start should be the same across both runs"
        )
        
        # Both should have same system prompt and tool schemas
        assert second_result.system_prompt == first_result.system_prompt
        assert len(second_result.tool_schemas) == len(first_result.tool_schemas)

    async def test_mutated_config_snapshot_dedup(self, session: AsyncSession, integration_agent: AgentRecord):
        """Mutated system prompt + toolset: proves each message gets its own snapshot."""
        test_model = TestModel(custom_output_text="Response", call_tools=[])
        agent_id = integration_agent.id  # Store ID before any commits
        
        # --- First run with original config ---
        await self._run_agent_turn(session, integration_agent, "First message", test_model)
        first_run_last_msg_id = await self._get_last_message_id(session, agent_id)
        
        # --- Mutate the agent config ---
        # Change system instructions
        new_instructions = "You are a MUTATED test agent with different personality."
        integration_agent.system_instructions = new_instructions
        
        # Change tool set (use subset of tools)
        new_config = AgentConfig(
            model_name="claude-sonnet-4-20250514",
            tool_names=["memory_replace"],  # Fewer tools than original
            soft_compaction_limit=10000,
        )
        integration_agent.agent_config = new_config
        
        # Recompile system prompt with new settings
        deps = AgentDeps(session, integration_agent)
        await compile_system_prompt(deps)
        await session.commit()
        
        # Refresh the agent record after commit
        await session.refresh(integration_agent)
        
        # --- Second run with mutated config ---
        await self._run_agent_turn(session, integration_agent, "Second message after mutation", test_model)
        second_run_last_msg_id = await self._get_last_message_id(session, integration_agent.id)
        
        # --- Verify first run's message still has original snapshot ---
        first_result = await reconstruct_context(session, first_run_last_msg_id)
        assert INTEGRATION_TEST_SYSTEM_INSTRUCTIONS in first_result.system_prompt
        assert len(first_result.tool_schemas) == len(ALL_TOOL_NAMES)
        
        # --- Verify second run's message has mutated snapshot ---
        second_result = await reconstruct_context(session, second_run_last_msg_id)
        assert new_instructions in second_result.system_prompt
        assert INTEGRATION_TEST_SYSTEM_INSTRUCTIONS not in second_result.system_prompt
        assert len(second_result.tool_schemas) == 1  # Only memory_replace
        
        # --- Verify the snapshots are actually different ---
        assert first_result.system_prompt != second_result.system_prompt
        assert first_result.tool_schemas != second_result.tool_schemas
