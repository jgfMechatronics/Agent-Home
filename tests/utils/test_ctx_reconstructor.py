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
from utils.ctx_reconstructor import ReconstructedContext, reconstruct_context

# Constants for integration tests
ALL_TOOL_NAMES = list(TOOL_REGISTRY.keys())
INTEGRATION_SYSTEM_INSTRUCTIONS = "You are an integration test agent."


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
        msg0, msg1, msg2 = self._create_and_add_context_window(self.agent_record.id, 3)
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

@pytest.mark.asyncio
class TestReconstructContextIntegration:
    """Integration tests: run_stateful_agent → DB persistence → reconstruct_context."""

    @pytest_asyncio.fixture
    async def agent(self, session: AsyncSession) -> AgentRecord:
        """AgentRecord with all tools attached and known system instructions."""
        agent = AgentRecord(
            name="integration-test-agent",
            agent_config=AgentConfig(model_name="claude-sonnet-4-20250514", tool_names=ALL_TOOL_NAMES, soft_compaction_limit=10000),
            system_instructions=INTEGRATION_SYSTEM_INSTRUCTIONS,
        )
        session.add(agent)
        await session.flush()
        await compile_system_prompt(AgentDeps(session, agent))
        return agent

    async def _run_and_reconstruct(self, session: AsyncSession, agent: AgentRecord, prompt: str,
                                   test_model: TestModel) -> ReconstructedContext:
        """Run agent turn and reconstruct context from last message."""
        agent_app_state_reg: dict[str, AgentAppState] = {}
        with patch("agent.factory.get_model", return_value=test_model):
            factory = AgentFactory(agent.id, agent_app_state_reg, session)
            async with factory.build_agent_and_deps() as (pydantic_agent, deps):
                async for _ in run_stateful_agent(pydantic_agent, deps, agent_app_state_reg[agent.id], prompt):
                    pass
        last_msg_id = (await session.execute(
            select(MessageRecord.id).where(MessageRecord.agent_id == agent.id).order_by(MessageRecord.seq_id.desc()).limit(1)
        )).scalar_one()
        return await reconstruct_context(session, last_msg_id)

    def _assert_standard_result(self, result: ReconstructedContext, agent: AgentRecord):
        """Common assertions for integration tests with original config."""
        assert INTEGRATION_SYSTEM_INSTRUCTIONS in result.system_prompt
        assert len(result.tool_schemas) == len(ALL_TOOL_NAMES)
        assert result.agent_id == agent.id

    @pytest.mark.parametrize("call_tools,min_messages", [([], 1), (["duckduckgo_search"], 3)], ids=["text_only", "with_tool"])
    async def test_round_trip(self, session: AsyncSession, agent: AgentRecord, call_tools: list, min_messages: int):
        """Round-trip: run agent, reconstruct context, verify structure."""
        result = await self._run_and_reconstruct(session, agent, "Hello", TestModel(custom_output_text="Hi!", call_tools=call_tools))
        self._assert_standard_result(result, agent)
        assert len(deserialize_messages(result.messages)) >= min_messages

    async def test_context_grows_across_runs(self, session: AsyncSession, agent: AgentRecord):
        """Second run includes first run's messages; context_window_start unchanged."""
        model = TestModel(custom_output_text="Response", call_tools=[])
        first = await self._run_and_reconstruct(session, agent, "First", model)
        second = await self._run_and_reconstruct(session, agent, "Second", model)

        assert len(second.messages) > len(first.messages)
        assert second.messages[0].id == first.messages[0].id  # Same context window start
        assert second.system_prompt == first.system_prompt

    async def test_mutated_config_snapshot_dedup(self, session: AsyncSession, agent: AgentRecord):
        """After config mutation, old messages keep original snapshot, new get updated."""
        model = TestModel(custom_output_text="Response", call_tools=[])

        # First run with original config
        first = await self._run_and_reconstruct(session, agent, "First", model)

        # Mutate config
        new_instructions = "MUTATED personality."
        agent.system_instructions = new_instructions
        agent.agent_config = AgentConfig(model_name="claude-sonnet-4-20250514", tool_names=["memory_replace"], soft_compaction_limit=10000)
        await compile_system_prompt(AgentDeps(session, agent))
        await session.commit()
        await session.refresh(agent)

        # Second run with mutated config
        second = await self._run_and_reconstruct(session, agent, "Second", model)

        # First run's snapshot unchanged
        self._assert_standard_result(first, agent)

        # Second run has mutated snapshot
        assert new_instructions in second.system_prompt
        assert INTEGRATION_SYSTEM_INSTRUCTIONS not in second.system_prompt
        assert len(second.tool_schemas) == 1
