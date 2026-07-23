"""
Tests for utils/ctx_reconstructor.py — context reconstruction from stored snapshots.
"""
import json
from uuid import uuid4

import pytest
import pytest_asyncio
from sqlalchemy.ext.asyncio import AsyncSession

from db.models import (
    AgentRecord,
    MessageRecord,
    SystemPromptSnapshot,
    ToolSchemaSnapshot,
    utcnow,
)
from messages.messages import _compute_sha256
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
