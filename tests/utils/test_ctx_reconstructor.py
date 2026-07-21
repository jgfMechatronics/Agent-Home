"""
Tests for utils/ctx_reconstructor.py — context reconstruction from stored snapshots.
"""
import hashlib
import json
from uuid import uuid4

import pytest
from sqlalchemy.ext.asyncio import AsyncSession

from db.models import (
    AgentRecord,
    MessageRecord,
    SystemPromptSnapshot,
    ToolSchemaSnapshot,
    utcnow,
)
from utils.ctx_reconstructor import reconstruct_context


def _hash_content(content: str) -> str:
    """
    SHA256 hash of content, matching the content-addressable storage pattern.
    TODO: Use the actual hash function that we use for generating the hashes on persistence here
    """
    return hashlib.sha256(content.encode("utf-8")).hexdigest()


def _make_message(
    agent_id: str,
    seq_id: int,
    system_prompt_hash: str,
    tool_schema_hash: str,
    context_window_start_msg_id: str,
    msg_id: str | None = None,
) -> MessageRecord:
    """Helper to create MessageRecord with sensible defaults."""
    msg_id = msg_id or str(uuid4())
    is_request = seq_id % 2 == 0  # Even = request, odd = response
    return MessageRecord(
        id=msg_id,
        agent_id=agent_id,
        type="ModelRequest" if is_request else "ModelResponse",
        content=f'{{"parts": [{{"type": "text", "content": "msg {seq_id}"}}]}}',
        total_tokens=None if is_request else 100,
        seq_id=seq_id,
        timestamp=utcnow(),
        system_prompt_hash=system_prompt_hash,
        tool_schema_hash=tool_schema_hash,
        context_window_start_msg_id=context_window_start_msg_id,
    )


async def _create_snapshots(session: AsyncSession) -> tuple[str, str, str, list[dict]]:
    """Create and persist snapshots. Returns (sys_hash, tool_hash, sys_content, tool_list)."""
    system_prompt = "You are a helpful assistant."
    system_prompt_hash = _hash_content(system_prompt)
    
    tool_list = [{"name": "test_tool", "description": "A test tool"}]
    tool_schemas_json = json.dumps(tool_list, sort_keys=True)
    tool_schema_hash = _hash_content(tool_schemas_json)
    
    session.add_all([
        SystemPromptSnapshot(id=system_prompt_hash, content=system_prompt, created_at=utcnow()),
        ToolSchemaSnapshot(id=tool_schema_hash, content=tool_schemas_json, created_at=utcnow()),
    ])
    await session.flush()
    
    return system_prompt_hash, tool_schema_hash, system_prompt, tool_list


@pytest.mark.asyncio
class TestReconstructContext:
    """Tests for reconstruct_context(session, message_id)."""
    # TODO: Iterate further when Opus back online.
    async def _assert_reconstruction(
        self,
        session: AsyncSession,
        target: MessageRecord,
        expected_messages: list[MessageRecord],
        expected_system_prompt: str,
        expected_tool_schemas: list[dict],
        expected_agent_id: str,
    ):
        """Common assertions for reconstruction tests."""
        result = await reconstruct_context(session, target.id)
        
        assert result.system_prompt == expected_system_prompt
        assert result.tool_schemas == expected_tool_schemas
        assert result.agent_id == expected_agent_id
        assert result.target_message.id == target.id
        assert [m.id for m in result.messages] == [m.id for m in expected_messages]

    async def test_reconstructs_context_clean_environment(
        self, session: AsyncSession, agent_record: AgentRecord
    ):
        """Basic case: 3 messages, target is last, context_window_start is first."""
        sys_hash, tool_hash, sys_prompt, tool_list = await _create_snapshots(session)
        
        # Create messages: msg0 (ctx start) -> msg1 -> msg2 (target)
        msg0_id = str(uuid4())
        msg0 = _make_message(agent_record.id, 0, sys_hash, tool_hash, msg0_id, msg_id=msg0_id)
        msg1 = _make_message(agent_record.id, 1, sys_hash, tool_hash, msg0_id)
        msg2 = _make_message(agent_record.id, 2, sys_hash, tool_hash, msg0_id)
        
        session.add_all([msg0, msg1, msg2])
        await session.flush()
        
        await self._assert_reconstruction(
            session, target=msg2, expected_messages=[msg0, msg1],
            expected_system_prompt=sys_prompt, expected_tool_schemas=tool_list,
            expected_agent_id=agent_record.id,
        )

    async def test_reconstructs_context_noisy_environment(
        self, session: AsyncSession, agent_record: AgentRecord
    ):
        """
        Same expected result as clean, but with noise:
        - Another agent's messages in the DB
        - Messages before context_window_start (earlier conversation)
        - Messages after target (later in same conversation)
        """
        sys_hash, tool_hash, sys_prompt, tool_list = await _create_snapshots(session)
        
        # --- Noise: other agent's messages ---
        other_agent = AgentRecord(
            name="other-agent",
            agent_config=agent_record.agent_config,
            system_instructions="Other agent.",
        )
        session.add(other_agent)
        await session.flush()
        
        other_msg0_id = str(uuid4())
        other_msgs = [
            _make_message(other_agent.id, i, sys_hash, tool_hash, other_msg0_id, 
                         msg_id=other_msg0_id if i == 0 else None)
            for i in range(3)
        ]
        session.add_all(other_msgs)
        
        # --- Noise: earlier conversation (before context_window_start) ---
        old_ctx_start_id = str(uuid4())
        old_msgs = [
            _make_message(agent_record.id, i, sys_hash, tool_hash, old_ctx_start_id,
                         msg_id=old_ctx_start_id if i == 0 else None)
            for i in range(3)
        ]
        session.add_all(old_msgs)
        
        # --- The actual context window we care about (seq_ids 3, 4, 5) ---
        msg0_id = str(uuid4())
        msg0 = _make_message(agent_record.id, 3, sys_hash, tool_hash, msg0_id, msg_id=msg0_id)
        msg1 = _make_message(agent_record.id, 4, sys_hash, tool_hash, msg0_id)
        msg2 = _make_message(agent_record.id, 5, sys_hash, tool_hash, msg0_id)
        session.add_all([msg0, msg1, msg2])
        
        # --- Noise: later messages (after target, different context window) ---
        later_ctx_start_id = str(uuid4())
        later_msgs = [
            _make_message(agent_record.id, 6 + i, sys_hash, tool_hash, later_ctx_start_id,
                         msg_id=later_ctx_start_id if i == 0 else None)
            for i in range(2)
        ]
        session.add_all(later_msgs)
        
        await session.flush()
        
        # Same assertions as clean environment
        await self._assert_reconstruction(
            session, target=msg2, expected_messages=[msg0, msg1],
            expected_system_prompt=sys_prompt, expected_tool_schemas=tool_list,
            expected_agent_id=agent_record.id,
        )

    async def test_returns_empty_messages_when_target_is_context_start(
        self, session: AsyncSession, agent_record: AgentRecord
    ):
        """Edge case: target message IS the context_window_start (points to itself)."""
        sys_hash, tool_hash, sys_prompt, tool_list = await _create_snapshots(session)
        
        msg_id = str(uuid4())
        msg = _make_message(agent_record.id, 0, sys_hash, tool_hash, msg_id, msg_id=msg_id)
        session.add(msg)
        await session.flush()
        
        await self._assert_reconstruction(
            session, target=msg, expected_messages=[],
            expected_system_prompt=sys_prompt, expected_tool_schemas=tool_list,
            expected_agent_id=agent_record.id,
        )

    async def test_raises_value_error_for_unknown_message_id(self, session: AsyncSession):
        """Requesting a non-existent message_id raises ValueError."""
        with pytest.raises(ValueError, match="Message not found"):
            await reconstruct_context(session, str(uuid4()))
