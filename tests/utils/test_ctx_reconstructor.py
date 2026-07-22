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

# REVIEW: Parametrize on snapshot env. The current soln only has the single targeted hash. we can strengthen the test by parameterizing on a callable in place of create snapshots, where the first callable behaves as create snapshots does now, and the second callable adds some noise to the snapshots. Basically just an extra set of snapshots, one system prompt and one schema, to make sure that the reconstructor can deal with there being multiple snapshots in there where it has to grab the correct one.
# The new helper function can just call the base create snapshots itself, and then within the new helper function it can just add some noise to the session.
@pytest.mark.asyncio
class TestReconstructContext:
    """Tests for reconstruct_context(session, message_id)."""

    @pytest_asyncio.fixture(autouse=True)
    async def setup_snapshots(self, session: AsyncSession):
        """Create snapshots once per test, store as member vars."""
        self.sys_hash, self.tool_hash, self.sys_prompt, self.tool_list = (
            await _create_snapshots(session)
        )

    def _make_message(
        self,
        agent_id: str,
        seq_id: int,
        context_window_start_msg_id: str,
        msg_id: str | None = None,
    ) -> MessageRecord:
        """Create MessageRecord using class snapshot hashes."""
        # REVIEW: the make_alternating_messages or whatever is going to need to be updated to support the new
        # If this particular helper is too specific and we would have to modify the behavior of make alternating messages just to make it work here, then skip this to-do and just delete the to-do.
        # Lets talk about this one before execution
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

    # REVIEW: Promote session to a class member, then any args passed to this fcn (inc session) that are common to all tests can just be removed as args and instead referenced via self
    # propagate self reference instead of per test fixture, and do the same for agent_record if possible (agent_record may not be relevant to this fcn)
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
        # Create messages: msg0 (ctx start) -> msg1 -> msg2 (target)
        msg0_id = str(uuid4())
        msg0 = self._make_message(agent_record.id, 0, msg0_id, msg_id=msg0_id)
        msg1 = self._make_message(agent_record.id, 1, msg0_id)
        msg2 = self._make_message(agent_record.id, 2, msg0_id)
        
        session.add_all([msg0, msg1, msg2])
        await session.flush()
        
        await self._assert_reconstruction(
            session, target=msg2, expected_messages=[msg0, msg1],
            expected_system_prompt=self.sys_prompt, expected_tool_schemas=self.tool_list,
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
        # --- Noise: other agent's messages ---
        other_agent = AgentRecord(
            name="other-agent",
            agent_config=agent_record.agent_config,
            system_instructions="Other agent.",
        )
        session.add(other_agent)
        await session.flush()
        
        other_msg0_id = str(uuid4())
        # REVIEW: not a fan of this for loop pattern, lets talk about what we can do to improve. Possibly involving the REVIEW item about using make_alternating
        other_msgs = [
            self._make_message(other_agent.id, i, other_msg0_id, 
                               msg_id=other_msg0_id if i == 0 else None)
            for i in range(3)
        ]
        session.add_all(other_msgs)
        
        # --- Noise: earlier conversation (before context_window_start) ---
        old_ctx_start_id = str(uuid4())
        old_msgs = [
            self._make_message(agent_record.id, i, old_ctx_start_id,
                               msg_id=old_ctx_start_id if i == 0 else None)
            for i in range(3)
        ]
        session.add_all(old_msgs)
        
        # --- The actual context window we care about (seq_ids 3, 4, 5) ---
        msg0_id = str(uuid4())
        msg0 = self._make_message(agent_record.id, 3, msg0_id, msg_id=msg0_id)
        msg1 = self._make_message(agent_record.id, 4, msg0_id)
        msg2 = self._make_message(agent_record.id, 5, msg0_id)
        session.add_all([msg0, msg1, msg2])
        
        # --- Noise: later messages (after target, different context window) ---
        later_ctx_start_id = str(uuid4())
        later_msgs = [
            self._make_message(agent_record.id, 6 + i, later_ctx_start_id,
                               msg_id=later_ctx_start_id if i == 0 else None)
            for i in range(2)
        ]
        session.add_all(later_msgs)
        
        await session.flush()
        
        # Same assertions as clean environment
        await self._assert_reconstruction(
            session, target=msg2, expected_messages=[msg0, msg1],
            expected_system_prompt=self.sys_prompt, expected_tool_schemas=self.tool_list,
            expected_agent_id=agent_record.id,
        )

    async def test_returns_empty_messages_when_target_is_context_start(
        self, session: AsyncSession, agent_record: AgentRecord
    ):
        """Edge case: target message IS the context_window_start (points to itself)."""
        msg_id = str(uuid4())
        msg = self._make_message(agent_record.id, 0, msg_id, msg_id=msg_id)
        session.add(msg)
        await session.flush()
        
        await self._assert_reconstruction(
            session, target=msg, expected_messages=[],
            expected_system_prompt=self.sys_prompt, expected_tool_schemas=self.tool_list,
            expected_agent_id=agent_record.id,
        )

    async def test_raises_value_error_for_unknown_message_id(self, session: AsyncSession):
        """Requesting a non-existent message_id raises ValueError."""
        with pytest.raises(ValueError, match="Message not found"):
            await reconstruct_context(session, str(uuid4()))
