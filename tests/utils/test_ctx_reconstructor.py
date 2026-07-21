"""
Tests for utils/ctx_reconstructor.py — context reconstruction from stored snapshots.
"""
import hashlib
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
from utils.ctx_reconstructor import ReconstructedContext, reconstruct_context

def _hash_content(content: str) -> str:
    """
    SHA256 hash of content, matching the content-addressable storage pattern.
    TODO: Use the actual hash function that we use for generating the hashes on persistence here
    """
    return hashlib.sha256(content.encode("utf-8")).hexdigest()


@pytest_asyncio.fixture
async def populated_context(session: AsyncSession, agent_record: AgentRecord) -> dict:
    """
    Set up a minimal but complete context for reconstruction testing.
    
    Creates:
    - 1 SystemPromptSnapshot
    - 1 ToolSchemaSnapshot  
    - 3 MessageRecords (seq_ids 0, 1, 2) all pointing to same snapshots
      - msg0: context_window_start (points to itself)
      - msg1: points to msg0 as context_window_start
      - msg2: points to msg0 as context_window_start (this will be our target)
    
    Returns dict with all created objects for test assertions.
    """
    # Create snapshots
    system_prompt = "You are a helpful assistant."
    system_prompt_hash = _hash_content(system_prompt)
    sys_snapshot = SystemPromptSnapshot(
        id=system_prompt_hash,
        content=system_prompt,
        created_at=utcnow(),
    )
    
    tool_schemas = json.dumps([{"name": "test_tool", "description": "A test tool"}], sort_keys=True)
    tool_schema_hash = _hash_content(tool_schemas)
    tool_snapshot = ToolSchemaSnapshot(
        id=tool_schema_hash,
        content=tool_schemas,
        created_at=utcnow(),
    )
    
    session.add_all([sys_snapshot, tool_snapshot])
    await session.flush()
    
    # Create messages
    msg0_id = str(uuid4())
    msg1_id = str(uuid4())
    msg2_id = str(uuid4())
    
    msg0 = MessageRecord(
        id=msg0_id,
        agent_id=agent_record.id,
        type="ModelRequest",
        content='{"parts": [{"type": "user-prompt", "content": "Hello"}]}',
        total_tokens=None,
        seq_id=0,
        timestamp=utcnow(),
        system_prompt_hash=system_prompt_hash,
        tool_schema_hash=tool_schema_hash,
        context_window_start_msg_id=msg0_id,  # points to itself
    )
    
    msg1 = MessageRecord(
        id=msg1_id,
        agent_id=agent_record.id,
        type="ModelResponse",
        content='{"parts": [{"type": "text", "content": "Hi there!"}]}',
        total_tokens=100,
        seq_id=1,
        timestamp=utcnow(),
        system_prompt_hash=system_prompt_hash,
        tool_schema_hash=tool_schema_hash,
        context_window_start_msg_id=msg0_id,
    )
    
    msg2 = MessageRecord(
        id=msg2_id,
        agent_id=agent_record.id,
        type="ModelRequest",
        content='{"parts": [{"type": "user-prompt", "content": "How are you?"}]}',
        total_tokens=None,
        seq_id=2,
        timestamp=utcnow(),
        system_prompt_hash=system_prompt_hash,
        tool_schema_hash=tool_schema_hash,
        context_window_start_msg_id=msg0_id,
    )
    
    session.add_all([msg0, msg1, msg2])
    await session.flush()
    
    return {
        "agent": agent_record,
        "sys_snapshot": sys_snapshot,
        "tool_snapshot": tool_snapshot,
        "messages": [msg0, msg1, msg2],
        "system_prompt": system_prompt,
        "tool_schemas": tool_schemas,
    }


@pytest.mark.asyncio
class TestReconstructContext:
    """Tests for reconstruct_context(session, message_id)."""

    async def test_happy_path_reconstructs_full_context(
        self, session: AsyncSession, populated_context: dict
    ):
        """
        Given a target message, reconstruct_context returns:
        - system_prompt from the target's snapshot
        - tool_schemas from the target's snapshot
        - messages from context_window_start up to (exclusive) target
        - target_message itself
        - agent_id
        """
        messages = populated_context["messages"]
        target = messages[2]  # msg2, seq_id=2
        
        result = await reconstruct_context(session, target.id)
        
        # Manually construct expected
        expected = ReconstructedContext(
            system_prompt=populated_context["system_prompt"],
            tool_schemas=json.loads(populated_context["tool_schemas"]),
            messages=[messages[0], messages[1]],  # msg0 and msg1, exclusive of target
            target_message=target,
            agent_id=populated_context["agent"].id,
        )
        
        assert result.system_prompt == expected.system_prompt
        assert result.tool_schemas == expected.tool_schemas
        assert result.agent_id == expected.agent_id
        assert result.target_message.id == expected.target_message.id
        
        # Compare message IDs (order matters)
        assert [m.id for m in result.messages] == [m.id for m in expected.messages]
