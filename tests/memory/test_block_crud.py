"""
Tests for block CRUD (memory/block_crud.py)

Read operations take (session, agent_id) — no lock required.
Write operations take (deps) — proves caller holds per-agent lock.
"""
import pytest, pytest_asyncio
from sqlalchemy.ext.asyncio import AsyncSession

from conftest import SAMPLE_AGENT_CONFIG
from db.models import AgentRecord, MemoryBlockRecord

from memory.block_crud import (
    get_blocks,
    get_block,
    update_block,
    create_block,
    delete_block,
    reorder_blocks,
)
from agent.types import AgentDeps


# --- Fixtures ---

@pytest.fixture
def block_fields():
    """Default fields for creating test blocks."""
    return {
        "description": "Test block",
        "char_limit": 2000,
    }

@pytest_asyncio.fixture
async def multi_tenant_agents_with_core_memory(session: AsyncSession):
    """
    Two agents, each with their own memory blocks.
    
    Agent A has: persona (pos 0), human (pos 1), system (pos 2)
    Agent B has: persona (pos 0), notes (pos 1)
    
    Returns dict with agents and their blocks for easy test access.
    """
    # Create two agents
    agent_a = AgentRecord(name="agent-a", agent_config=SAMPLE_AGENT_CONFIG, system_instructions="Agent A instructions")
    agent_b = AgentRecord(name="agent-b", agent_config=SAMPLE_AGENT_CONFIG, system_instructions="Agent B instructions")
    session.add_all([agent_a, agent_b])
    await session.flush()
    
    # Agent A's blocks (out of order to verify position sorting)
    block_a_human = MemoryBlockRecord(agent_id=agent_a.id, label="human", content="Human info for A", description="", char_limit=2000, position=1)
    block_a_persona = MemoryBlockRecord(agent_id=agent_a.id, label="persona", content="Persona for A", description="", char_limit=2000, position=0)
    block_a_system = MemoryBlockRecord(agent_id=agent_a.id, label="system", content="System for A", description="", char_limit=2000, position=2)
    
    # Agent B's blocks
    block_b_persona = MemoryBlockRecord(agent_id=agent_b.id, label="persona", content="Persona for B", description="", char_limit=2000, position=0)
    block_b_notes = MemoryBlockRecord(agent_id=agent_b.id, label="notes", content="Notes for B", description="", char_limit=2000, position=1)
    
    session.add_all([block_a_human, block_a_persona, block_a_system, block_b_persona, block_b_notes])
    await session.flush()
    
    return {
        "agent_a": agent_a,
        "agent_b": agent_b,
        "blocks_a": [block_a_persona, block_a_human, block_a_system],  # in position order
        "blocks_b": [block_b_persona, block_b_notes],
    }


# --- Shared read operation tests ---

@pytest.mark.parametrize("fn,args,expected", [
    pytest.param(get_blocks, (), [], id="get_blocks_returns_empty_list"),
    pytest.param(get_block, ("any_label",), None, id="get_block_returns_none"),
])
async def test_read_ops_handle_nonexistent_agent_gracefully(session: AsyncSession, fn, args, expected):
    """Both read functions should return empty/None for nonexistent agent_id, not raise."""
    nonexistent_agent_id = "agent-does-not-exist"
    result = await fn(session, nonexistent_agent_id, *args)
    assert result == expected


# --- get_blocks tests ---

async def test_get_blocks_in_order_from_correct_agent(session: AsyncSession, multi_tenant_agents_with_core_memory: dict):
    agent_id = multi_tenant_agents_with_core_memory["agent_a"].id
    expected_blocks = multi_tenant_agents_with_core_memory["blocks_a"]
    got_blocks = await get_blocks(session, agent_id)
    assert got_blocks == expected_blocks


async def test_get_blocks_returns_empty_list_for_agent_with_no_blocks(session: AsyncSession, multi_tenant_agents_with_core_memory: dict):
    # Create a third agent with no blocks
    agent_c = AgentRecord(name="agent-c", agent_config=SAMPLE_AGENT_CONFIG, system_instructions="Agent C instructions")
    session.add(agent_c)
    await session.flush()
    
    got_blocks = await get_blocks(session, agent_c.id)
    assert got_blocks == []


# --- get_block tests ---

async def test_get_block_returns_correct_block_by_label(session: AsyncSession, multi_tenant_agents_with_core_memory: dict):
    agent_a = multi_tenant_agents_with_core_memory["agent_a"]
    expected_block = multi_tenant_agents_with_core_memory["blocks_a"][1]  # human block at position 1
    
    got_block = await get_block(session, agent_a.id, expected_block.label)
    assert got_block == expected_block


async def test_get_block_returns_none_for_nonexistent_label(session: AsyncSession, multi_tenant_agents_with_core_memory: dict):
    agent_a = multi_tenant_agents_with_core_memory["agent_a"]
    
    got_block = await get_block(session, agent_a.id, "nonexistent")
    assert got_block is None


async def test_get_block_isolates_by_agent(session: AsyncSession, multi_tenant_agents_with_core_memory: dict):
    """Agent A's 'persona' block should not be returned when querying Agent B."""
    agent_b = multi_tenant_agents_with_core_memory["agent_b"]
    
    # Both agents have 'persona', but Agent B's 'system' doesn't exist
    got_block = await get_block(session, agent_b.id, "system")
    assert got_block is None


# --- update_block tests ---

# TODO: Modifies content, updates updated_at
# TODO: Enforces char_limit (rejects content exceeding limit)
# TODO: On nonexistent block raises appropriate error


# --- create_block tests ---

# TODO: Inserts new block with correct defaults
# TODO: With duplicate label raises/fails (unique constraint)
# TODO: Assigns position (auto-increment or explicitly specified)


# --- delete_block tests ---

# TODO: Removes block
# TODO: On nonexistent block raises error


# --- reorder_blocks tests ---

# TODO: Assigns positions 0, 1, 2... based on list order
# TODO: Updates all positions atomically
# TODO: Raises error if list doesn't include all blocks for agent
# TODO: Raises error if list contains unknown label


# --- Multi-tenant isolation (common to all write ops) ---

# TODO: Write operations only affect blocks for specified agent_id
