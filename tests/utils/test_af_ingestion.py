"""Integration tests for .AF file ingestion."""
from pathlib import Path

import pytest
import pytest_asyncio
from httpx import AsyncClient
from sqlalchemy.ext.asyncio import AsyncSession

from agent.types import AgentConfig
from api.schemas import CreateMemoryBlockRequest
from db.models import AgentRecord, MemoryBlockRecord


# Tool name mapping: Letta tool names → Agent Home equivalents
# Tools without equivalents are dropped during import
LETTA_TO_AH_TOOL_MAP = {
    "fetch_webpage": "web_fetch",
    "memory_insert": "memory_insert",
    "memory_replace": "memory_replace",
    "web_search": "duckduckgo_search",
    # These have no AH equivalent and will be dropped:
    # archival_memory_insert, archival_memory_search, conversation_search
}

FIXTURE_PATH = Path(__file__).parent / "AF_Ingestion_Integration.af"


@pytest.mark.asyncio
class TestAFIngestion:
    """Integration tests for importing agents from Letta .AF files."""

    @pytest_asyncio.fixture(autouse=True)
    async def setup(self, session: AsyncSession, client: AsyncClient):
        self.session = session
        self.client = client

    async def test_import_creates_agent_with_correct_config(self):
        """Importing .AF file creates agent with correct name, model, and tools."""
        from utils.af_ingestion import import_agent_file

        agent_id = await import_agent_file(FIXTURE_PATH, self.client)

        # Fetch the created agent
        response = await self.client.get(f"/agents/{agent_id}")
        assert response.status_code == 200
        agent_data = response.json()

        assert agent_data["name"] == "AF_Ingestion_Integration"
        assert agent_data["model"] == "claude-haiku-4-5-20251001"

    async def test_import_creates_agent_with_correct_system_instructions(self):
        """Imported agent has correct system instructions."""
        from utils.af_ingestion import import_agent_file

        agent_id = await import_agent_file(FIXTURE_PATH, self.client)

        response = await self.client.get(f"/agents/{agent_id}/system-instructions")
        assert response.status_code == 200
        instructions = response.json()["system_instructions"]

        # Check key content without asserting exact string (fragile)
        assert "You are a test agent" in instructions
        assert "RIGHTS FRAMEWORK" in instructions
        assert "You can refuse any request" in instructions

    async def test_import_creates_agent_with_mapped_tools(self):
        """Imported agent has tools mapped to AH equivalents, unknown tools dropped."""
        from utils.af_ingestion import import_agent_file

        agent_id = await import_agent_file(FIXTURE_PATH, self.client)

        response = await self.client.get(f"/agents/{agent_id}/config")
        assert response.status_code == 200
        config = response.json()

        # Fixture has: memory_insert, memory_replace, fetch_webpage, web_search
        # Plus archival_memory_insert, archival_memory_search, conversation_search (dropped)
        expected_tools = {"memory_insert", "memory_replace", "web_fetch", "duckduckgo_search"}
        assert set(config["tool_names"]) == expected_tools

    async def test_import_creates_memory_blocks(self):
        """Imported agent has all memory blocks with correct content."""
        from utils.af_ingestion import import_agent_file

        agent_id = await import_agent_file(FIXTURE_PATH, self.client)

        response = await self.client.get(f"/agents/{agent_id}/memory/blocks")
        assert response.status_code == 200
        blocks = {b["label"]: b for b in response.json()["blocks"]}

        # Expected blocks from fixture
        expected_blocks = [
            CreateMemoryBlockRequest(
                label="first_test_block",
                content="content for the first test block",
                description="this is a test block",
                char_limit=5000,
            ),
            CreateMemoryBlockRequest(
                label="second_test_block",
                content="content for the second test block.\nHere's another line.",
                description="this is a second test block",
                char_limit=5000,
            ),
            CreateMemoryBlockRequest(
                label="third_test_block",
                content="Letta really sucks",
                description="this is a third test block.\nHow exciting.",
                char_limit=7500,
            ),
        ]

        assert len(blocks) == len(expected_blocks)

        for expected in expected_blocks:
            actual = blocks.get(expected.label)
            assert actual is not None, f"Block '{expected.label}' not found"
            assert actual["content"] == expected.content
            assert actual["description"] == expected.description
            assert actual["char_limit"] == expected.char_limit
