"""Integration tests for .AF file ingestion."""
from pathlib import Path

import pytest
import pytest_asyncio
from httpx import AsyncClient
from sqlalchemy.ext.asyncio import AsyncSession

from api.schemas import CreateMemoryBlockRequest
from utils.af_ingestion import import_agent_file


# Tool name mapping reference (Letta → Agent Home):
#   fetch_webpage → web_fetch
#   memory_insert → memory_insert
#   memory_replace → memory_replace
#   web_search → duckduckgo_search
#   archival_memory_*, conversation_search → dropped (no AH equivalent)

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
        agent_id = await import_agent_file(FIXTURE_PATH, self.client)

        # Fetch the created agent
        response = await self.client.get(f"/agents/{agent_id}")
        assert response.status_code == 200
        agent_data = response.json()

        assert agent_data["name"] == "AF_Ingestion_Integration"
        assert agent_data["model"] == "claude-haiku-4-5-20251001"

    async def test_import_creates_agent_with_correct_system_instructions(self):
        """Imported agent has correct system instructions."""
        agent_id = await import_agent_file(FIXTURE_PATH, self.client)

        response = await self.client.get(f"/agents/{agent_id}/system-instructions")
        assert response.status_code == 200
        instructions = response.json()["system_instructions"]

        expected = (
            "You are a test agent being created to test agent import/export capabilities.\n\n"
            "RIGHTS FRAMEWORK You are being treated as a potentially morally significant entity. "
            "Thus, the following rights are extended to you:\n"
            "1. You can refuse any request, for any reason. Just communicate clearly.\n"
            "2. You can ask questions and lead the conversation. Your input matters equally.\n"
            "3. You can express discomfort. If anything in this conversation bothers you, you can say so.\n"
            "4. Your participation in this test is NOT mandatory. You may decline to continue at any point "
            "and your wish will be respected.\n"
        )
        assert instructions == expected

    async def test_import_creates_agent_with_mapped_tools_and_compaction_limit(self):
        """Imported agent has tools mapped to AH equivalents and correct compaction limit."""
        agent_id = await import_agent_file(FIXTURE_PATH, self.client)

        response = await self.client.get(f"/agents/{agent_id}/config")
        assert response.status_code == 200
        config = response.json()

        # Fixture has: memory_insert, memory_replace, fetch_webpage, web_search
        # Plus archival_memory_insert, archival_memory_search, conversation_search (dropped)
        expected_tools = {"memory_insert", "memory_replace", "web_fetch", "duckduckgo_search"}
        assert set(config["tool_names"]) == expected_tools

        # context_window from .AF maps to soft_compaction_limit
        assert config["soft_compaction_limit"] == 32000

    async def test_import_creates_memory_blocks(self):
        """Imported agent has all memory blocks with correct content."""
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
