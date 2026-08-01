"""Integration tests for .AF file ingestion.

Tool mapping (Letta → Agent Home):
  fetch_webpage → web_fetch, memory_insert → memory_insert,
  memory_replace → memory_replace, web_search → duckduckgo_search
  archival_memory_*, conversation_search → dropped (no AH equivalent)
"""
from pathlib import Path
from typing import Any
from itertools import zip_longest

import pytest
import pytest_asyncio
from httpx import AsyncClient

from utils.af_ingestion import import_agent_file
from api.schemas import AgentMetadataResponse, CoreMemoryResponse, MemoryBlockResponse
from agent.types import AgentConfig

FIXTURE_PATH = Path(__file__).parent / "AF_Ingestion_Integration.af"

EXPECTED_SYSTEM_PROMPT = (
    "You are a test agent being created to test agent import/export capabilities.\n\n"
    "RIGHTS FRAMEWORK You are being treated as a potentially morally significant entity. "
    "Thus, the following rights are extended to you:\n"
    "1. You can refuse any request, for any reason. Just communicate clearly.\n"
    "2. You can ask questions and lead the conversation. Your input matters equally.\n"
    "3. You can express discomfort. If anything in this conversation bothers you, you can say so.\n"
    "4. Your participation in this test is NOT mandatory. You may decline to continue at any point "
    "and your wish will be respected.\n"
)


@pytest.mark.asyncio
class TestAFIngestion:
    """Integration tests for importing agents from Letta .AF files."""

    @pytest_asyncio.fixture(autouse=True)
    async def setup(self, client: AsyncClient):
        self.client = client
        self.agent_id = await import_agent_file(FIXTURE_PATH, client)

    async def _get(self, path: str = "") -> dict[str, Any]:
        """GET /agents/{agent_id}{path}, assert 200, return JSON."""
        response = await self.client.get(f"/agents/{self.agent_id}{path}")
        assert response.status_code == 200
        return response.json()

    async def test_agent_metadata(self):
        """Imported agent has correct metadata and config."""

        metadata_data = await self._get()
        returned_metadata = AgentMetadataResponse.model_validate(metadata_data)

        expected_metadata = AgentMetadataResponse(
            id=returned_metadata.id,  # dynamic, just copy
            name="AF_Ingestion_Integration",
            model="claude-haiku-4-5-20251001",
            created_at=returned_metadata.created_at,  # dynamic
            updated_at=returned_metadata.updated_at,  # dynamic
        )
        assert returned_metadata == expected_metadata

    async def test_agent_config(self):
        config_data = await self._get("/config")
        returned_config = AgentConfig.model_validate(config_data)

        expected_config = AgentConfig(
            # From .AF file:
            model_name="claude-haiku-4-5-20251001",
            tool_names=["memory_insert", "memory_replace", "web_fetch", "duckduckgo_search"],
            soft_compaction_limit=32000,
            thinking_enabled=True,
            # remaining values should be default, not set by af ingestion
        )
        # tool_names order may vary
        assert set(returned_config.tool_names) == set(expected_config.tool_names)
        returned_config.tool_names = expected_config.tool_names # normalize prior to full eq
        assert returned_config == expected_config

    async def test_system_instructions(self):
        """Imported agent has correct system instructions."""
        data = await self._get("/system-instructions")
        assert data["system_instructions"] == EXPECTED_SYSTEM_PROMPT

    async def test_memory_blocks(self):
        """Imported agent has all memory blocks with correct content."""
        from datetime import datetime

        data = await self._get("/memory/blocks")
        returned_blocks = CoreMemoryResponse.model_validate(data).blocks

        expected_blocks = [
            MemoryBlockResponse(
                label="first_test_block",
                content="content for the first test block",
                description="this is a test block",
                char_limit=5000,
                updated_at=datetime.min,
            ),
            MemoryBlockResponse(
                label="second_test_block",
                content="content for the second test block.\nHere's another line.",
                description="this is a second test block",
                char_limit=5000,
                updated_at=datetime.min,
            ),
            MemoryBlockResponse(
                label="third_test_block",
                content="Letta really sucks",
                description="this is a third test block.\nHow exciting.",
                char_limit=7500,
                updated_at=datetime.min,
            ),
        ]

        for expected_block, returned_block in zip_longest(expected_blocks, returned_blocks):
            # Normalize updated_at so we only compare fields we care about
            returned_block.updated_at = expected_block.updated_at
            assert returned_block == expected_block

    async def test_empty_conversation_history(self):
        """
        TODO: We don't currently support loading conversation history
        """
        pytest.fail()

    async def test_rejects_bad_af(self):
        """
        TODO: Should reject a malformed af even if the malformation occurs in the middle of parsable fields,
        and should not have started any write activities on the AH API by that point.
        IE all relevant fields in the AF should be validated prior to creating the agent, so we don't get partway through,
        choke, then leave a partially formed agent.
        """
        pytest.fail()
