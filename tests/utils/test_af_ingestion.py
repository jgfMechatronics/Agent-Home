"""Integration tests for .AF file ingestion.

Tool mapping (Letta → Agent Home):
  fetch_webpage → web_fetch, memory_insert → memory_insert,
  memory_replace → memory_replace, web_search → duckduckgo_search
  archival_memory_*, conversation_search → dropped (no AH equivalent)
"""
from pathlib import Path
from typing import Any

import pytest
import pytest_asyncio
from httpx import AsyncClient

from utils.af_ingestion import import_agent_file

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

EXPECTED_BLOCKS = [
    {"label": "first_test_block", "content": "content for the first test block",
     "description": "this is a test block", "char_limit": 5000},
    {"label": "second_test_block", "content": "content for the second test block.\nHere's another line.",
     "description": "this is a second test block", "char_limit": 5000},
    {"label": "third_test_block", "content": "Letta really sucks",
     "description": "this is a third test block.\nHow exciting.", "char_limit": 7500},
]


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
        """Imported agent has correct name and model."""
        data = await self._get()
        assert data["name"] == "AF_Ingestion_Integration"
        assert data["model"] == "claude-haiku-4-5-20251001"

    async def test_system_instructions(self):
        """Imported agent has correct system instructions."""
        data = await self._get("/system-instructions")
        assert data["system_instructions"] == EXPECTED_SYSTEM_PROMPT

    async def test_config(self):
        """Imported agent has mapped tools and correct compaction limit."""
        config = await self._get("/config")
        assert set(config["tool_names"]) == {"memory_insert", "memory_replace", "web_fetch", "duckduckgo_search"}
        assert config["soft_compaction_limit"] == 32000

    async def test_memory_blocks(self):
        """Imported agent has all memory blocks with correct content."""
        data = await self._get("/memory/blocks")
        # Strip updated_at (dynamic) for comparison
        actual = sorted(
            [{k: v for k, v in b.items() if k != "updated_at"} for b in data["blocks"]],
            key=lambda b: b["label"]
        )
        expected = sorted(EXPECTED_BLOCKS, key=lambda b: b["label"])
        assert actual == expected
