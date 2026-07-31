"""
.AF file ingestion — imports Letta agent exports into Agent Home.

Parses Letta's AgentFile (.af) JSON format and creates the equivalent
agent with memory blocks via the Agent Home API.

Tool mapping:
- fetch_webpage → web_fetch
- memory_insert → memory_insert  
- memory_replace → memory_replace
- web_search → duckduckgo_search
- archival_memory_*, conversation_search → dropped (no AH equivalent)
"""
from pathlib import Path
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from httpx import AsyncClient


# Letta tool name → Agent Home tool name
# Tools not in this map are dropped during import
TOOL_NAME_MAP = {
    "fetch_webpage": "web_fetch",
    "memory_insert": "memory_insert",
    "memory_replace": "memory_replace",
    "web_search": "duckduckgo_search",
}


async def import_agent_file(af_path: Path, client: "AsyncClient") -> str:
    """Import a Letta .AF file and create the agent via API.

    Args:
        af_path: Path to the .af file
        client: AsyncClient configured for the Agent Home API

    Returns:
        The created agent's ID

    Raises:
        NotImplementedError: Implementation pending (Sonnet's task)
    """
    raise NotImplementedError("AF ingestion not yet implemented")
