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
import json
from pathlib import Path
from typing import TYPE_CHECKING, Any

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


class AFIngestionError(ValueError):
    """Raised when the .AF file is malformed or missing required fields."""


def _extract_or_raise(obj: dict, key: str, context: str = "") -> Any:
    """Extract a required key from a dict, raising AFIngestionError if missing."""
    if (not isinstance(obj, dict)) or (key not in obj):
        raise AFIngestionError(f"Missing required field '{key}'{f' in {context}' if context else ''}")
    return obj[key]


def _parse_af(data: dict) -> tuple[dict, list[dict]]:
    """Parse and validate the .AF structure, returning (agent_payload, blocks_payload).

    Validates all required fields before returning. Raises AFIngestionError on any
    missing or malformed data so callers can fail before making any API calls.
    """
    agents = _extract_or_raise(data, "agents", context=".AF root")
    if not isinstance(agents, list) or len(agents) == 0:
        raise AFIngestionError("'agents' must be a non-empty list")
    elif not len(agents) == 1:
        raise AFIngestionError(f"only one agent per AF supported. Provided file contained {len(agents)} agents")

    agent = agents[0]

    # Required agent fields
    name = _extract_or_raise(agent, "name", context="agents[0]")
    system = _extract_or_raise(agent, "system", context="agents[0]")
    llm_config = _extract_or_raise(agent, "llm_config", context="agents[0]")
    model = _extract_or_raise(llm_config, "model", context="agents[0].llm_config")
    context_window = _extract_or_raise(llm_config, "context_window", context="agents[0].llm_config")
    enable_reasoner = _extract_or_raise(llm_config, "enable_reasoner", context="agents[0].llm_config")

    # Resolve tool names via tool_ids → tools array
    raw_tool_ids = _extract_or_raise(agent, "tool_ids", context="agents[0]")
    if not isinstance(raw_tool_ids, list):
        raise AFIngestionError("'tool_ids' in agents[0] must be a list")
    letta_tool_ids = set(raw_tool_ids)
    letta_tools_list = _extract_or_raise(agent, "tools", context="agents[0]")
    letta_tool_names = {t["name"] for t in letta_tools_list if t.get("id") in letta_tool_ids and t.get("name")}
    tool_names = [
        TOOL_NAME_MAP[name] for name in letta_tool_names if name in TOOL_NAME_MAP
    ]

    agent_payload = {
        "name": name,
        "system_instructions": system,
        "config": {
            "model_name": model,
            "tool_names": tool_names,
            "soft_compaction_limit": context_window,
            "thinking_enabled": enable_reasoner,
        },
    }

    # Resolve blocks via block_ids → blocks array
    raw_block_ids = _extract_or_raise(agent, "block_ids", context="agents[0]")
    if not isinstance(raw_block_ids, list):
        raise AFIngestionError("'block_ids' in agents[0] must be a list")
    block_ids = set(raw_block_ids)
    blocks_list = data.get("blocks") or []
    referenced_blocks = [b for b in blocks_list if b.get("id") in block_ids]

    blocks_payload = []
    for block in referenced_blocks:
        label = _extract_or_raise(block, "label", context="block")
        blocks_payload.append({
            "label": label,
            "content": block.get("value", ""),
            "description": block.get("description", ""),
            "char_limit": block.get("limit", 20000),
        })

    return agent_payload, blocks_payload


async def import_agent_file(af_path: Path, client: "AsyncClient") -> str:
    """Import a Letta .AF file and create the agent via API.

    Validates all fields from the .AF before making any API calls.
    Memory blocks are created after the agent.

    Args:
        af_path: Path to the .af file
        client: AsyncClient configured for the Agent Home API

    Returns:
        The created agent's ID

    Raises:
        AFIngestionError: If the .AF file is malformed or missing required fields
        httpx.HTTPStatusError: If any API call fails
    """
    data = json.loads(af_path.read_text())

    # Validate everything before touching the API
    agent_payload, blocks_payload = _parse_af(data)

    # Create agent
    response = await client.post("/agents", json=agent_payload)
    response.raise_for_status()
    agent_id = response.json()["id"]

    # Create memory blocks
    for block in blocks_payload:
        block_response = await client.post(f"/agents/{agent_id}/memory/blocks", json=block)
        block_response.raise_for_status()

    return agent_id
