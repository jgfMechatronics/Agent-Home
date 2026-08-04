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
    # technically the tools in the AF may not all be attached to the agent. The agent's tools is determined by tool_ids_for_agent
    raw_tool_ids_for_agent = _extract_or_raise(agent, "tool_ids", context="agents[0]")
    if not isinstance(raw_tool_ids_for_agent, list):
        raise AFIngestionError("'tool_ids' in agents[0] must be a list")
    tool_ids_for_agent = set(raw_tool_ids_for_agent)
    all_letta_tools = _extract_or_raise(data, "tools", context=".AF root")
    # Letta stores tool dicts in a straight list — build a lookup dict to resolve by ID
    letta_tools_by_id = {t["id"]: t["name"] for t in all_letta_tools}
    letta_tool_names_for_agent = [letta_tools_by_id[tid] for tid in tool_ids_for_agent]
    # this is where we drop tools outside our explicit mapping
    tool_names = [TOOL_NAME_MAP[n] for n in letta_tool_names_for_agent if n in TOOL_NAME_MAP]

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
    blocks_list = _extract_or_raise(data, "blocks", context=".AF root")
    referenced_blocks = [b for b in blocks_list if b["id"] in block_ids]

    blocks_payload = []
    for block in referenced_blocks:
        blocks_payload.append({
            "label": _extract_or_raise(block, "label", context="block"),
            "content": _extract_or_raise(block, "value", context="block"),
            "description": _extract_or_raise(block, "description", context="block"),
            "char_limit": _extract_or_raise(block, "limit", context="block"),
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
    TODO: If something is wrong with the af in a way that it can still parse here but fail validation server side (could happen)
    we end up with a partially created agent. The easiest thing to do here would be to add a delete agent route then delete the agent
    we created if any of the steps fail. Otherwise we have to pull a lot of validation infrastructure into this script (like db/pyd models)
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
