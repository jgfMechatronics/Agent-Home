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


def _get_referenced_items(agent: dict, id_key: str, data: dict, items_key: str) -> list[dict]:
    """
    Return items from data[items_key] whose 'id' is referenced by agent[id_key].
    AF files track what resources are attached to a particular agent by recording an ID on the agent dict
    Technically the full list of tools and blocks in the AF could include items not attached to that agent
    NOTE: this may not happen in practice with only a single agent in the file, which we enforce elsewhere
    """
    raw_ids = _extract_or_raise(agent, id_key, context="agents[0]")
    if not isinstance(raw_ids, list):
        raise AFIngestionError(f"'{id_key}' in agents[0] must be a list")
    ids = set(raw_ids)
    all_items = _extract_or_raise(data, items_key, context=".AF root")
    return [item for item in all_items if item["id"] in ids]


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
    # Resolve tools — not all tools in the AF are necessarily attached to this agent
    letta_tools_for_agent = _get_referenced_items(agent, "tool_ids", data, "tools")
    # this is where we drop tools outside our explicit mapping
    tool_names = [TOOL_NAME_MAP[t["name"]] for t in letta_tools_for_agent if t["name"] in TOOL_NAME_MAP]

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

    # Resolve blocks
    referenced_blocks = _get_referenced_items(agent, "block_ids", data, "blocks")

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
