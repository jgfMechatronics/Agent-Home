"""HTTP helpers for sending slash commands to an Agent Home server and verifying results.

Provides both sync (httpx.Client) and async (httpx.AsyncClient) variants.

TODO: propagate httpx-sse usage to bridge.py and cli.py, which currently
use manual aiter_lines() + startswith('data:') SSE parsing.
"""
import json

import httpx
from httpx_sse import aconnect_sse, connect_sse


def send_slash_cmd(client: httpx.Client, agent_id: str, cmd: str) -> None:
    """Send a slash command and verify it succeeds.

    Streams the SSE response, finds the SlashCommandResult event for the command,
    and raises RuntimeError if status is not 'success'.
    """
    with connect_sse(client, "POST", f"/agents/{agent_id}/messages", json={"message": f"/{cmd}"}) as event_source:
        for event in event_source.iter_sse():
            if event.event != "SlashCommandResult":
                continue
            data = json.loads(event.data)
            if data.get("name") != f"user_{cmd}":
                continue
            if data.get("status") != "success":
                raise RuntimeError(f"/{cmd} failed: {data.get('result')}")
            break


async def async_send_slash_cmd(client: httpx.AsyncClient, agent_id: str, cmd: str) -> None:
    """Async version of send_slash_cmd."""
    async with aconnect_sse(client, "POST", f"/agents/{agent_id}/messages", json={"message": f"/{cmd}"}) as event_source:
        async for event in event_source.aiter_sse():
            if event.event != "SlashCommandResult":
                continue
            data = json.loads(event.data)
            if data.get("name") != f"user_{cmd}":
                continue
            if data.get("status") != "success":
                raise RuntimeError(f"/{cmd} failed: {data.get('result')}")
            break
