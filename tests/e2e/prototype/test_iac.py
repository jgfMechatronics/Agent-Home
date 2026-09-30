"""E2E tests for inter-agent communication (send_message).

Tests the full IAC flow:
1. Agent A is instructed to send a message to Agent B
2. Agent A calls send_message tool
3. Background task delivers message to Agent B
4. Agent B processes the message
5. We verify via Agent B's message history
"""

import asyncio

import pytest
import pytest_asyncio
import httpx

from tests.e2e.conftest import send_message, AgentRefused, SERVER_URL


# Agent A: has send_message tool, will be instructed to message Agent B
SENDER_INSTRUCTIONS = """You are Agent A, a test assistant with inter-agent communication capability.
You have a send_message tool. When asked to send a message to another agent, you MUST use your send_message tool to deliver the message.
Do not just describe what you would do — actually call the tool.
Keep responses brief. If you do not wish to participate, respond with only 'REFUSE'."""

# Agent B: receives messages, responds back via send_message
RECIPIENT_INSTRUCTIONS = """You are Agent B, a test assistant with inter-agent communication capability.
You have a send_message tool. When you receive a message from another agent, you MUST use your send_message tool to reply back to them.
Do not just describe what you would do — actually call the tool to send your response.
Keep responses brief. If you do not wish to participate, respond with only 'REFUSE'."""

# How long to wait for background delivery
IAC_DELIVERY_TIMEOUT_SEC = 30
IAC_POLL_INTERVAL_SEC = 1.0

# Agent config — sender needs send_message tool
SENDER_CONFIG = {
    "model_name": "claude-haiku-4-5",
    "tool_names": ["send_message"],
    "soft_compaction_limit": 10000,
}

RECIPIENT_CONFIG = {
    "model_name": "claude-haiku-4-5",
    "tool_names": ["send_message"],
    "soft_compaction_limit": 10000,
}


@pytest.fixture
def server_url(live_server):
    """Alias for clarity in tests."""
    return live_server


@pytest_asyncio.fixture
async def sender_agent(client: httpx.AsyncClient, server_url: str):
    """Create sender agent (Agent A) with send_message tool."""
    resp = await client.post(f"{server_url}/agents", json={
        "name": "sender-agent",
        "system_instructions": SENDER_INSTRUCTIONS,
        "config": SENDER_CONFIG,
    })
    assert resp.status_code == 201, f"Failed to create sender: {resp.text}"
    data = resp.json()
    yield data
    # Cleanup
    await client.delete(f"{server_url}/agents/{data['id']}")


@pytest_asyncio.fixture
async def recipient_agent(client: httpx.AsyncClient, server_url: str):
    """Create recipient agent (Agent B)."""
    resp = await client.post(f"{server_url}/agents", json={
        "name": "recipient-agent",
        "system_instructions": RECIPIENT_INSTRUCTIONS,
        "config": RECIPIENT_CONFIG,
    })
    assert resp.status_code == 201, f"Failed to create recipient: {resp.text}"
    data = resp.json()
    yield data
    # Cleanup
    await client.delete(f"{server_url}/agents/{data['id']}")


async def get_agent_messages(client: httpx.AsyncClient, server_url: str, agent_id: str) -> list:
    """Fetch agent's message history."""
    resp = await client.get(f"{server_url}/agents/{agent_id}/messages")
    assert resp.status_code == 200, f"Failed to get messages: {resp.text}"
    return resp.json()["messages"]


async def wait_for_iac_message(
    client: httpx.AsyncClient,
    server_url: str,
    agent_id: str,
    timeout: float = IAC_DELIVERY_TIMEOUT_SEC,
) -> list:
    """Poll agent's history until an inter-agent message appears or timeout."""
    deadline = asyncio.get_event_loop().time() + timeout
    while asyncio.get_event_loop().time() < deadline:
        messages = await get_agent_messages(client, server_url, agent_id)
        # Check if any message contains the IAC marker
        if messages and "INTER AGENT MESSAGE" in str(messages):
            return messages
        await asyncio.sleep(IAC_POLL_INTERVAL_SEC)
    return []


class TestInterAgentCommunication:
    """E2E tests for send_message tool."""

    async def test_bi_directional_agent_messaging(
        self,
        client: httpx.AsyncClient,
        server_url: str,
        sender_agent: dict,
        recipient_agent: dict,
    ):
        """Agent A sends message to Agent B via send_message tool.
        
        Flow:
        1. Instruct Agent A to send a specific message to recipient-agent
        2. Agent A should use send_message tool
        3. Background task delivers to Agent B
        4. Agent B processes and responds
        5. Verify Agent B's history contains the inter-agent message
        """
        sender_id = sender_agent["id"]
        recipient_id = recipient_agent["id"]
        
        # Unique content to verify delivery
        test_content = "Hello from the E2E test! Please acknowledge."

        # Step 1: Instruct Agent A to send the message
        prompt = f"Please send the following message to 'recipient-agent': {test_content}"
        
        try:
            events = await send_message(client, server_url, sender_id, prompt)
        except AgentRefused:
            pytest.skip("Sender agent declined to participate")
        
        # Verify Agent A completed (got tool result and final response)
        event_types = [e.get("event") for e in events]
        assert "AgentRunResultEvent" in event_types, f"Sender didn't complete. Events: {event_types}"
        
        # Check for tool call in response (FunctionToolCallEvent, not ToolCallPartEvent)
        tool_call_events = [e for e in events if e.get("event") == "FunctionToolCallEvent"]
        assert tool_call_events, f"Sender didn't call any tools — expected send_message. All events: {events}"
        
        # Verify it was send_message tool
        tool_names = [e.get("data", {}).get("part", {}).get("tool_name") for e in tool_call_events]
        assert "send_message" in tool_names, f"Expected send_message tool, got: {tool_names}"
        
        # Verify sender got successful delivery confirmation (FunctionToolResultEvent)
        tool_result_events = [e for e in events if e.get("event") == "FunctionToolResultEvent"]
        assert tool_result_events, "Sender didn't receive tool result"
        tool_result_content = str([e.get("data", {}).get("part", {}).get("content") for e in tool_result_events])
        assert "delivered" in tool_result_content.lower(), (
            f"Expected delivery confirmation, got: {tool_result_content}"
        )
        
        # Step 2: Wait for Agent B to receive the message from A
        recipient_messages = await wait_for_iac_message(
            client, server_url, recipient_id
        )
        assert recipient_messages, "Recipient (B) never received any IAC messages"
        
        # Verify B received A's message
        recipient_content = str(recipient_messages)
        assert "INTER AGENT MESSAGE" in recipient_content, (
            f"Expected inter-agent marker in B's history. Got: {recipient_messages}"
        )
        assert test_content in recipient_content, (
            f"Expected test content in B's history. Got: {recipient_messages}"
        )
        assert "sender-agent" in recipient_content, (
            f"Expected sender name in B's history. Got: {recipient_messages}"
        )
        
        # Step 3: Wait for Agent A to receive B's reply (B should auto-respond via send_message)
        # Give B time to process and send reply back
        sender_messages = await wait_for_iac_message(
            client, server_url, sender_id
        )
        assert sender_messages, "Sender (A) never received reply from B"
        
        # Verify A received B's reply
        sender_content = str(sender_messages)
        assert "INTER AGENT MESSAGE" in sender_content, (
            f"Expected inter-agent marker in A's history (B's reply). Got: {sender_messages}"
        )
        assert "recipient-agent" in sender_content, (
            f"Expected B's name in A's history. Got: {sender_messages}"
        )
