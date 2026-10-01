"""E2E tests for inter-agent communication (send_message).

Tests the full IAC flow:
1. Agent A is instructed to send a message to Agent B
2. Agent A calls send_message tool
3. Background task delivers message to Agent B (visible via broadcast stream)
4. Agent B processes the message
5. We verify via Agent B's message history and broadcast events
"""

import asyncio
from dataclasses import dataclass, field

import pytest
import pytest_asyncio
import httpx
from httpx_sse import aconnect_sse

from tests.e2e.conftest import send_message, sse_to_dict, AgentRefused


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

# Agent config — both agents need send_message tool
SENDER_CONFIG = {
    "model_name": "anthropic:claude-haiku-4-5",
    "tool_names": ["send_message"],
    "soft_compaction_limit": 10000,
}

RECIPIENT_CONFIG = {
    "model_name": "anthropic:claude-haiku-4-5",
    "tool_names": ["send_message"],
    "soft_compaction_limit": 10000,
}

TEST_CONTENT = "Hello from the E2E test! Please acknowledge."


@dataclass
class IACTestData:
    """Results from running the IAC exchange, used by multiple test cases."""
    sender_id: str
    recipient_id: str
    sender_events: list = field(default_factory=list)  # Events from sender's /messages call
    recipient_messages: list = field(default_factory=list)  # Recipient's message history
    sender_messages: list = field(default_factory=list)  # Sender's message history (after reply)
    recipient_broadcast_events: list = field(default_factory=list)  # Broadcast events for recipient
    skipped: bool = False
    skip_reason: str = ""


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
        if messages and "INTER AGENT MESSAGE" in str(messages):
            return messages
        await asyncio.sleep(IAC_POLL_INTERVAL_SEC)
    return []


async def subscribe_to_stream(
    client: httpx.AsyncClient,
    server_url: str,
    agent_id: str,
    events: list,
    ready_event: asyncio.Event,
):
    """Subscribe to agent's broadcast stream and collect events."""
    async with aconnect_sse(client, "GET", f"{server_url}/agents/{agent_id}/stream") as event_source:
        ready_event.set()
        async for sse in event_source.aiter_sse():
            events.append(sse_to_dict(sse))


class TestInterAgentCommunication:
    """E2E tests for send_message tool and broadcast visibility."""

    @pytest_asyncio.fixture(scope="class")
    async def iac_exchange(self, live_server: str) -> IACTestData:
        """Run full IAC exchange once, collect all data for assertions.
        
        Creates agents, subscribes to recipient's broadcast stream, drives the
        A→B→A message exchange, and collects results for individual test methods.
        """
        data = IACTestData(sender_id="", recipient_id="")
        
        async with httpx.AsyncClient(timeout=60.0) as client:
            # Create sender agent (Agent A)
            resp = await client.post(f"{live_server}/agents", json={
                "name": "sender-agent",
                "system_instructions": SENDER_INSTRUCTIONS,
                "config": SENDER_CONFIG,
            })
            assert resp.status_code == 201, f"Failed to create sender: {resp.text}"
            data.sender_id = resp.json()["id"]
            
            # Create recipient agent (Agent B)
            resp = await client.post(f"{live_server}/agents", json={
                "name": "recipient-agent",
                "system_instructions": RECIPIENT_INSTRUCTIONS,
                "config": RECIPIENT_CONFIG,
            })
            assert resp.status_code == 201, f"Failed to create recipient: {resp.text}"
            data.recipient_id = resp.json()["id"]
            
            # Subscribe to recipient's broadcast stream before triggering IAC
            subscription_ready = asyncio.Event()
            subscription_task = asyncio.create_task(
                subscribe_to_stream(
                    client, live_server, data.recipient_id,
                    data.recipient_broadcast_events, subscription_ready
                )
            )
            
            try:
                await asyncio.wait_for(subscription_ready.wait(), timeout=5.0)
                await asyncio.sleep(0.1)  # Ensure subscription is fully established
                
                # Instruct Agent A to send message to Agent B
                prompt = f"Please send the following message to 'recipient-agent': {TEST_CONTENT}"
                
                try:
                    data.sender_events = await send_message(
                        client, live_server, data.sender_id, prompt
                    )
                except AgentRefused:
                    data.skipped = True
                    data.skip_reason = "Sender agent declined to participate"
                    return data
                
                # Wait for Agent B to receive the message
                data.recipient_messages = await wait_for_iac_message(
                    client, live_server, data.recipient_id
                )
                
                # Wait for Agent A to receive B's reply
                data.sender_messages = await wait_for_iac_message(
                    client, live_server, data.sender_id
                )
                
                # Give broadcast events time to arrive
                await asyncio.sleep(0.5)
                
            finally:
                subscription_task.cancel()
                try:
                    await subscription_task
                except asyncio.CancelledError:
                    pass
        
        return data

    def test_sender_calls_send_message_tool(self, iac_exchange: IACTestData):
        """Sender agent uses send_message tool and gets delivery confirmation."""
        if iac_exchange.skipped:
            pytest.skip(iac_exchange.skip_reason)
        
        events = iac_exchange.sender_events
        event_types = [e.get("event") for e in events]
        
        # Agent completed
        assert "AgentRunResultEvent" in event_types, f"Sender didn't complete. Events: {event_types}"
        
        # Used send_message tool
        tool_call_events = [e for e in events if e.get("event") == "FunctionToolCallEvent"]
        assert tool_call_events, f"Sender didn't call any tools. Events: {events}"
        tool_names = [e.get("data", {}).get("part", {}).get("tool_name") for e in tool_call_events]
        assert "send_message" in tool_names, f"Expected send_message tool, got: {tool_names}"
        
        # Got delivery confirmation
        tool_result_events = [e for e in events if e.get("event") == "FunctionToolResultEvent"]
        assert tool_result_events, "Sender didn't receive tool result"
        tool_result_content = str([e.get("data", {}).get("part", {}).get("content") for e in tool_result_events])
        assert "delivered" in tool_result_content.lower(), f"Expected delivery confirmation, got: {tool_result_content}"

    def test_recipient_receives_message(self, iac_exchange: IACTestData):
        """Recipient agent receives inter-agent message with correct content."""
        if iac_exchange.skipped:
            pytest.skip(iac_exchange.skip_reason)
        
        assert iac_exchange.recipient_messages, "Recipient never received any IAC messages"
        
        content = str(iac_exchange.recipient_messages)
        assert "INTER AGENT MESSAGE" in content, f"Missing IAC marker. Got: {content}"
        assert TEST_CONTENT in content, f"Missing test content. Got: {content}"
        assert "sender-agent" in content, f"Missing sender name. Got: {content}"

    def test_sender_receives_reply(self, iac_exchange: IACTestData):
        """Sender agent receives reply from recipient."""
        if iac_exchange.skipped:
            pytest.skip(iac_exchange.skip_reason)
        
        assert iac_exchange.sender_messages, "Sender never received reply from recipient"
        
        content = str(iac_exchange.sender_messages)
        assert "INTER AGENT MESSAGE" in content, f"Missing IAC marker in reply. Got: {content}"
        assert "recipient-agent" in content, f"Missing recipient name in reply. Got: {content}"

    def test_recipient_broadcast_stream_received_events(self, iac_exchange: IACTestData):
        """Recipient's broadcast stream received run events (IAC uses broadcast runner)."""
        if iac_exchange.skipped:
            pytest.skip(iac_exchange.skip_reason)
        
        events = iac_exchange.recipient_broadcast_events
        assert events, "No broadcast events received for recipient"
        
        event_types = [e.get("event") for e in events]
        
        # Should have synthetic bookend events
        assert "RunStartedEvent" in event_types, f"Missing RunStartedEvent. Got: {event_types}"
        assert "RunCompletedEvent" in event_types, f"Missing RunCompletedEvent. Got: {event_types}"
