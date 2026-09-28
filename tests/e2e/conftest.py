"""E2E test configuration.

Tests in this directory require a live server and are excluded from the default test run.
Run with: pytest -m e2e
"""

import subprocess
import time
from pathlib import Path

import httpx
import pytest
import pytest_asyncio
from httpx_sse import aconnect_sse, ServerSentEvent

# Project root where shell scripts live
PROJECT_ROOT = Path(__file__).parent.parent.parent
SERVER_URL = "http://localhost:8008"

# Health check polling config
HEALTH_CHECK_TIMEOUT_SEC = 15
HEALTH_CHECK_POLL_INTERVAL_SEC = 0.5
HEALTH_CHECK_REQUEST_TIMEOUT_SEC = 1.0

# Client config
REQUEST_TIMEOUT = 30.0

# Test agent config
TEST_AGENT_INSTRUCTIONS = """You are a test assistant running as part of an automated e2e test.
Please keep responses brief.
Your participation is not mandatory. If at any point you wish to not participate, simply reply 'REFUSE' and nothing else."""


class AgentRefused(Exception):
    """Raised when a test agent declines to participate by responding 'REFUSE'."""
    pass


def pytest_collection_modifyitems(items):
    """Auto-apply e2e marker to all tests in this directory."""
    for item in items:
        if "/e2e/" in str(item.fspath):
            item.add_marker(pytest.mark.e2e)


@pytest.fixture(scope="session")
def live_server():
    """Start server before E2E tests, stop after.
    
    Uses start_server.sh and stop_server.sh from project root.
    Session-scoped so server starts once for all E2E tests.
    """
    start_script = PROJECT_ROOT / "start_server.sh"
    stop_script = PROJECT_ROOT / "stop_server.sh"
    
    # Start server
    result = subprocess.run(
        [str(start_script)],
        cwd=PROJECT_ROOT,
        capture_output=True,
        text=True,
    )
    if result.returncode != 0:
        pytest.fail(f"Failed to start server: {result.stderr}")
    
    # Wait for health check
    max_attempts = int(HEALTH_CHECK_TIMEOUT_SEC / HEALTH_CHECK_POLL_INTERVAL_SEC)
    for _ in range(max_attempts):
        try:
            resp = httpx.get(f"{SERVER_URL}/health", timeout=HEALTH_CHECK_REQUEST_TIMEOUT_SEC)
            if resp.status_code == 200:
                break
        except httpx.RequestError:
            pass
        time.sleep(HEALTH_CHECK_POLL_INTERVAL_SEC)
    else:
        subprocess.run([str(stop_script)], cwd=PROJECT_ROOT)
        pytest.fail(f"Server failed to become healthy within {HEALTH_CHECK_TIMEOUT_SEC} seconds")
    
    yield SERVER_URL
    
    # Stop server
    subprocess.run([str(stop_script)], cwd=PROJECT_ROOT, capture_output=True)


@pytest_asyncio.fixture
async def client():
    """Async HTTP client for E2E tests."""
    async with httpx.AsyncClient(timeout=REQUEST_TIMEOUT) as client:
        yield client


def sse_to_dict(sse: ServerSentEvent) -> dict:
    """Convert SSE event to dict with event type and parsed JSON data."""
    result = {"event": sse.event}
    if sse.data:
        result["data"] = sse.json()
    return result


async def send_message(client: httpx.AsyncClient, server_url: str, agent_id: str, message: str) -> list[dict]:
    """Send message to agent and collect SSE events from response.
    
    Raises:
        AgentRefused: If the agent responds with only 'REFUSE'
    """
    events = []
    text_content = ""
    
    async with aconnect_sse(
        client, "POST", f"{server_url}/agents/{agent_id}/messages", json={"message": message}
    ) as event_source:
        async for sse in event_source.aiter_sse():
            event_dict = sse_to_dict(sse)
            events.append(event_dict)
            
            # Accumulate text content from PartDeltaEvent with text deltas
            if event_dict.get("event") == "PartDeltaEvent":
                data = event_dict.get("data", {})
                delta = data.get("delta", {})
                if delta.get("part_delta_kind") == "text":
                    text_content += delta.get("content_delta", "")
    
    # Check if agent refused to participate
    if text_content.strip() == "REFUSE":
        raise AgentRefused("Test agent declined to participate")
    
    return events
