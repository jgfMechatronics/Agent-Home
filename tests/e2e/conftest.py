"""E2E test configuration.

Tests in this directory require a live server and are excluded from the default test run.
Run with: pytest tests/e2e -m e2e
"""

import os
import socket
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


def pytest_configure(config):
    """Force serial execution for e2e runs.

    E2E tests manage live servers on fixed ports — parallel xdist workers
    would each spawn a conflicting server. This conftest only loads on the
    controller when tests/e2e is given as a run path, so clearing the xdist
    options here disables parallelism for e2e runs while leaving the default
    suite parallel. All three options must be cleared: xdist populates
    option.tx (worker specs) during cmdline_main, before configure hooks run,
    and worker spawning is driven by tx/dist — numprocesses alone is ignored.
    """
    config.option.numprocesses = 0
    config.option.dist = "no"
    config.option.tx = []


def _require_port_free(server_url: str) -> None:
    """Fail fast with a helpful message if the target port is already occupied.

    A stray server (e.g. orphaned by a crashed previous run) on the port makes
    e2e tests fail in confusing ways — health checks pass against the WRONG
    server, or the fresh server fails to bind. Detecting it up front turns
    that into an immediate, actionable error.
    """
    parsed = httpx.URL(server_url)
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        s.settimeout(0.5)
        if s.connect_ex((str(parsed.host), parsed.port)) != 0:
            return  # Port is free

    # Something is listening — classify it for a better message.
    what = "another process"
    try:
        resp = httpx.get(f"{server_url}/health", timeout=1.0)
        if resp.status_code == 200:
            what = "an Agent Home server (likely a stray from a previous run)"
    except httpx.RequestError:
        pass

    pytest.fail(
        f"Port {parsed.port} is already in use by {what}.\n"
        f"Find it:    ps aux | grep -E 'uvicorn|start_server' | grep -v grep\n"
        f"Kill it:    kill <PID>   (or run stop_server.sh if the PID file exists)"
    )


@pytest.fixture(scope="session")
def live_server(tmp_path_factory):
    """Start server before E2E tests, stop after.

    Uses start_server.sh and stop_server.sh from project root.
    Session-scoped so server starts once for all E2E tests.
    Uses a temp directory for the database to ensure test isolation.
    """
    # Pre-flight: fail fast (with remediation) if the port is already occupied —
    # a stray server from a previous run makes tests fail in confusing ways.
    _require_port_free(SERVER_URL)

    start_script = PROJECT_ROOT / "start_server.sh"
    stop_script = PROJECT_ROOT / "stop_server.sh"
    
    # Use temp directory for test database
    tmp_dir = tmp_path_factory.mktemp("agent_home_e2e")
    db_path = tmp_dir / "db.sqlite"
    env = os.environ.copy()
    env["AGENT_HOME_DB_PATH"] = str(db_path)
    
    # Start server
    result = subprocess.run(
        [str(start_script)],
        cwd=PROJECT_ROOT,
        capture_output=True,
        text=True,
        env=env,
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


async def get_or_create_agent(client: httpx.AsyncClient, server_url: str, model_name: str) -> str:
    """Get or create an agent for the given model, named after it. Returns agent_id.

    Name is derived from the model string so parametrized tests don't collide.
    Agents persist for the lifetime of the session-scoped live_server (tmp DB).
    """
    agent_name = "e2e-" + model_name.replace(":", "-").replace("/", "-")

    resp = await client.get(f"{server_url}/agents")
    resp.raise_for_status()
    for agent in resp.json():
        if agent.get("name") == agent_name:
            return agent["id"]

    resp = await client.post(
        f"{server_url}/agents",
        json={
            "name": agent_name,
            "system_instructions": TEST_AGENT_INSTRUCTIONS,
            "config": {
                "model_name": model_name,
                "tool_names": [],
                "soft_compaction_limit": 100000,
            },
        },
    )
    resp.raise_for_status()
    return resp.json()["id"]


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
