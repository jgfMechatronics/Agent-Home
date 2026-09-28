"""E2E test configuration.

Tests in this directory require a live server and are excluded from the default test run.
Run with: pytest -m e2e
"""

import subprocess
import time
from pathlib import Path

import httpx
import pytest

# Project root where shell scripts live
PROJECT_ROOT = Path(__file__).parent.parent.parent
SERVER_URL = "http://localhost:8008"


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
    for _ in range(30):
        try:
            resp = httpx.get(f"{SERVER_URL}/health", timeout=1.0)
            if resp.status_code == 200:
                break
        except httpx.RequestError:
            pass
        time.sleep(0.5)
    else:
        subprocess.run([str(stop_script)], cwd=PROJECT_ROOT)
        pytest.fail("Server failed to become healthy within 15 seconds")
    
    yield SERVER_URL
    
    # Stop server
    subprocess.run([str(stop_script)], cwd=PROJECT_ROOT, capture_output=True)
