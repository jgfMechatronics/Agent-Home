"""E2E tests for multi-provider support.

Verifies that a single-turn agent run completes successfully across providers.
Each provider case is skipped if the required API key env var is not set.

Run with: pytest tests/e2e -m e2e
"""
import os

import httpx
import pytest

from tests.e2e.conftest import get_or_create_agent, send_message


# ---------------------------------------------------------------------------
# Provider matrix
# ---------------------------------------------------------------------------

PROVIDERS = [
    pytest.param(
        ("anthropic:claude-haiku-4-5", "ANTHROPIC_API_KEY"),
        id="anthropic",
    ),
    pytest.param(
        ("together:zai-org/GLM-5.3-Flash", "TOGETHER_API_KEY"),
        id="together-glm-5.3-flash",
    ),
]


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture(params=PROVIDERS)
def provider_config(request) -> tuple[str, str]:
    """Yield (model_name, api_key_env) for each provider, skipping if key absent."""
    model_name, key_env = request.param
    if not os.environ.get(key_env):
        pytest.skip(f"{key_env} not set")
    return model_name, key_env


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------

async def test_single_turn_responds(
    provider_config: tuple[str, str],
    live_server: str,
    client: httpx.AsyncClient,
):
    """Agent completes a single-turn exchange for each provider.

    send_message streams the /messages SSE response and validates internally
    that the response is not a refusal. A non-empty event list confirms the
    run completed and the provider is wired correctly end-to-end.
    """
    model_name, _ = provider_config
    agent_id = await get_or_create_agent(client, live_server, model_name)
    events = await send_message(client, live_server, agent_id, "Say 'hello' and nothing else.")

    assert len(events) > 0, f"No events received from {model_name}"
