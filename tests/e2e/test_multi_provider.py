"""E2E tests for multi-provider support.

Verifies that a single-turn agent run completes successfully across providers.
Requires API keys for each provider (via .env or exported env) — missing keys
fail loudly rather than skipping.

Run with: pytest tests/e2e -m e2e
"""
import httpx
import pytest

from tests.e2e.conftest import get_or_create_agent, send_message


@pytest.mark.parametrize(
    "model_name",
    [
        pytest.param("anthropic:claude-haiku-4-5", id="anthropic"),
        pytest.param("together:zai-org/GLM-5.3-Flash", id="together-glm-5.3-flash"),
        pytest.param("openrouter:z-ai/glm-5.3-flash", id="openrouter-glm-5.3-flash"),
    ],
)
async def test_single_turn_responds(
    model_name: str,
    live_server: str,
    client: httpx.AsyncClient,
):
    """Agent completes a single-turn exchange for each provider.

    send_message streams the /messages SSE response and validates internally
    that the response is not a refusal. Success requires an error-free stream
    that reaches a result — a non-empty event list alone is NOT sufficient
    (auth failures mid-stream still yield RunStarted + Error + RunCompleted).
    """
    agent_id = await get_or_create_agent(client, live_server, model_name)
    events = await send_message(client, live_server, agent_id, "Say 'hello' and nothing else.")

    error_events = [e for e in events if e.get("event") == "Error"]
    assert not error_events, f"Run errored for {model_name}: {error_events}"

    assert any(e.get("event") == "AgentRunResultEvent" for e in events), (
        f"Run did not complete for {model_name}: {[e['event'] for e in events]}"
    )
