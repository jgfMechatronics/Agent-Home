"""
Tests for api/deps.py — FastAPI dependency functions.

Uses a minimal test app with a single route to exercise the full
dependency injection chain, including exception→HTTP status translation.

Tests are currently red (TDD) — get_agent_and_deps is a stub pending James's implementation.

Fixtures from conftest used here:
- session: Test DB session

Strategy: mock AgentFactory.build_agent_and_deps at the boundary.
  - get_agent_and_deps is responsible for: calling build_agent_and_deps, yielding
    the result, and translating domain exceptions to HTTP. Nothing more.
  - Lock management, DB lookups, agent construction → build_agent_and_deps's concern,
    covered in tests/agent/test_factory.py.
"""
from collections.abc import AsyncGenerator
from contextlib import asynccontextmanager
from unittest.mock import MagicMock, patch

import pytest
import pytest_asyncio
from fastapi import Depends, FastAPI
from httpx import ASGITransport, AsyncClient
from sqlalchemy.ext.asyncio import AsyncSession

from agent.factory import AgentFactory, AgentLockedError, AgentNotFoundError
from api.deps import get_agent_and_deps, get_lock_reg, get_session_dep


# --- Fixtures ---

@pytest.fixture
def lock_reg() -> dict:
    """Empty lock registry for each test."""
    return {}


@pytest.fixture
def captured() -> dict:
    """Populated by the test route with the resolved (agent, deps) on each request.

    Allows the happy-path test to assert on the actual objects yielded by
    get_agent_and_deps, not just the HTTP response.
    """
    return {}


@pytest_asyncio.fixture
async def test_client(
    session: AsyncSession,
    lock_reg: dict,
    captured: dict,
) -> AsyncGenerator[AsyncClient, None]:
    """Minimal FastAPI app wired to get_agent_and_deps, with real DB session and lock registry.

    Overrides get_session_dep and get_lock_reg so tests control the infrastructure
    without a running server. Route captures resolved objects into `captured` for inspection.
    """
    app = FastAPI()

    @app.get("/test/{agent_id}")
    async def _route(agent_and_deps: tuple = Depends(get_agent_and_deps)):
        agent, deps = agent_and_deps
        captured["agent"] = agent
        captured["deps"] = deps
        return {}

    async def _override_session():
        yield session

    app.dependency_overrides[get_session_dep] = _override_session
    app.dependency_overrides[get_lock_reg] = lambda: lock_reg

    async with AsyncClient(
        transport=ASGITransport(app=app), base_url="http://test"
    ) as client:
        yield client


# --- Tests ---

class TestGetAgentAndDeps:
    """get_agent_and_deps: yields (Agent, AgentDeps) and translates domain exceptions to HTTP."""

    async def test_yields_correct_agent_and_deps(
        self, test_client: AsyncClient, captured: dict
    ):
        """Happy path: passes through exactly the (agent, deps) returned by build_agent_and_deps."""
        mock_agent = MagicMock()
        mock_deps = MagicMock()

        @asynccontextmanager
        async def mock_build(*args, **kwargs):
            yield mock_agent, mock_deps

        with patch.object(AgentFactory, "build_agent_and_deps", mock_build):
            response = await test_client.get("/test/any-id")

        assert response.status_code == 200
        assert captured["agent"] is mock_agent
        assert captured["deps"] is mock_deps

    @pytest.mark.parametrize("error,expected_status", [
        (AgentNotFoundError("Agent 'any-id' not found"), 404),
        (AgentLockedError("Agent 'any-id' did not become available"), 503),
    ])
    async def test_translates_domain_exceptions_to_http(
        self, test_client: AsyncClient, error: Exception, expected_status: int
    ):
        """AgentNotFoundError → 404, AgentLockedError → 503."""
        @asynccontextmanager
        async def mock_build(*args, **kwargs):
            raise error
            yield  # makes it a valid async generator

        with patch.object(AgentFactory, "build_agent_and_deps", mock_build):
            response = await test_client.get("/test/any-id")

        assert response.status_code == expected_status
