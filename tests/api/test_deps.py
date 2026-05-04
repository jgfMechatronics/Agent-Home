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
    """Shared dict between tests and the test route.

    The route populates "agent" and "dep" on each request.
    Tests can set "route_raises" to an exception before the request to make
    the route raise after deps are resolved (used for sad-path CM lifecycle tests).
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
        if exc := captured.get("route_raises"):
            raise exc
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

    @pytest.fixture(autouse=True)
    def mock_build_success(self):
        """Patches build_agent_and_deps with a success CM for all tests.

        Provides self.mock_agent and self.mock_deps (the yielded objects) and
        self.cm_state ({"entered": bool, "exited": bool}) for CM lifecycle assertions.
        Error tests override this locally with a second patch.object call.
        """
        self.mock_agent = MagicMock()
        self.mock_deps = MagicMock()
        self.cm_state = {"entered": False, "exited": False}

        @asynccontextmanager
        async def _build(*args, **kwargs):
            self.cm_state["entered"] = True
            try:
                yield self.mock_agent, self.mock_deps
            finally:
                self.cm_state["exited"] = True

        with patch.object(AgentFactory, "build_agent_and_deps", _build):
            yield

    async def test_yields_correct_agent_and_deps(
        self, test_client: AsyncClient, captured: dict
    ):
        """Happy path: passes through exactly the (agent, deps) returned by build_agent_and_deps."""
        response = await test_client.get("/test/any-id")
        assert response.status_code == 200
        assert captured["agent"] is self.mock_agent
        assert captured["deps"] is self.mock_deps

    @pytest.mark.parametrize("route_raises", [
        pytest.param(None, id="happy_path"),
        pytest.param(RuntimeError("route exploded"), id="route_raises"),
    ])
    async def test_enters_and_exits_build_cm(
        self, test_client: AsyncClient, captured: dict, route_raises: Exception | None
    ):
        """build_agent_and_deps CM is entered and exited after request, whether or not the route raised."""
        if route_raises:
            captured["route_raises"] = route_raises
        await test_client.get("/test/any-id")
        assert self.cm_state["entered"], "build_agent_and_deps should have been entered as a CM"
        assert self.cm_state["exited"], "build_agent_and_deps CM should be exited after request completes"

    @pytest.mark.parametrize("error,expected_status", [
        (AgentNotFoundError("Agent 'any-id' not found"), 404),
        (AgentLockedError(f"Agent 'any-id' did not become available within {AgentFactory.LOCK_TIMEOUT_SECONDS}s"), 503),
    ])
    async def test_translates_domain_exceptions_to_http(
        self, test_client: AsyncClient, error: Exception, expected_status: int
    ):
        """AgentNotFoundError → 404, AgentLockedError → 503."""
        @asynccontextmanager
        async def mock_error_during_build_agent_and_deps(*args, **kwargs):
            raise error
            yield  # unreachable — makes this an async generator function

        with patch.object(AgentFactory, "build_agent_and_deps", mock_error_during_build_agent_and_deps):
            response = await test_client.get("/test/any-id")

        assert response.status_code == expected_status
        assert response.json()["detail"] == str(error)
