"""
Tests for api/fastapi_deps.py — FastAPI dependency functions.

Uses a minimal test app with a single route to exercise the full
dependency injection chain.

Strategy: mock AgentFactory.build_agent_and_deps / build_deps at the boundary.
  - get_agent_and_deps is responsible for: calling build_agent_and_deps, yielding
    the result, and propagating exceptions. Nothing more.
  - get_agent_deps mirrors get_agent_and_deps but yields AgentDeps only (no Agent).
  - Exception → HTTP mapping is the app's concern, tested in tests/api/test_app.py.
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
from agent.types import AgentAppState, AgentDeps
from api.fastapi_deps import get_agent_and_deps, get_agent_app_state_reg, get_agent_deps, get_session_dep


# --- Fixtures ---

@pytest.fixture
def agent_app_state_reg() -> dict:
    """Empty agent state registry for each test."""
    return {}


@pytest.fixture
def captured() -> dict:
    """Shared dict between tests and the test route.

    The route populates "agent" and/or "deps" on each request.
    Tests can set "route_raises" to an exception before the request to make
    the route raise after deps are resolved (used for route propagation tests).
    """
    return {}


@asynccontextmanager
async def _build_test_client(
    route,
    session: AsyncSession,
    agent_app_state_reg: dict,
) -> AsyncGenerator[AsyncClient, None]:
    """Minimal FastAPI test app with session and agent state registry overridden, as an AsyncClient.

    No exception handlers registered — exceptions propagate uncaught.
    Accepts any route handler so each fixture can define its own dep and capture logic.

    The fixtures for testing get_agent_deps and get_agent_and_deps both share app setup, only the specifics of
    the route differ
    """
    app = FastAPI()
    app.get("/test/{agent_id}")(route)

    async def _override_session():
        yield session

    app.dependency_overrides[get_session_dep] = _override_session
    app.dependency_overrides[get_agent_app_state_reg] = lambda: agent_app_state_reg

    async with AsyncClient(
        transport=ASGITransport(app=app), base_url="http://test"
    ) as client:
        yield client


@pytest_asyncio.fixture
async def agent_and_deps_client(
    session: AsyncSession,
    agent_app_state_reg: dict,
    captured: dict,
) -> AsyncGenerator[AsyncClient, None]:
    """Test client wired to get_agent_and_deps — route captures both Agent and AgentDeps."""
    async def _route(agent_and_deps: tuple = Depends(get_agent_and_deps)):
        agent, deps = agent_and_deps
        captured["agent"] = agent
        captured["deps"] = deps
        if exc := captured.get("route_raises"):
            raise exc
        return {}

    async with _build_test_client(_route, session, agent_app_state_reg) as client:
        yield client


@pytest_asyncio.fixture
async def deps_only_client(
    session: AsyncSession,
    agent_app_state_reg: dict,
    captured: dict,
) -> AsyncGenerator[AsyncClient, None]:
    """Test client wired to get_agent_deps — route captures AgentDeps only (no Agent)."""
    async def _route(deps: AgentDeps = Depends(get_agent_deps)):
        captured["deps"] = deps
        if exc := captured.get("route_raises"):
            raise exc
        return {}

    async with _build_test_client(_route, session, agent_app_state_reg) as client:
        yield client


# --- Tests ---

class TestGetAgentAndDeps:
    """get_agent_and_deps: yields (Agent, AgentDeps) and propagates all exceptions uncaught."""

    @pytest.fixture(autouse=True)
    def mock_build_success(self):
        """Patches build_agent_and_deps with a success CM for all tests.

        Provides self.mock_agent, self.mock_deps, and self.contextman_state for assertions.
        Error tests override locally with a second patch.object call.
        """
        self.mock_agent = MagicMock()
        self.mock_deps = MagicMock()
        self.contextman_state = {"entered": False, "exited": False}

        @asynccontextmanager
        async def _build(*args, **kwargs):
            self.contextman_state["entered"] = True
            try:
                yield self.mock_agent, self.mock_deps
            finally:
                self.contextman_state["exited"] = True

        with patch.object(AgentFactory, "build_agent_and_deps", _build):
            yield

    async def test_yields_correct_agent_and_deps(
        self, agent_and_deps_client: AsyncClient, captured: dict
    ):
        """Happy path: passes through exactly the (agent, deps) returned by build_agent_and_deps."""
        response = await agent_and_deps_client.get("/test/any-id")
        assert response.status_code == 200
        assert captured["agent"] is self.mock_agent
        assert captured["deps"] is self.mock_deps

    async def test_enters_and_exits_build_cm(self, agent_and_deps_client: AsyncClient):
        """build_agent_and_deps CM is entered and exited after a successful request.

        Exception-path teardown is tested at the factory level (test_factory.py).
        """
        await agent_and_deps_client.get("/test/any-id")
        assert self.contextman_state["entered"], "build_agent_and_deps should have been entered as a CM"
        assert self.contextman_state["exited"], "build_agent_and_deps CM should be exited after request completes"

    # Below are two exceptions we know we want to propagate, followed by a generic exception
    @pytest.mark.parametrize("error", [
        AgentNotFoundError("agent not found"),
        AgentLockedError("agent locked"),
        RuntimeError("unexpected error"),
    ], ids=["not_found", "locked", "runtime"])
    async def test_propagates_construction_exception(
        self, agent_and_deps_client: AsyncClient, error: Exception
    ):
        """Exceptions raised during dep construction propagate uncaught to the caller."""
        @asynccontextmanager
        async def _raise(*args, **kwargs):
            raise error
            yield  # unreachable — makes this an async generator function

        with patch.object(AgentFactory, "build_agent_and_deps", _raise):
            with pytest.raises(type(error)):
                await agent_and_deps_client.get("/test/any-id")

    async def test_propagates_route_exception(
        self, agent_and_deps_client: AsyncClient, captured: dict
    ):
        """Exceptions raised inside the route after deps resolve propagate uncaught."""
        # route will inspect this dict entry and throw any exception present
        captured["route_raises"] = RuntimeError("route exploded")
        with pytest.raises(RuntimeError, match="route exploded"):
            await agent_and_deps_client.get("/test/any-id")


class TestGetAgentDeps:
    """get_agent_deps: yields AgentDeps (no Agent) with short lock timeout, propagates all exceptions uncaught."""

    @pytest.fixture(autouse=True)
    def mock_build_success(self):
        """Patches build_deps with a success CM for all tests.

        Provides self.mock_deps and self.contextman_state for assertions.
        Error tests override locally with a second patch.object call.

        NOTE: this could probably be deduplicated with TestGetAgentAndDeps but the exercise is of questionable value
        """
        self.mock_deps = MagicMock()
        self.contextman_state = {"entered": False, "exited": False}

        @asynccontextmanager
        async def _build(*args, **kwargs):
            self.contextman_state["entered"] = True
            try:
                yield self.mock_deps
            finally:
                self.contextman_state["exited"] = True

        with patch.object(AgentFactory, "build_deps", _build):
            yield

    async def test_yields_correct_deps(
        self, deps_only_client: AsyncClient, captured: dict
    ):
        """Happy path: passes through exactly the AgentDeps returned by build_deps."""
        response = await deps_only_client.get("/test/any-id")
        assert response.status_code == 200
        assert captured["deps"] is self.mock_deps

    async def test_enters_and_exits_build_cm(self, deps_only_client: AsyncClient):
        """build_deps CM is entered and exited after a successful request.

        Exception-path teardown is tested at the factory level (test_factory.py).
        """
        await deps_only_client.get("/test/any-id")
        assert self.contextman_state["entered"], "build_deps should have been entered as a CM"
        assert self.contextman_state["exited"], "build_deps CM should be exited after request completes"

    @pytest.mark.parametrize("error", [
        AgentNotFoundError("agent not found"),
        AgentLockedError("agent locked"),
        RuntimeError("unexpected error"),
    ], ids=["not_found", "locked", "runtime"])
    async def test_propagates_construction_exception(
        self, deps_only_client: AsyncClient, error: Exception
    ):
        """Exceptions raised during dep construction propagate uncaught to the caller."""
        @asynccontextmanager
        async def _raise(*args, **kwargs):
            raise error
            yield  # unreachable — makes this an async generator function

        with patch.object(AgentFactory, "build_deps", _raise):
            with pytest.raises(type(error)):
                await deps_only_client.get("/test/any-id")

    async def test_propagates_route_exception(
        self, deps_only_client: AsyncClient, captured: dict
    ):
        """Exceptions raised inside the route after deps resolve propagate uncaught."""
        captured["route_raises"] = RuntimeError("route exploded")
        with pytest.raises(RuntimeError, match="route exploded"):
            await deps_only_client.get("/test/any-id")


# These two test classes are a little disjointed with above as they were written later, but it lets us be more unit-testey
class TestGetAgentAppStates:
    """get_agent_app_state_reg: returns the app-wide agent state registry from request.app.state."""

    def test_returns_app_state_agent_app_state_reg(self):
        """Returns exactly the agent state registry stored on app.state."""
        mock_request = MagicMock()
        registry = {"agent-1": MagicMock(spec=AgentAppState)}
        mock_request.app.state.agent_app_state_reg = registry
        assert get_agent_app_state_reg(mock_request) is registry


class TestGetSessionDep:
    """get_session_dep: yields a session from get_session(app.state.engine), one per call."""

    @pytest.fixture
    def mock_request(self) -> MagicMock:
        req = MagicMock()
        req.app.state.engine = MagicMock()
        return req

    async def test_yields_session_from_engine(self, mock_request: MagicMock):
        """Yields the session returned by get_session, called with app.state.engine."""
        mock_session = MagicMock(spec=AsyncSession)
        captured_engine = None

        @asynccontextmanager
        async def mock_get_session(engine):
            nonlocal captured_engine
            captured_engine = engine
            yield mock_session

        with patch("api.fastapi_deps.get_session", mock_get_session):
            session = await get_session_dep(mock_request).__anext__()

        assert session is mock_session
        assert captured_engine is mock_request.app.state.engine

    async def test_new_session_per_call(self, mock_request: MagicMock):
        """Each invocation of get_session_dep yields a distinct session object."""
        @asynccontextmanager
        async def mock_get_session(engine):
            yield MagicMock(spec=AsyncSession)

        with patch("api.fastapi_deps.get_session", mock_get_session):
            session1 = await get_session_dep(mock_request).__anext__()
            session2 = await get_session_dep(mock_request).__anext__()

        assert session1 is not session2
