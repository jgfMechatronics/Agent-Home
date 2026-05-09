"""HTTP route tests for Section 4.1.

Tests the FastAPI routes using httpx AsyncClient. Uses dependency_overrides
to inject mock factories, avoiding real DB lookups in route tests.

Fixtures are defined here (not in conftest) because only this file uses them.

Fixtures from conftest used here:
- session: Test DB session (function-scoped, rolled back after each test)
- agent_record: Pre-created agent for tests that need an existing agent
- agent_with_blocks: Agent with memory blocks attached
"""
# Standard library
import json
from contextlib import asynccontextmanager
from unittest.mock import AsyncMock, Mock, patch
from uuid import uuid4

# Third-party
import pytest
import pytest_asyncio
from fastapi import FastAPI, HTTPException
from httpx import ASGITransport, AsyncClient, Response
from pydantic_ai import AgentRunResultEvent
from sqlalchemy.ext.asyncio import AsyncSession
from pydantic_ai.messages import (
    FunctionToolCallEvent,
    FunctionToolResultEvent,
    PartDeltaEvent,
    PartStartEvent,
    TextPart,
    TextPartDelta,
    ToolCallPart,
    ToolReturnPart,
)

# Local
from api.deps import get_agent_and_deps, get_session_dep
from agent.crud import create_agent_record
from conftest import make_deps
from db.models import AgentRecord, _utcnow
from api.schemas import AgentMetadataResponse, CoreMemoryResponse, MemoryBlockResponse

# --- Module-level test data ---

TOOL_CALL_PART = ToolCallPart(tool_name="memory_replace", args={"label": "notes"}, tool_call_id="call-1")
TOOL_RETURN_PART = ToolReturnPart(tool_name="memory_replace", content="Updated.", tool_call_id="call-1")

# Common event sequences for reuse across tests
MINIMAL_STREAM = lambda: [AgentRunResultEvent(result=Mock())]
TEXT_STREAM = lambda: [
    PartStartEvent(index=0, part=TextPart(content="Hello")),
    PartDeltaEvent(index=0, delta=TextPartDelta(content_delta=" world")),
    AgentRunResultEvent(result=Mock()),
]
TOOL_STREAM = lambda: [
    FunctionToolCallEvent(part=TOOL_CALL_PART),
    FunctionToolResultEvent(result=TOOL_RETURN_PART),
    AgentRunResultEvent(result=Mock()),
]


# --- Test Fixtures ---

@pytest.fixture
def app() -> FastAPI:
    """Fresh app instance per test — avoids state contamination."""
    from api.app import _create_app
    return _create_app()


@pytest_asyncio.fixture
async def client(app: FastAPI) -> AsyncClient:
    """Async HTTP client bound to the test app."""
    async with AsyncClient(
        transport=ASGITransport(app=app),
        base_url="http://test"
    ) as c:
        yield c


def make_mock_agent(events: list | None = None, raises_mid_stream: Exception | None = None) -> Mock:
    """Create a mock agent whose run_stream_events yields the given events.

    The mock is a plain async generator, matching Pydantic AI's current API.
    If raises_mid_stream is set, the exception is raised after all events are yielded.
    """
    agent = Mock()

    async def _stream(*args, **kwargs):
        for event in (events or []):
            yield event
        if raises_mid_stream is not None:
            raise raises_mid_stream

    agent.run_stream_events = _stream
    return agent


@pytest.fixture(autouse=True)
def override_db_session(app: FastAPI, session: AsyncSession):
    """Ensure routes use the test session, not a separate DB connection.
    
    Without this, routes call get_session_dep() -> new connection -> test data invisible.
    """
    async def _get_test_session():
        yield session
    
    app.dependency_overrides[get_session_dep] = _get_test_session
    yield
    app.dependency_overrides.pop(get_session_dep)



# --- Helpers ---

VALID_SSE_PREFIXES = ("data: ", "event: ", "id: ", "retry: ", ":")

async def collect_sse_events(response: Response) -> list[dict]:
    """Parse SSE events from a streaming response.

    Returns a list of dicts with 'event' (SSE event field) and 'data' (parsed JSON):
        {"event": "PartDeltaEvent", "data": {"index": 0, ...}}

    Assumes event: field precedes data: field within each event block (standard SSE order).
    Asserts that every non-empty line is a recognised SSE field — catches
    garbage or malformed content that a silent filter would miss.
    """
    events = []
    current_event_type: str | None = None
    async for line in response.aiter_lines():
        if not line:
            current_event_type = None  # blank line = event separator
            continue
        assert any(line.startswith(prefix) for prefix in VALID_SSE_PREFIXES), (
            f"Unexpected content in SSE stream: {line!r}"
        )
        if line.startswith("event: "):
            current_event_type = line[7:]
        elif line.startswith("data: "):
            events.append({"event": current_event_type, "data": json.loads(line[6:])})
    return events


async def stream_and_collect(client: AsyncClient, agent_id, message: str = "test") -> list[dict]:
    """POST to messages endpoint, assert 200, return parsed SSE events.
    Reduces boilerplate in streaming tests.
    """
    async with client.stream(
        "POST",
        f"/agents/{agent_id}/messages",
        json={"message": message},
    ) as response:
        assert response.status_code == 200
        return await collect_sse_events(response)


# --- Test Classes ---

class TestSendMessage:
    """
    POST /agents/{agent_id}/messages — main streaming endpoint.
    TODO (Low priority): This test class got a bit confusing, consider simplifying if possible
    """

    @pytest.fixture(autouse=True)
    def mock_agent_dep(self, app: FastAPI, agent_record: AgentRecord):
        """Provides self.agent_record, self.mock_session, and self.configure_mock_get_agent_and_deps.

        self.mock_session is a Mock with commit/rollback/refresh as AsyncMocks. The route calls
        session.commit() and session.rollback() — using the real AsyncSession inside httpx's
        ASGITransport would trigger MissingGreenlet (SQLAlchemy's greenlet bridge isn't set up there).

        A MINIMAL_STREAM() default is installed automatically. Tests that need specific events,
        exceptions, or mid-stream failures call self.configure_mock_get_agent_and_deps() to override.
        """
        self.agent_record = agent_record
        self.mock_session = Mock()
        self.mock_session.commit = AsyncMock()
        self.mock_session.rollback = AsyncMock()
        self.mock_session.refresh = AsyncMock()

        def _configure(events=None, raise_exc=None, raises_mid_stream=None):
            async def _mock_dep():
                if raise_exc is not None:
                    raise raise_exc
                yield make_mock_agent(events, raises_mid_stream), make_deps(self.mock_session, agent_record)

            app.dependency_overrides[get_agent_and_deps] = _mock_dep

        self.configure_mock_get_agent_and_deps = _configure

        # Default: most tests just need a minimal valid stream. Tests that require specific
        # events, exceptions, or mid-stream failures call configure again to override.
        _configure(events=MINIMAL_STREAM())
        yield

        app.dependency_overrides.pop(get_agent_and_deps)

    @pytest.fixture(autouse=True)
    def mock_route_side_effects(self):
        """Patch route-level side effects for all TestSendMessage tests.

        Uses create=True so existing tests stay green before routes.py imports these
        functions. Once implementation adds the imports, create=True is a harmless no-op.

        Provides self.mock_persist, self.mock_needs_compact, self.mock_compact
        for tests that assert on persistence/compaction behavior.
        """
        with (
            patch("api.routes.load_in_context_messages", new_callable=AsyncMock, create=True) as mock_load,
            patch("api.routes.persist_messages", new_callable=AsyncMock, create=True) as mock_persist,
            patch("api.routes.is_compaction_needed", create=True) as mock_needs_compact,
            patch("api.routes.compact", new_callable=AsyncMock, create=True) as mock_compact,
        ):
            mock_load.return_value = []  # safe default — no prior history
            mock_needs_compact.return_value = False  # safe default — no compaction unless overridden
            self.mock_load_in_context = mock_load
            self.mock_persist = mock_persist
            self.mock_needs_compact = mock_needs_compact
            self.mock_compact = mock_compact
            yield

    @pytest.mark.parametrize("events_factory,expected_types", [
        pytest.param(TEXT_STREAM, ["PartStartEvent", "PartDeltaEvent", "AgentRunResultEvent"], id="text-stream"),
        pytest.param(TOOL_STREAM, ["FunctionToolCallEvent", "FunctionToolResultEvent", "AgentRunResultEvent"], id="tool-stream"),
    ])
    async def test_event_types_are_forwarded_as_sse(
        self, client: AsyncClient, events_factory, expected_types
    ):
        """All run_stream_events event types are serialized and forwarded as SSE.

        Covers both text and tool-call streams to verify forwarding is event-agnostic.
        Tool-only stream also confirms zero-text-output runs complete normally.
        """
        self.configure_mock_get_agent_and_deps(events=events_factory())
        sse_events = await stream_and_collect(client, self.agent_record.id)
        assert [e["event"] for e in sse_events] == expected_types

    @pytest.mark.parametrize("exc,expected_status", [
        (HTTPException(status_code=404, detail="not found"), 404),
        (HTTPException(status_code=503, detail="in use"), 503),
    ])
    async def test_dep_http_exception_returns_appropriate_status(
        self, client: AsyncClient, exc, expected_status
    ):
        """HTTPExceptions raised by get_agent_and_deps propagate to the correct HTTP status.

        The actual domain→HTTP translation (AgentNotFoundError→404 etc.) is tested in test_deps.py.
        This test confirms the route correctly surfaces HTTPException from the dependency.
        (This test is somewhat redendant with the one in test_deps, but confirms we're using the dependency as expected)
        """
        self.configure_mock_get_agent_and_deps(raise_exc=exc)
        response = await client.post(f"/agents/{uuid4()}/messages", json={"message": "hi"})
        assert response.status_code == expected_status
    
    async def test_content_type_is_event_stream(self, client: AsyncClient):
        """Response Content-Type is text/event-stream for SSE."""
        async with client.stream(
            "POST",
            f"/agents/{self.agent_record.id}/messages",
            json={"message": "test"},
        ) as response:
            assert "text/event-stream" in response.headers["content-type"]
            await collect_sse_events(response)  # Consume to avoid warnings

    async def test_returns_400_for_malformed_body(self, client: AsyncClient):
        """Missing required 'message' field returns 400 or 422.

        The factory mock is required because FastAPI resolves dependencies before body
        validation — the stub get_agent_factory would otherwise raise NotImplementedError
        and mask the validation error.
        """
        self.configure_mock_get_agent_and_deps(events=[])
        response = await client.post(f"/agents/{uuid4()}/messages", json={})
        assert response.status_code in (400, 422)

    async def test_persists_messages_on_agent_run_result_event(self, client: AsyncClient):
        """persist_messages is called exactly once, triggered by AgentRunResultEvent.

        Three events in the stream — persist must not be called on the earlier two.
        """
        self.configure_mock_get_agent_and_deps(events=TEXT_STREAM())

        await stream_and_collect(client, self.agent_record.id)

        self.mock_persist.assert_called_once()

    @pytest.mark.parametrize("needs_compact,expect_compact", [(True, True), (False, False)])
    async def test_compaction_called_based_on_check(
        self, client: AsyncClient, needs_compact, expect_compact
    ):
        """compact is called iff is_compaction_needed returns True."""
        self.mock_needs_compact.return_value = needs_compact

        await stream_and_collect(client, self.agent_record.id)

        assert self.mock_compact.called == expect_compact

    async def test_yields_error_event_on_exception(self, client: AsyncClient):
        """Exception mid-stream yields Error SSE event after partial output, then closes."""
        # Simulate partial stream before exception (more realistic than immediate failure)
        self.configure_mock_get_agent_and_deps(
            events=[PartStartEvent(index=0, part=TextPart(content="Starting..."))],
            raises_mid_stream=RuntimeError("something went wrong"),
        )

        sse_events = await stream_and_collect(client, self.agent_record.id)

        # Should see partial event(s) + Error
        assert len(sse_events) == 2
        assert sse_events[0]["event"] == "PartStartEvent"
        assert sse_events[1]["event"] == "Error"
        assert sse_events[1]["data"]["message"] == "Unexpected internal server error: 'RuntimeError: something went wrong'"

    async def test_persist_called_when_new_messages_empty(self, client: AsyncClient):
        """
        persist_messages is called even when new_messages() returns [] — no-op for DB layer.
        TODO: Idk why we actually set this requirement
        """
        mock_result = Mock()
        mock_result.new_messages.return_value = []
        self.configure_mock_get_agent_and_deps(events=[AgentRunResultEvent(result=mock_result)])

        await stream_and_collect(client, self.agent_record.id)

        self.mock_persist.assert_called_once()

    # --- Transaction behaviour ---

    async def test_commits_session_on_happy_path(self, client: AsyncClient):
        """Session is committed exactly once after a successful run. No rollback."""
        await stream_and_collect(client, self.agent_record.id)

        self.mock_session.commit.assert_called_once()
        self.mock_session.rollback.assert_not_called()

    async def test_rollback_and_error_sse_on_persist_failure(self, client: AsyncClient):
        """persist_messages failure triggers rollback and yields Error SSE. Session is not committed."""
        self.mock_persist.side_effect = RuntimeError("DB write failed")

        sse_events = await stream_and_collect(client, self.agent_record.id)

        self.mock_session.rollback.assert_called_once()
        self.mock_session.commit.assert_not_called()
        assert sse_events[-1]["event"] == "Error"

    async def test_commits_before_compaction_failure_so_turn_is_preserved(self, client: AsyncClient):
        """Session is committed before compaction is attempted.

        If compaction fails, the turn's message writes survive — commit runs before
        the compaction check. The outer except then rolls back only the empty
        post-commit transaction, leaving the messages intact.
        """
        self.mock_needs_compact.return_value = True
        self.mock_compact.side_effect = RuntimeError("compaction failed")

        sse_events = await stream_and_collect(client, self.agent_record.id)

        self.mock_session.commit.assert_called_once()
        assert sse_events[-1]["event"] == "Error"

    async def test_yields_error_sse_on_mid_stream_exception(self, client: AsyncClient):
        """Error SSE is delivered to the client when an exception occurs mid-stream.

        Verifies that partial events (e.g. a TextPart that started streaming) are followed
        by an Error event — clients always receive an explicit signal that the request failed.
        """
        self.configure_mock_get_agent_and_deps(
            events=[PartStartEvent(index=0, part=TextPart(content="Starting..."))],
            raises_mid_stream=RuntimeError("mid-stream crash"),
        )

        sse_events = await stream_and_collect(client, self.agent_record.id)

        assert sse_events[-1]["event"] == "Error"


class TestCreateAgent:
    """POST /agents/ — create a new agent."""

    _NAME = "test-agent"
    _MODEL = "claude-sonnet-4-20250514"
    _VALID_BODY: dict = {
        "name": _NAME,
        "system_instructions": "Be helpful.",
        "config": {
            "model_name": _MODEL,
            "tool_names": [],
            "soft_compaction_limit": 1000,
        },
    }

    @pytest.fixture(autouse=True)
    def mock_create_agent_deps(self):
        with patch("api.routes.create_agent_record", new_callable=AsyncMock, create=True) as mock_create:
            self.mock_create_agent_record = mock_create
            yield

    async def test_creates_agent_and_returns_metadata(self, client: AsyncClient) -> None:
        """Creating an agent returns full metadata and 201 status."""
        expected_id = str(uuid4())
        DATETIME_NOW = _utcnow()

        mock_record = Mock()
        mock_record.id = expected_id
        mock_record.name = self._NAME
        mock_record.agent_config.model_name = self._MODEL
        mock_record.created_at = DATETIME_NOW
        mock_record.updated_at = DATETIME_NOW
        self.mock_create_agent_record.return_value = mock_record

        expected_metadata = AgentMetadataResponse(
            id=expected_id,
            name=self._NAME,
            model=self._MODEL,
            created_at=DATETIME_NOW,
            updated_at=DATETIME_NOW,
        )

        response = await client.post("/agents/", json=self._VALID_BODY)

        assert response.status_code == 201
        self.mock_create_agent_record.assert_called_once()
        assert AgentMetadataResponse.model_validate(response.json()) == expected_metadata

    async def test_returns_500_when_create_agent_fails(self, client: AsyncClient):
        """Internal failure in create_agent returns 500 with exception details."""
        self.mock_create_agent_record.side_effect = RuntimeError("DB failure")
        response = await client.post("/agents/", json=self._VALID_BODY)
        assert response.status_code == 500
        assert response.json()["detail"] == "Exception during agent creation: RuntimeError: DB failure"

    async def test_returns_400_for_invalid_config(self, client: AsyncClient):
        """Missing required fields result in 400 before route logic is reached."""
        response = await client.post(
            "/agents/",
            json={"name": "incomplete"},  # missing system_instructions and config
        )
        
        assert response.status_code in (400, 422)  # FastAPI validation error


class TestGetAgent:
    """GET /agents/{agent_id} — agent metadata."""
    
    async def test_returns_agent_metadata(self, client: AsyncClient, agent_record: AgentRecord):
        """
        Returns agent metadata: name, model, created_at, updated_at.
        TODO: Should this assert that calls the appropriate internal function?
        Might be an impl detail we *don't* want to test actually
        """
        response = await client.get(f"/agents/{agent_record.id}")
        metadata = AgentMetadataResponse.model_validate(response.json())
        expected_metadata = AgentMetadataResponse(
            id=agent_record.id,
            name=agent_record.name,
            model=agent_record.agent_config.model_name,
            created_at=agent_record.created_at,
            updated_at=agent_record.updated_at,
        )

        assert response.status_code == 200
        assert metadata == expected_metadata
    
    # 404 tested via parametrized test_get_endpoints_return_404_for_unknown_agent


class TestGetCoreMemory:
    """GET /agents/{agent_id}/core_memory — memory blocks."""
    
    async def test_returns_memory_blocks(self, client: AsyncClient, agent_with_blocks: dict):
        """Returns blocks in position order with all schema fields present."""
        agent = agent_with_blocks["agent"]
        blocks = agent_with_blocks["blocks"]

        response = await client.get(f"/agents/{agent.id}/core_memory")

        assert response.status_code == 200
        actual = CoreMemoryResponse.model_validate(response.json())
        expected = CoreMemoryResponse(blocks=[
            MemoryBlockResponse(
                label=block.label,
                description=block.description,
                content=block.content,
                char_limit=block.char_limit,
                updated_at=block.updated_at,
            )
            for block in blocks
        ])
        assert actual == expected

    async def test_returns_empty_blocks_list_when_no_blocks(self, client: AsyncClient, agent_record: AgentRecord):
        """Returns empty blocks list when agent has no memory blocks."""
        response = await client.get(f"/agents/{agent_record.id}/core_memory")

        assert response.status_code == 200
        data = response.json()
        assert data["blocks"] == []

    # 404 tested via parametrized test_get_endpoints_return_404_for_unknown_agent


class TestGetMessages:
    """
    GET /agents/{agent_id}/messages — conversation history.
    TODO: This is OK for now but we will likely rework the endpoint after defining what is most useful for the frontend in terms of message format
    """

    @pytest.fixture(autouse=True)
    def mock_message_loaders(self):
        """Patch message-loading functions for all TestGetMessages tests.

        Uses create=True so tests stay green before routes.py imports these functions.

        Provides self.mock_in_context and self.mock_full for loader-routing assertions.
        """
        with (
            patch("api.routes.load_in_context_messages", new_callable=AsyncMock, create=True) as mock_in_context,
            patch("api.routes.load_message_history", new_callable=AsyncMock, create=True) as mock_full,
        ):
            mock_in_context.return_value = []
            mock_full.return_value = []
            self.mock_in_context = mock_in_context
            self.mock_full = mock_full
            yield

    async def test_default_loads_context_window_and_returns_messages(self, client: AsyncClient, agent_record: AgentRecord):
        """Without ?full=true: calls load_in_context_messages, returns its result."""
        expected_messages = [{"role": "user", "content": "test"}]
        self.mock_in_context.return_value = expected_messages

        response = await client.get(f"/agents/{agent_record.id}/messages")

        assert response.status_code == 200
        assert response.json()["messages"] == expected_messages
        self.mock_in_context.assert_called_once()
        self.mock_full.assert_not_called()

    async def test_full_true_returns_complete_history(self, client: AsyncClient, agent_record: AgentRecord):
        """With ?full=true: calls load_message_history, returns its result."""
        expected_messages = [{"role": "user", "content": "old"}, {"role": "assistant", "content": "reply"}]
        self.mock_full.return_value = expected_messages

        response = await client.get(f"/agents/{agent_record.id}/messages?full=true")

        assert response.status_code == 200
        assert response.json()["messages"] == expected_messages
        self.mock_full.assert_called_once()
        self.mock_in_context.assert_not_called()

    @pytest.mark.xfail(reason="TODO: Finalize MessageItemFormat")
    async def test_returns_reasonable_format(self):
        # TODO: finalize MessageItem format, constrain MessageResponse (or whatever it is) to be list[MessageItem]
        pytest.fail()

    # 404 tested via parametrized test_get_endpoints_return_404_for_unknown_agent


class TestHealthCheck:
    """GET /health — service health."""
    
    async def test_returns_200_ok(self, client: AsyncClient):
        """Health endpoint returns 200 with status."""
        response = await client.get("/health")
        
        assert response.status_code == 200
        assert response.json()["status"] == "ok"

    @pytest.mark.xfail(reason="TODO: requires DB integration in app lifespan — need to determine how to simulate unreachable DB")
    async def test_returns_503_when_db_unreachable(self, client: AsyncClient):
        """Health endpoint should return 503 when the DB is unreachable."""
        response = await client.get("/health")
        assert response.status_code == 503


class TestNotFound:
    """404 behavior for unknown agent_id across all endpoints."""

    @pytest.mark.parametrize("path", [
        "/agents/{agent_id}",
        "/agents/{agent_id}/core_memory",
        "/agents/{agent_id}/messages",
    ])
    async def test_get_endpoints_return_404_for_unknown_agent(self, client: AsyncClient, path: str):
        """All GET endpoints with agent_id return 404 for unknown agents."""
        url = path.format(agent_id=uuid4())
        response = await client.get(url)
        assert response.status_code == 404
