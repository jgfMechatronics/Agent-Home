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
import asyncio
import importlib.metadata
import json
from contextlib import asynccontextmanager
from datetime import datetime
from unittest.mock import AsyncMock, Mock, patch
from uuid import uuid4

# Third-party
import pytest
import pytest_asyncio
from packaging.version import Version
from fastapi import FastAPI, HTTPException
from httpx import ASGITransport, AsyncClient, Response
from pydantic_ai import Agent, AgentRunResultEvent
from sqlalchemy.ext.asyncio import AsyncSession
from pydantic_ai.messages import (
    FunctionToolCallEvent,
    FunctionToolResultEvent,
    ModelRequest,
    ModelResponse,
    PartDeltaEvent,
    PartStartEvent,
    RetryPromptPart,
    TextPart,
    TextPartDelta,
    ToolCallPart,
    ToolReturnPart,
    UserPromptPart,
)
from pydantic_ai.models.function import AgentInfo, DeltaToolCall, DeltaToolCalls, FunctionModel

# Local
from agent.factory import AgentNotFoundError
from api.fastapi_deps import get_agent_and_deps, get_deps_dep, get_session_dep
from agent.crud import create_agent_record
from conftest import make_deps
from db.models import AgentRecord, MemoryBlockRecord, utcnow
from api.schemas import AgentMetadataResponse, CoreMemoryResponse, MemoryBlockResponse
from memory.block_crud import DuplicateBlockError

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
    FunctionToolResultEvent(part=TOOL_RETURN_PART),
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
    """Async HTTP client bound to the test app.

    raise_app_exceptions=False so that app-level Exception handlers (ServerErrorMiddleware)
    just return the 500 response rather than re-raising into the test.
    """
    async with AsyncClient(
        transport=ASGITransport(app=app, raise_app_exceptions=False),
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


@pytest.fixture
def _base_route_patches():
    """Patch 4 route-level side effects (everything except persist_messages).

    Shared by TestSendMessage.mock_route_side_effects and
    TestPersistenceAcrossInterruptions.persist_spy so the patch set stays DRY.

    Yields a dict: {"load", "deserialize", "needs_compact", "compact"}
    """
    with (
        patch("api.routes.load_messages", new_callable=AsyncMock) as mock_load,
        patch("api.routes.deserialize_messages") as mock_deserialize,
        patch("api.routes.is_compaction_needed") as mock_needs_compact,
        patch("api.routes.compact", new_callable=AsyncMock) as mock_compact,
    ):
        mock_load.return_value = []
        mock_deserialize.return_value = []
        mock_needs_compact.return_value = False
        yield {
            "load": mock_load,
            "deserialize": mock_deserialize,
            "needs_compact": mock_needs_compact,
            "compact": mock_compact,
        }


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
        TODO: Test gap, we don't actually test that the pyddantic AI agent is contructed for the loop with the expected inputs
        IE we could not pass the right deps, pass the wrong message history, etc. Pretty sure we also don't check that the right
        method is called on the agent. We already have a mock agent so we can just use that.
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
    def mock_route_side_effects(self, _base_route_patches):
        """Patch route-level side effects for all TestSendMessage tests.

        Provides self.mock_persist, self.mock_needs_compact, self.mock_compact
        for tests that assert on persistence/compaction behavior.
        """
        with patch("api.routes.persist_messages", new_callable=AsyncMock) as mock_persist:
            self.mock_load_messages = _base_route_patches["load"]
            self.mock_deserialize = _base_route_patches["deserialize"]
            self.mock_needs_compact = _base_route_patches["needs_compact"]
            self.mock_compact = _base_route_patches["compact"]
            self.mock_persist = mock_persist
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
        with patch("api.routes.create_agent_record", new_callable=AsyncMock) as mock_create:
            self.mock_create_agent_record = mock_create
            yield

    async def test_creates_agent_and_returns_metadata(self, client: AsyncClient) -> None:
        """Creating an agent returns full metadata and 201 status."""
        expected_id = str(uuid4())
        DATETIME_NOW = utcnow()

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
        """Route propagates unexpected exceptions to the app-level handler, returning 500."""
        self.mock_create_agent_record.side_effect = RuntimeError("DB failure")
        response = await client.post("/agents/", json=self._VALID_BODY)
        assert response.status_code == 500
        assert response.json()["detail"] == "RuntimeError: DB failure"

    async def test_returns_400_for_invalid_config(self, client: AsyncClient):
        """Missing required fields result in 400 before route logic is reached."""
        response = await client.post(
            "/agents/",
            json={"name": "incomplete"},  # missing system_instructions and config
        )
        assert response.status_code in (400, 422)  # FastAPI validation error


# ---------------------------------------------------------------------------
# Persistence helpers (used by TestPersistenceAcrossInterruptions)
# ---------------------------------------------------------------------------

def _union_of_persisted(spy) -> list:
    """Concatenate messages from all persist_messages calls, in call order."""
    union = []
    for call in spy.call_args_list:
        msgs = call.kwargs.get("messages") or (call.args[1] if len(call.args) > 1 else [])
        union.extend(msgs)
    return union


def _select(union: list, msg_type: type, part_type: type) -> list:
    """Messages of msg_type in the union carrying at least one part of part_type."""
    return [
        m for m in union
        if isinstance(m, msg_type) and any(isinstance(p, part_type) for p in m.parts)
    ]


def _assert_no_duplicates(union: list) -> None:
    """Each message object must appear at most once in the persist union."""
    for i in range(len(union)):
        for j in range(i + 1, len(union)):
            assert union[i] is not union[j], (
                f"Duplicate message at indices {i} and {j}: {union[i]!r}"
            )


def _assert_no_orphans(union: list) -> None:
    """Every ToolCallPart must have a matching ToolReturnPart or RetryPromptPart."""
    tool_call_ids: set[str] = set()
    response_ids: set[str] = set()
    for msg in union:
        if isinstance(msg, ModelResponse):
            for part in msg.parts:
                if isinstance(part, ToolCallPart):
                    tool_call_ids.add(part.tool_call_id)
        elif isinstance(msg, ModelRequest):
            for part in msg.parts:
                if isinstance(part, (ToolReturnPart, RetryPromptPart)):
                    response_ids.add(part.tool_call_id)
    orphans = tool_call_ids - response_ids
    assert not orphans, f"Orphaned ToolCallPart IDs with no matching return: {orphans}"


def _make_mock_session() -> Mock:
    """Build a mock AsyncSession with async commit/rollback/refresh."""
    session = Mock()
    session.commit = AsyncMock()
    session.rollback = AsyncMock()
    session.refresh = AsyncMock()
    return session


def _make_function_agent(stream_fn) -> Agent:
    """Build a real pydantic-ai Agent backed by FunctionModel with a record_thing tool."""
    agent: Agent = Agent(FunctionModel(stream_function=stream_fn))

    def record_thing(thing: str) -> str:
        return f"recorded: {thing}"

    agent.tool_plain(record_thing)
    return agent


def _has_tool_return(messages) -> bool:
    """True when messages include a ToolReturnPart — i.e. the stream is in step 2."""
    return any(
        isinstance(part, ToolReturnPart)
        for msg in messages if isinstance(msg, ModelRequest)
        for part in msg.parts
    )


def _tool_call_delta(tool_name: str, tool_call_id: str) -> DeltaToolCalls:
    """A single-tool-call streaming delta. The tool arg is asserted nowhere, so fixed."""
    return DeltaToolCalls(
        {0: DeltaToolCall(name=tool_name, json_args='{"thing": "x"}', tool_call_id=tool_call_id)}
    )


async def _two_step_stream(messages, info: AgentInfo):
    """Step 1 → emit tool call; step 2 (after tool return) → yield final text."""
    if _has_tool_return(messages):
        yield "Turn complete."
    else:
        yield _tool_call_delta("record_thing", "tc-a1")


async def _exception_after_tool_return_stream(messages, info: AgentInfo):
    """Step 1 → emit tool call; step 2 → yield one text chunk then raise.

    The leading yield is required: FunctionModel calls peek() during request_stream
    __aenter__, which must succeed for setup to complete.  The exception on the
    *next* __anext__ then propagates through the event iteration loop (where the
    route's try/except can catch it) rather than during context-manager setup
    (which bypasses it).
    """
    if _has_tool_return(messages):
        yield "start..."  # let peek() succeed; exception fires on the next __anext__
        raise RuntimeError("Simulated crash mid-stream")
    yield _tool_call_delta("record_thing", "tc-b1")


# ---------------------------------------------------------------------------
# TestPersistenceAcrossInterruptions
# ---------------------------------------------------------------------------

class _PersistSpyMixin:
    """Shared autouse fixture: spy on persist_messages (autospec) + expose base patches.

    Inherited by the persistence and cancellation test classes, which both assert on
    the persisted-message union via self.spy.
    """

    @pytest.fixture(autouse=True)
    def persist_spy(self, _base_route_patches):
        """Spy on persist_messages (autospec) + expose base patches as attrs."""
        with patch("api.routes.persist_messages", autospec=True) as spy:
            self.spy = spy
            self.mock_load_messages = _base_route_patches["load"]
            self.mock_deserialize = _base_route_patches["deserialize"]
            self.mock_needs_compact = _base_route_patches["needs_compact"]
            self.mock_compact = _base_route_patches["compact"]
            yield


class TestPersistenceAcrossInterruptions(_PersistSpyMixin):
    """Persistence contract tests using a real pydantic-ai Agent + FunctionModel.

    Test 1 (happy path) documents existing persist-at-terminal-event behavior and
    will stay green as the implementation changes.

    Test 2 (exception mid-stream) is a CONTRACT-DEFINING RED TEST.  Current impl
    rolls back on exception and never persists.  The test encodes the desired
    behavior: persist the completed portion even when the run crashes mid-stream.
    """

    @pytest.fixture(autouse=True)
    def real_agent_dep(self, app, agent_record):
        """Inject a real Agent + mock session via get_agent_and_deps override.

        Exposes self.agent_record, self.mock_session.
        Call self.set_stream_fn(fn) before the request to swap the stream function.
        """
        self.agent_record = agent_record
        self.mock_session = _make_mock_session()

        self._stream_fn = _two_step_stream

        def set_stream_fn(fn):
            self._stream_fn = fn

        self.set_stream_fn = set_stream_fn

        async def _dep():
            agent = _make_function_agent(self._stream_fn)
            deps = make_deps(self.mock_session, agent_record)
            yield agent, deps

        app.dependency_overrides[get_agent_and_deps] = _dep
        yield
        app.dependency_overrides.pop(get_agent_and_deps)

    @pytest.mark.asyncio
    async def test_happy_path_persists_full_message_union(self, client: AsyncClient):
        """Persist union contains all four message types in causal order.

        Uses a real pydantic-ai Agent so new_messages() reflects actual pydantic-ai
        message structure.  Validates the full pipeline against the real library.
        """
        fake_history = [ModelRequest(parts=[UserPromptPart(content="prior turn")])]
        self.mock_deserialize.return_value = fake_history

        events = await stream_and_collect(client, self.agent_record.id)
        event_types = [e["event"] for e in events]
        assert "Error" not in event_types, f"Unexpected Error event: {events}"

        union = _union_of_persisted(self.spy)

        user_reqs = _select(union, ModelRequest, UserPromptPart)
        tool_resps = _select(union, ModelResponse, ToolCallPart)
        tool_return_reqs = _select(union, ModelRequest, ToolReturnPart)
        text_resps = _select(union, ModelResponse, TextPart)

        assert len(user_reqs) == 1, "Expected exactly one UserPrompt in union"
        assert len(tool_resps) == 1, "Expected exactly one ToolCall response in union"
        assert len(tool_return_reqs) == 1, "Expected exactly one ToolReturn request in union"
        assert len(text_resps) == 1, "Expected exactly one final Text response in union"

        causal_order = [union.index(m) for m in (user_reqs[0], tool_resps[0], tool_return_reqs[0], text_resps[0])]
        assert causal_order == sorted(causal_order), "Messages are not in causal order in union"

        _assert_no_duplicates(union)
        _assert_no_orphans(union)

        for h_msg in fake_history:
            assert h_msg not in union, f"History message leaked into persist union: {h_msg!r}"

        assert self.mock_session.commit.called, "Session must be committed on happy path"
        assert not self.mock_session.rollback.called, "Session must NOT be rolled back on happy path"

    @pytest.mark.asyncio
    async def test_persist_survives_mid_run_exception(self, client: AsyncClient):
        """CONTRACT-DEFINING RED TEST.

        Current behavior: exception → rollback, persist_messages never called.
        Desired behavior: persist the completed portion of the run (up to the crash),
        commit it, surface Error SSE — do NOT discard completed work.

        Will fail with current implementation.  Goes green when the
        cancellation+improved-persistence implementation lands.
        """
        self.set_stream_fn(_exception_after_tool_return_stream)

        events = await stream_and_collect(client, self.agent_record.id)
        event_types = [e["event"] for e in events]
        assert "Error" in event_types, "Expected Error SSE after mid-run exception"

        # --- All assertions below are RED with current impl ---

        assert self.spy.call_count >= 1, (
            "persist_messages must be called even when the run crashes mid-stream"
        )

        union = _union_of_persisted(self.spy)

        tool_resps = _select(union, ModelResponse, ToolCallPart)
        tool_return_reqs = _select(union, ModelRequest, ToolReturnPart)
        text_resps = _select(union, ModelResponse, TextPart)

        assert len(tool_resps) >= 1, "Completed tool call must appear in persist union"
        assert len(tool_return_reqs) >= 1, "Completed tool return must appear in persist union"
        assert len(text_resps) == 0, "No final text expected: crash before step 2 yielded text"

        _assert_no_orphans(union)

        assert self.mock_session.commit.called, (
            "Completed work must be committed even when the run crashes"
        )
        assert not self.mock_session.rollback.called, (
            "Rollback discards completed work — must not roll back on mid-run exception"
        )


async def _blocking_tool_stream(messages, info: AgentInfo):
    """Step 1 → emit blocking_tool call; step 2 (after tool return) → yield final text."""
    if _has_tool_return(messages):
        yield "Done."
    else:
        yield _tool_call_delta("blocking_tool", "tc-c1")


def _make_blocking_agent(tool_entered: asyncio.Event, release: asyncio.Event) -> Agent:
    """Build a real Agent whose 'blocking_tool' signals entry and waits for release.

    Used for deterministic mid-tool cancellation timing: await tool_entered to confirm
    the tool is provably in-flight, then act (cancel/etc), then release.set() to unblock.
    """
    agent: Agent = Agent(FunctionModel(stream_function=_blocking_tool_stream))

    async def blocking_tool(thing: str) -> str:
        tool_entered.set()
        await release.wait()
        return f"recorded: {thing}"

    agent.tool_plain(blocking_tool)
    return agent


# ---------------------------------------------------------------------------
# Version constants for rendezvous regression guard
# ---------------------------------------------------------------------------

_PYDANTIC_AI_VERSION = importlib.metadata.version("pydantic-ai")
# Rendezvous semantics confirmed on pydantic-ai 1.104.0 (PR #5313, merged May 2026).
# If the rendezvous test starts failing after a pydantic-ai upgrade, our cancel
# strategy must be re-evaluated before shipping.
_RENDEZVOUS_MIN_VERSION = "1.104.0"


# ---------------------------------------------------------------------------
# TestCancellation
# ---------------------------------------------------------------------------

class TestCancellation(_PersistSpyMixin):
    """Cancellation contract tests.

    Uses a blocking tool for deterministic timing: the tool signals 'tool_entered'
    when it starts and waits on 'release' before returning, letting the test act
    precisely while the tool is in flight.

    test_graceful_cancel: xfail pending cancel route + _cancel_signals implementation.
    """

    @pytest.fixture(autouse=True)
    def real_agent_dep(self, app, agent_record):
        """Inject a real blocking-tool Agent + mock session via get_agent_and_deps."""
        self.agent_record = agent_record
        self.tool_entered = asyncio.Event()
        self.release = asyncio.Event()
        self.mock_session = _make_mock_session()

        async def _dep():
            agent = _make_blocking_agent(self.tool_entered, self.release)
            deps = make_deps(self.mock_session, agent_record)
            yield agent, deps

        app.dependency_overrides[get_agent_and_deps] = _dep
        yield
        app.dependency_overrides.pop(get_agent_and_deps)

    @pytest.mark.xfail(
        strict=True,
        reason=(
            "cancel route (POST /agents/{id}/cancel) and _cancel_signals mechanism "
            "not yet implemented; xfail strict=True forces cleanup when impl lands"
        ),
    )
    @pytest.mark.asyncio
    async def test_graceful_cancel(self, client: AsyncClient):
        """CONTRACT: graceful cancel lets the in-flight tool complete, persists the
        tool call+return pair and a cancellation notice, then commits — without rolling back.

        Covers requirements:
          #4 — active tool is allowed to complete before cancel takes effect
          #5 — cancellation notice wrapped in <system_message> tags is persisted
          #7 — cancel is delivered via POST /agents/{id}/cancel

        xfail: will pass once the cancel route and _cancel_signals mechanism land.
        strict=True: when this test unexpectedly passes, pytest errors, forcing removal
        of this marker.
        """
        # Start the SSE stream as a background task so we can interleave cancel.
        stream_task = asyncio.create_task(
            stream_and_collect(client, self.agent_record.id)
        )

        # Wait until the tool is provably in-flight before sending cancel.
        await asyncio.wait_for(self.tool_entered.wait(), timeout=5.0)

        # Deliver cancel (currently 404 — no route yet).
        cancel_response = await client.post(f"/agents/{self.agent_record.id}/cancel")

        # Always release so the stream task can finish regardless of cancel outcome.
        self.release.set()
        events = await asyncio.wait_for(stream_task, timeout=5.0)

        # --- Contract assertions (all RED without cancel implementation) ---

        assert cancel_response.status_code == 200, (
            f"Cancel route must return 200; got {cancel_response.status_code}"
        )

        assert self.spy.call_count >= 1, (
            "persist_messages must be called even on graceful cancel"
        )

        union = _union_of_persisted(self.spy)

        # Tool call + return pair (tool was allowed to complete — req #4).
        assert len(_select(union, ModelResponse, ToolCallPart)) >= 1, (
            "Tool call must be persisted on graceful cancel"
        )
        assert len(_select(union, ModelRequest, ToolReturnPart)) >= 1, (
            "Tool return must be persisted — tool must complete before cancel takes effect"
        )
        _assert_no_orphans(union)

        # No final assistant text: post-tool model step must not run after cancel (req #4).
        assert len(_select(union, ModelResponse, TextPart)) == 0, (
            "Post-tool model step must not run after cancel"
        )

        # Cancellation notice persisted as <system_message> (req #5).
        cancel_notices = [
            m for m in union
            if isinstance(m, ModelRequest)
            and any(
                isinstance(p, UserPromptPart) and "<system_message>" in p.content
                for p in m.parts
            )
        ]
        assert len(cancel_notices) >= 1, (
            "A cancellation notice wrapped in <system_message> tags must be persisted"
        )

        # Commit, no rollback — completed work must not be discarded.
        assert self.mock_session.commit.called, (
            "Completed work must be committed on graceful cancel"
        )
        assert not self.mock_session.rollback.called, (
            "Must not rollback completed work on graceful cancel"
        )


# ---------------------------------------------------------------------------
# Rendezvous regression guard (standalone — no class fixtures needed)
# ---------------------------------------------------------------------------

@pytest.mark.asyncio
@pytest.mark.skipif(
    Version(_PYDANTIC_AI_VERSION) < Version(_RENDEZVOUS_MIN_VERSION),
    reason=f"Rendezvous semantics not verified before pydantic-ai {_RENDEZVOUS_MIN_VERSION}",
)
async def test_rendezvous_tool_does_not_start_before_event_consumed():
    """Regression guard: pydantic-ai rendezvous semantics.

    Property: a tool does NOT begin executing until its FunctionToolCallEvent has
    been consumed by the caller.  Our cancel strategy depends entirely on this —
    we can break out of the event stream BEFORE consuming FunctionToolCallEvent
    and guarantee the tool has not started (and therefore cancel without orphaning it).

    Drives a real Agent directly (no HTTP route) to isolate the pydantic-ai behaviour.

    Pinned to pydantic-ai {_PYDANTIC_AI_VERSION} — see _RENDEZVOUS_MIN_VERSION.
    If a pydantic-ai upgrade breaks this test, the cancellation strategy must be
    re-evaluated before shipping.
    """
    tool_entered = asyncio.Event()
    release = asyncio.Event()
    release.set()  # release immediately; we only need the entry signal here

    agent = _make_blocking_agent(tool_entered, release)

    async with asyncio.timeout(5.0):
        async with agent.run_stream_events("test", message_history=[]) as stream:
            events_iter = stream.__aiter__()

            # Consume events strictly before FunctionToolCallEvent.
            # FunctionToolCallEvent is emitted only after the stream function's generator
            # is exhausted, so at least one PartStartEvent/PartDeltaEvent precedes it.
            event = await events_iter.__anext__()
            pre_tool_event_seen = False

            while not isinstance(event, FunctionToolCallEvent):
                pre_tool_event_seen = True
                # THE INVARIANT: until FunctionToolCallEvent is consumed, the tool
                # must not have started (producer is blocked on the rendezvous send).
                # Yield real time to the event loop to let the producer attempt to advance.
                await asyncio.sleep(0.2)
                assert not tool_entered.is_set(), (
                    f"Rendezvous violated: blocking_tool started before "
                    f"FunctionToolCallEvent was consumed "
                    f"(pydantic-ai {_PYDANTIC_AI_VERSION}). "
                    f"Cancel strategy must be re-evaluated."
                )
                event = await events_iter.__anext__()

            assert pre_tool_event_seen, (
                "No events appeared before FunctionToolCallEvent — "
                "the rendezvous invariant was never exercised. "
                "Check whether pydantic-ai changed its event ordering."
            )

            # FunctionToolCallEvent consumed: tool is now allowed to start.
            # Drain remaining events to allow clean context-manager teardown.
            async for _ in stream:
                pass

    # After FunctionToolCallEvent consumed and release already set, tool should have run.
    assert tool_entered.is_set(), (
        "Tool must execute after FunctionToolCallEvent is consumed"
    )


async def test_TODO_decide_cancel_orphan_tool_call_handling():
    """DECISION REQUIRED before the persist-on-cancel implementation ships.

    When a cancel lands on a tool call that was *generated but never run*
    (rendezvous guarantees the tool didn't start — see
    test_rendezvous_tool_does_not_start_before_event_consumed), the tail of the
    captured/persisted messages is a lone ToolCallPart with no matching
    ToolReturnPart. We must choose ONE of:

      (A) TRIM it in the cursor/persist logic — drop the lone ToolCallPart before
          persisting; the tool is simply re-requested on the next turn. Preserves
          any sibling text/thinking parts on that ModelResponse. Requires a unit
          test of the trim/cursor logic.

      (B) LET persist_messages EAT it — its orphan sanitizer already replaces an
          unmatched ToolCallPart with an "[Orphaned tool call(s) dropped]" record,
          keeping the history API-valid. Cheaper, but first verify the sanitizer
          does not also discard accompanying text/thinking parts on the same
          ModelResponse.

    Both options yield API-valid history (no dangling tool_use), so this is a
    narrative-cleanliness call, deferred deliberately. Resolve it, update the plan
    doc's Test Coverage section, and REPLACE this marker with the real test.
    """
    pytest.fail(
        "TODO: decide cancel-orphan tool-call handling (trim vs. let "
        "persist_messages eat it), then replace this marker with the real "
        "persistence unit test. "
        "See docs/Planning/cancellation_and_improved_persistence_plan.md."
    )


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


class TestGetMemoryBlocks:
    """GET /agents/{agent_id}/memory/blocks — memory blocks."""
    
    async def test_returns_memory_blocks(self, client: AsyncClient, agent_with_blocks: dict):
        """Returns blocks in position order with all schema fields present."""
        agent = agent_with_blocks["agent"]
        blocks = agent_with_blocks["blocks"]

        response = await client.get(f"/agents/{agent.id}/memory/blocks")

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
        response = await client.get(f"/agents/{agent_record.id}/memory/blocks")

        assert response.status_code == 200
        data = response.json()
        assert data["blocks"] == []

    # 404 tested via parametrized test_get_endpoints_return_404_for_unknown_agent


@pytest.mark.xfail(reason="get_messages endpoint format TBD — will be reworked once coding CLI/harness is selected")
class TestGetMessages:
    """
    GET /agents/{agent_id}/messages — conversation history.
    TODO: This is OK for now but we will likely rework the endpoint after defining what is most useful for the frontend in terms of message format
    """

    @pytest.fixture(autouse=True)
    def mock_message_loaders(self):
        """Patch message-loading functions for all TestGetMessages tests.

        Provides self.mock_load_messages for loader-routing assertions.
        """
        with (
            patch("api.routes.load_messages", new_callable=AsyncMock) as mock_load,
        ):
            mock_load.return_value = []
            self.mock_load_messages = mock_load
            yield

    async def test_default_loads_context_window_and_returns_messages(self, client: AsyncClient, agent_record: AgentRecord, session: AsyncSession):
        """Without ?full=true: calls load_messages with context_window_start as start_timestamp."""
        expected_messages = [{"role": "user", "content": "test"}]
        self.mock_load_messages.return_value = expected_messages

        response = await client.get(f"/agents/{agent_record.id}/messages")

        assert response.status_code == 200
        assert response.json()["messages"] == expected_messages
        self.mock_load_messages.assert_called_once_with(
            session, agent_record.id, start_timestamp=agent_record.context_window_start
        )

    async def test_full_true_returns_complete_history(self, client: AsyncClient, agent_record: AgentRecord, session: AsyncSession):
        """With ?full=true: calls load_messages with start_timestamp=None for full history."""
        expected_messages = [{"role": "user", "content": "old"}, {"role": "assistant", "content": "reply"}]
        self.mock_load_messages.return_value = expected_messages

        response = await client.get(f"/agents/{agent_record.id}/messages?full=true")

        assert response.status_code == 200
        assert response.json()["messages"] == expected_messages
        self.mock_load_messages.assert_called_once_with(
            session, agent_record.id, start_timestamp=None
        )

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
        "/agents/{agent_id}/memory/blocks",
        "/agents/{agent_id}/messages",
    ])
    async def test_get_endpoints_return_404_for_unknown_agent(self, client: AsyncClient, path: str):
        """All GET endpoints with agent_id return 404 for unknown agents."""
        url = path.format(agent_id=uuid4())
        response = await client.get(url)
        assert response.status_code == 404


class TestCreateMemoryBlock:
    """POST /agents/{agent_id}/memory/blocks — create a memory block."""

    _VALID_BODY = {
        "label": "notes",
        "content": "Some content.",
        "description": "A scratch pad.",
        "char_limit": 5000,
    }
    _MOCK_UPDATED_AT = datetime(2026, 1, 1, 12, 0, 0)

    @pytest.fixture(autouse=True)
    def mock_create_block_dep(self, app: FastAPI, agent_record: AgentRecord):
        """Overrides get_deps_dep and patches create_block for all tests.

        Provides self.configure_mock_get_deps_dep() to change dep behavior (e.g. raise
        AgentNotFoundError for 404 tests). Default: yields a valid AgentDeps.
        """
        self.agent_record = agent_record
        self.mock_session = Mock()

        def _configure(raise_exc=None):
            async def _mock_dep():
                if raise_exc is not None:
                    raise raise_exc
                yield make_deps(self.mock_session, agent_record)
                
            app.dependency_overrides[get_deps_dep] = _mock_dep

        self.configure_mock_get_deps_dep = _configure
        _configure()  # default: happy path

        with patch("api.routes.create_block", new_callable=AsyncMock) as mock:
            self.mock_create_block = mock
            yield

        app.dependency_overrides.pop(get_deps_dep)

    async def test_calls_create_block_and_returns_201(self, client: AsyncClient):
        """Successful creation calls create_block and returns 201 with block data."""
        mock_block_record = MemoryBlockRecord(
            agent_id="dummy", position=0, updated_at=self._MOCK_UPDATED_AT, **self._VALID_BODY
        )
        self.mock_create_block.return_value = mock_block_record

        response = await client.post(
            f"/agents/{self.agent_record.id}/memory/blocks",
            json=self._VALID_BODY,
        )

        assert response.status_code == 201
        self.mock_create_block.assert_called_once()
        assert MemoryBlockResponse.model_validate(response.json()) == MemoryBlockResponse.from_record(mock_block_record)

    async def test_returns_404_for_unknown_agent(self, client: AsyncClient):
        """
        Returns 404 before calling create_block when agent does not exist.
        Exception is propagated by the route and caught by app level handler
        """
        self.configure_mock_get_deps_dep(raise_exc=AgentNotFoundError(f"Agent not found"))

        response = await client.post(
            f"/agents/{uuid4()}/memory/blocks",
            json=self._VALID_BODY,
        )

        assert response.status_code == 404
        self.mock_create_block.assert_not_called()

    async def test_returns_400_for_duplicate_block(self, client: AsyncClient):
        """
        Returns 400 with label in detail when block label already exists.
        This one is mapped internally by the route since this is the only place we expect it to occur....
        
        TODO: The above could be wrong, what if the agent tries to make a duplicate block with a tool call (future intended tool)?
        Then send_messages could raise this exception! Consider moving to an app level handler like some of the others
        """
        self.mock_create_block.side_effect = DuplicateBlockError("block with label 'notes' already exists")

        response = await client.post(
            f"/agents/{self.agent_record.id}/memory/blocks",
            json=self._VALID_BODY,
        )

        assert response.status_code == 400
        assert response.json()["detail"] == "Duplicate block: block with label 'notes' already exists"

    async def test_returns_500_for_unexpected_error(self, client: AsyncClient):
        """
        Route propagates unexpected exceptions to the app-level handler, returning 500.
        Caught by an app level exception handler
        """
        self.mock_create_block.side_effect = RuntimeError("DB failure")

        response = await client.post(
            f"/agents/{self.agent_record.id}/memory/blocks",
            json=self._VALID_BODY,
        )

        assert response.status_code == 500
        assert response.json()["detail"] == "RuntimeError: DB failure"
