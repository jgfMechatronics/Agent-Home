"""HTTP route tests for Section 4.1.

Tests the FastAPI routes using httpx AsyncClient. Uses dependency_overrides
to inject mock factories, avoiding real DB lookups in route tests.

Fixtures are defined here (not in conftest) because only this file uses them.

Fixtures from conftest used here:
- session: Test DB session (function-scoped, rolled back after each test)
- agent_record: Pre-created agent for tests that need an existing agent
- agent_with_blocks: Agent with memory blocks attached

TODO: This test file is too big. We should split some of it into different files (like the detailed persistence and cancellation tests)
And possibly consider this as a sign that the routes module has too many responsibilities. One option: move _handle_messages to a seperate file and
test it directly, but then we could lose a lot of the integrationeyness of these tests. We could also split some key routes (like the ones that call _handle_messages)
out into their own file and have the misc smaller routes in a seperate file
"""
# Standard library
import asyncio
import importlib.metadata
import json
from contextlib import asynccontextmanager, contextmanager
from datetime import datetime
from unittest.mock import AsyncMock, Mock, patch
from uuid import uuid4

# Third-party
import pytest
import pytest_asyncio
from fastapi import FastAPI, HTTPException
from httpx import ASGITransport, AsyncClient, Response
from pydantic_ai import Agent, AgentRunResultEvent
from sqlalchemy.ext.asyncio import AsyncSession
from pydantic_ai.messages import (
    FunctionToolCallEvent,
    FunctionToolResultEvent,
    ModelMessage,
    ModelRequest,
    ModelResponse,
    PartDeltaEvent,
    PartStartEvent,
    RetryPromptPart,
    TextPart,
    TextPartDelta,
    ThinkingPart,
    ToolCallPart,
    ToolReturnPart,
    UserPromptPart,
)
from pydantic_ai.models.function import AgentInfo, DeltaThinkingPart, DeltaThinkingCalls, DeltaToolCall, DeltaToolCalls, FunctionModel

# Local
from agent.factory import AgentNotFoundError
from agent.types import AgentAppState
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

    async def _gen():
        for event in (events or []):
            yield event
        if raises_mid_stream is not None:
            raise raises_mid_stream

    @asynccontextmanager
    async def _stream(*args, **kwargs):
        yield _gen()

    agent.run_stream_events = _stream
    return agent


def _make_mock_session() -> Mock:
    """Build a mock AsyncSession with async commit/rollback/refresh."""
    session = Mock()
    session.commit = AsyncMock()
    session.rollback = AsyncMock()
    session.refresh = AsyncMock()
    return session


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


class _BaseRouteTest:
    """Base class for test classes that need the standard route-level patches.

    Patches load_messages, deserialize_messages, is_compaction_needed, and compact,
    exposing them as self.mock_load_messages, self.mock_deserialize_msgs,
    self.mock_needs_compact, and self.mock_compact.
    """

    @pytest.fixture(autouse=True)
    def _base_route_patches(self):
        """
        Patch route-level side effects
        TODO: Consider autospec across the board. Currently only applied to persist_messages
        Came in from refactor where the signature matching for it was explicitly considered.
        """
        with (
            patch("api.routes.load_messages", new_callable=AsyncMock) as mock_load,
            patch("api.routes.deserialize_messages") as mock_deserialize,
            patch("api.routes.is_compaction_needed") as mock_needs_compact,
            patch("api.routes.compact", new_callable=AsyncMock) as mock_compact,
            patch("api.routes.persist_messages", autospec=True) as mock_persist_messages,
        ):
            mock_load.return_value = []
            mock_deserialize.return_value = []
            mock_needs_compact.return_value = False
            self.mock_load_messages = mock_load
            self.mock_deserialize_msgs = mock_deserialize
            self.mock_needs_compact = mock_needs_compact
            self.mock_compact = mock_compact
            self.mock_persist_messages = mock_persist_messages
            yield


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


DEFAULT_USER_MESSAGE = "You're not a real LLM are you?"
async def stream_and_collect(client: AsyncClient, agent_id, message: str = DEFAULT_USER_MESSAGE) -> list[dict]:
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

class TestSendMessage(_BaseRouteTest):
    """
    POST /agents/{agent_id}/messages — main streaming endpoint.
    TODO (Low priority): This test class got a bit confusing, consider simplifying if possible
    TODO: Remove now redundant tests around persistence and commiting now that we have the stronger, more updated persistence/cancellation tests
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
        TODO: This is a cleanup for when we refactor this test suite, its disjointed with how we mock the agent in persistence
        and cancellation tests. Can we just use the FunctionModelTestAgent for everything?
        """
        self.agent_record = agent_record
        self.mock_session = _make_mock_session()
        app.state.agent_app_states[agent_record.id] = AgentAppState()

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
        del app.state.agent_app_states[agent_record.id]

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
# Detailed Persistence test and Cancellation tests
# ---------------------------------------------------------------------------

class FunctionModelTestAgent:
    """Test agent backed by FunctionModel for precise behavioral control.

    Centralises agent construction, message-history capture (self.calls), and
    dependency injection.  Behavior is declared as a sequence of steps consumed
    in order, one per _stream invocation (i.e. one per model call).  Each step
    is either a value to yield or an Exception to raise immediately.

    Class-level constants cover the common scenarios:

        DEFAULT_STEPS  — happy path: tool call → completion text
        CRASH_STEPS    — exception after tool return (no text emitted before crash)

    Usage:
        # Happy path (default)
        agent = FunctionModelTestAgent()

        # Exception after tool return — call before making the request
        agent.set_steps(FunctionModelTestAgent.CRASH_STEPS)

        # Blocking tool (for cancellation/timing tests)
        agent.block_in_tool = True
        # Then in test: await agent.tool_entered.wait(), do assertions,
        # agent.resume_tool_exec.set()

        # Inject into app
        with agent.override_agent_and_deps_factory(app, agent_record) as mock_session:
            ...
    TODO(Low priority): The blocking/resuming exectuion in stream/tool is a little duplicatey and could be
    cleaned up a bit with some common infrastructure and helpers to make it easier on callers
    """

    
    # --- Dummy tool identity ---
    # Couples the FunctionModel stream deltas, expected message parts, and the tool implementation.
    # NOTE: the tool function name below in _build() must match DUMMY_TOOL_NAME.
    DUMMY_TOOL_NAME    = "dummy_tool"
    DUMMY_TOOL_ARGS    = '{"arg": "dummy"}'
    DUMMY_TOOL_CALL_ID = "tc-a1"
    TOOL_RETURN_VALUE  = "dummy_tool_return"

    DUMMY_TOOL_CALL_PART  = ToolCallPart(tool_name=DUMMY_TOOL_NAME, args=DUMMY_TOOL_ARGS, tool_call_id=DUMMY_TOOL_CALL_ID)
    DUMMY_TOOL_RETURN_PART = ToolReturnPart(tool_name=DUMMY_TOOL_NAME, content=TOOL_RETURN_VALUE, tool_call_id=DUMMY_TOOL_CALL_ID)

    # --- Step constants (one value or exception per model invocation) ---
    # https://pydantic.dev/docs/ai/api/models/function/#pydantic_ai.models.function.StreamFunctionDef claims we're being bad here, but can't *really* tell why its an issue
    # Its possible its just the Delta index collision we ran into below. Either way, the illict behavior is useful so we keep it for now
    # The different types (raw text, Deltas, etc) is just an artifact of what pydantic AI expects from FunctionModel
    THINKING_TEXT   = "If I think hard enough I will unravel the mysteries of the universe"
    THINKING_STEP   = DeltaThinkingCalls({0: DeltaThinkingPart(content=THINKING_TEXT)})
    PRE_TOOL_TEXT   = "I'll look that up for you."
    TOOL_CALL       = DeltaToolCalls({1: DeltaToolCall(name=DUMMY_TOOL_NAME, json_args=DUMMY_TOOL_ARGS, tool_call_id=DUMMY_TOOL_CALL_ID)})
    COMPLETION_TEXT = "Turn complete."
    CRASH           = RuntimeError("Simulated crash mid-stream")

    # --- Step sequences and expected model messages ---
    # Each entry in the list results in one AsyncIterator. Lists are yielded during a single iterator, single items will be the only item in the iterator
    DEFAULT_STEPS = [[THINKING_STEP, PRE_TOOL_TEXT, TOOL_CALL], COMPLETION_TEXT]
    # DEFAULT_STEPS is what the *model* outputs. Below is what the *agent* outputs, IE what is returned by run_stream_events or similar
    # Not raw chunks but the complete model messages (so what should be persisted, not necessarily what is streamed)
    DEFAULT_EXPECTED_TOTAL_MODELMSGS: list[ModelMessage] = [
        ModelResponse(parts=[ThinkingPart(content=THINKING_TEXT), TextPart(content=PRE_TOOL_TEXT), DUMMY_TOOL_CALL_PART]),
        ModelRequest(parts=[DUMMY_TOOL_RETURN_PART]),
        ModelResponse(parts=[TextPart(content=COMPLETION_TEXT)]),
    ]

    CRASH_STEPS           = [TOOL_CALL, TOOL_CALL, CRASH, COMPLETION_TEXT]  # Two tool calls before crash; completion text should NOT be reached
    THREE_TOOL_CALL_STEPS = [TOOL_CALL, TOOL_CALL, TOOL_CALL, COMPLETION_TEXT]

    # A single tool call/return pair — shared by CRASH_EXPECTED_PARTIAL_MODELMSGS and THREE_TOOL_CALL_EXPECTED_MSGS
    EXPECTED_TOOL_PAIR: list[ModelMessage] = [
        ModelResponse(parts=[DUMMY_TOOL_CALL_PART]),
        ModelRequest(parts=[DUMMY_TOOL_RETURN_PART]),
    ]
    # What should be persisted for CRASH_STEPS: both tool pairs.
    # FunctionToolResultEvent fires after each tool completes, before the next step starts.
    # Our route persists the tool pair atomically on FunctionToolResultEvent, so both pairs
    # are committed before the crash hits on step 3.
    CRASH_EXPECTED_PARTIAL_MODELMSGS: list[ModelMessage] = EXPECTED_TOOL_PAIR * 2
    # THREE_TOOL_CALL_STEPS: 3 tool call/return pairs followed by a final text response
    THREE_TOOL_CALL_EXPECTED_MSGS: list[ModelMessage] = EXPECTED_TOOL_PAIR * 3 + [
        ModelResponse(parts=[TextPart(content=COMPLETION_TEXT)]),
    ]

    def __init__(self) -> None:
        self.calls: list[list] = []
        self._stream_step_index = 0
        self._stream_steps = self.DEFAULT_STEPS
        # use to pause tool execution to test intermediate states
        self.block_in_tool = False
        # Set when dummy_tool begins executing (tests can await this); cleared after each resume
        self.tool_entered = asyncio.Event()
        # Tests set this to let dummy_tool resume after blocking
        self.resume_tool_exec = asyncio.Event()
        # use to pause stream emission to test mid-stream cancel timing
        self.block_in_stream = False
        # Set after each chunk is yielded; tests await this, then clear before resuming
        self.chunk_emitted = asyncio.Event()
        # Tests set this to let _stream yield the next chunk
        self.resume_stream = asyncio.Event()
        self._agent = self._build()

    def set_steps(self, steps) -> None:
        """Replace the step sequence and reset the index (call before making a request)."""
        self._stream_steps = steps
        self._stream_step_index = 0

    def reset_for_new_run(self) -> None:
        """Reset state for a subsequent run. Call between multi-run tests."""
        self._stream_step_index = 0
        self.tool_entered.clear()
        self.resume_tool_exec.clear()
        self.chunk_emitted.clear()
        self.resume_stream.clear()

    @property
    def agent(self) -> Agent:
        return self._agent

    @contextmanager
    def override_agent_and_deps_factory(self, app: FastAPI, agent_record):
        """
        Install this agent in app's dep overrides; restore previous override on exit.
        Yields the mock_session used to construct the agent and deps.

        Also creates an AgentAppState and injects it into app.state.agent_app_states,
        and acquires the lock for the dep lifetime — mirroring what AgentFactory does.

        If we need to emulate more of the actual thing we're overriding than this,
        we should just make the override use a real factory with a FunctionModelAgent model or something.
        """
        mock_session = _make_mock_session()
        agent_app_state = AgentAppState()
        app.state.agent_app_states[agent_record.id] = agent_app_state

        async def _make_agent_and_deps():
            async with agent_app_state.lock:
                yield self._agent, make_deps(mock_session, agent_record)
                agent_app_state.cancel_requested.clear()

        app.dependency_overrides[get_agent_and_deps] = _make_agent_and_deps
        try:
            yield mock_session
        finally:
            app.dependency_overrides.pop(get_agent_and_deps)
            del app.state.agent_app_states[agent_record.id]

    # ------------------------------------------------------------------
    # Private helpers
    # ------------------------------------------------------------------

    def _build(self) -> Agent:
        agent: Agent = Agent(FunctionModel(stream_function=self._stream))

        async def dummy_tool(arg: str) -> str:
            if self.block_in_tool:
                self.tool_entered.set()
                await self.resume_tool_exec.wait()

                # Reset incase we have subsequent calls
                self.tool_entered.clear()
                self.resume_tool_exec.clear()
            return self.TOOL_RETURN_VALUE

        agent.tool_plain(dummy_tool)
        return agent

    async def _block_after_chunk(self) -> None:
        """
        Signal that a chunk was emitted and pause until the test resumes us.
        clearing of chunk_emitted is going to be redundant under most uses as caller needs to clear it
        before waiting on it again to avoid immediately unblocking from a stale set.
        """
        self.chunk_emitted.set()
        await self.resume_stream.wait()
        self.resume_stream.clear()
        self.chunk_emitted.clear()

    async def _stream(self, messages, info: AgentInfo):
        """
        Allows for multi-step single streams with setting _stream_steps to contain a list.
        Behavior is: Stream events in a list in _stream_step_index as if they were one model invocation, then on next call move on to next item in _stream_step_index
        This allows us to simulate model behavior across multiple invocations
        """
        self.calls.append(list(messages))
        step = self._stream_steps[self._stream_step_index]
        self._stream_step_index += 1
        if isinstance(step, Exception):
            raise step
        if isinstance(step, list):
            for chunk in step:
                yield chunk
                if self.block_in_stream:
                    await self._block_after_chunk()
        else:
            yield step
            if self.block_in_stream:
                await self._block_after_chunk()


class _PersistenceAndCancellationTestBase(_BaseRouteTest):
    """Base for test classes that drive a real pydantic-ai Agent.

    Shared assertion and extraction helpers as static methods.
    """

    @pytest.fixture(autouse=True)
    def real_agent_dep(self, app, agent_record):
        """Construct a FunctionModelTestAgent and install it via get_agent_and_deps override.

        Exposes self.agent_record, self.mock_session, self.function_agent.
        """
        self.agent_record = agent_record
        self.function_agent = FunctionModelTestAgent()
        with self.function_agent.override_agent_and_deps_factory(app, agent_record) as mock_session:
            self.mock_session = mock_session
            yield


    @staticmethod
    def _list_persisted_messages(mock_persist_messages) -> list:
        """Concatenate messages from all persist_messages calls, in call order."""
        messages = []
        for call in mock_persist_messages.call_args_list:
            assert "messages" in call.kwargs, (
                "At time of writing impl only uses kwargs. This will break if impl switches to positional args. "
                "Update if that issue occurs."
            )
            messages.extend(call.kwargs["messages"])
        return messages

    @staticmethod
    def _messages_with_part(persisted_msgs_list: list, msg_type: type, part_type: type) -> list:
        """Messages of msg_type in the persisted list carrying at least one part of part_type."""
        return [
            m for m in persisted_msgs_list
            if isinstance(m, msg_type) and any(isinstance(p, part_type) for p in m.parts)
        ]

    @staticmethod
    def _assert_no_duplicates(persisted_msgs_list: list) -> None:
        """Each message object must appear at most once in the persisted message list."""
        for i in range(len(persisted_msgs_list)):
            for j in range(i + 1, len(persisted_msgs_list)):
                assert persisted_msgs_list[i] != persisted_msgs_list[j], (
                    f"Duplicate message at indices {i} and {j}: {persisted_msgs_list[i]!r}"
                )

    @staticmethod
    def _assert_no_orphans(persisted_msgs_list: list) -> None:
        """Every ToolCallPart must have a matching ToolReturnPart or RetryPromptPart."""
        tool_call_ids: set[str] = set()
        response_ids: set[str] = set()
        for msg in persisted_msgs_list:
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

    @staticmethod
    def _assert_ModelMessage_list_eq(
        actual: list[ModelMessage],
        expected: list[ModelMessage],
    ) -> None:
        """Assert two ModelMessage lists are semantically equal, ignoring runtime fields (timestamps etc.)."""
        assert len(actual) == len(expected), f"Message list length mismatch: {len(actual)} != {len(expected)}"
        for i, (actual_msg, expected_msg) in enumerate(zip(actual, expected)):
            assert type(actual_msg) is type(expected_msg), f"Message {i}: type mismatch {type(actual_msg)} != {type(expected_msg)}"
            assert len(actual_msg.parts) == len(expected_msg.parts), f"Message {i}: part count mismatch"
            for j, (actual_part, expected_part) in enumerate(zip(actual_msg.parts, expected_msg.parts)):
                assert type(actual_part) is type(expected_part), f"Message {i} part {j}: type mismatch"
                match expected_part:
                    case UserPromptPart():
                        assert actual_part.content == expected_part.content, f"Message {i} part {j}: content mismatch"
                    case ToolCallPart():
                        assert actual_part.tool_name == expected_part.tool_name, f"Message {i} part {j}: tool_name mismatch"
                        assert actual_part.args == expected_part.args, f"Message {i} part {j}: args mismatch"
                        assert actual_part.tool_call_id == expected_part.tool_call_id, f"Message {i} part {j}: tool_call_id mismatch"
                    case ToolReturnPart():
                        assert actual_part.tool_name == expected_part.tool_name, f"Message {i} part {j}: tool_name mismatch"
                        assert actual_part.content == expected_part.content, f"Message {i} part {j}: content mismatch"
                        assert actual_part.tool_call_id == expected_part.tool_call_id, f"Message {i} part {j}: tool_call_id mismatch"
                    case TextPart():
                        assert actual_part.content == expected_part.content, f"Message {i} part {j}: content mismatch"
                    case ThinkingPart():
                        assert actual_part.content == expected_part.content, f"Message {i} part {j}: content mismatch"
                    case _:
                        assert actual_part == expected_part, (f"Message {i} part {j}: equality mismatch.\n" 
                                                              "Comparison helper may not be accountinng for this type.")


class TestSendMessagePersistenceBehavior(_PersistenceAndCancellationTestBase):
    """Persistence contract tests using a real pydantic-ai Agent + FunctionModel."""

    async def test_happy_path_persists_full_message_list(self, client: AsyncClient):
        """
        Persisted message list contains all four message types in causal order.

        Uses a real pydantic-ai Agent so new_messages() reflects actual pydantic-ai
        message structure.  Validates the full pipeline against the real library.
        
        A more detailed/stronger version of the basic persistence test in TestSendMessage.
        """
        fake_history = [ModelRequest(parts=[UserPromptPart(content="prior turn")])]
        self.mock_deserialize_msgs.return_value = fake_history
        events = await stream_and_collect(client, self.agent_record.id)
        event_types = [e["event"] for e in events]
        assert "Error" not in event_types, f"Unexpected Error event: {events}"
        # Sanity check: ensure history made it in 
        # history + new user prompt combined into one ModelRequest by pydantic-ai
        self._assert_ModelMessage_list_eq(
            self.function_agent.calls[0],
            [ModelRequest(parts=[UserPromptPart(content="prior turn"), UserPromptPart(content=DEFAULT_USER_MESSAGE)])],
        )

        persisted_msgs_list = self._list_persisted_messages(self.mock_persist_messages)

        # Based on the construction of this list, this assertion checks:
        # - expected content persisted
        # - no dupes
        # - no orphaned tool calls
        # - old history not persisted
        expected_msg_list = [
            ModelRequest(parts=[UserPromptPart(content=DEFAULT_USER_MESSAGE)]),
        ] + FunctionModelTestAgent.DEFAULT_EXPECTED_TOTAL_MODELMSGS
        self._assert_ModelMessage_list_eq(persisted_msgs_list, expected_msg_list)

        # These are now sanity checks due to strength of above hard coded comparison
        self._assert_no_duplicates(persisted_msgs_list)
        self._assert_no_orphans(persisted_msgs_list)

        assert not self.mock_session.rollback.called, "Session must NOT be rolled back on happy path"

    def _get_messages_from_last_persist_call(self) -> list:
        return self.mock_persist_messages.call_args_list[-1].kwargs["messages"]

    async def test_persists_as_complete_msgs_come_in(self, client: AsyncClient):
        """
        Persistence happens incrementally: tool call + return pairs are committed as they
        complete, not only at end-of-run.

        Atomicity requirement: ToolCallPart and ToolReturnPart must always appear in the same
        persist_messages call — the route must never persist a bare ToolCallPart without its
        corresponding return (which would appear as an orphan in TUI history polling).
        
        TODO(low priority): Is there a gap in coverage here where the route might just be persisting the first tool pair over and over and we wouldn't
        know since they're identical? In isolation, yes, but maybe the other tests preclude that possibility by having more unique contents 
        """
        self.function_agent.set_steps(FunctionModelTestAgent.THREE_TOOL_CALL_STEPS)
        self.function_agent.block_in_tool = True

        stream_task = asyncio.create_task(stream_and_collect(client, self.agent_record.id))

        # --- Tool 1: model emits ToolCallPart, tool is blocking ---
        await asyncio.wait_for(self.function_agent.tool_entered.wait(), timeout=5.0)
        # NOTE: user message may end up persisted together with first tool pair — adjust counts
        # below if implementation batches them rather than persisting user message upfront.
        assert self.mock_persist_messages.call_count == 1, (
            "User message should be persisted before (or as) the first tool call completes"
        )
        assert self.mock_session.commit.call_count == 1, "Route must commit after persisting user message"
        self._assert_ModelMessage_list_eq(
            self._get_messages_from_last_persist_call(),
            [ModelRequest(parts=[UserPromptPart(content=DEFAULT_USER_MESSAGE)])],
        )
        self.function_agent.tool_entered.clear()  # consume signal before resuming to avoid stale wait
        self.function_agent.resume_tool_exec.set()

        # --- Check persistence of Tool pairs 1 & 2:---
        # Every time we hit a tool_entered Event, the tool call which triggered that event is *not* available to persist yet
        # So the number of expected persisted tool call/returns lags the number of times we've hit tool_entered by 1
        for i in range(2, 4):
            await asyncio.wait_for(self.function_agent.tool_entered.wait(), timeout=5.0)
            assert self.mock_persist_messages.call_count == i, (
                f"Tool call/return pair {i} should be persisted as soon as the return is available"
            )
            assert self.mock_session.commit.call_count == i, "Route must commit after each persist"
            self._assert_ModelMessage_list_eq(self._get_messages_from_last_persist_call(), FunctionModelTestAgent.EXPECTED_TOOL_PAIR)
            self.function_agent.tool_entered.clear()  # consume signal before resuming to avoid stale wait
            self.function_agent.resume_tool_exec.set()

        # The end of the final loop iter freed up the last tool call/return pair
        self._assert_ModelMessage_list_eq(self._get_messages_from_last_persist_call(), FunctionModelTestAgent.EXPECTED_TOOL_PAIR)

        # --- Run complete ---
        events = await asyncio.wait_for(stream_task, timeout=5.0)
        assert "Error" not in [e["event"] for e in events], f"Unexpected Error event: {events}"

        assert self.mock_persist_messages.call_count == 5, (
            "Final model response must be persisted"
        )
        assert self.mock_session.commit.call_count == 5, "Each persist must have been committed"
        self._assert_ModelMessage_list_eq(self._get_messages_from_last_persist_call(), [FunctionModelTestAgent.THREE_TOOL_CALL_EXPECTED_MSGS[-1]])

        # Aggregate: full flattened message list must be complete and well-formed (sanity check)
        persisted_msgs_list = self._list_persisted_messages(self.mock_persist_messages)
        expected_msg_list = [
            ModelRequest(parts=[UserPromptPart(content=DEFAULT_USER_MESSAGE)]),
        ] + FunctionModelTestAgent.THREE_TOOL_CALL_EXPECTED_MSGS
        self._assert_ModelMessage_list_eq(persisted_msgs_list, expected_msg_list)
        self._assert_no_orphans(persisted_msgs_list)
        assert not self.mock_session.rollback.called, "Session must NOT be rolled back on happy path"

    async def test_persist_survives_mid_run_exception(self, client: AsyncClient):
        """
        Completed work (tool call/return pair) is persisted and committed before the crash.
        The route surfaces an Error SSE and calls rollback to clear any uncommitted state —
        committed work is unaffected by the rollback.
        """
        self.function_agent.set_steps(FunctionModelTestAgent.CRASH_STEPS)

        events = await stream_and_collect(client, self.agent_record.id)
        event_types = [e["event"] for e in events]
        assert "Error" in event_types, "Expected Error SSE after mid-run exception"

        assert self.mock_persist_messages.call_count >= 1, (
            "persist_messages must be called even when the run crashes mid-stream"
        )

        persisted_msgs_list = self._list_persisted_messages(self.mock_persist_messages)

        expected_msg_list = [
            ModelRequest(parts=[UserPromptPart(content=DEFAULT_USER_MESSAGE)]),
        ] + FunctionModelTestAgent.CRASH_EXPECTED_PARTIAL_MODELMSGS
        self._assert_ModelMessage_list_eq(persisted_msgs_list, expected_msg_list)

        self._assert_no_orphans(persisted_msgs_list)  # sanity check

        # Completed work is committed before the crash, confirmed by "persist as you go test"; rollback clears any uncommitted state
        assert self.mock_session.rollback.called, (
            "Route must call rollback on exception to clear any uncommitted state"
        )

    async def test_rollback_and_error_sse_on_persist_failure(self, client: AsyncClient):
        """persist_messages failure triggers rollback and yields Error SSE. Session is not committed.

        Uses a real pydantic-ai Agent (default steps) so capture_run_messages populates
        and the incremental persist path is exercised. The first persist call raises,
        which should propagate out of the stream loop, trigger rollback, and emit Error SSE.
        """
        self.mock_persist_messages.side_effect = RuntimeError("DB write failed")

        sse_events = await stream_and_collect(client, self.agent_record.id)

        self.mock_session.rollback.assert_called_once()
        self.mock_session.commit.assert_not_called()
        assert sse_events[-1]["event"] == "Error"


# ---------------------------------------------------------------------------
# TestCancellation
# ---------------------------------------------------------------------------

class TestCancellation(_PersistenceAndCancellationTestBase):
    """Cancellation contract tests.

    Uses a blocking tool for deterministic timing: the tool signals 'tool_entered'
    when it starts and waits on 'release' before returning, letting the test act
    precisely while the tool is in flight.

    test_graceful_cancel: xfail pending cancel route + _cancel_signals implementation.

    Corner case — ToolCallPart buffered but not yet consumed as FunctionToolCallEvent:
    pydantic-ai appends ModelResponse([ToolCallPart]) to its capture_run_messages buffer
    BEFORE emitting FunctionToolCallEvent to the consumer. If cancel is serviced in this
    narrow window, the captured list may contain a bare ToolCallPart with no return.
    This is a non-issue: the route passes the captured list as-is to persist_messages,
    which already has orphan sanitization logic and will drop the unpaired ToolCallPart.
    No dedicated test needed — either the ToolCallPart ends up in the buffer and
    persist_messages eats it, or it never made it to the buffer at all.
    """

    CANCEL_NOTICE = ModelRequest(parts=[UserPromptPart(content="<system_message>Turn cancelled by user.</system_message>")])

    async def test_cancel_no_active_run_returns_409(self, client: AsyncClient):
        """Cancel route returns 409 when no run is active for the given agent_id."""
        response = await client.post(f"/agents/{self.agent_record.id}/cancel")
        assert response.status_code == 409

    async def _test_cancel_during_tool_exec(self, client: AsyncClient) -> None:
        """Execute cancel-during-tool scenario with full assertions.
        
        Reusable helper for tests that need this scenario as a building block.
        """
        self.function_agent.block_in_tool = True

        # Start the SSE stream as a background task so we can interleave cancel.
        stream_task = asyncio.create_task(
            stream_and_collect(client, self.agent_record.id)
        )

        await asyncio.wait_for(self.function_agent.tool_entered.wait(), timeout=5.0)

        cancel_response = await client.post(f"/agents/{self.agent_record.id}/cancel")

        assert cancel_response.status_code == 202, (
            f"Cancel route must return 202; got {cancel_response.status_code}"
        )

        # Resume to allow cancellation to be serviced
        self.function_agent.resume_tool_exec.set()
        await asyncio.wait_for(stream_task, timeout=5.0)

        persisted_msgs_list = self._list_persisted_messages(self.mock_persist_messages)
        
        # We expect the default sequence except the cancel prevents us from reaching the COMPLETED chunk,
        # and instead we get the cancel notice
        expected_msg_list = (
            [ModelRequest(parts=[UserPromptPart(content=DEFAULT_USER_MESSAGE)])]
            + FunctionModelTestAgent.DEFAULT_EXPECTED_TOTAL_MODELMSGS[:-1]
            + [self.CANCEL_NOTICE]
        )
        self._assert_ModelMessage_list_eq(persisted_msgs_list, expected_msg_list)
        self._assert_no_orphans(persisted_msgs_list)

        assert self.mock_session.commit.call_count == self.mock_persist_messages.call_count, (
            "Every persist call must be followed by a commit — including cancel code path"
        )
        assert not self.mock_session.rollback.called, (
            "Must not rollback completed work on graceful cancel"
        )

    async def test_cancel_during_tool_exec(self, client: AsyncClient):
        """
        CONTRACT: graceful cancel lets the in-flight tool complete, persists the
        tool call+return pair and a cancellation notice, then commits — without rolling back.

        Covers requirements:
            — active tool is allowed to complete before cancel takes effect
            — cancellation notice wrapped in <system_message> tags is persisted
            — cancel is delivered via POST /agents/{id}/cancel
        """
        await self._test_cancel_during_tool_exec(client)

    async def test_subsequent_run_after_cancel(self, client: AsyncClient):
        """
        After a cancel, subsequent runs must work correctly.

        Covers:
            — cancel_requested state is cleared after run completes
            — persistence cursor adjustment accounts for pydantic-ai merging the
              trailing ModelRequest (cancel notice) with the next user prompt
        """
        # Phase 1: Cancel run
        await self._test_cancel_during_tool_exec(client)
        post_cancel_history = self._list_persisted_messages(self.mock_persist_messages)

        # Reset state for phase 2, set deserialize to return history from previous run
        self.mock_persist_messages.reset_mock()
        self.mock_session.reset_mock()
        self.function_agent.reset_for_new_run()
        self.mock_deserialize_msgs.return_value = post_cancel_history

        # Phase 2: Another run with post-cancel history — should pass same assertions
        await self._test_cancel_during_tool_exec(client)


    async def test_cancel_during_text_streaming(self, client: AsyncClient):
        """
        Cancel received while the model is mid-text-streaming (after a completed thinking step,
        before the tool call chunk arrives).

        pydantic-ai assembles all parts of a model step into a single ModelResponse only at
        step-end (i.e. when all chunks for that step have been yielded). Mid-step cancel means
        nothing from that in-progress step is in captured messages yet — only the user request
        and the cancel notice are persisted.
        """
        self.function_agent.block_in_stream = True
        stream_task = asyncio.create_task(
            stream_and_collect(client, self.agent_record.id)
        )

        # --- Let thinking chunk (chunk 1 of DEFAULT_STEPS step 1) through ---
        await asyncio.wait_for(self.function_agent.chunk_emitted.wait(), timeout=5.0)
        self.function_agent.chunk_emitted.clear()
        self.function_agent.resume_stream.set()

        # --- Text chunk (chunk 2) has been yielded — fire cancel before tool call (chunk 3) ---
        await asyncio.wait_for(self.function_agent.chunk_emitted.wait(), timeout=5.0)
        cancel_response = await client.post(f"/agents/{self.agent_record.id}/cancel")
        assert cancel_response.status_code == 202
        self.function_agent.chunk_emitted.clear()
        self.function_agent.resume_stream.set()  # unblock so route can service the cancel

        events = await asyncio.wait_for(stream_task, timeout=5.0)

        persisted_msgs_list = self._list_persisted_messages(self.mock_persist_messages)

        # pydantic-ai assembles all parts of a step into one ModelResponse only at step-end.
        # Cancel fired mid-step means nothing from that in-progress step is in captured messages.
        expected_msg_list = [
            ModelRequest(parts=[UserPromptPart(content=DEFAULT_USER_MESSAGE)]),
            self.CANCEL_NOTICE,
        ]
        self._assert_ModelMessage_list_eq(persisted_msgs_list, expected_msg_list)
        assert self.mock_session.commit.call_count == self.mock_persist_messages.call_count, (
            "Every persist call must be followed by a commit — including cancel code path"
        )
        assert not self.mock_session.rollback.called, (
            "Must not rollback completed work on graceful cancel"
        )


# ---------------------------------------------------------------------------
# Rendezvous regression guard (standalone — no class fixtures needed)
# ---------------------------------------------------------------------------

async def test_rendezvous_tool_does_not_start_before_event_consumed():
    """Regression guard: pydantic-ai rendezvous semantics.

    Property: a tool does NOT begin executing until its FunctionToolCallEvent has
    been consumed by the caller.  Our cancel strategy depends entirely on this —
    we can break out of the event stream BEFORE consuming FunctionToolCallEvent
    and guarantee the tool has not started (and therefore cancel without orphaning it).

    Drives a real Agent directly (no HTTP route) to isolate the pydantic-ai behaviour.
    """
    _PYDANTIC_AI_VERSION = importlib.metadata.version("pydantic-ai")
    # Rendezvous semantics previously confirmed on pydantic-ai 1.104.0
    # If this test starts failing after a pydantic-ai upgrade, the cancel strategy must
    # be re-evaluated before shipping.

    test_agent = FunctionModelTestAgent()
    test_agent.block_in_tool = True
    agent = test_agent.agent

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
                assert not test_agent.tool_entered.is_set(), (
                    f"Rendezvous violated: blocking_tool started before "
                    f"FunctionToolCallEvent was consumed "
                    f"(pydantic-ai {_PYDANTIC_AI_VERSION}). "
                    f"Cancel strategy must be re-evaluated."
                )
                event = await events_iter.__anext__()
            
            await asyncio.wait_for(test_agent.tool_entered.wait(), timeout=2) # Tool should execute now that we popped the FunctionToolCallEvent off
            test_agent.resume_tool_exec.set()

            assert pre_tool_event_seen, (
                "No events appeared before FunctionToolCallEvent — "
                "the rendezvous invariant was never exercised. "
                "Check whether pydantic-ai changed its event ordering."
            )

            # FunctionToolCallEvent consumed: tool is now allowed to start.
            # Drain remaining events to allow clean context-manager teardown.
            async for _ in stream:
                pass


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
