"""
HTTP route tests

Tests the FastAPI routes using httpx AsyncClient. Uses dependency_overrides
to inject mock factories, avoiding real DB lookups in route tests.

Fixtures are defined here (not in conftest) because only this file uses them.

Fixtures from conftest used here:
- session: Test DB session (function-scoped, rolled back after each test)
- agent_record: Pre-created agent for tests that need an existing agent
- agent_with_blocks: Agent with memory blocks attached

handle_message test is currently in agent.test_runner.py as those tests are currently entangled with the runner
"""
# Standard library
import json
from datetime import datetime
from unittest.mock import AsyncMock, Mock, patch
from uuid import uuid4

# Third-party
import pytest
import pytest_asyncio
from fastapi import FastAPI, HTTPException
from httpx import ASGITransport, AsyncClient, Response
from sqlalchemy.ext.asyncio import AsyncSession

# Local
from agent.factory import AgentNotFoundError, LOCK_TIMEOUT_FAST
from agent.types import AgentAppState, AgentConfig, AgentDeps
from api.fastapi_deps import get_agent_deps, get_agent_and_deps, get_session_dep
from agent.crud import create_agent_record
from conftest import make_deps, SAMPLE_AGENT_CONFIG
from db.models import AgentRecord, MemoryBlockRecord, utcnow
from api.schemas import AgentMetadataResponse, CoreMemoryResponse, MemoryBlockResponse
from memory.block_crud import DuplicateBlockError
from api.routes import _parse_slash_cmd, _is_slash_cmd, _handle_slash_cmd, _handle_recompile, SlashCommandDef
from fastapi.sse import ServerSentEvent
from pydantic_ai import AgentRunResultEvent
from pydantic_ai.messages import (
    FunctionToolCallEvent, FunctionToolResultEvent,
    PartDeltaEvent, PartStartEvent, TextPart, TextPartDelta, ToolCallPart, ToolReturnPart,
)

# --- Module-level test data ---

TOOL_CALL_PART = ToolCallPart(tool_name="memory_replace", args={"label": "notes"}, tool_call_id="call-1")
TOOL_RETURN_PART = ToolReturnPart(tool_name="memory_replace", content="Updated.", tool_call_id="call-1")

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


# --- Fixtures ---

@pytest.fixture
def app() -> FastAPI:
    """Fresh app instance per test — avoids state contamination."""
    from api.app import _create_app
    return _create_app()


@pytest_asyncio.fixture
async def client(app: FastAPI) -> AsyncClient:
    async with AsyncClient(
        transport=ASGITransport(app=app, raise_app_exceptions=False),
        base_url="http://test"
    ) as c:
        yield c


def make_mock_agent(events: list | None = None, raises_mid_stream: Exception | None = None) -> Mock:
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
    async def _get_test_session():
        yield session
    app.dependency_overrides[get_session_dep] = _get_test_session
    yield
    app.dependency_overrides.pop(get_session_dep)


VALID_SSE_PREFIXES = ("data: ", "event: ", "id: ", "retry: ", ":")


async def collect_sse_events(response: Response) -> list[dict]:
    events = []
    current_event_type: str | None = None
    async for line in response.aiter_lines():
        if not line:
            current_event_type = None
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
    def mock_route_side_effects(self):
        """Patch route-level side effects for all TestSendMessage tests.

        Provides self.mock_persist, self.mock_needs_compact, self.mock_compact
        for tests that assert on persistence/compaction behavior.
        """
        with (
            patch("api.routes.load_messages", new_callable=AsyncMock) as mock_load,
            patch("api.routes.deserialize_messages") as mock_deserialize,
            patch("api.routes.persist_messages", new_callable=AsyncMock) as mock_persist,
            patch("api.routes.is_compaction_needed") as mock_needs_compact,
            patch("api.routes.compact", new_callable=AsyncMock) as mock_compact,
        ):
            mock_load.return_value = []  # safe default — no prior history
            mock_deserialize.return_value = []  # safe default — empty deserialized history
            mock_needs_compact.return_value = False  # safe default — no compaction unless overridden
            self.mock_load_messages = mock_load
            self.mock_deserialize = mock_deserialize
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

    async def test_slash_command_recompile_returns_result_sse_and_skips_agent(self, client: AsyncClient):
        """/recompile bypasses the agent run entirely and returns a single SlashCommandResult SSE."""
        with patch("api.routes.compile_system_prompt", new_callable=AsyncMock):
            events = await stream_and_collect(client, self.agent_record.id, message="/recompile")

        assert len(events) == 1
        assert events[0]["event"] == "SlashCommandResult"
        assert events[0]["data"]["name"] == "user_recompile"
        assert events[0]["data"]["status"] == "success"



class TestCreateAgent:
    """POST /agents — create a new agent."""

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

        response = await client.post("/agents", json=self._VALID_BODY)

        assert response.status_code == 201
        self.mock_create_agent_record.assert_called_once()
        assert AgentMetadataResponse.model_validate(response.json()) == expected_metadata

    async def test_returns_500_when_create_agent_fails(self, client: AsyncClient):
        """Route propagates unexpected exceptions to the app-level handler, returning 500."""
        self.mock_create_agent_record.side_effect = RuntimeError("DB failure")
        response = await client.post("/agents", json=self._VALID_BODY)
        assert response.status_code == 500
        assert response.json()["detail"] == "RuntimeError: DB failure"


# =============================================================================
# Slash Command Unit Tests
# =============================================================================


class TestParseSlashCmd:
    """_parse_slash_cmd parsing — pure function, no I/O."""

    @pytest.mark.parametrize("msg,expected", [
        ("/recompile", ("recompile", "")),
        ("/recompile some args", ("recompile", "some args")),
        ("/RECOMPILE", ("recompile", "")),
        ("recompile", None),       # no leading slash
        ("/unknown_cmd", None),    # unrecognized command passes through to model
        ("/", None),               # slash with no command
        ("", None),                # empty string
    ])
    def test_parse_slash_cmd(self, msg, expected):
        assert _parse_slash_cmd(msg) == expected


class TestIsSlashCmd:
    """_is_slash_cmd — boolean wrapper around _parse_slash_cmd."""

    @pytest.mark.parametrize("msg,expected", [
        ("/recompile", True),
        ("/unknown_cmd", False),   # unrecognized → passes to model, not a slash cmd
        ("plain text", False),
    ])
    def test_is_slash_cmd(self, msg, expected):
        assert _is_slash_cmd(msg) == expected


class TestHandleSlashCmd:
    """_handle_slash_cmd dispatch logic."""

    @pytest.mark.parametrize("msg,expected_args", [
        ("/recompile", ""),
        ("/recompile some args", "some args"),
    ])
    async def test_dispatches_to_handler_with_correct_args(self, msg, expected_args):
        """_handle_slash_cmd parses the message and calls the registered handler with the right args."""
        deps = Mock()
        expected_sse = ServerSentEvent(
            data={"name": "user_recompile", "args": expected_args, "result": "ok", "status": "success"},
            event="SlashCommandResult",
        )
        mock_handler = AsyncMock(return_value=expected_sse)

        mock_def = SlashCommandDef(handler=mock_handler, description="test")
        with patch.dict("api.routes.SLASH_COMMANDS", {"recompile": mock_def}):
            result = await _handle_slash_cmd(deps, msg)

        mock_handler.assert_awaited_once_with(deps, expected_args)
        assert result is expected_sse

    async def test_returns_error_sse_on_handler_exception(self):
        """Handler exceptions are caught and returned as a SlashCommandResult with status=error."""
        deps = Mock()
        mock_handler = AsyncMock(side_effect=RuntimeError("boom"))

        mock_def = SlashCommandDef(handler=mock_handler, description="test")
        with patch.dict("api.routes.SLASH_COMMANDS", {"recompile": mock_def}):
            result = await _handle_slash_cmd(deps, "/recompile")

        assert result.event == "SlashCommandResult"
        assert result.data["status"] == "error"
        assert "boom" in result.data["result"]

    async def test_returns_error_sse_when_precondition_violated(self):
        """Gracefully handles being called with a non-slash-command (precondition violated)."""
        deps = Mock()
        result = await _handle_slash_cmd(deps, "not_a_slash_cmd")
        assert result.event == "SlashCommandResult"
        assert result.data["status"] == "error"


class TestHandleRecompile:
    """_handle_recompile handler — calls compile + commit, returns success SSE."""

    async def test_calls_compile_and_commit_then_returns_success_sse(self):
        deps = Mock()
        deps.commit_changes_refresh_agent_record = AsyncMock()

        with patch("api.routes.compile_system_prompt", new_callable=AsyncMock) as mock_compile:
            result = await _handle_recompile(deps, "")

        mock_compile.assert_awaited_once_with(deps)
        deps.commit_changes_refresh_agent_record.assert_awaited_once()
        assert result.event == "SlashCommandResult"
        assert result.data["name"] == "user_recompile"
        assert result.data["status"] == "success"
        assert result.data["result"]  # non-empty result message

    async def test_returns_400_for_invalid_config(self, client: AsyncClient):
        """Missing required fields result in 400 before route logic is reached."""
        response = await client.post(
            "/agents",
            json={"name": "incomplete"},  # missing system_instructions and config
        )
        assert response.status_code in (400, 422)  # FastAPI validation error


class TestCreateAgentNameUniqueness:
    """POST /agents — duplicate name handling."""

    async def test_returns_409_for_duplicate_name(self, client: AsyncClient):
        """Creating a second agent with an already-used name returns 409."""
        first = await client.post("/agents", json=TestCreateAgent._VALID_BODY)
        assert first.status_code == 201

        second = await client.post("/agents", json=TestCreateAgent._VALID_BODY)
        assert second.status_code == 409
        assert second.json()["detail"] == f"Agent name already in use: {TestCreateAgent._NAME!r}"


class TestGetConfig:
    """GET /agents/{agent_id}/config — agent config."""

    async def test_returns_config(self, client: AsyncClient, agent_record: AgentRecord):
        """Returns the agent's AgentConfig as JSON."""
        response = await client.get(f"/agents/{agent_record.id}/config")

        assert response.status_code == 200
        assert AgentConfig.model_validate(response.json()) == agent_record.agent_config

    # 404 tested via parametrized TestNotFound


class TestGetSystemInstructions:
    """GET /agents/{agent_id}/system-instructions — agent system instructions."""

    async def test_returns_system_instructions(self, client: AsyncClient, agent_record: AgentRecord):
        """Returns the agent's system instructions wrapped in a response object."""
        response = await client.get(f"/agents/{agent_record.id}/system-instructions")

        assert response.status_code == 200
        assert response.json() == {"system_instructions": agent_record.system_instructions}

    # 404 tested via parametrized TestNotFound


class _PutEndpointBase:
    """Base for PUT endpoint tests that patch a crud function and override get_agent_deps."""
    crud_patch_target: str  # subclasses define

    @staticmethod
    def _make_deps_override(agent_deps: AgentDeps):
        """Returns an async dep generator that yields agent_deps, for overriding get_agent_deps in tests."""
        async def _dep():
            yield agent_deps
        return _dep

    @pytest_asyncio.fixture(autouse=True)
    async def _setup(self, app: FastAPI, agent_deps: AgentDeps):
        app.dependency_overrides[get_agent_deps] = _PutEndpointBase._make_deps_override(agent_deps)
        self.agent_deps = agent_deps
        with patch(self.crud_patch_target, new_callable=AsyncMock) as mock:
            self.mock_crud_fcn = mock
            yield
        app.dependency_overrides.pop(get_agent_deps)


class TestPutConfig(_PutEndpointBase):
    """PUT /agents/{agent_id}/config — replace agent config."""
    crud_patch_target = "api.routes.replace_agent_config"

    @pytest.fixture()
    def mutated_config_copy(self, agent_record: AgentRecord):
        original_config = agent_record.agent_config
        return original_config.model_copy(update={"retries": original_config.retries + 1})



    async def test_calls_replace_agent_config_with_correct_args(
        self, client: AsyncClient, agent_record: AgentRecord, mutated_config_copy: AgentConfig
    ):
        """Calls replace_agent_config with the request body config, not the one already on agent_deps."""
        self.mock_crud_fcn.return_value = mutated_config_copy

        await client.put(f"/agents/{agent_record.id}/config", json=mutated_config_copy.model_dump())

        self.mock_crud_fcn.assert_called_once_with(self.agent_deps, mutated_config_copy)

    async def test_returns_200_with_echoed_config(
        self, client: AsyncClient, agent_record: AgentRecord, mutated_config_copy: AgentConfig
    ):
        """Echoes the value returned by replace_agent_config, not just the input."""
        sent_config = agent_record.agent_config
        # Return a different config to confirm we echo the crud result, not the raw input
        self.mock_crud_fcn.return_value = mutated_config_copy

        response = await client.put(
            f"/agents/{agent_record.id}/config",
            json=sent_config.model_dump(),
        )

        assert response.status_code == 200
        assert AgentConfig.model_validate(response.json()) == mutated_config_copy

    async def test_returns_422_for_invalid_config(
        self, client: AsyncClient, agent_record: AgentRecord
    ):
        """Returns 422 when config fails AgentConfig validation."""
        invalid_config = {"model_name": 12345}  # model_name should be string

        response = await client.put(
            f"/agents/{agent_record.id}/config",
            json=invalid_config,
        )

        assert response.status_code == 422
        # Crud function should not be called when validation fails
        self.mock_crud_fcn.assert_not_called()

    # 404/423 checked in common tests (TestNotFound, TestAgentLocked)


class TestPutSystemInstructions(_PutEndpointBase):
    """PUT /agents/{agent_id}/system-instructions — replace system instructions."""
    crud_patch_target = "api.routes.replace_system_instructions"

    async def test_calls_replace_system_instructions_with_correct_args(
        self, client: AsyncClient, agent_record: AgentRecord
    ):
        """Calls replace_system_instructions with the agent's deps and the instructions string."""
        instructions = "instructions updated by route"
        self.mock_crud_fcn.return_value = instructions

        await client.put(
            f"/agents/{agent_record.id}/system-instructions",
            json={"system_instructions": instructions},
        )

        self.mock_crud_fcn.assert_called_once_with(self.agent_deps, instructions)

    async def test_returns_200_with_echoed_instructions(
        self, client: AsyncClient, agent_record: AgentRecord
    ):
        """Echoes the value returned by replace_system_instructions, not just the input."""
        original = agent_record.system_instructions
        mutated = "mutated instructions"
        self.mock_crud_fcn.return_value = mutated

        response = await client.put(
            f"/agents/{agent_record.id}/system-instructions",
            json={"system_instructions": original},
        )

        assert response.status_code == 200
        assert response.json() == {"system_instructions": mutated}

    # 404/423 checked in common tests (TestNotFound, TestAgentLocked)


class TestListAgents:
    """GET /agents — list all agents on the server."""

    @pytest.mark.parametrize("n_agents", list(range(4)))
    async def test_returns_all_agents(
        self, client: AsyncClient, session: AsyncSession, n_agents: int
    ):
        """Returns all agents as AgentMetadataResponse objects; empty list when none exist."""
        expected = []
        for i in range(n_agents):
            record = await create_agent_record(
                session, name=f"agent-{i}", system_instructions="", config=SAMPLE_AGENT_CONFIG
            )
            expected.append(AgentMetadataResponse.from_record(record))

        response = await client.get("/agents")

        assert response.status_code == 200
        result = sorted(
            [AgentMetadataResponse(**item) for item in response.json()],
            key=lambda r: str(r.id),
        )
        assert result == sorted(expected, key=lambda r: str(r.id))


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

@pytest.mark.note("These tests are REFERENCE ONLY For the TUI PROTOTYPE BRANCH. They are now a mix of official reviewed tests (on main) and tests added for the prototype")
class TestGetMessages:
    """
    GET /agents/{agent_id}/messages — conversation history.
    TODO: This is OK for now but we will likely rework the endpoint after defining what is most useful for the frontend in terms of message format
    """

    @staticmethod
    def _make_message_record(
        id: str = "msg-1",
        type: str = "ModelResponse",
        content: str = '{"kind": "response", "parts": []}',
        timestamp: datetime | None = None,
    ) -> Mock:
        """Build a mock MessageRecord with the attributes the route accesses."""
        m = Mock()
        m.id = id
        m.type = type
        m.content = content
        m.timestamp = timestamp or datetime(2026, 6, 9, 12, 0, 0)
        return m

    @pytest.fixture(autouse=True)
    def mock_message_loaders(self):
        """Patch load_messages for all TestGetMessages tests.

        Provides self.mock_load_messages for loader-routing assertions.
        """
        with patch("api.routes.load_messages", new_callable=AsyncMock) as mock_load:
            mock_load.return_value = []
            self.mock_load_messages = mock_load
            yield

    async def test_default_loads_context_window(self, client: AsyncClient, agent_record: AgentRecord, session: AsyncSession):
        """Without params: calls load_messages with context_window_start as start_timestamp."""
        self.mock_load_messages.return_value = [self._make_message_record()]

        response = await client.get(f"/agents/{agent_record.id}/messages")

        assert response.status_code == 200
        self.mock_load_messages.assert_called_once_with(
            session, agent_record.id, start_timestamp=agent_record.context_window_start, start_exclusive=False
        )

    async def test_full_true_loads_complete_history(self, client: AsyncClient, agent_record: AgentRecord, session: AsyncSession):
        """With ?full=true: calls load_messages with start_timestamp=None for full history."""
        records = [self._make_message_record(id="msg-1"), self._make_message_record(id="msg-2", type="ModelRequest")]
        self.mock_load_messages.return_value = records

        response = await client.get(f"/agents/{agent_record.id}/messages?full=true")

        assert response.status_code == 200
        assert len(response.json()["messages"]) == 2
        self.mock_load_messages.assert_called_once_with(
            session, agent_record.id, start_timestamp=None, start_exclusive=False
        )

    async def test_response_uses_message_item_format(self, client: AsyncClient, agent_record: AgentRecord, session: AsyncSession):
        """Response items use MessageItem format: id, type, content (raw JSON string), timestamp (ISO string)."""
        ts = datetime(2026, 6, 9, 12, 0, 0)
        raw_content = '{"kind": "response", "parts": [{"part_kind": "text", "content": "hello"}]}'
        self.mock_load_messages.return_value = [
            self._make_message_record(id="msg-42", type="ModelResponse", content=raw_content, timestamp=ts)
        ]

        response = await client.get(f"/agents/{agent_record.id}/messages")

        assert response.status_code == 200
        messages = response.json()["messages"]
        assert len(messages) == 1
        item = messages[0]
        assert item["id"] == "msg-42"
        assert item["type"] == "ModelResponse"
        assert item["content"] == raw_content  # raw JSON string — NOT parsed by the route
        assert item["timestamp"] == ts.isoformat()

    async def test_after_param_filters_exclusively(self, client: AsyncClient, agent_record: AgentRecord, session: AsyncSession):
        """?after=<timestamp>: calls load_messages with that timestamp and start_exclusive=True."""
        cutoff = datetime(2026, 6, 9, 12, 0, 0)

        response = await client.get(f"/agents/{agent_record.id}/messages?after={cutoff.isoformat()}")

        assert response.status_code == 200
        self.mock_load_messages.assert_called_once_with(
            session, agent_record.id, start_timestamp=cutoff, start_exclusive=True
        )

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


# --- Shared test data for parametrized PUT endpoint tests ---
_VALID_CONFIG_BODY = {
    "model_name": "claude-sonnet-4-20250514",
    "tool_names": [],
    "soft_compaction_limit": 1000,
}
_PUT_ENDPOINT_PARAMS = [
    ("/agents/{agent_id}/config", _VALID_CONFIG_BODY),
    ("/agents/{agent_id}/system-instructions", "some instructions"),
]


class TestNotFound:
    """404 behavior for unknown agent_id across all endpoints."""

    @pytest.mark.parametrize("path", [
        "/agents/{agent_id}",
        "/agents/{agent_id}/memory/blocks",
        "/agents/{agent_id}/messages",
        "/agents/{agent_id}/config",
        "/agents/{agent_id}/system-instructions",
    ])
    async def test_get_endpoints_return_404_for_unknown_agent(self, client: AsyncClient, path: str):
        """All GET endpoints with agent_id return 404 for unknown agents."""
        url = path.format(agent_id=uuid4())
        response = await client.get(url)
        assert response.status_code == 404

    @pytest.mark.parametrize("path,body", _PUT_ENDPOINT_PARAMS)
    async def test_put_endpoints_return_404_for_unknown_agent(
        self, client: AsyncClient, path: str, body
    ):
        """All PUT endpoints with agent_id return 404 for unknown agents."""
        url = path.format(agent_id=uuid4())
        response = await client.put(url, json=body)
        assert response.status_code == 404


class TestAgentLocked:
    """423 behavior when agent has an active run in progress."""

    @pytest.mark.parametrize("path,body", _PUT_ENDPOINT_PARAMS)
    async def test_put_endpoints_return_423_when_agent_locked(
        self, app: FastAPI, client: AsyncClient, agent_record: AgentRecord, path: str, body
    ):
        """All write endpoints that modify agent state return 423 when agent is locked."""
        # Set up locked state for this agent
        agent_state = AgentAppState()
        await agent_state.lock.acquire()
        app.state.agent_app_state_reg[agent_record.id] = agent_state

        url = path.format(agent_id=agent_record.id)
        response = await client.put(url, json=body)

        assert response.status_code == 423
        assert response.json()["detail"] == f"AgentLockedError: Agent {agent_record.id!r} did not become available within {LOCK_TIMEOUT_FAST}s"


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
        """Overrides get_agent_deps and patches create_block for all tests.

        Provides self.configure_mock_get_agent_deps() to change dep behavior (e.g. raise
        AgentNotFoundError for 404 tests). Default: yields a valid AgentDeps.
        """
        self.agent_record = agent_record
        self.mock_session = Mock()

        def _configure(raise_exc=None):
            async def _mock_dep():
                if raise_exc is not None:
                    raise raise_exc
                yield make_deps(self.mock_session, agent_record)
                
            app.dependency_overrides[get_agent_deps] = _mock_dep

        self.configure_mock_get_agent_deps = _configure
        _configure()  # default: happy path

        with patch("api.routes.create_block", new_callable=AsyncMock) as mock:
            self.mock_create_block = mock
            yield

        app.dependency_overrides.pop(get_agent_deps)

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
        self.configure_mock_get_agent_deps(raise_exc=AgentNotFoundError(f"Agent not found"))

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
        Then handle_message could raise this exception! Consider moving to an app level handler like some of the others
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
