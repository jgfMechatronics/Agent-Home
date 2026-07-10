import os
from contextlib import asynccontextmanager
from datetime import datetime
from unittest.mock import AsyncMock, Mock

import pytest
import pytest_asyncio
from fastapi import FastAPI
from httpx import ASGITransport, AsyncClient
from sqlalchemy import event

# Prevent AnthropicModel construction from failing in tests that don't make real API calls
os.environ.setdefault("ANTHROPIC_API_KEY", "test")
from sqlalchemy.ext.asyncio import AsyncSession, create_async_engine
from sqlalchemy.pool import StaticPool

from agent.types import AgentConfig, AgentDeps
from api.fastapi_deps import get_session_dep
from db.models import AgentRecord, Base, MemoryBlockRecord
from pydantic_ai import RunContext
from pydantic_ai.messages import (
    ModelRequest,
    ModelResponse,
    RetryPromptPart,
    TextPart,
    ToolCallPart,
    ToolReturnPart,
    UserPromptPart,
)


def mock_run_context(deps: AgentDeps):
    """Create a mock RunContext with deps attached, for testing tools."""
    # spoof for isinstance checks
    ctx = Mock(spec=RunContext)
    ctx.deps = deps
    return ctx


SAMPLE_AGENT_CONFIG_DATA = { "model_name": "claude-sonnet-4-20250514",
    "tool_names": ["memory_replace", "memory_insert"],
    "soft_compaction_limit": 10000,
}

SAMPLE_AGENT_CONFIG = AgentConfig(**SAMPLE_AGENT_CONFIG_DATA)

def make_deps(session: AsyncSession, agent: AgentRecord) -> AgentDeps:
    """Construct AgentDeps from a session and agent record."""
    return AgentDeps(session=session, agent_record=agent)


# ---------------------------------------------------------------------------
# Pydantic-AI message factory helpers — shared across test modules
# ---------------------------------------------------------------------------

# Fixed naive UTC timestamp used in tool-pair request messages.  ModelRequest
# doesn't auto-assign a timestamp the way ModelResponse does, so we pin one to
# make assertions about orphan-warning log output predictable without bracketing.
_TOOL_PAIR_REQUEST_TS = datetime(2026, 1, 1, 12, 0, 0)


def make_request(content: str = "hello") -> ModelRequest:
    """Minimal ModelRequest with a single UserPromptPart."""
    return ModelRequest(parts=[UserPromptPart(content=content)])


def make_response(content: str = "hi") -> ModelResponse:
    """Minimal ModelResponse with a single TextPart."""
    return ModelResponse(parts=[TextPart(content=content)])


def make_tool_pair() -> tuple[ModelResponse, ModelRequest]:
    """A matched tool-call / tool-return pair.

    Returns (ModelResponse(ToolCallPart), ModelRequest(ToolReturnPart)).
    The ModelRequest has a fixed timestamp so tests can assert on it without
    time-bracketing.
    
    NOTE: These message shapes are hand-crafted to match pydantic-ai's internal format.
    A possibly more robust alternative would be to use FunctionModel (pydantic_ai.models.function) to
    run a real agent turn and capture the actual message sequence via result.all_messages().
    FunctionModel is useful for valid sequences; orphan tests still need hand-crafted invalid
    sequences (deliberately incomplete pairs) which could be produced by mutating the results of FunctionModel
    """
    call_part = ToolCallPart(tool_name="mem_replace", args='{"label":"x"}', tool_call_id="tc1")
    return_part = ToolReturnPart(tool_name="mem_replace", content="ok", tool_call_id="tc1")
    return (
        ModelResponse(parts=[call_part]),
        ModelRequest(parts=[return_part], timestamp=_TOOL_PAIR_REQUEST_TS),
    )


def make_retry_pair() -> tuple[ModelResponse, ModelRequest]:
    """A matched tool-call / retry-prompt pair (ModelRetry path).

    The call side is identical to make_tool_pair() — only the response side differs
    (RetryPromptPart instead of ToolReturnPart).  Same fixed-timestamp convention.
    """
    call_response, _ = make_tool_pair()
    retry_part = RetryPromptPart(
        content="block 'x' not found",
        tool_name="mem_replace",
        tool_call_id="tc1",
    )
    return (
        call_response,
        ModelRequest(parts=[retry_part], timestamp=_TOOL_PAIR_REQUEST_TS),
    )


@pytest_asyncio.fixture
async def session():
    """Fresh in-memory SQLite database per test. StaticPool ensures create_all and
    the session share the same connection; engine disposal destroys the DB."""
    engine = create_async_engine(
        "sqlite+aiosqlite:///:memory:",
        poolclass=StaticPool,
    )

    @event.listens_for(engine.sync_engine, "connect")
    def enable_foreign_keys(dbapi_conn, _):
        cursor = dbapi_conn.cursor()
        cursor.execute("PRAGMA foreign_keys=ON")
        cursor.close()

    try:
        async with engine.begin() as conn:
            await conn.run_sync(Base.metadata.create_all)
        
        async with AsyncSession(engine) as async_session:
            yield async_session
    finally:
        await engine.dispose()


@pytest_asyncio.fixture
async def agent_record(session: AsyncSession) -> AgentRecord:
    """
    A persisted AgentRecord for use in tests that require an existing agent.
    The underlying session can be used by dependents by seperately requesting that fixture. Pytest will
    cache it resulting in session pointing to the temp DB which contains the persisted agent
    """
    agent = AgentRecord(
        name="test-agent",
        agent_config=SAMPLE_AGENT_CONFIG,
        system_instructions="You are a test agent.",
    )
    session.add(agent)
    await session.flush()
    return agent


@pytest_asyncio.fixture
async def agent_with_blocks(session: AsyncSession):
    """
    Agent with system_instructions and three memory blocks in known positions.
    
    Blocks have descriptions (for XML formatting tests) and varied char_limits
    (for limit enforcement tests). Created out of position order to verify sorting.
    
    Returns dict with agent and blocks for test access.
    """
    agent = AgentRecord(
        name="agent-with-blocks",
        agent_config=SAMPLE_AGENT_CONFIG,
        system_instructions="You are a helpful assistant.",
    )
    session.add(agent)
    await session.flush()
    
    block_persona = MemoryBlockRecord(
        agent_id=agent.id,
        label="persona",
        description="The agent's identity",
        content="I am a test agent.",
        char_limit=1000,
        position=0,
    )
    block_human = MemoryBlockRecord(
        agent_id=agent.id,
        label="human",
        description="Information about the user",
        content="The user's name is Alice.",
        char_limit=500,
        position=1,
    )
    block_notes = MemoryBlockRecord(
        agent_id=agent.id,
        label="notes",
        description="Scratch space",
        content="Remember to be helpful.",
        char_limit=2000,
        position=2,
    )
    
    # Insert out of position order to verify queries sort by position, not insertion order
    session.add_all([block_human, block_notes, block_persona])
    blocks = [block_persona, block_human, block_notes]  # position order for test assertions
    await session.flush()
    
    return {"agent": agent, "blocks": blocks}


# ---------------------------------------------------------------------------
# Shared route test helpers and fixtures
# Used by both tests/api/test_routes.py and tests/agent/test_runner.py
# ---------------------------------------------------------------------------

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
