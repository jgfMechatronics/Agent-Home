import hashlib
import os
import tempfile
import uuid
from pathlib import Path
from zoneinfo import ZoneInfo

# Must be set before any import of api.app (which reads it at module level).
# Conftest is imported first by pytest, making this the right place for it.
os.environ.setdefault("AGENT_HOME_DB_PATH", os.path.expanduser("~/.agent-home_pytest/db.sqlite"))

from contextlib import asynccontextmanager
from datetime import datetime
from unittest.mock import AsyncMock, Mock

import pytest
import pytest_asyncio
import time_machine
from fastapi import FastAPI
from httpx import ASGITransport, AsyncClient
from sqlalchemy.orm import object_session
from db.connection import ReadOnlyEngine, create_readonly_sqlite_engine, create_sqlite_engine

from sqlalchemy.ext.asyncio import AsyncEngine, AsyncSession

from agent.deps import AgentDeps
from agent.types import AgentAppState, AgentConfig
from agent.broadcast_streaming import BroadcastHub
from pydantic import ConfigDict
from api.app import _create_app
from db.models import (
    AgentConfigSnapshot,
    AgentRecord,
    Base,
    MemoryBlockRecord,
    SystemPromptSnapshot,
    ToolDefinitionSnapshot,
    utcnow,
)
from pydantic_ai import RunContext
from pydantic_ai.messages import (
    ModelMessage,
    ModelRequest,
    ModelResponse,
    RetryPromptPart,
    TextPart,
    ThinkingPart,
    ToolCallPart,
    ToolReturnPart,
    UserPromptPart,
)
from pydantic_ai.usage import RequestUsage


def mock_run_context(deps: AgentDeps):
    """Create a mock RunContext with deps attached, for testing tools."""
    # spoof for isinstance checks
    ctx = Mock(spec=RunContext)
    ctx.deps = deps
    return ctx


FROZEN_NOW = datetime(2026, 10, 10, 12, 0, 0, tzinfo=ZoneInfo("UTC"))
"""The frozen instant for timestamp tests. Importable so tests can do math on
it (e.g. .astimezone(...)) when asserting on stamps in other zones."""


@pytest.fixture
def frozen_stamp_clock():
    """Freeze the clock (time-machine) for timestamp determinism.

    NOT autouse: some tests need real, incrementing timestamps. Modules/classes
    that assert on stamped content opt in via
    `pytestmark = pytest.mark.usefixtures("frozen_stamp_clock")`.
    """
    with time_machine.travel(FROZEN_NOW):
        yield


SAMPLE_AGENT_CONFIG_DATA = { "model_name": "anthropic:claude-sonnet-4-20250514",
    "tool_names": ["memory_replace", "memory_insert"],
    "soft_compaction_limit": 10000,
}


class _FrozenAgentConfig(AgentConfig):
    """Immutable subclass of AgentConfig for use as a shared test constant.

    Raises FrozenInstanceError on any in-place mutation, catching mistakes at
    the point of the error rather than silently corrupting shared test state.
    model_copy() returns a regular (mutable) AgentConfig as expected.

    __eq__ compares field values against any AgentConfig instance so that DB
    round-trips (which deserialize to plain AgentConfig) compare equal when the
    field values match.
    """
    model_config = ConfigDict(frozen=True)

    def __eq__(self, other: object) -> bool:
        if isinstance(other, AgentConfig):
            return self.model_dump() == other.model_dump()
        return NotImplemented


SAMPLE_AGENT_CONFIG = _FrozenAgentConfig(**SAMPLE_AGENT_CONFIG_DATA)


@pytest.fixture
def fake_provider_keys(monkeypatch):
    """Fake API keys for provider client construction.

    Building agents via infer_model() instantiates provider clients, which
    require API keys at construction time. No API calls are made in unit
    tests — keys are only needed to satisfy client constructors.

    The rest of the suite runs keyless, matching standalone DB readers
    (integrity checker, CLI tools) and guarding against validation
    regressing to require keys.
    """
    for key in ("ANTHROPIC_API_KEY", "OPENAI_API_KEY", "TOGETHER_API_KEY", "GROQ_API_KEY", "OPENROUTER_API_KEY"):
        monkeypatch.setenv(key, "test-fake-key")


# ---------------------------------------------------------------------------
# Stub snapshot constants — satisfy MessageRecord FK constraints in tests
# ---------------------------------------------------------------------------

def _sha256(s: str) -> str:
    return hashlib.sha256(s.encode("utf-8")).hexdigest()


STUB_SYS_PROMPT = ""
STUB_SYS_HASH = _sha256(STUB_SYS_PROMPT)
STUB_TOOL_JSON = "[]"
STUB_TOOL_HASH = _sha256(STUB_TOOL_JSON)
STUB_CONFIG_JSON = SAMPLE_AGENT_CONFIG.model_dump_json()
STUB_CONFIG_HASH = _sha256(STUB_CONFIG_JSON)
STUB_CTX_MSG_ID = "00000000-0000-0000-0000-000000000001"

PARTIAL_MEMORY_BLOCK_FIELDS = {
    "description": "",
    "char_limit": 2000,
    "position": 0,
}

PARTIAL_MESSAGE_FIELDS = {
    "type": "ModelRequest",
    "content": "{}",
    "total_tokens": None,
    "system_prompt_hash": STUB_SYS_HASH,
    "tool_definition_hash": STUB_TOOL_HASH,
    "agent_config_hash": STUB_CONFIG_HASH,
    "context_window_start_msg_id": STUB_CTX_MSG_ID,
}


@pytest_asyncio.fixture
async def seed_stub_snapshots(session: AsyncSession):
    """Pre-seed stub snapshot rows so MessageRecord FK constraints are satisfied.
    
    Use @pytest.mark.usefixtures("seed_stub_snapshots") on test classes that
    insert raw MessageRecords, or request it directly in test functions.
    """
    session.add_all([
        SystemPromptSnapshot(id=STUB_SYS_HASH, content=STUB_SYS_PROMPT, created_at=utcnow()),
        ToolDefinitionSnapshot(id=STUB_TOOL_HASH, content=STUB_TOOL_JSON, created_at=utcnow()),
        AgentConfigSnapshot(id=STUB_CONFIG_HASH, content=STUB_CONFIG_JSON, created_at=utcnow()),
    ])
    await session.flush()


def make_ro_engine(engine) -> ReadOnlyEngine:
    """Derive a genuine read-only engine (SQLite mode=ro) from a test RW engine.

    Same DB file, exact prod constructor: writes through the ro side fail in
    tests with the same readonly-database error prod would raise. ReadOnlySession
    is only a NewType (type-level intent, no runtime enforcement), so this engine
    is the ONLY runtime write protection on the read path. NullPool: no teardown
    needed. Used by make_deps and by sites building build_deps/AgentFactory directly.
    """
    return create_readonly_sqlite_engine(engine.url.database)


def make_deps(engine, agent_record) -> AgentDeps:
    """Test-side factory for AgentDeps — bypasses the factory guard (valid in tests only).

    Requires a real SQLAlchemy engine (will fail loudly on engine.url if passed a mock).
    Derives a genuine mode=ro ro engine (see make_ro_engine) so the rw/ro split is
    enforced in tests as in prod. Sites explicitly testing the rw/ro split itself use
    the raw constructor. Centralizes the guard-bypass so constructor signature changes
    touch one place.

    Expunges the record if it's still attached to a session (belt-and-suspenders —
    commit_and_detach_fixture_record already detaches; guards against raw flush+add
    callers): deps' invariant is a detached record between scopes, and write_scope's
    re-attach fails loudly on a doubly-attached record.
    """
    ro_engine = make_ro_engine(engine)
    if (attached_session := object_session(agent_record)) is not None:
        attached_session.expunge(agent_record)
    return AgentDeps(engine, ro_engine, agent_record, bypass_factory_guard=True)


@pytest_asyncio.fixture
async def agent_deps(test_engine, agent_record: AgentRecord) -> AgentDeps:
    """AgentDeps on the test engine, with a genuine mode=ro ro engine (see make_deps)."""
    return make_deps(test_engine, agent_record)


# ---------------------------------------------------------------------------
# Pydantic-AI message factory helpers — shared across test modules
# ---------------------------------------------------------------------------


def make_request(content: str = "hello") -> ModelRequest:
    """Minimal ModelRequest with a single UserPromptPart."""
    return ModelRequest(parts=[UserPromptPart(content=content)])


def make_response(content: str = "hi", *, total_tokens: int | None = None) -> ModelResponse:
    """Minimal ModelResponse with a single TextPart.
    
    Pass total_tokens to simulate real API usage data (stored in MessageRecord.total_tokens
    when persisted via persist_messages). RequestUsage.total_tokens is a computed property
    (input_tokens + output_tokens), so total_tokens is carried entirely in input_tokens.
    """
    usage = RequestUsage(input_tokens=total_tokens, output_tokens=0) if total_tokens is not None else RequestUsage()
    return ModelResponse(parts=[TextPart(content=content)], usage=usage)


def make_alternating_messages(count: int, prefix: str = "msg") -> list[ModelMessage]:
    """Generate count alternating request/response messages."""
    return [
        make_request(f"{prefix} {i}") if i % 2 == 0 else make_response(f"{prefix} {i}")
        for i in range(count)
    ]


def make_tool_pair() -> tuple[ModelResponse, ModelRequest]:
    """A matched tool-call / tool-return pair.

    Returns (ModelResponse(ToolCallPart), ModelRequest(ToolReturnPart)).

    NOTE: These message shapes are hand-crafted to match pydantic-ai's internal format.
    A possibly more robust alternative would be to use FunctionModel (pydantic_ai.models.function) to
    run a real agent turn and capture the actual message sequence via result.all_messages().
    FunctionModel is useful for valid sequences; orphan tests still need hand-crafted invalid
    sequences (deliberately incomplete pairs) which could be produced by mutating the results of FunctionModel
    """
    tool_call_id = str(uuid.uuid4())
    call_part = ToolCallPart(tool_name="mem_replace", args='{"label":"x"}', tool_call_id=tool_call_id)
    return_part = ToolReturnPart(tool_name="mem_replace", content="ok", tool_call_id=tool_call_id)
    return (
        ModelResponse(parts=[call_part]),
        ModelRequest(parts=[return_part]),
    )


def make_retry_pair() -> tuple[ModelResponse, ModelRequest]:
    """A matched tool-call / retry-prompt pair (ModelRetry path).

    The call side is identical to make_tool_pair() — only the response side differs
    (RetryPromptPart instead of ToolReturnPart).
    """
    call_response, _ = make_tool_pair()
    retry_part = RetryPromptPart(
        content="block 'x' not found",
        tool_name="mem_replace",
        tool_call_id=call_response.parts[0].tool_call_id,
    )
    return (
        call_response,
        ModelRequest(parts=[retry_part]),
    )


@pytest_asyncio.fixture
async def test_engine():
    """Per-test engine backed by a real file DB in a per-test tmpfs directory.

    File-backed + create_sqlite_engine gives tests the exact production SQLite
    configuration (WAL, busy_timeout, foreign_keys): committed-visibility and
    snapshot semantics, with no read_uncommitted divergence. The directory
    lives under /dev/shm (tmpfs) — RAM speed, no disk wear. TemporaryDirectory
    gives a unique per-test dir (isolation incl. xdist workers) and removes it
    automatically on teardown; the engine is disposed first.
    """
    tmpfs = "/dev/shm" if Path("/dev/shm").is_dir() else None
    with tempfile.TemporaryDirectory(prefix="agent-home-test-", dir=tmpfs, ignore_cleanup_errors=True) as db_dir:
        engine = create_sqlite_engine(str(Path(db_dir) / "test.db"))
        try:
            async with engine.begin() as conn:
                await conn.run_sync(Base.metadata.create_all)
            yield engine
        finally:
            await engine.dispose()


@pytest_asyncio.fixture
async def session(test_engine):
    """Session bound to the test engine, for test setup and assertions.

    Fixture setup that other connections must see: use commit_and_detach_fixture_record
    (add + commit + refresh — the same idiom as prod write_scope).
    """
    async with AsyncSession(test_engine) as async_session:
        yield async_session


async def commit_and_detach_fixture_record(session: AsyncSession, record):
    """Fixture setup idiom: add + commit + refresh + expunge.

    Commit publishes the record to other connections (read_scope queries) AND
    autoflushes any pending inserts (no explicit flush() needed). Refresh reloads
    attributes wiped by commit-expiry. Expunge detaches the record so it can be
    passed to make_deps or write_scope without double-attach errors.

    Use this for fixture records accessed via separate sessions (e.g., read_scope
    in tested functions). If you need DB-generated values BEFORE committing (rare),
    do explicit flush+commit by hand instead (see TestListAgents.test_returns_all_agents).
    """
    session.add(record)
    await session.commit()
    await session.refresh(record)
    session.expunge(record)
    return record


@pytest.fixture
def agent_app_state_reg() -> dict[str, AgentAppState]:
    """Fresh agent state registry for each test. Must be an empty dict."""
    return {}


@pytest_asyncio.fixture
async def agent_record(session: AsyncSession) -> AgentRecord:
    """
    A persisted, committed, detached AgentRecord for use in tests.
    Committed so build_deps (which creates its own session) can find it.
    Refreshed and expunged by commit_and_detach_fixture_record so attribute
    reads work on the detached record.
    """
    agent = AgentRecord(
        name="test-agent",
        agent_config=SAMPLE_AGENT_CONFIG,
        system_instructions="You are a test agent.",
    )
    return await commit_and_detach_fixture_record(session, agent)
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
    # Commit + refresh + expunge: fixtures hand out detached, attribute-loaded records
    # (ORM objects never outlive their session). Mutate via merge + commit in tests.
    await session.commit()
    await session.refresh(agent)
    for block in blocks:
        await session.refresh(block)
    session.expunge(agent)
    for block in blocks:
        session.expunge(block)

    return {"agent": agent, "blocks": blocks}


# ---------------------------------------------------------------------------
# Shared route test helpers and fixtures
# Used by both tests/api/test_routes.py and tests/agent/test_runner.py
# ---------------------------------------------------------------------------

TEST_BASE_URL = "http://localhost"


def make_mock_agent(events: list | None = None, raises_mid_stream: Exception | None = None) -> Mock:
    """Create a mock agent whose run_stream_events yields the given events.

    The mock is a plain async generator, matching Pydantic AI's current API.
    If raises_mid_stream is set, the exception is raised after all events are yielded.
    """
    agent = Mock()
    agent.toolsets = []  # _extract_tool_definitions iterates toolsets; mock agents have none

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


@pytest_asyncio.fixture
async def app(test_engine) -> FastAPI:
    """Fresh app instance per test — avoids state contamination.

    Sets rw_engine/ro_engine on app.state (normally done in lifespan, which doesn't run in tests).
    """
    app = _create_app()
    app.state.broadcast_hub = BroadcastHub()
    app.state.rw_engine = test_engine
    app.state.ro_engine = make_ro_engine(test_engine)
    return app


@pytest_asyncio.fixture
async def client(app: FastAPI) -> AsyncClient:
    """Async HTTP client bound to the test app.

    raise_app_exceptions=False so that app-level Exception handlers (ServerErrorMiddleware)
    just return the 500 response rather than re-raising into the test.
    """
    async with AsyncClient(
        transport=ASGITransport(app=app, raise_app_exceptions=False),
        base_url=TEST_BASE_URL,
    ) as c:
        yield c


# ---------------------------------------------------------------------------
# MCP test helpers (shared between test_runner.py and test_messages.py)
# ---------------------------------------------------------------------------

async def local_dummy_tool(ctx: RunContext, text: str) -> str:
    """A local function tool used in MCP integration tests."""
    return text


class ScriptedFunction:
    """FunctionModel non-streamed function consuming one step per model invocation.

    Each step is either a ModelResponse to return, or a callable receiving the
    live message history (list[ModelMessage]) and returning a ModelResponse —
    used to script tool calls that depend on state from earlier in the same run.
    Running out of steps raises IndexError: fail loudly. `invocation` counts
    model calls, assertable as a no-retry check.

    TODO: We should consider if this ScriptedFunction and the corresponding FunctionModel build from it could replace the FunctionModelTestAgent
    or at least inspire it.
    """

    def __init__(self, steps: list):
        self._steps = steps
        self.invocation = 0

    def __call__(self, messages, info) -> ModelResponse:
        step = self._steps[self.invocation]
        self.invocation += 1
        if callable(step):
            step = step(list(messages))
        return step


def assert_ModelMessage_list_eq(
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


def in_process_mcp_toolset_factory():
    """Real in-process FastMCP server exposing a known tool — no HTTP, no mocking."""
    from fastmcp import FastMCP
    from pydantic_ai.mcp import MCPToolset

    mcp = FastMCP("test-mcp-server")

    @mcp.tool()
    def mcp_read_file(path: str) -> str:
        """Read a file from disk."""
        return f"contents of {path}"

    return MCPToolset(mcp)


@pytest.fixture
def in_process_mcp_toolset():
    return in_process_mcp_toolset_factory()
