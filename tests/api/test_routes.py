"""
HTTP route tests

Tests the FastAPI routes using httpx AsyncClient. Uses dependency_overrides
to inject mock factories, avoiding real DB lookups in route tests.

Fixtures are defined here (not in conftest) because only this file uses them.

Fixtures from conftest used here:
- session: Test DB session (function-scoped, rolled back after each test)
- agent_record: Pre-created agent for tests that need an existing agent
- agent_with_blocks: Agent with memory blocks attached

send_message test is currently in agent.test_runner.py as those tests are currently entangled with the runner
"""
# Standard library
from datetime import datetime
from unittest.mock import AsyncMock, Mock, patch
from uuid import uuid4

# Third-party
import pytest
from fastapi import FastAPI
from httpx import AsyncClient
from sqlalchemy.ext.asyncio import AsyncSession

# Local
from agent.factory import AgentNotFoundError
from agent.types import AgentConfig
from api.fastapi_deps import get_deps_dep
from conftest import make_deps
from db.models import AgentRecord, MemoryBlockRecord, utcnow
from api.schemas import AgentMetadataResponse, CoreMemoryResponse, MemoryBlockResponse
from memory.block_crud import DuplicateBlockError


# --- Test Classes ---

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
        """Returns the agent's system instructions as a JSON string."""
        response = await client.get(f"/agents/{agent_record.id}/system-instructions")

        assert response.status_code == 200
        assert response.json() == agent_record.system_instructions

    # 404 tested via parametrized TestNotFound


class TestPutConfig:
    """PUT /agents/{agent_id}/config — replace agent config."""

    @pytest.fixture(autouse=True)
    def mock_replace_agent_config_dep(self):
        with patch("api.routes.replace_agent_config", new_callable=AsyncMock) as mock:
            self.mock_replace_agent_config = mock
            yield

    async def test_calls_replace_agent_config_with_correct_args(
        self, client: AsyncClient, agent_record: AgentRecord, session: AsyncSession
    ):
        """Calls replace_agent_config with the agent_id and a validated AgentConfig (not raw dict)."""
        config = agent_record.agent_config
        self.mock_replace_agent_config.return_value = config

        response = await client.put(
            f"/agents/{agent_record.id}/config",
            json=config.model_dump(),
        )

        assert response.status_code == 200
        self.mock_replace_agent_config.assert_called_once_with(
            session, agent_record.id, config
        )

    # 404, 409, 422 tested separately


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
        "/agents/{agent_id}/config",
        "/agents/{agent_id}/system-instructions",
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
