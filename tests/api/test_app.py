import pytest
import asyncio
from contextlib import asynccontextmanager
from asgi_lifespan import LifespanManager
from httpx import AsyncClient, ASGITransport
from unittest.mock import patch, AsyncMock, MagicMock

from api.app import app # TODO: Are all the tests sharing the same app?


class TestLifespan:

    @pytest.fixture(autouse=True)
    async def setup_and_teardown(self):
        self.mock_db_engine = MagicMock()
        self.mock_db_engine.dispose = MagicMock()

        with (patch('api.app.create_sqlite_engine', new_callable=AsyncMock) as mock_create_engine,
              patch('api.app.init_db', new_callable=AsyncMock) as mock_init_db):
            self.mock_create_engine = mock_create_engine
            self.mock_init_db = mock_init_db
            self.mock_create_engine.return_value = self.mock_db_engine

            async with LifespanManager(app):  # Triggers ASGI lifespan startup/shutdown
                async with AsyncClient(transport=ASGITransport(app=app), base_url="http://test") as client:
                    self.client = client
                    yield

            # lifespan shutdown should have disposed engine
            self.mock_db_engine.dispose.assert_called_once()
    
    async def test_happy_path(self):
        expected_db_path = "/data/db.sqlite"
        
        self.mock_create_engine.assert_called_once_with(expected_db_path)
        self.mock_init_db.assert_called_once_with(self.mock_db_engine)

        assert app.state.engine is self.mock_db_engine
        assert app.state.lock_reg == {} # TODO: should this be a TypedDict? or do we trust the reg users?
        # teardown asserts cleanup activity
