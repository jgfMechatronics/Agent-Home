import pytest
from asgi_lifespan import LifespanManager
from unittest.mock import patch, AsyncMock, MagicMock

from api.app import _create_app
from api.routes import router


def test_create_app_includes_router():
    """Sanity check that _create_app() wires up the routes."""
    app = _create_app()
    
    app_paths = {r.path for r in app.routes}
    router_paths = {r.path for r in router.routes}
    
    assert router_paths.issubset(app_paths)
    assert "/health" in app_paths


class TestLifespan:

    @pytest.fixture(autouse=True)
    async def setup_and_teardown(self):
        # Fresh app instance per test - no state contamination
        self.app = _create_app()
        
        # set up mocks and handle patching
        self.mock_db_engine = MagicMock()
        self.mock_db_engine.dispose = AsyncMock()

        with (patch('api.app.create_sqlite_engine') as mock_create_engine,  # sync function
              patch('api.app.init_db', new_callable=AsyncMock) as mock_init_db):
            self.mock_create_engine = mock_create_engine
            self.mock_init_db = mock_init_db
            self.mock_create_engine.return_value = self.mock_db_engine
            yield
    
    async def startup_and_shutdown_lifespan(self) -> None:
        try:
            async with LifespanManager(self.app):  # Triggers ASGI lifespan startup/shutdown
                pass
        finally:
            # lifespan shutdown should have disposed engine
            self.mock_db_engine.dispose.assert_called_once()

    async def test_happy_path(self):
        await self.startup_and_shutdown_lifespan()

        expected_db_path = "/data/db.sqlite"
        
        self.mock_create_engine.assert_called_once_with(expected_db_path)
        self.mock_init_db.assert_called_once_with(self.mock_db_engine)

        assert self.app.state.engine is self.mock_db_engine
        assert self.app.state.agent_lock_reg == {}
        # teardown asserts cleanup activity

    async def test_init_db_failure_still_disposes(self):
        """If init_db raises, engine.dispose() should still be called for cleanup."""
        self.mock_init_db.side_effect = RuntimeError("DB init failed")

        with pytest.raises(RuntimeError, match="DB init failed"):
            await self.startup_and_shutdown_lifespan()
        # dispose assertion happens in startup_and_shutdown_lifespan's finally block
