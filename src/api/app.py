"""FastAPI application and lifespan"""
import logging
import os
from contextlib import asynccontextmanager
from pathlib import Path
from urllib.parse import urlparse

from fastapi import FastAPI, Request
from fastapi.responses import JSONResponse, Response
from starlette.datastructures import Headers
from starlette.middleware.trustedhost import TrustedHostMiddleware
from starlette.types import ASGIApp, Receive, Scope, Send

from agent.broadcast_streaming import BroadcastHub
from agent.types import AgentLockedError, AgentNotFoundError
from memory.block_crud import BlockNotFoundError, DuplicateBlockError, DuplicatePositionError
from api.routes import router
from prototype.iac.send_message import configure_iac  # TODO: Remove with IAC redesign
from api.schemas import HealthResponse
from db.connection import create_readonly_sqlite_engine, create_sqlite_engine, init_db
from common.constants import INTEGRITY_LOCKFILE_NAME

logger = logging.getLogger(__name__)

DB_PATH = os.environ["AGENT_HOME_DB_PATH"]


_ALLOWED_HOSTS: tuple[str, ...] = ("localhost", "127.0.0.1")
_ALLOWED_ORIGINS: tuple[str, ...] = _ALLOWED_HOSTS


class OriginValidationMiddleware:
    """Reject browser-originated cross-site requests by validating the Origin header.

    Browsers send an Origin header on cross-origin requests. If it is present and does
    not resolve to an allowed host, the request is rejected with a 403. Requests without
    an Origin header (direct API calls from toad, curl, scripts) are always allowed.

    This prevents a malicious web page from making requests to the server on behalf of
    a user who happens to visit it while the server is running locally — a relevant concern
    as endpoints become more capable.

    Note: TrustedHostMiddleware handles DNS rebinding; this middleware handles CSRF.
    Together they cover the two main browser-based attack vectors.
    """

    def __init__(self, app: ASGIApp) -> None:
        self.app = app

    async def __call__(self, scope: Scope, receive: Receive, send: Send) -> None:
        if scope["type"] not in ("http", "websocket"):
            await self.app(scope, receive, send)
            return
        origin = Headers(scope=scope).get("origin")
        if origin is not None and urlparse(origin).hostname not in _ALLOWED_ORIGINS:
            await Response(status_code=403)(scope, receive, send)
            return
        await self.app(scope, receive, send)


@asynccontextmanager
async def lifespan(app: FastAPI):
    lockfile = Path(DB_PATH).parent / INTEGRITY_LOCKFILE_NAME
    if lockfile.exists():
        msg = (
            f"Integrity check failed — server startup blocked. "
            f"Inspect {lockfile.parent / 'integrity_checker_results.txt'} and resolve all issues, "
            f"then delete {lockfile} to allow the server to start."
        )
        logger.critical(msg)
        raise RuntimeError(msg)
    rw_engine = create_sqlite_engine(DB_PATH)
    ro_engine = create_readonly_sqlite_engine(DB_PATH)
    hub = BroadcastHub()
    try:
        await init_db(rw_engine)
        # TODO: rw/ro engine wiring through app.state is invisible to static type checking.
        # Starlette's State.__getattr__ returns Any, so these two assignments and every
        # `request.app.state.{rw,ro}_engine` read (fastapi_deps.py x4, routes.py create_agent x2)
        # accept a swapped engine without complaint, even with an annotated target
        # (`x: ReadOnlyEngine = request.app.state.rw_engine` type-checks). Everything downstream is
        # typed, but the checker trusts the declared types and nothing verifies them at runtime
        # (NewType is erased). A rw engine in the ro slot is the silent failure: reads work, the ro
        # protection is just gone. (ro in the rw slot fails loudly on first write.)
        # Fix: store one typed object and read it through one accessor, like get_broadcast_hub:
        #     @dataclass(frozen=True)
        #     class Engines:              # probably in db/connection.py, next to ReadOnlyEngine
        #         rw: AsyncEngine
        #         ro: ReadOnlyEngine
        #         def __post_init__(self):  # runtime check, once at startup
        #             if self.ro.url.query.get("mode") != "ro":
        #                 raise RuntimeError("Engines.ro is not a mode=ro engine")
        #     app.state._engines = Engines(rw=rw_engine, ro=ro_engine)  # keyword args: typed
        #     def get_engines(request: Request) -> Engines:
        #         # app.state._engines is private on purpose: this function is meant to be the only
        #         # reader, so all engine access goes through one typed (and checked) accessor.
        #         return request.app.state._engines
        # Then forbid `app.state.*engine` access outside that accessor. The test fixtures that set
        # app.state.rw_engine/ro_engine (conftest, api/test_routes, api/test_fastapi_deps,
        # api/test_app, prototype/api/test_routes, utils/test_memory_cleanup) move to a shared
        # helper. Known remaining gap: ReadOnlyEngine is a NewType of AsyncEngine, so
        # Engines(rw=ro_engine, ...) still type-checks, but that fails loudly on first write.
        app.state.rw_engine = rw_engine
        app.state.ro_engine = ro_engine
        app.state.broadcast_hub = hub
        configure_iac(app.state.agent_app_state_reg, hub)  # TODO: Remove with IAC redesign
        yield
    finally:
        await hub.shutdown()
        await rw_engine.dispose()
        await ro_engine.dispose()


# App-level handlers commonize exception → HTTP response mapping. Without them, each route or
# dep that raises these exceptions would need its own mapping, making it easy for behavior to
# drift across the codebase. Handlers here apply consistently regardless of raise site.
# TODO: commonize with other exception formatting in the codebase
def _exc_detail(exc: Exception) -> str:
    return f"{type(exc).__name__}: {exc}"


async def agent_not_found_handler(request: Request, exc: AgentNotFoundError) -> JSONResponse:
    return JSONResponse(status_code=404, content={"detail": _exc_detail(exc)})


async def agent_locked_handler(request: Request, exc: AgentLockedError) -> JSONResponse:
    return JSONResponse(status_code=423, content={"detail": _exc_detail(exc)})


async def block_not_found_handler(request: Request, exc: BlockNotFoundError) -> JSONResponse:
    return JSONResponse(status_code=404, content={"detail": _exc_detail(exc)})


async def duplicate_block_handler(request: Request, exc: DuplicateBlockError) -> JSONResponse:
    return JSONResponse(status_code=422, content={"detail": _exc_detail(exc)})


async def duplicate_position_handler(request: Request, exc: DuplicatePositionError) -> JSONResponse:
    return JSONResponse(status_code=422, content={"detail": _exc_detail(exc)})


# Since this app is intended for self hosters, we want exception details to pass on to the client
async def unexpected_error_handler(request: Request, exc: Exception) -> JSONResponse:
    return JSONResponse(status_code=500, content={"detail": _exc_detail(exc)})


def _create_app() -> FastAPI:
    """Factory function for creating the FastAPI app. Enables fresh instances per test."""
    app = FastAPI(lifespan=lifespan)
    app.state.agent_app_state_reg = {}
    app.include_router(router)
    app.add_exception_handler(AgentNotFoundError, agent_not_found_handler)
    app.add_exception_handler(AgentLockedError, agent_locked_handler)
    app.add_exception_handler(BlockNotFoundError, block_not_found_handler)
    app.add_exception_handler(DuplicateBlockError, duplicate_block_handler)
    app.add_exception_handler(DuplicatePositionError, duplicate_position_handler)
    app.add_exception_handler(Exception, unexpected_error_handler)
    # Prevent DNS rebinding (validates Host header)
    app.add_middleware(TrustedHostMiddleware, allowed_hosts=_ALLOWED_HOSTS)
    # Prevent CSRF (validates Origin header on browser-originated requests)
    app.add_middleware(OriginValidationMiddleware)

    @app.get("/health")
    async def health() -> HealthResponse:
        # TODO: Shallow check right now, add check that DB is reachable and impl the associated test 
        return HealthResponse(status="ok")
    
    return app


app = _create_app()
