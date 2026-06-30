"""FastAPI application and lifespan"""
from contextlib import asynccontextmanager

from fastapi import FastAPI, Request
from fastapi.responses import JSONResponse

from agent.factory import AgentLockedError, AgentNotFoundError
from api.routes import router
from api.schemas import HealthResponse
from db.connection import create_sqlite_engine, init_db


DB_PATH = "/data/db.sqlite"


@asynccontextmanager
async def lifespan(app: FastAPI):
    engine = create_sqlite_engine(DB_PATH)
    try:
        await init_db(engine)
        app.state.engine = engine
        yield
    finally:
        await engine.dispose()


# App-level handlers commonize exception → HTTP response mapping. Without them, each route or
# dep that raises these exceptions would need its own mapping, making it easy for behavior to
# drift across the codebase. Handlers here apply consistently regardless of raise site.
# TODO: commonize with other exception formatting in the codebase
def _exc_detail(exc: Exception) -> str:
    return f"{type(exc).__name__}: {exc}"


async def agent_not_found_handler(request: Request, exc: AgentNotFoundError) -> JSONResponse:
    return JSONResponse(status_code=404, content={"detail": _exc_detail(exc)})


async def agent_locked_handler(request: Request, exc: AgentLockedError) -> JSONResponse:
    return JSONResponse(status_code=503, content={"detail": _exc_detail(exc)})


# Since this app is intended for self hosters, we want exception details to pass on to the client
async def unexpected_error_handler(request: Request, exc: Exception) -> JSONResponse:
    return JSONResponse(status_code=500, content={"detail": _exc_detail(exc)})


def _create_app() -> FastAPI:
    """Factory function for creating the FastAPI app. Enables fresh instances per test."""
    app = FastAPI(lifespan=lifespan)
    app.state.agent_app_states = {}
    app.include_router(router)
    app.add_exception_handler(AgentNotFoundError, agent_not_found_handler)
    app.add_exception_handler(AgentLockedError, agent_locked_handler)
    app.add_exception_handler(Exception, unexpected_error_handler)

    @app.get("/health")
    async def health() -> HealthResponse:
        # TODO: Shallow check right now, add check that DB is reachable and impl the associated test 
        return HealthResponse(status="ok")
    
    return app


app = _create_app()
