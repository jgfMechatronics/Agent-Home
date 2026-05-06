"""FastAPI application and lifespan"""
from contextlib import asynccontextmanager

from fastapi import FastAPI

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
        app.state.agent_lock_reg = {}
        yield
    finally:
        await engine.dispose()

app = FastAPI(lifespan=lifespan)
app.include_router(router)


@app.get("/health")
async def health() -> HealthResponse:
    # TODO: Shallow check right now, add check that DB is reachable and impl the associated test 
    return HealthResponse(status="ok")
