"""FastAPI application and lifespan"""
from contextlib import asynccontextmanager

from fastapi import FastAPI

from api.routes import router
from api.schemas import HealthResponse
from db.connection import create_sqlite_engine, init_db


@asynccontextmanager
async def lifespan(app: FastAPI):
    raise NotImplementedError
    yield

app = FastAPI(lifespan=lifespan)
app.include_router(router)


@app.get("/health")
async def health() -> HealthResponse:
    # TODO: Shallow check right now, add check that DB is reachable and impl the associated test 
    return HealthResponse(status="ok")
