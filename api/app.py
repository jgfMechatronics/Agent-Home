"""FastAPI application and lifespan"""
from fastapi import FastAPI

from api.routes import router
from api.schemas import HealthResponse

app = FastAPI()
app.include_router(router)


@app.get("/health")
async def health() -> HealthResponse:
    # TODO: Shallow check right now, add check that DB is reachable and impl the associated test 
    return HealthResponse(status="ok")
