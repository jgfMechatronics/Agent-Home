"""FastAPI application and lifespan — Section 4.3."""
from fastapi import FastAPI

from api.routes import router
from api.schemas import HealthResponse

app = FastAPI()
app.include_router(router)


@app.get("/health")
async def health() -> HealthResponse:
    raise NotImplementedError
