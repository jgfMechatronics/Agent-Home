"""Entry point — re-exports the FastAPI app for uvicorn and tests."""
from api.app import app

__all__ = ["app"]
