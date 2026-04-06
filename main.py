# ============================================================
# FILE: app/main.py
# PURPOSE: FastAPI application factory with lifespan, CORS, and router registration
# ============================================================

from contextlib import asynccontextmanager
from datetime import datetime, timezone

import redis.asyncio as aioredis
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from app.config import settings
from app.models import HealthResponse
from app.routers import query, upload
from app.services.cache import set_redis_client
from app.utils.logger import get_logger

logger = get_logger(__name__)


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Manage application startup and shutdown lifecycle events."""
    # --- Startup ---
    logger.info("App starting up — connecting to Redis at %s", settings.REDIS_URL)
    redis_client = aioredis.from_url(
        settings.REDIS_URL,
        encoding="utf-8",
        decode_responses=False,
    )
    set_redis_client(redis_client)
    logger.info("App started")

    yield

    # --- Shutdown ---
    logger.info("App shutting down — closing Redis connection")
    await redis_client.aclose()
    logger.info("App shutdown complete")


def create_app() -> FastAPI:
    """Construct and return the configured FastAPI application."""
    app = FastAPI(
        title="RAG Document Q&A API",
        description="Upload PDFs and ask questions via a retrieval-augmented generation pipeline.",
        version="1.0.0",
        lifespan=lifespan,
    )

    app.add_middleware(
        CORSMiddleware,
        allow_origins=["*"],
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
    )

    app.include_router(upload.router, prefix="/api/v1", tags=["upload"])
    app.include_router(query.router, prefix="/api/v1", tags=["query"])

    @app.get("/health", response_model=HealthResponse, tags=["health"])
    async def health_check() -> HealthResponse:
        """Return service health status and current UTC timestamp."""
        return HealthResponse(
            status="ok",
            timestamp=datetime.now(timezone.utc).isoformat(),
        )

    return app


app = create_app()
