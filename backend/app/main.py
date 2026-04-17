"""FastAPI application entry-point."""

from __future__ import annotations

import logging
from contextlib import asynccontextmanager

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from app.config import get_settings
from app.database import create_all_tables

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)-8s  %(name)s  %(message)s",
)
logger = logging.getLogger(__name__)


@asynccontextmanager
async def lifespan(application: FastAPI):
    """Startup / shutdown lifecycle."""
    settings = get_settings()
    logger.info("Starting Fake Currency Detection API on port %s …", settings.SERVER_PORT)

    # 1. Create database + tables
    await create_all_tables()

    # 2. Pre-load ML model (if available)
    try:
        from app.ml.inference import get_inference_engine
        engine = get_inference_engine()
        logger.info("ML inference engine ready (model loaded: %s).", engine.model_loaded)
    except Exception as exc:
        logger.warning("ML model not loaded at startup (will use fallback): %s", exc)

    yield

    logger.info("Shutting down …")


def create_app() -> FastAPI:
    settings = get_settings()

    application = FastAPI(
        title="Fake Currency Detection API",
        description="LeJEPA-based Indian currency authenticity verification",
        version="1.0.0",
        lifespan=lifespan,
    )

    # CORS
    application.add_middleware(
        CORSMiddleware,
        allow_origins=settings.cors_origins_list,
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
    )

    # Health check
    @application.get("/health", tags=["Health"])
    async def health():
        return {"status": "ok", "version": "1.0.0"}

    # Mount API routers
    from app.api.router import api_router
    application.include_router(api_router, prefix="/api/v1")

    return application


app = create_app()
