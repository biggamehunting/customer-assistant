"""FastAPI application factory.

Call create_app() to get a configured FastAPI instance.
Uvicorn entry point: app.main:app
"""
from __future__ import annotations

from contextlib import asynccontextmanager

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from app.api.middleware import (
    RequestIDMiddleware,
    StructuredLoggingMiddleware,
    register_exception_handlers,
)
from app.api.routes import admin, chat, documents, health
from app.core.config import get_settings
from app.core.logging import configure_logging, get_logger
from app.core.telemetry import instrument_app, setup_telemetry
from app.storage.database import close_engine

logger = get_logger(__name__)


@asynccontextmanager
async def lifespan(app: FastAPI):
    # ── Startup ───────────────────────────────────────────────────────────
    logger.info("app.starting", environment=get_settings().environment.value)
    yield
    # ── Shutdown ──────────────────────────────────────────────────────────
    await close_engine()
    logger.info("app.stopped")


def create_app() -> FastAPI:
    configure_logging()
    setup_telemetry()
    settings = get_settings()

    app = FastAPI(
        title="Customer Assistant API",
        version=settings.app_version,
        description="Production-grade agentic RAG customer support assistant",
        docs_url="/docs" if not settings.is_production else None,
        redoc_url="/redoc" if not settings.is_production else None,
        lifespan=lifespan,
    )

    # ── Middleware (order matters: outer → inner) ──────────────────────────
    app.add_middleware(StructuredLoggingMiddleware)
    app.add_middleware(RequestIDMiddleware)
    app.add_middleware(
        CORSMiddleware,
        allow_origins=[str(o) for o in settings.cors_origins],
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
    )

    # ── Exception handlers ────────────────────────────────────────────────
    register_exception_handlers(app)

    # ── Routers ───────────────────────────────────────────────────────────
    app.include_router(health.router)
    app.include_router(chat.router)
    app.include_router(documents.router)
    app.include_router(admin.router)

    # ── OpenTelemetry FastAPI instrumentation ─────────────────────────────
    instrument_app(app)

    logger.info("app.ready", version=settings.app_version)
    return app


app = create_app()
