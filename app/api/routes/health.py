"""Health and readiness endpoints."""
from __future__ import annotations

import time

from fastapi import APIRouter
from sqlalchemy import text

from app.core.config import get_settings
from app.core.logging import get_logger
from app.domain.schemas import DependencyHealth, HealthResponse, HealthStatus
from app.storage.database import get_db_context

router = APIRouter(tags=["health"])
logger = get_logger(__name__)


@router.get("/health/live", response_model=HealthResponse, include_in_schema=False)
async def liveness() -> HealthResponse:
    """Kubernetes liveness probe — just verifies the process is alive."""
    settings = get_settings()
    return HealthResponse(
        status=HealthStatus.ok,
        version=settings.app_version,
        environment=settings.environment.value,
    )


@router.get("/health/ready", response_model=HealthResponse)
async def readiness() -> HealthResponse:
    """Kubernetes readiness probe — checks critical dependencies."""
    settings = get_settings()
    deps: list[DependencyHealth] = []
    overall = HealthStatus.ok

    # Database check
    t0 = time.monotonic()
    try:
        async with get_db_context() as session:
            await session.execute(text("SELECT 1"))
        deps.append(
            DependencyHealth(
                name="postgresql",
                status=HealthStatus.ok,
                latency_ms=round((time.monotonic() - t0) * 1000, 1),
            )
        )
    except Exception as exc:
        logger.warning("health.db_check_failed", error=str(exc))
        deps.append(
            DependencyHealth(
                name="postgresql",
                status=HealthStatus.down,
                latency_ms=round((time.monotonic() - t0) * 1000, 1),
                detail="Database unreachable",
            )
        )
        overall = HealthStatus.down

    return HealthResponse(
        status=overall,
        version=settings.app_version,
        environment=settings.environment.value,
        dependencies=deps,
    )


@router.get("/health", response_model=HealthResponse)
async def health_summary() -> HealthResponse:
    """Combined health summary (alias for readiness)."""
    return await readiness()
