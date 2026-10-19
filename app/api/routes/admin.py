"""Administrative endpoints — protected by admin scope."""
from __future__ import annotations

import uuid

from fastapi import APIRouter, HTTPException
from sqlalchemy import func, select

from app.api.dependencies import AdminUser, DbSession
from app.core.logging import get_logger
from app.domain.schemas import IngestionJobResponse, MetricsResponse
from app.storage.orm import ConversationORM, DocumentORM, IngestionJobORM, MessageORM

router = APIRouter(prefix="/api/v1/admin", tags=["admin"])
logger = get_logger(__name__)


@router.get("/metrics", response_model=MetricsResponse)
async def get_metrics(
    db: DbSession,
    current_user: AdminUser,
) -> MetricsResponse:
    """Corpus and system metrics for the current tenant."""
    tenant_id = current_user.tenant_id

    doc_counts = await db.execute(
        select(DocumentORM.status, func.count(DocumentORM.id))
        .where(DocumentORM.tenant_id == tenant_id)
        .group_by(DocumentORM.status)
    )
    status_map: dict[str, int] = {row[0]: row[1] for row in doc_counts}

    conv_count = await db.execute(
        select(func.count(ConversationORM.id)).where(ConversationORM.tenant_id == tenant_id)
    )
    msg_count = await db.execute(
        select(func.count(MessageORM.id)).join(
            ConversationORM, MessageORM.conversation_id == ConversationORM.id
        ).where(ConversationORM.tenant_id == tenant_id)
    )
    pending_jobs = await db.execute(
        select(func.count(IngestionJobORM.id)).where(
            IngestionJobORM.tenant_id == tenant_id,
            IngestionJobORM.status == "pending",
        )
    )
    failed_jobs = await db.execute(
        select(func.count(IngestionJobORM.id)).where(
            IngestionJobORM.tenant_id == tenant_id,
            IngestionJobORM.status == "failed",
        )
    )

    return MetricsResponse(
        total_documents=sum(status_map.values()),
        indexed_documents=status_map.get("indexed", 0),
        failed_documents=status_map.get("failed", 0),
        total_conversations=conv_count.scalar() or 0,
        total_messages=msg_count.scalar() or 0,
        pending_jobs=pending_jobs.scalar() or 0,
        failed_jobs=failed_jobs.scalar() or 0,
    )


@router.get("/ingestion/jobs", response_model=list[IngestionJobResponse])
async def list_ingestion_jobs(
    db: DbSession,
    current_user: AdminUser,
    status_filter: str | None = None,
    limit: int = 50,
    offset: int = 0,
) -> list[IngestionJobResponse]:
    """List ingestion jobs with optional status filter."""
    query = select(IngestionJobORM).where(IngestionJobORM.tenant_id == current_user.tenant_id)
    if status_filter:
        query = query.where(IngestionJobORM.status == status_filter)
    result = await db.execute(query.order_by(IngestionJobORM.created_at.desc()).limit(limit).offset(offset))
    jobs = result.scalars().all()
    return [
        IngestionJobResponse(
            id=j.id,
            document_id=j.document_id,
            status=j.status,
            attempt_count=j.attempt_count,
            created_at=j.created_at,
            updated_at=j.updated_at,
            error_message=j.error_message,
            started_at=j.started_at,
            completed_at=j.completed_at,
        )
        for j in jobs
    ]


@router.post("/ingestion/jobs/{job_id}/retry", response_model=IngestionJobResponse)
async def retry_job(
    job_id: uuid.UUID,
    db: DbSession,
    current_user: AdminUser,
) -> IngestionJobResponse:
    """Reset a failed job back to pending for retry."""
    result = await db.execute(
        select(IngestionJobORM).where(
            IngestionJobORM.id == job_id,
            IngestionJobORM.tenant_id == current_user.tenant_id,
        )
    )
    job = result.scalar_one_or_none()
    if not job:
        raise HTTPException(status_code=404, detail="Job not found")
    if job.status not in ("failed", "cancelled"):
        raise HTTPException(status_code=400, detail=f"Cannot retry job in status: {job.status}")

    job.status = "pending"
    job.error_message = None
    job.attempt_count = 0

    try:
        from app.workers.ingestion_tasks import process_document_task
        process_document_task.delay(str(job.document_id), str(job_id), str(current_user.tenant_id))
    except Exception as exc:
        logger.warning("admin.retry_enqueue_failed", job_id=str(job_id), error=str(exc))

    return IngestionJobResponse(
        id=job.id,
        document_id=job.document_id,
        status=job.status,
        attempt_count=job.attempt_count,
        created_at=job.created_at,
        updated_at=job.updated_at,
    )
