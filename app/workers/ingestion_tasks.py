"""Celery ingestion worker tasks.

Each task is idempotent: re-running it on an already-processed document
is safe.  Worker crashes are recovered by the stale-lease recovery job.

Task lifecycle:
  PENDING → PROCESSING → (INDEXED | FAILED | RETRYING)

A task claims a job via a DB lease before doing any work.  If the worker
dies mid-task, the lease expires and another worker picks it up.
"""
from __future__ import annotations

import asyncio
import uuid
from datetime import datetime, timezone

from celery import Celery
from celery.utils.log import get_task_logger

from app.core.config import get_settings

settings = get_settings()

celery_app = Celery(
    "customer-assistant",
    broker=str(settings.redis_url),
    backend=str(settings.redis_url),
)

celery_app.conf.update(
    task_serializer="json",
    result_serializer="json",
    accept_content=["json"],
    task_acks_late=True,
    task_reject_on_worker_lost=True,
    worker_prefetch_multiplier=1,           # one task at a time per worker
    task_soft_time_limit=1800,              # 30 min soft limit
    task_time_limit=2100,                   # 35 min hard limit
    worker_max_tasks_per_child=settings.celery_max_tasks_per_child,
    task_routes={
        "app.workers.ingestion_tasks.process_document_task": {"queue": "ingestion"},
        "app.workers.ingestion_tasks.recover_stale_jobs_task": {"queue": "maintenance"},
    },
)

task_logger = get_task_logger(__name__)


@celery_app.task(
    name="app.workers.ingestion_tasks.process_document_task",
    bind=True,
    max_retries=3,
    default_retry_delay=60,
)
def process_document_task(
    self,
    document_id: str,
    job_id: str,
    tenant_id: str,
) -> dict:
    """Process a single document through the full ingestion pipeline."""
    return asyncio.get_event_loop().run_until_complete(
        _async_process_document(self, document_id, job_id, tenant_id)
    )


async def _async_process_document(
    task,
    document_id: str,
    job_id: str,
    tenant_id: str,
) -> dict:
    from app.storage.database import get_db_context
    from app.storage.orm import DocumentORM, IngestionJobORM
    from sqlalchemy import select, update

    doc_uuid = uuid.UUID(document_id)
    job_uuid = uuid.UUID(job_id)
    tenant_uuid = uuid.UUID(tenant_id)
    now = datetime.now(timezone.utc)

    task_logger.info(f"Starting ingestion: doc={document_id} job={job_id}")

    async with get_db_context() as db:
        # Claim job with a lease
        result = await db.execute(
            select(IngestionJobORM).where(IngestionJobORM.id == job_uuid)
        )
        job = result.scalar_one_or_none()
        if not job:
            task_logger.warning(f"Job {job_id} not found — skipping")
            return {"status": "skipped"}

        if job.status == "completed":
            task_logger.info(f"Job {job_id} already completed — idempotent skip")
            return {"status": "already_completed"}

        job.status = "processing"
        job.started_at = now
        job.attempt_count += 1
        job.lease_expires_at = datetime.fromtimestamp(
            now.timestamp() + 300, tz=timezone.utc
        )

        # Fetch document
        doc_result = await db.execute(
            select(DocumentORM).where(
                DocumentORM.id == doc_uuid,
                DocumentORM.tenant_id == tenant_uuid,
            )
        )
        doc = doc_result.scalar_one_or_none()
        if not doc:
            job.status = "failed"
            job.error_message = "Document record not found"
            return {"status": "failed", "reason": "document_not_found"}

        await db.execute(
            update(DocumentORM)
            .where(DocumentORM.id == doc_uuid)
            .values(status="processing", processing_started_at=now)
        )
        await db.flush()

    # ── Pipeline execution ────────────────────────────────────────────────
    try:
        from app.ingestion.pipeline import IngestionPipeline
        from app.storage.s3 import get_storage_backend

        storage = get_storage_backend()
        pipeline = IngestionPipeline(storage=storage)

        stats = await pipeline.run(
            document_id=doc_uuid,
            tenant_id=tenant_uuid,
            storage_uri=doc.storage_uri,
        )

        # Mark success
        async with get_db_context() as db:
            completed = datetime.now(timezone.utc)
            await db.execute(
                update(DocumentORM)
                .where(DocumentORM.id == doc_uuid)
                .values(
                    status="indexed",
                    page_count=stats.get("page_count"),
                    chunk_count=stats.get("chunk_count"),
                    processing_completed_at=completed,
                    parser_version=stats.get("parser_version"),
                    embedding_model=stats.get("embedding_model"),
                    error_message=None,
                )
            )
            result2 = await db.execute(
                select(IngestionJobORM).where(IngestionJobORM.id == job_uuid)
            )
            job2 = result2.scalar_one_or_none()
            if job2:
                job2.status = "completed"
                job2.completed_at = completed
                job2.lease_expires_at = None

        task_logger.info(
            f"Ingestion completed: doc={document_id} "
            f"pages={stats.get('page_count')} chunks={stats.get('chunk_count')}"
        )
        return {"status": "completed", **stats}

    except Exception as exc:
        task_logger.error(f"Ingestion failed: doc={document_id} error={exc}")

        async with get_db_context() as db:
            attempt = getattr(task, "request", None)
            retry_count = attempt.retries if attempt else 0
            final_failure = retry_count >= 2  # max_retries=3, 0-indexed

            new_status = "failed" if final_failure else "retrying"
            await db.execute(
                update(IngestionJobORM)
                .where(IngestionJobORM.id == job_uuid)
                .values(
                    status=new_status,
                    error_message=str(exc)[:500],
                    lease_expires_at=None,
                )
            )
            if final_failure:
                await db.execute(
                    update(DocumentORM)
                    .where(DocumentORM.id == doc_uuid)
                    .values(status="failed", error_message=str(exc)[:500])
                )

        if not (getattr(task, "request", None) and task.request.retries >= 2):
            raise task.retry(exc=exc, countdown=60 * (2 ** (task.request.retries + 1)))

        return {"status": "failed", "error": str(exc)}


@celery_app.task(name="app.workers.ingestion_tasks.recover_stale_jobs_task")
def recover_stale_jobs_task() -> dict:
    """Reset jobs whose lease expired without completion."""
    return asyncio.get_event_loop().run_until_complete(_async_recover_stale())


async def _async_recover_stale() -> dict:
    from app.storage.database import get_db_context
    from app.storage.orm import DocumentORM, IngestionJobORM
    from sqlalchemy import and_, update

    now = datetime.now(timezone.utc)
    async with get_db_context() as db:
        result = await db.execute(
            update(IngestionJobORM)
            .where(
                and_(
                    IngestionJobORM.status == "processing",
                    IngestionJobORM.lease_expires_at < now,
                    IngestionJobORM.attempt_count < IngestionJobORM.max_attempts,
                )
            )
            .values(status="pending", lease_expires_at=None)
            .returning(IngestionJobORM.id)
        )
        recovered = result.fetchall()

    count = len(recovered)
    if count:
        task_logger.warning(f"Recovered {count} stale ingestion jobs")
    return {"recovered": count}
