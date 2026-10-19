"""Document upload, listing, status, and management endpoints."""
from __future__ import annotations

import uuid
from datetime import datetime, timezone
from typing import Annotated

import xxhash
from fastapi import APIRouter, File, HTTPException, Query, UploadFile, status
from sqlalchemy import select, update

from app.api.dependencies import AuthUser, DbSession, Storage
from app.core.config import get_settings
from app.core.exceptions import (
    DuplicateDocumentError,
    FileTooLargeError,
    InvalidFileTypeError,
)
from app.core.logging import get_logger
from app.domain.models import DocumentStatus, IngestionJobStatus
from app.domain.schemas import (
    DocumentResponse,
    DocumentStatusResponse,
    DocumentUploadResponse,
    IngestionJobResponse,
    PaginatedResponse,
)
from app.storage.orm import DocumentORM, IngestionJobORM

router = APIRouter(prefix="/api/v1/documents", tags=["documents"])
logger = get_logger(__name__)

ALLOWED_MIME_TYPES = {"application/pdf", "application/x-pdf"}


@router.post("", response_model=DocumentUploadResponse, status_code=status.HTTP_201_CREATED)
async def upload_document(
    file: Annotated[UploadFile, File(description="PDF file to ingest")],
    db: DbSession,
    current_user: AuthUser,
    storage: Storage,
    title: str | None = None,
    category: str | None = None,
    description: str | None = None,
) -> DocumentUploadResponse:
    """Upload a single PDF for asynchronous ingestion."""
    settings = get_settings()

    # Validate content type
    content_type = file.content_type or ""
    if not (content_type in ALLOWED_MIME_TYPES or (file.filename or "").lower().endswith(".pdf")):
        raise InvalidFileTypeError(
            f"Only PDF files are accepted, got: {content_type}",
            detail=f"Filename: {file.filename}",
        )

    # Read with size limit
    data = await file.read(settings.max_upload_size_bytes + 1)
    if len(data) > settings.max_upload_size_bytes:
        raise FileTooLargeError(
            f"File exceeds {settings.max_upload_size_mb} MB limit",
            detail=f"Size: {len(data)} bytes",
        )

    # Compute content hash for deduplication
    content_hash = xxhash.xxh3_128(data).hexdigest()

    # Duplicate check within tenant
    existing = await db.execute(
        select(DocumentORM).where(
            DocumentORM.content_hash == content_hash,
            DocumentORM.tenant_id == current_user.tenant_id,
        )
    )
    if existing.scalar_one_or_none():
        raise DuplicateDocumentError(
            "A document with identical content already exists in this workspace",
            detail=f"Content hash: {content_hash}",
        )

    # Upload to storage
    doc_id = uuid.uuid4()
    storage_key = f"documents/{current_user.tenant_id}/{doc_id}/{file.filename}"
    await storage.upload(storage_key, data, content_type="application/pdf")

    # Persist document record
    now = datetime.now(timezone.utc)
    doc = DocumentORM(
        id=doc_id,
        tenant_id=current_user.tenant_id,
        uploaded_by=current_user.user_id,
        filename=file.filename or "document.pdf",
        title=title,
        description=description,
        category=category,
        storage_uri=f"documents/{current_user.tenant_id}/{doc_id}/{file.filename}",
        content_hash=content_hash,
        file_size_bytes=len(data),
        mime_type="application/pdf",
        status=DocumentStatus.queued.value,
        created_at=now,
        updated_at=now,
    )
    db.add(doc)

    # Create ingestion job
    job_id = uuid.uuid4()
    job = IngestionJobORM(
        id=job_id,
        document_id=doc_id,
        tenant_id=current_user.tenant_id,
        status=IngestionJobStatus.pending.value,
        created_at=now,
        updated_at=now,
    )
    db.add(job)
    await db.flush()

    # Enqueue Celery task (import here to avoid circular imports)
    try:
        from app.workers.ingestion_tasks import process_document_task
        process_document_task.delay(str(doc_id), str(job_id), str(current_user.tenant_id))
    except Exception as exc:
        logger.warning("documents.enqueue_failed", doc_id=str(doc_id), error=str(exc))
        # Job will be picked up by the worker's polling loop if Celery is unavailable

    logger.info(
        "documents.uploaded",
        doc_id=str(doc_id),
        filename=file.filename,
        size_bytes=len(data),
        tenant_id=str(current_user.tenant_id),
    )

    return DocumentUploadResponse(
        document_id=doc_id,
        filename=file.filename or "document.pdf",
        status=DocumentStatus.queued,
        content_hash=content_hash,
        file_size_bytes=len(data),
        job_id=job_id,
    )


@router.get("", response_model=PaginatedResponse)
async def list_documents(
    db: DbSession,
    current_user: AuthUser,
    limit: int = Query(default=20, ge=1, le=100),
    offset: int = Query(default=0, ge=0),
    status_filter: str | None = Query(default=None, alias="status"),
) -> PaginatedResponse:
    """List documents for the current tenant."""
    query = select(DocumentORM).where(DocumentORM.tenant_id == current_user.tenant_id)
    if status_filter:
        query = query.where(DocumentORM.status == status_filter)

    count_query = select(DocumentORM).where(DocumentORM.tenant_id == current_user.tenant_id)
    if status_filter:
        count_query = count_query.where(DocumentORM.status == status_filter)

    result = await db.execute(query.order_by(DocumentORM.created_at.desc()).limit(limit).offset(offset))
    docs = result.scalars().all()

    count_result = await db.execute(count_query)
    total = len(count_result.scalars().all())

    return PaginatedResponse(
        total=total,
        limit=limit,
        offset=offset,
        items=[_doc_to_response(d) for d in docs],
    )


@router.get("/{document_id}", response_model=DocumentResponse)
async def get_document(
    document_id: uuid.UUID,
    db: DbSession,
    current_user: AuthUser,
) -> DocumentResponse:
    """Get a single document by ID."""
    result = await db.execute(
        select(DocumentORM).where(
            DocumentORM.id == document_id,
            DocumentORM.tenant_id == current_user.tenant_id,
        )
    )
    doc = result.scalar_one_or_none()
    if not doc:
        raise HTTPException(status_code=404, detail="Document not found")
    return _doc_to_response(doc)


@router.get("/{document_id}/status", response_model=DocumentStatusResponse)
async def get_document_status(
    document_id: uuid.UUID,
    db: DbSession,
    current_user: AuthUser,
) -> DocumentStatusResponse:
    """Poll processing status for a document."""
    result = await db.execute(
        select(DocumentORM).where(
            DocumentORM.id == document_id,
            DocumentORM.tenant_id == current_user.tenant_id,
        )
    )
    doc = result.scalar_one_or_none()
    if not doc:
        raise HTTPException(status_code=404, detail="Document not found")

    return DocumentStatusResponse(
        document_id=doc.id,
        filename=doc.filename,
        status=doc.status,
        error_message=doc.error_message,
        updated_at=doc.updated_at,
    )


@router.delete("/{document_id}", status_code=status.HTTP_204_NO_CONTENT, response_model=None)
async def delete_document(
    document_id: uuid.UUID,
    db: DbSession,
    current_user: AuthUser,
    storage: Storage,
) -> None:
    """Delete a document and all its chunks and embeddings."""
    result = await db.execute(
        select(DocumentORM).where(
            DocumentORM.id == document_id,
            DocumentORM.tenant_id == current_user.tenant_id,
        )
    )
    doc = result.scalar_one_or_none()
    if not doc:
        raise HTTPException(status_code=404, detail="Document not found")

    # Delete from storage (best-effort)
    try:
        await storage.delete(doc.storage_uri)
    except Exception as exc:
        logger.warning("documents.storage_delete_failed", doc_id=str(document_id), error=str(exc))

    # Cascade delete handles chunks and embeddings via FK constraints
    await db.delete(doc)
    logger.info("documents.deleted", doc_id=str(document_id), tenant_id=str(current_user.tenant_id))


@router.post("/{document_id}/reprocess", response_model=IngestionJobResponse, status_code=status.HTTP_202_ACCEPTED)
async def reprocess_document(
    document_id: uuid.UUID,
    db: DbSession,
    current_user: AuthUser,
) -> IngestionJobResponse:
    """Queue a document for reprocessing (e.g. after config changes)."""
    result = await db.execute(
        select(DocumentORM).where(
            DocumentORM.id == document_id,
            DocumentORM.tenant_id == current_user.tenant_id,
        )
    )
    doc = result.scalar_one_or_none()
    if not doc:
        raise HTTPException(status_code=404, detail="Document not found")

    now = datetime.now(timezone.utc)
    job = IngestionJobORM(
        id=uuid.uuid4(),
        document_id=document_id,
        tenant_id=current_user.tenant_id,
        status=IngestionJobStatus.pending.value,
        created_at=now,
        updated_at=now,
    )
    db.add(job)
    await db.execute(
        update(DocumentORM)
        .where(DocumentORM.id == document_id)
        .values(status=DocumentStatus.queued.value, updated_at=now)
    )
    await db.flush()

    try:
        from app.workers.ingestion_tasks import process_document_task
        process_document_task.delay(str(document_id), str(job.id), str(current_user.tenant_id))
    except Exception as exc:
        logger.warning("documents.reprocess_enqueue_failed", doc_id=str(document_id), error=str(exc))

    return IngestionJobResponse(
        id=job.id,
        document_id=document_id,
        status=job.status,
        attempt_count=0,
        created_at=now,
        updated_at=now,
    )


def _doc_to_response(doc: DocumentORM) -> DocumentResponse:
    duration = None
    if doc.processing_started_at and doc.processing_completed_at:
        duration = (doc.processing_completed_at - doc.processing_started_at).total_seconds()
    return DocumentResponse(
        id=doc.id,
        filename=doc.filename,
        title=doc.title,
        status=doc.status,
        page_count=doc.page_count,
        chunk_count=doc.chunk_count,
        file_size_bytes=doc.file_size_bytes,
        created_at=doc.created_at,
        updated_at=doc.updated_at,
        error_message=doc.error_message,
        processing_duration_seconds=duration,
    )
