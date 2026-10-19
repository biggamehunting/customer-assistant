"""Pydantic v2 API request and response schemas.

These are separate from domain models so the API contract can evolve
independently of the internal domain representation.
"""
from __future__ import annotations

from datetime import datetime
from enum import Enum
from typing import Any
from uuid import UUID

from pydantic import BaseModel, ConfigDict, Field, field_validator


# ── Shared base ───────────────────────────────────────────────────────────────

class APIModel(BaseModel):
    model_config = ConfigDict(from_attributes=True, populate_by_name=True)


# ── Health ────────────────────────────────────────────────────────────────────

class HealthStatus(str, Enum):
    ok = "ok"
    degraded = "degraded"
    down = "down"


class DependencyHealth(APIModel):
    name: str
    status: HealthStatus
    latency_ms: float | None = None
    detail: str | None = None


class HealthResponse(APIModel):
    status: HealthStatus
    version: str
    environment: str
    dependencies: list[DependencyHealth] = Field(default_factory=list)


# ── Auth ──────────────────────────────────────────────────────────────────────

class TokenResponse(APIModel):
    access_token: str
    refresh_token: str
    token_type: str = "bearer"
    expires_in: int


class LoginRequest(APIModel):
    email: str
    password: str


# ── Pagination ────────────────────────────────────────────────────────────────

class PaginatedResponse(APIModel):
    total: int
    limit: int
    offset: int
    items: list[Any]


# ── Documents ─────────────────────────────────────────────────────────────────

class DocumentStatus(str, Enum):
    pending = "pending"
    queued = "queued"
    processing = "processing"
    indexed = "indexed"
    failed = "failed"
    cancelled = "cancelled"


class DocumentUploadResponse(APIModel):
    document_id: UUID
    filename: str
    status: DocumentStatus
    content_hash: str
    file_size_bytes: int
    job_id: UUID
    message: str = "Document queued for processing"


class DocumentResponse(APIModel):
    id: UUID
    filename: str
    title: str | None
    status: DocumentStatus
    page_count: int | None
    chunk_count: int | None
    file_size_bytes: int
    created_at: datetime
    updated_at: datetime
    error_message: str | None = None
    processing_duration_seconds: float | None = None


class DocumentStatusResponse(APIModel):
    document_id: UUID
    filename: str
    status: DocumentStatus
    progress_pct: float | None = None
    pages_processed: int | None = None
    total_pages: int | None = None
    error_message: str | None = None
    updated_at: datetime


# ── Ingestion Jobs ────────────────────────────────────────────────────────────

class JobStatus(str, Enum):
    pending = "pending"
    processing = "processing"
    retrying = "retrying"
    completed = "completed"
    failed = "failed"
    cancelled = "cancelled"


class IngestionJobResponse(APIModel):
    id: UUID
    document_id: UUID
    status: JobStatus
    attempt_count: int
    created_at: datetime
    updated_at: datetime
    error_message: str | None = None
    started_at: datetime | None = None
    completed_at: datetime | None = None


# ── Citations ─────────────────────────────────────────────────────────────────

class CitationResponse(APIModel):
    document_id: UUID
    document_title: str
    page_start: int
    page_end: int
    chunk_id: UUID
    passage: str
    section_path: list[str] = Field(default_factory=list)
    relevance_score: float
    view_url: str | None = None


# ── Chat ──────────────────────────────────────────────────────────────────────

class ChatRequest(APIModel):
    message: str = Field(min_length=1, max_length=4000)
    conversation_id: UUID | None = None
    document_ids: list[UUID] | None = None  # restrict retrieval to these docs
    include_charts: bool = True

    @field_validator("message")
    @classmethod
    def strip_message(cls, v: str) -> str:
        return v.strip()


class StreamEventType(str, Enum):
    status = "status"
    token = "token"
    citations = "citations"
    chart = "chart"
    done = "done"
    error = "error"


class ChatResponse(APIModel):
    conversation_id: UUID
    message_id: UUID
    answer: str
    citations: list[CitationResponse] = Field(default_factory=list)
    chart_id: UUID | None = None
    abstained: bool = False
    abstention_reason: str | None = None
    token_usage: dict[str, int] | None = None
    latency_ms: int | None = None


# ── Conversations ─────────────────────────────────────────────────────────────

class ConversationResponse(APIModel):
    id: UUID
    title: str | None
    created_at: datetime
    updated_at: datetime
    message_count: int = 0


class MessageResponse(APIModel):
    id: UUID
    role: str
    content: str
    created_at: datetime
    citations: list[CitationResponse] = Field(default_factory=list)
    chart_id: UUID | None = None


# ── Analytics ─────────────────────────────────────────────────────────────────

class AnalyticsQueryRequest(APIModel):
    query: str = Field(min_length=1, max_length=2000)
    conversation_id: UUID | None = None
    document_ids: list[UUID] | None = None


class ChartResponse(APIModel):
    id: UUID
    chart_type: str
    title: str
    created_at: datetime
    description: str | None = None
    image_url: str | None = None
    csv_url: str | None = None
    citations: list[CitationResponse] = Field(default_factory=list)


# ── Admin ─────────────────────────────────────────────────────────────────────

class MetricsResponse(APIModel):
    total_documents: int
    indexed_documents: int
    failed_documents: int
    total_conversations: int
    total_messages: int
    pending_jobs: int
    failed_jobs: int
    storage_bytes: int | None = None


# ── Error responses ───────────────────────────────────────────────────────────

class ErrorResponse(APIModel):
    error_code: str
    message: str
    detail: str | None = None
    request_id: str | None = None
