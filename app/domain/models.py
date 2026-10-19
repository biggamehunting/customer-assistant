"""Domain models — plain Python dataclasses with no ORM or infrastructure imports.

These are the canonical representations of the domain objects that flow through
the application.  Infrastructure layers (SQLAlchemy, Pydantic API schemas) map
to/from these classes.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from uuid import UUID


# ── Enumerations ──────────────────────────────────────────────────────────────

class DocumentStatus(str, Enum):
    pending = "pending"
    queued = "queued"
    processing = "processing"
    indexed = "indexed"
    failed = "failed"
    cancelled = "cancelled"


class IngestionJobStatus(str, Enum):
    pending = "pending"
    processing = "processing"
    retrying = "retrying"
    completed = "completed"
    failed = "failed"
    cancelled = "cancelled"


class ContentType(str, Enum):
    text = "text"
    table = "table"
    figure = "figure"
    mixed = "mixed"


class MessageRole(str, Enum):
    user = "user"
    assistant = "assistant"
    system = "system"


class UserRole(str, Enum):
    customer = "customer"
    admin = "admin"
    super_admin = "super_admin"


class ChartType(str, Enum):
    bar = "bar"
    grouped_bar = "grouped_bar"
    line = "line"
    pie = "pie"
    scatter = "scatter"
    histogram = "histogram"
    table = "table"


# ── Core domain objects ───────────────────────────────────────────────────────

@dataclass
class Tenant:
    id: UUID
    name: str
    slug: str
    is_active: bool
    created_at: datetime
    max_documents: int = 10_000
    max_storage_bytes: int = 50 * 1024 * 1024 * 1024  # 50 GB


@dataclass
class User:
    id: UUID
    tenant_id: UUID
    email: str
    role: UserRole
    is_active: bool
    created_at: datetime
    full_name: str | None = None


@dataclass
class Document:
    id: UUID
    tenant_id: UUID
    filename: str
    storage_uri: str
    content_hash: str
    file_size_bytes: int
    status: DocumentStatus
    uploaded_by: UUID
    created_at: datetime
    updated_at: datetime
    page_count: int | None = None
    chunk_count: int | None = None
    mime_type: str = "application/pdf"
    title: str | None = None
    description: str | None = None
    category: str | None = None
    version: str | None = None
    language: str | None = None
    parser_version: str | None = None
    embedding_model: str | None = None
    error_message: str | None = None
    processing_started_at: datetime | None = None
    processing_completed_at: datetime | None = None


@dataclass
class DocumentChunk:
    id: UUID
    document_id: UUID
    tenant_id: UUID
    content: str
    content_type: ContentType
    page_start: int
    page_end: int
    chunk_index: int
    content_hash: str
    token_count: int
    section_path: list[str] = field(default_factory=list)
    metadata: dict = field(default_factory=dict)
    embedding_model: str | None = None
    parser_version: str | None = None


@dataclass
class ExtractedTable:
    id: UUID
    document_id: UUID
    tenant_id: UUID
    page_start: int
    page_end: int
    caption: str | None
    headers: list[str]
    rows: list[list[str]]
    extraction_method: str
    confidence: float
    section_context: str | None = None


@dataclass
class Citation:
    document_id: UUID
    document_title: str
    page_start: int
    page_end: int
    chunk_id: UUID
    passage: str
    section_path: list[str] = field(default_factory=list)
    relevance_score: float = 0.0
    view_url: str | None = None


@dataclass
class RetrievedPassage:
    chunk_id: UUID
    document_id: UUID
    tenant_id: UUID
    content: str
    content_type: ContentType
    page_start: int
    page_end: int
    section_path: list[str]
    document_title: str
    dense_score: float = 0.0
    lexical_score: float = 0.0
    fused_score: float = 0.0
    rerank_score: float | None = None


@dataclass
class Conversation:
    id: UUID
    tenant_id: UUID
    user_id: UUID
    created_at: datetime
    updated_at: datetime
    title: str | None = None
    is_active: bool = True


@dataclass
class Message:
    id: UUID
    conversation_id: UUID
    role: MessageRole
    content: str
    created_at: datetime
    citations: list[Citation] = field(default_factory=list)
    chart_id: UUID | None = None
    token_usage: dict | None = None
    latency_ms: int | None = None


@dataclass
class IngestionJob:
    id: UUID
    document_id: UUID
    tenant_id: UUID
    status: IngestionJobStatus
    created_at: datetime
    updated_at: datetime
    attempt_count: int = 0
    max_attempts: int = 3
    error_message: str | None = None
    started_at: datetime | None = None
    completed_at: datetime | None = None
    lease_expires_at: datetime | None = None


@dataclass
class ChartArtifact:
    id: UUID
    conversation_id: UUID
    tenant_id: UUID
    chart_type: ChartType
    title: str
    storage_uri: str
    created_at: datetime
    source_citations: list[Citation] = field(default_factory=list)
    csv_uri: str | None = None
    description: str | None = None
