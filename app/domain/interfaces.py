"""Service interfaces (abstract base classes).

Concrete implementations live in app/storage/, app/ingestion/, etc.
This inversion keeps domain logic independent of infrastructure.
"""
from __future__ import annotations

from abc import ABC, abstractmethod
from typing import AsyncIterator
from uuid import UUID

from app.domain.models import (
    ChartArtifact,
    Conversation,
    Document,
    DocumentChunk,
    ExtractedTable,
    IngestionJob,
    Message,
    RetrievedPassage,
    Tenant,
    User,
)


class StorageBackend(ABC):
    """Object storage: S3 in production, local filesystem in development."""

    @abstractmethod
    async def upload(
        self, key: str, data: bytes, content_type: str = "application/octet-stream"
    ) -> str:
        """Upload bytes and return the storage URI."""

    @abstractmethod
    async def download(self, key: str) -> bytes:
        """Download and return bytes for the given key."""

    @abstractmethod
    async def delete(self, key: str) -> None:
        """Delete the object at key."""

    @abstractmethod
    async def presigned_url(self, key: str, expires_seconds: int = 3600) -> str:
        """Return a time-limited pre-signed download URL."""

    @abstractmethod
    async def exists(self, key: str) -> bool:
        """Return True if the key exists."""


class OCRBackend(ABC):
    """OCR abstraction: AWS Textract in production, Tesseract in development."""

    @abstractmethod
    async def extract_text(
        self, image_bytes: bytes, *, mime_type: str = "image/png"
    ) -> tuple[str, float]:
        """Return (extracted_text, confidence_score 0–100)."""

    @abstractmethod
    async def extract_tables(
        self, image_bytes: bytes, *, mime_type: str = "image/png"
    ) -> list[dict]:
        """Return a list of table dicts with headers, rows, confidence."""


class EmbeddingService(ABC):
    """Generate dense vector embeddings for text chunks."""

    @abstractmethod
    async def embed_texts(self, texts: list[str]) -> list[list[float]]:
        """Embed a batch of texts. Retries on transient errors."""

    @abstractmethod
    async def embed_query(self, text: str) -> list[float]:
        """Embed a single query string."""

    @property
    @abstractmethod
    def dimensions(self) -> int:
        """Embedding vector dimension."""


class VectorStore(ABC):
    """Dense retrieval using pgvector."""

    @abstractmethod
    async def upsert(self, chunk: DocumentChunk, embedding: list[float]) -> None:
        """Insert or replace a chunk+embedding."""

    @abstractmethod
    async def query(
        self,
        embedding: list[float],
        *,
        tenant_id: UUID,
        top_k: int = 20,
        filters: dict | None = None,
    ) -> list[RetrievedPassage]:
        """ANN search returning ranked passages."""

    @abstractmethod
    async def delete_document(self, document_id: UUID) -> None:
        """Remove all vectors for a document."""


class RerankerService(ABC):
    """Cross-encoder reranking of retrieved passages."""

    @abstractmethod
    async def rerank(
        self,
        query: str,
        passages: list[RetrievedPassage],
        *,
        top_k: int,
    ) -> list[RetrievedPassage]:
        """Return top_k passages sorted by rerank score (descending)."""


class DocumentRepository(ABC):
    @abstractmethod
    async def create(self, document: Document) -> Document: ...

    @abstractmethod
    async def get(self, document_id: UUID, tenant_id: UUID) -> Document | None: ...

    @abstractmethod
    async def update_status(
        self,
        document_id: UUID,
        *,
        status: str,
        error_message: str | None = None,
        **kwargs,
    ) -> None: ...

    @abstractmethod
    async def list_for_tenant(
        self, tenant_id: UUID, *, limit: int = 50, offset: int = 0
    ) -> list[Document]: ...

    @abstractmethod
    async def delete(self, document_id: UUID, tenant_id: UUID) -> None: ...

    @abstractmethod
    async def find_by_content_hash(
        self, content_hash: str, tenant_id: UUID
    ) -> Document | None: ...


class ConversationRepository(ABC):
    @abstractmethod
    async def create(self, conversation: Conversation) -> Conversation: ...

    @abstractmethod
    async def get(
        self, conversation_id: UUID, tenant_id: UUID
    ) -> Conversation | None: ...

    @abstractmethod
    async def list_for_user(
        self, user_id: UUID, tenant_id: UUID, *, limit: int = 20, offset: int = 0
    ) -> list[Conversation]: ...

    @abstractmethod
    async def delete(self, conversation_id: UUID, tenant_id: UUID) -> None: ...


class MessageRepository(ABC):
    @abstractmethod
    async def create(self, message: Message) -> Message: ...

    @abstractmethod
    async def list_for_conversation(
        self, conversation_id: UUID, *, limit: int = 50, offset: int = 0
    ) -> list[Message]: ...


class IngestionJobRepository(ABC):
    @abstractmethod
    async def create(self, job: IngestionJob) -> IngestionJob: ...

    @abstractmethod
    async def get(self, job_id: UUID) -> IngestionJob | None: ...

    @abstractmethod
    async def claim_pending(
        self, *, worker_id: str, lease_seconds: int = 300
    ) -> IngestionJob | None:
        """Atomically claim one pending job.  Returns None if queue empty."""

    @abstractmethod
    async def update_status(
        self,
        job_id: UUID,
        *,
        status: str,
        error_message: str | None = None,
    ) -> None: ...

    @abstractmethod
    async def refresh_lease(
        self, job_id: UUID, *, lease_seconds: int = 300
    ) -> None: ...

    @abstractmethod
    async def recover_stale_jobs(self) -> int:
        """Reset jobs whose lease expired without completion.  Returns count."""
