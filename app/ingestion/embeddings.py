"""Batched embedding generation and chunk indexing.

Responsibilities:
- Batch texts to respect OpenAI token limits
- Retry with exponential backoff and jitter on transient errors
- Insert chunks + embeddings into PostgreSQL atomically
- Skip chunks whose content_hash is already indexed (idempotent)
- Never log API keys or token content
"""
from __future__ import annotations

import asyncio
import random
import uuid
from typing import Any

from openai import APIStatusError, AsyncOpenAI, RateLimitError

from app.core.config import get_settings
from app.core.exceptions import EmbeddingError
from app.core.logging import get_logger

logger = get_logger(__name__)

_BATCH_SIZE = 100      # texts per OpenAI embeddings call
_MAX_RETRIES = 5


class EmbeddingBatcher:
    def __init__(self) -> None:
        self._settings = get_settings()
        self._client = AsyncOpenAI(
            api_key=self._settings.openai_api_key.get_secret_value(),
            timeout=self._settings.openai_timeout,
            max_retries=0,  # We handle retries ourselves
        )

    async def embed_and_index(
        self,
        chunks: list[dict[str, Any]],
        document_id: uuid.UUID,
        tenant_id: uuid.UUID,
    ) -> int:
        """Embed chunks in batches and write to the database.

        Returns the number of chunks successfully indexed.
        """
        if not chunks:
            return 0

        indexed = 0
        for batch_start in range(0, len(chunks), _BATCH_SIZE):
            batch = chunks[batch_start : batch_start + _BATCH_SIZE]
            texts = [c["content"] for c in batch]

            try:
                embeddings = await self._embed_with_retry(texts)
            except EmbeddingError as exc:
                logger.error(
                    "embeddings.batch_failed",
                    batch_start=batch_start,
                    batch_size=len(batch),
                    error=str(exc),
                )
                # Continue with remaining batches rather than aborting entirely
                continue

            indexed += await self._write_batch(batch, embeddings, document_id, tenant_id)

        return indexed

    async def _embed_with_retry(self, texts: list[str]) -> list[list[float]]:
        settings = self._settings
        for attempt in range(_MAX_RETRIES):
            try:
                response = await self._client.embeddings.create(
                    input=texts,
                    model=settings.openai_embedding_model,
                    dimensions=settings.openai_embedding_dimensions,
                )
                # Sort by index in case API returns out of order
                return [e.embedding for e in sorted(response.data, key=lambda x: x.index)]
            except RateLimitError as exc:
                if attempt == _MAX_RETRIES - 1:
                    raise EmbeddingError(f"Rate limit exceeded after {_MAX_RETRIES} retries") from exc
                wait = _backoff(attempt)
                logger.warning("embeddings.rate_limited", attempt=attempt, wait_seconds=wait)
                await asyncio.sleep(wait)
            except APIStatusError as exc:
                if exc.status_code and exc.status_code >= 500:
                    if attempt == _MAX_RETRIES - 1:
                        raise EmbeddingError(f"OpenAI server error: {exc}") from exc
                    await asyncio.sleep(_backoff(attempt))
                else:
                    raise EmbeddingError(f"OpenAI API error: {exc}") from exc
        raise EmbeddingError("Embedding failed after all retries")

    async def _write_batch(
        self,
        batch: list[dict],
        embeddings: list[list[float]],
        document_id: uuid.UUID,
        tenant_id: uuid.UUID,
    ) -> int:
        from sqlalchemy import select
        from app.storage.database import get_db_context
        from app.storage.orm import ChunkEmbeddingORM, ChunkORM

        settings = self._settings
        written = 0

        async with get_db_context() as db:
            for chunk_data, embedding in zip(batch, embeddings):
                # Idempotency: skip if content_hash already indexed
                existing = await db.execute(
                    select(ChunkORM).where(
                        ChunkORM.content_hash == chunk_data["content_hash"],
                        ChunkORM.document_id == document_id,
                    )
                )
                if existing.scalar_one_or_none():
                    continue

                chunk = ChunkORM(
                    id=uuid.uuid4(),
                    document_id=document_id,
                    tenant_id=tenant_id,
                    content=chunk_data["content"],
                    content_type=chunk_data.get("content_type", "text"),
                    content_hash=chunk_data["content_hash"],
                    page_start=chunk_data["page_start"],
                    page_end=chunk_data["page_end"],
                    chunk_index=chunk_data["chunk_index"],
                    token_count=chunk_data["token_count"],
                    section_path=chunk_data.get("section_path", []),
                    metadata_=chunk_data.get("metadata", {}),
                    embedding_model=settings.openai_embedding_model,
                    parser_version=chunk_data.get("parser_version"),
                )
                db.add(chunk)
                await db.flush()

                emb_record = ChunkEmbeddingORM(
                    chunk_id=chunk.id,
                    embedding_model=settings.openai_embedding_model,
                    embedding=embedding,
                )
                db.add(emb_record)
                written += 1

        return written


def _backoff(attempt: int) -> float:
    """Exponential backoff with full jitter."""
    return min(2 ** attempt * 1.0 + random.uniform(0, 1), 60.0)
