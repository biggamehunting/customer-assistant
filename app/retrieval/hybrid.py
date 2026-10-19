"""Hybrid retrieval: dense (pgvector cosine) + lexical (pg full-text) + RRF fusion.

Retrieval pipeline:
  1. Embed query with OpenAI
  2. Dense ANN search via pgvector
  3. Lexical BM25-like search via PostgreSQL tsvector
  4. Reciprocal Rank Fusion (RRF) to merge ranked lists
  5. Deduplication by chunk_id
  6. Optional score threshold filter
"""
from __future__ import annotations

import uuid
from typing import Any

from openai import AsyncOpenAI
from sqlalchemy import select, text
from sqlalchemy.ext.asyncio import AsyncSession

from app.core.config import get_settings
from app.core.logging import get_logger
from app.core.telemetry import span
from app.domain.models import ContentType, RetrievedPassage
from app.storage.database import get_db_context
from app.storage.orm import ChunkEmbeddingORM, ChunkORM, DocumentORM

logger = get_logger(__name__)

_RRF_K = 60  # RRF rank constant


class HybridRetriever:
    def __init__(self) -> None:
        self._settings = get_settings()

    async def retrieve(
        self,
        query: str,
        tenant_id: uuid.UUID,
        *,
        top_k: int | None = None,
        document_ids: list[uuid.UUID] | None = None,
        filters: dict[str, Any] | None = None,
    ) -> list[RetrievedPassage]:
        settings = self._settings
        top_k = top_k or settings.retrieval_top_k

        with span("retrieval.hybrid", query_len=len(query), top_k=top_k):
            query_embedding = await self._embed_query(query)

            async with get_db_context() as db:
                dense = await self._dense_search(
                    db, query_embedding, tenant_id, top_k=top_k, document_ids=document_ids
                )
                lexical = await self._lexical_search(
                    db, query, tenant_id, top_k=top_k, document_ids=document_ids
                )

        fused = self._rrf_fuse(dense, lexical, top_k=top_k)
        logger.info(
            "retrieval.completed",
            dense_count=len(dense),
            lexical_count=len(lexical),
            fused_count=len(fused),
        )
        return fused

    async def _embed_query(self, query: str) -> list[float]:
        settings = self._settings
        with span("retrieval.embed_query"):
            client = AsyncOpenAI(
                api_key=settings.openai_api_key.get_secret_value(),
                timeout=settings.openai_timeout,
                max_retries=settings.openai_max_retries,
            )
            response = await client.embeddings.create(
                input=query,
                model=settings.openai_embedding_model,
                dimensions=settings.openai_embedding_dimensions,
            )
            return response.data[0].embedding

    async def _dense_search(
        self,
        db: AsyncSession,
        embedding: list[float],
        tenant_id: uuid.UUID,
        top_k: int,
        document_ids: list[uuid.UUID] | None,
    ) -> list[RetrievedPassage]:
        """pgvector cosine similarity search."""
        with span("retrieval.dense"):
            embedding_str = "[" + ",".join(str(v) for v in embedding) + "]"

            filters = [
                f"c.tenant_id = '{tenant_id}'",
                "ce.embedding IS NOT NULL",
            ]
            if document_ids:
                ids = ",".join(f"'{d}'" for d in document_ids)
                filters.append(f"c.document_id IN ({ids})")

            where_clause = " AND ".join(filters)

            sql = text(f"""
                SELECT
                    c.id            AS chunk_id,
                    c.document_id,
                    c.tenant_id,
                    c.content,
                    c.content_type,
                    c.page_start,
                    c.page_end,
                    c.section_path,
                    d.title         AS document_title,
                    1 - (ce.embedding <=> '{embedding_str}'::vector) AS score
                FROM chunks c
                JOIN chunk_embeddings ce ON ce.chunk_id = c.id
                JOIN documents d ON d.id = c.document_id
                WHERE {where_clause}
                ORDER BY ce.embedding <=> '{embedding_str}'::vector
                LIMIT :top_k
            """)

            result = await db.execute(sql, {"top_k": top_k})
            rows = result.fetchall()

            return [
                RetrievedPassage(
                    chunk_id=row.chunk_id,
                    document_id=row.document_id,
                    tenant_id=row.tenant_id,
                    content=row.content,
                    content_type=ContentType(row.content_type),
                    page_start=row.page_start,
                    page_end=row.page_end,
                    section_path=row.section_path or [],
                    document_title=row.document_title or "Unknown",
                    dense_score=float(row.score),
                )
                for row in rows
            ]

    async def _lexical_search(
        self,
        db: AsyncSession,
        query: str,
        tenant_id: uuid.UUID,
        top_k: int,
        document_ids: list[uuid.UUID] | None,
    ) -> list[RetrievedPassage]:
        """PostgreSQL full-text search (tsvector / plainto_tsquery)."""
        with span("retrieval.lexical"):
            filters = [f"c.tenant_id = '{tenant_id}'"]
            if document_ids:
                ids = ",".join(f"'{d}'" for d in document_ids)
                filters.append(f"c.document_id IN ({ids})")
            where_clause = " AND ".join(filters)

            sql = text(f"""
                SELECT
                    c.id            AS chunk_id,
                    c.document_id,
                    c.tenant_id,
                    c.content,
                    c.content_type,
                    c.page_start,
                    c.page_end,
                    c.section_path,
                    d.title         AS document_title,
                    ts_rank_cd(
                        to_tsvector('english', c.content),
                        plainto_tsquery('english', :query)
                    ) AS score
                FROM chunks c
                JOIN documents d ON d.id = c.document_id
                WHERE {where_clause}
                  AND to_tsvector('english', c.content) @@
                      plainto_tsquery('english', :query)
                ORDER BY score DESC
                LIMIT :top_k
            """)

            result = await db.execute(sql, {"query": query, "top_k": top_k})
            rows = result.fetchall()

            return [
                RetrievedPassage(
                    chunk_id=row.chunk_id,
                    document_id=row.document_id,
                    tenant_id=row.tenant_id,
                    content=row.content,
                    content_type=ContentType(row.content_type),
                    page_start=row.page_start,
                    page_end=row.page_end,
                    section_path=row.section_path or [],
                    document_title=row.document_title or "Unknown",
                    lexical_score=float(row.score),
                )
                for row in rows
            ]

    def _rrf_fuse(
        self,
        dense: list[RetrievedPassage],
        lexical: list[RetrievedPassage],
        top_k: int,
    ) -> list[RetrievedPassage]:
        """Reciprocal Rank Fusion — combines two ranked lists."""
        settings = self._settings
        scores: dict[uuid.UUID, float] = {}
        passages: dict[uuid.UUID, RetrievedPassage] = {}

        for rank, p in enumerate(dense, 1):
            scores[p.chunk_id] = scores.get(p.chunk_id, 0.0) + (
                settings.retrieval_dense_weight / (_RRF_K + rank)
            )
            passages[p.chunk_id] = p

        for rank, p in enumerate(lexical, 1):
            scores[p.chunk_id] = scores.get(p.chunk_id, 0.0) + (
                settings.retrieval_lexical_weight / (_RRF_K + rank)
            )
            if p.chunk_id not in passages:
                passages[p.chunk_id] = p
            else:
                passages[p.chunk_id].lexical_score = p.lexical_score

        ranked = sorted(scores.items(), key=lambda x: x[1], reverse=True)[:top_k]

        result: list[RetrievedPassage] = []
        for chunk_id, fused_score in ranked:
            if fused_score < settings.retrieval_score_threshold:
                continue
            p = passages[chunk_id]
            p.fused_score = fused_score
            result.append(p)

        return result
