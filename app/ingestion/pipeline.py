"""Main ingestion pipeline.

Orchestrates:
  1. Download PDF from storage
  2. PDF classification (text vs scanned vs mixed, page level)
  3. Text extraction (PyMuPDF for digital, OCR for scanned pages)
  4. Table extraction
  5. Structure-aware chunking
  6. Embedding (OpenAI, batched)
  7. Indexing (chunks + embeddings → PostgreSQL + pgvector)

Designed to handle 5,000+ documents without loading all content into memory.
Each page is processed and discarded before the next begins.
"""
from __future__ import annotations

import hashlib
import uuid
from dataclasses import dataclass, field
from typing import Any

import fitz  # PyMuPDF

from app.core.config import get_settings
from app.core.logging import get_logger
from app.domain.interfaces import StorageBackend
from app.ingestion.chunker import Chunker
from app.ingestion.embeddings import EmbeddingBatcher
from app.ingestion.ocr import OCRRouter
from app.ingestion.table_extractor import TableExtractor

logger = get_logger(__name__)

PARSER_VERSION = "1.0.0"


@dataclass
class PageExtractionResult:
    page_number: int
    text: str
    extraction_method: str  # "digital" | "ocr" | "skipped"
    confidence: float = 1.0
    tables: list[dict] = field(default_factory=list)
    is_scanned: bool = False


class IngestionPipeline:
    def __init__(self, storage: StorageBackend) -> None:
        self._storage = storage
        self._settings = get_settings()

    async def run(
        self,
        document_id: uuid.UUID,
        tenant_id: uuid.UUID,
        storage_uri: str,
    ) -> dict[str, Any]:
        """Process a document and return ingestion stats."""
        logger.info("pipeline.start", document_id=str(document_id))

        pdf_bytes = await self._storage.download(storage_uri)
        doc = fitz.open(stream=pdf_bytes, filetype="pdf")
        page_count = len(doc)

        if page_count > self._settings.max_pages_per_document:
            raise ValueError(
                f"Document has {page_count} pages, "
                f"exceeding limit of {self._settings.max_pages_per_document}"
            )

        ocr_router = OCRRouter()
        table_extractor = TableExtractor()
        chunker = Chunker(
            chunk_size_tokens=self._settings.chunk_size_tokens,
            overlap_tokens=self._settings.chunk_overlap_tokens,
        )
        embedder = EmbeddingBatcher()

        all_chunks: list[dict] = []
        total_tables: list[dict] = []

        for page_num in range(page_count):
            page = doc[page_num]
            result = await self._process_page(page, page_num + 1, ocr_router, table_extractor)
            if result.text.strip():
                page_chunks = chunker.chunk_page(
                    text=result.text,
                    document_id=document_id,
                    tenant_id=tenant_id,
                    page_number=result.page_number,
                    extraction_method=result.extraction_method,
                )
                all_chunks.extend(page_chunks)
            total_tables.extend(result.tables)

        doc.close()

        # Batch embed and index
        chunk_count = await embedder.embed_and_index(all_chunks, document_id=document_id, tenant_id=tenant_id)

        # Index extracted tables
        if total_tables:
            await self._index_tables(total_tables, document_id=document_id, tenant_id=tenant_id)

        stats = {
            "page_count": page_count,
            "chunk_count": chunk_count,
            "table_count": len(total_tables),
            "parser_version": PARSER_VERSION,
            "embedding_model": self._settings.openai_embedding_model,
        }
        logger.info("pipeline.completed", document_id=str(document_id), **stats)
        return stats

    async def _process_page(
        self,
        page: fitz.Page,
        page_number: int,
        ocr_router: OCRRouter,
        table_extractor: TableExtractor,
    ) -> PageExtractionResult:
        """Extract text from a single page, routing to OCR if needed."""
        # Attempt digital text extraction first
        text = page.get_text("text")
        char_count = len(text.strip())
        word_count = len(text.split())

        # Heuristic: fewer than 50 words on a page with a non-trivial area → scanned
        page_area = page.rect.width * page.rect.height
        is_likely_scanned = (word_count < 50) and (page_area > 100_000)

        if is_likely_scanned or char_count < 100:
            # Render page to image and OCR
            mat = fitz.Matrix(2.0, 2.0)  # 2x scale for better OCR quality
            pix = page.get_pixmap(matrix=mat)
            img_bytes = pix.tobytes("png")
            ocr_text, confidence = await ocr_router.extract_text(img_bytes)

            # Also attempt table extraction from the image
            tables = await ocr_router.extract_tables(img_bytes)

            return PageExtractionResult(
                page_number=page_number,
                text=ocr_text,
                extraction_method="ocr",
                confidence=confidence,
                tables=tables,
                is_scanned=True,
            )

        # Digital text extraction — also try table detection
        tables = table_extractor.extract_from_page(page, page_number)

        return PageExtractionResult(
            page_number=page_number,
            text=text,
            extraction_method="digital",
            confidence=1.0,
            tables=tables,
        )

    async def _index_tables(
        self,
        tables: list[dict],
        document_id: uuid.UUID,
        tenant_id: uuid.UUID,
    ) -> None:
        from app.storage.database import get_db_context
        from app.storage.orm import ExtractedTableORM

        async with get_db_context() as db:
            for table in tables:
                orm_table = ExtractedTableORM(
                    id=uuid.uuid4(),
                    document_id=document_id,
                    tenant_id=tenant_id,
                    page_start=table.get("page_start", 0),
                    page_end=table.get("page_end", 0),
                    caption=table.get("caption"),
                    section_context=table.get("section_context"),
                    headers=table.get("headers", []),
                    rows=table.get("rows", []),
                    extraction_method=table.get("extraction_method", "unknown"),
                    confidence=table.get("confidence", 1.0),
                )
                db.add(orm_table)
