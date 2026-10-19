"""Unit tests for the structure-aware chunker."""
from __future__ import annotations

import uuid

import pytest

from app.ingestion.chunker import Chunker


@pytest.fixture
def chunker() -> Chunker:
    return Chunker(chunk_size_tokens=50, overlap_tokens=10)


class TestChunker:
    def test_empty_text_returns_no_chunks(self, chunker):
        result = chunker.chunk_page(
            text="",
            document_id=uuid.uuid4(),
            tenant_id=uuid.uuid4(),
            page_number=1,
            extraction_method="digital",
        )
        assert result == []

    def test_short_text_fits_one_chunk(self, chunker):
        text = "This is a short sentence."
        result = chunker.chunk_page(
            text=text,
            document_id=uuid.uuid4(),
            tenant_id=uuid.uuid4(),
            page_number=5,
            extraction_method="digital",
        )
        assert len(result) >= 1
        assert result[0]["page_start"] == 5
        assert result[0]["content_type"] == "text"

    def test_long_text_splits_into_multiple_chunks(self):
        chunker = Chunker(chunk_size_tokens=20, overlap_tokens=5)
        text = " ".join(["word"] * 200)
        result = chunker.chunk_page(
            text=text,
            document_id=uuid.uuid4(),
            tenant_id=uuid.uuid4(),
            page_number=1,
            extraction_method="ocr",
        )
        assert len(result) > 1

    def test_chunk_index_is_sequential(self):
        chunker = Chunker(chunk_size_tokens=20, overlap_tokens=5)
        text = " ".join(["word"] * 200)
        result = chunker.chunk_page(
            text=text,
            document_id=uuid.uuid4(),
            tenant_id=uuid.uuid4(),
            page_number=1,
            extraction_method="digital",
        )
        indices = [c["chunk_index"] for c in result]
        assert indices == list(range(len(indices)))

    def test_content_hash_is_deterministic(self, chunker):
        text = "Consistent content for hashing."
        doc_id = uuid.uuid4()
        tenant_id = uuid.uuid4()
        result_a = chunker.chunk_page(text=text, document_id=doc_id, tenant_id=tenant_id,
                                       page_number=1, extraction_method="digital")
        result_b = chunker.chunk_page(text=text, document_id=doc_id, tenant_id=tenant_id,
                                       page_number=1, extraction_method="digital")
        assert result_a[0]["content_hash"] == result_b[0]["content_hash"]

    def test_chunk_metadata_contains_required_fields(self, chunker):
        result = chunker.chunk_page(
            text="Some content here.",
            document_id=uuid.uuid4(),
            tenant_id=uuid.uuid4(),
            page_number=3,
            extraction_method="ocr",
        )
        chunk = result[0]
        required = {"content", "document_id", "tenant_id", "page_start", "page_end",
                    "chunk_index", "content_hash", "token_count", "section_path"}
        assert required.issubset(chunk.keys())

    def test_section_path_provided_is_preserved(self, chunker):
        result = chunker.chunk_page(
            text="Policy details here.",
            document_id=uuid.uuid4(),
            tenant_id=uuid.uuid4(),
            page_number=1,
            extraction_method="digital",
            section_path=["Chapter 1", "Section 2"],
        )
        assert "Chapter 1" in result[0]["section_path"]
