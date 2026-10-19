"""Unit tests for retrieval — RRF fusion and score threshold."""
from __future__ import annotations

import uuid

import pytest

from app.domain.models import ContentType, RetrievedPassage
from app.retrieval.hybrid import HybridRetriever


def _passage(chunk_id: uuid.UUID | None = None, dense: float = 0.0, lexical: float = 0.0) -> RetrievedPassage:
    return RetrievedPassage(
        chunk_id=chunk_id or uuid.uuid4(),
        document_id=uuid.uuid4(),
        tenant_id=uuid.uuid4(),
        content="Sample passage content.",
        content_type=ContentType.text,
        page_start=1,
        page_end=1,
        section_path=[],
        document_title="Test Doc",
        dense_score=dense,
        lexical_score=lexical,
    )


class TestRRFFusion:
    def setup_method(self):
        self.retriever = HybridRetriever()

    def test_empty_lists_return_empty(self):
        result = self.retriever._rrf_fuse([], [], top_k=10)
        assert result == []

    def test_passage_appearing_in_both_lists_scores_higher(self):
        shared_id = uuid.uuid4()
        dense = [_passage(chunk_id=shared_id, dense=0.9), _passage(dense=0.5)]
        lexical = [_passage(chunk_id=shared_id, lexical=0.8), _passage(lexical=0.4)]
        result = self.retriever._rrf_fuse(dense, lexical, top_k=10)
        # The shared passage should rank first
        assert result[0].chunk_id == shared_id

    def test_top_k_is_respected(self):
        dense = [_passage() for _ in range(20)]
        lexical = []
        result = self.retriever._rrf_fuse(dense, lexical, top_k=5)
        assert len(result) <= 5

    def test_fused_score_is_set_on_results(self):
        p = _passage(dense=0.8)
        result = self.retriever._rrf_fuse([p], [], top_k=10)
        assert len(result) == 1
        assert result[0].fused_score > 0

    def test_duplicate_chunk_ids_not_doubled(self):
        chunk_id = uuid.uuid4()
        p = _passage(chunk_id=chunk_id, dense=0.9)
        dense = [p]
        lexical = [_passage(chunk_id=chunk_id, lexical=0.7)]
        result = self.retriever._rrf_fuse(dense, lexical, top_k=10)
        ids = [r.chunk_id for r in result]
        assert ids.count(chunk_id) == 1

    def test_score_threshold_filters_low_results(self, monkeypatch):
        monkeypatch.setenv("RETRIEVAL_SCORE_THRESHOLD", "0.99")
        from app.core.config import invalidate_settings
        invalidate_settings()
        retriever = HybridRetriever()
        # All passages will have low fused scores far below 0.99
        passages = [_passage() for _ in range(5)]
        result = retriever._rrf_fuse(passages, [], top_k=10)
        assert result == []
        invalidate_settings()
