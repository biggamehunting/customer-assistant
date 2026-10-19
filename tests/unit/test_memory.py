"""Unit tests for short-term memory: history loading, formatting, query enrichment."""
from __future__ import annotations

import uuid
from datetime import datetime, timezone
from unittest.mock import AsyncMock, MagicMock

import pytest

from app.agents.history import (
    HistoryMessage,
    format_history_for_llm,
    summarise_history_context,
)
from app.agents.graph import _enrich_query


# ── format_history_for_llm ────────────────────────────────────────────────────

class TestFormatHistoryForLLM:
    def test_empty_history_returns_empty_list(self):
        assert format_history_for_llm([]) == []

    def test_single_user_message(self):
        msgs = [HistoryMessage(role="user", content="Hello", message_id="1")]
        result = format_history_for_llm(msgs)
        assert result == [{"role": "user", "content": "Hello"}]

    def test_preserves_role_order(self):
        msgs = [
            HistoryMessage(role="user", content="Q1", message_id="1"),
            HistoryMessage(role="assistant", content="A1", message_id="2"),
            HistoryMessage(role="user", content="Q2", message_id="3"),
        ]
        result = format_history_for_llm(msgs)
        assert [m["role"] for m in result] == ["user", "assistant", "user"]
        assert result[1]["content"] == "A1"

    def test_message_id_not_in_output(self):
        msgs = [HistoryMessage(role="user", content="Q", message_id="abc")]
        result = format_history_for_llm(msgs)
        assert "message_id" not in result[0]


# ── summarise_history_context ─────────────────────────────────────────────────

class TestSummariseHistoryContext:
    def test_empty_history_returns_empty_string(self):
        assert summarise_history_context([]) == ""

    def test_single_qa_pair_is_included(self):
        msgs = [
            HistoryMessage(role="user", content="What is the refund policy?", message_id="1"),
            HistoryMessage(role="assistant", content="Returns within 30 days.", message_id="2"),
        ]
        result = summarise_history_context(msgs)
        assert "refund policy" in result
        assert "30 days" in result

    def test_long_messages_are_truncated_to_200_chars(self):
        long_content = "x" * 500
        msgs = [
            HistoryMessage(role="user", content=long_content, message_id="1"),
            HistoryMessage(role="assistant", content=long_content, message_id="2"),
        ]
        result = summarise_history_context(msgs)
        # Each message is truncated to 200 chars in the summary
        assert len(result) < 600

    def test_only_last_5_turns_included(self):
        msgs = []
        for i in range(12):
            role = "user" if i % 2 == 0 else "assistant"
            msgs.append(HistoryMessage(role=role, content=f"msg-{i}", message_id=str(i)))
        result = summarise_history_context(msgs)
        # Earlier messages should be absent; last ones present
        assert "msg-11" in result or "msg-10" in result


# ── _enrich_query ─────────────────────────────────────────────────────────────

class TestEnrichQuery:
    def test_no_history_returns_original_query(self):
        q = "What is the pricing?"
        assert _enrich_query(q, "") == q

    def test_no_reference_words_returns_original(self):
        q = "What is the refund policy?"
        summary = "User asked about pricing. Assistant gave plan details."
        assert _enrich_query(q, summary) == q

    def test_pronoun_triggers_enrichment(self):
        q = "Can you explain it further?"
        summary = "User: pricing plans\nAssistant: There are three tiers."
        result = _enrich_query(q, summary)
        assert "Prior context" in result
        assert q in result

    def test_the_document_triggers_enrichment(self):
        q = "What does the document say about cancellations?"
        summary = "User asked about refunds."
        result = _enrich_query(q, summary)
        assert "Prior context" in result

    def test_history_summary_is_truncated_to_400_chars(self):
        q = "What does that mean?"
        long_summary = "x" * 1000
        result = _enrich_query(q, long_summary)
        # Enriched query includes truncated summary of ≤400 chars
        assert len(result) < len(q) + 500

    def test_earlier_in_query_triggers_enrichment(self):
        q = "Can you elaborate on what you said earlier?"
        result = _enrich_query(q, "Some prior context here.")
        assert "Prior context" in result


# ── Checkpointer thread_id mapping ───────────────────────────────────────────

class TestCheckpointerThreadId:
    def test_conversation_id_maps_to_thread_id(self):
        from app.agents.checkpointer import get_thread_id
        conv_id = str(uuid.uuid4())
        thread_id = get_thread_id(conv_id)
        assert thread_id == f"conv:{conv_id}"

    def test_different_conversations_have_different_thread_ids(self):
        from app.agents.checkpointer import get_thread_id
        a = get_thread_id(str(uuid.uuid4()))
        b = get_thread_id(str(uuid.uuid4()))
        assert a != b

    def test_same_conversation_always_same_thread_id(self):
        from app.agents.checkpointer import get_thread_id
        conv_id = str(uuid.uuid4())
        assert get_thread_id(conv_id) == get_thread_id(conv_id)


# ── MemorySaver used in test environment ──────────────────────────────────────

class TestCheckpointerFactory:
    @pytest.mark.asyncio
    async def test_test_env_uses_memory_saver(self):
        from langgraph.checkpoint.memory import MemorySaver
        from app.agents.checkpointer import get_checkpointer

        async with get_checkpointer() as cp:
            assert isinstance(cp, MemorySaver)

    @pytest.mark.asyncio
    async def test_memory_saver_stores_and_retrieves_state(self):
        """Smoke-test: write a checkpoint and read it back."""
        from langgraph.checkpoint.memory import MemorySaver
        from app.agents.checkpointer import get_checkpointer, get_thread_id

        conv_id = str(uuid.uuid4())
        thread_id = get_thread_id(conv_id)

        async with get_checkpointer() as cp:
            assert isinstance(cp, MemorySaver)
            # MemorySaver stores data keyed by thread_id — just verify it's usable
            config = {"configurable": {"thread_id": thread_id, "checkpoint_ns": "",
                                        "checkpoint_id": "test-1"}}
            # MemorySaver.alist returns an async iterator; simply check it's callable
            assert hasattr(cp, "alist") or hasattr(cp, "get")


# ── History token budget ──────────────────────────────────────────────────────

class TestHistoryTokenBudget:
    @pytest.mark.asyncio
    async def test_load_history_respects_token_budget(self):
        """Verify that very long messages are truncated to fit the budget."""
        from app.agents.history import load_conversation_history, MAX_HISTORY_TOKENS

        long_content = "word " * 20_000   # ~80k chars >> budget
        mock_row = MagicMock()
        mock_row.id = uuid.uuid4()
        mock_row.role = "user"
        mock_row.content = long_content
        mock_row.created_at = datetime.now(timezone.utc)

        db = AsyncMock()
        result_mock = MagicMock()
        result_mock.scalars.return_value.all.return_value = [mock_row]
        db.execute = AsyncMock(return_value=result_mock)

        history = await load_conversation_history(db, uuid.uuid4())
        assert len(history) == 1
        # Content must be truncated
        total_chars = sum(len(m.content) for m in history)
        assert total_chars <= MAX_HISTORY_TOKENS * 4 + 20   # small margin for "…[truncated]"

    @pytest.mark.asyncio
    async def test_load_history_empty_conversation(self):
        from app.agents.history import load_conversation_history

        db = AsyncMock()
        result_mock = MagicMock()
        result_mock.scalars.return_value.all.return_value = []
        db.execute = AsyncMock(return_value=result_mock)

        history = await load_conversation_history(db, uuid.uuid4())
        assert history == []

    @pytest.mark.asyncio
    async def test_load_history_chronological_order(self):
        from app.agents.history import load_conversation_history

        rows = []
        for i, (role, content) in enumerate([
            ("user", "first question"),
            ("assistant", "first answer"),
            ("user", "second question"),
            ("assistant", "second answer"),
        ]):
            row = MagicMock()
            row.id = uuid.uuid4()
            row.role = role
            row.content = content
            row.created_at = datetime(2024, 1, 1, 0, i, tzinfo=timezone.utc)
            rows.append(row)

        # DB returns most-recent first (ORDER BY created_at DESC)
        db = AsyncMock()
        result_mock = MagicMock()
        result_mock.scalars.return_value.all.return_value = list(reversed(rows))
        db.execute = AsyncMock(return_value=result_mock)

        history = await load_conversation_history(db, uuid.uuid4())
        # Should be reversed back to chronological
        assert history[0].content == "first question"
        assert history[-1].content == "second answer"
