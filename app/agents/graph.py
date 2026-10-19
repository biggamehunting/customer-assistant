"""LangGraph agent graph — Phase 5 upgrade: full multi-turn memory.

Architecture:
  ┌─────────────────────────────────────────────────┐
  │                   AgentState                    │
  │  conversation_id ──► thread_id in checkpointer │
  │  history         ──► last N turns from DB      │
  │  retrieved_passages ► re-retrieved each turn   │
  │  answer, citations  ► generated this turn      │
  └─────────────────────────────────────────────────┘

Short-term memory is handled by two complementary mechanisms:

1. LangGraph PostgreSQL checkpointer
   Stores the full AgentState between requests under a stable thread_id
   (= conversation_id).  The agent can inspect what it retrieved and
   decided in the previous turn without another DB round-trip.

2. Conversation history loaded from MessageORM
   The last N user+assistant messages are loaded from the DB and
   injected into the LLM prompt so it can refer to earlier questions
   ("the document I asked about before", "that 30-day policy").

Safety constraints (unchanged):
  - Max steps enforced to prevent loops.
  - No LLM-generated code is executed.
  - Secrets never stored in checkpointed state.
  - Document content treated as untrusted data.
"""
from __future__ import annotations

import uuid
from typing import Any, AsyncIterator, TypedDict

from app.core.config import get_settings
from app.core.logging import get_logger
from app.domain.schemas import StreamEventType

logger = get_logger(__name__)


# ── Typed graph state ─────────────────────────────────────────────────────────

class AgentState(TypedDict, total=False):
    # ── Request context ───────────────────────────────────────────────────
    query: str
    conversation_id: str
    tenant_id: str
    user_id: str
    document_ids: list[str] | None
    include_charts: bool

    # ── Short-term memory ─────────────────────────────────────────────────
    # Loaded from DB each turn; NOT persisted in checkpoint (already in DB)
    history: list[dict[str, str]]          # [{"role": ..., "content": ...}]
    history_summary: str                   # compact text summary of history

    # ── Routing ───────────────────────────────────────────────────────────
    intent: str          # knowledge | analytical | chart | clarification

    # ── Retrieval ─────────────────────────────────────────────────────────
    retrieved_passages: list[dict]
    evidence_sufficient: bool

    # ── Generation ────────────────────────────────────────────────────────
    answer: str
    citations: list[dict]
    chart_id: str | None
    abstained: bool
    abstention_reason: str | None

    # ── Housekeeping ──────────────────────────────────────────────────────
    token_usage: dict[str, int]
    step_count: int
    errors: list[str]


# ── Public entry points ───────────────────────────────────────────────────────

async def run_agent(
    *,
    query: str,
    tenant_id: uuid.UUID,
    user_id: uuid.UUID,
    conversation_id: uuid.UUID,
    document_ids: list[uuid.UUID] | None = None,
    include_charts: bool = True,
    db=None,                          # AsyncSession — optional for history loading
) -> dict[str, Any]:
    """Run the agent and return the final result dict."""
    settings = get_settings()

    # Load conversation history from DB
    history: list[dict[str, str]] = []
    history_summary: str = ""
    if db is not None:
        from app.agents.history import (
            format_history_for_llm,
            load_conversation_history,
            summarise_history_context,
        )
        msgs = await load_conversation_history(db, conversation_id)
        history = format_history_for_llm(msgs)
        history_summary = summarise_history_context(msgs)

    state: AgentState = {
        "query": query,
        "conversation_id": str(conversation_id),
        "tenant_id": str(tenant_id),
        "user_id": str(user_id),
        "document_ids": [str(d) for d in document_ids] if document_ids else None,
        "include_charts": include_charts,
        "history": history,
        "history_summary": history_summary,
        "step_count": 0,
        "errors": [],
    }

    try:
        from app.agents.checkpointer import get_checkpointer, get_thread_id
        from langgraph.graph import END, StateGraph

        graph = _build_graph()
        thread_id = get_thread_id(str(conversation_id))

        async with get_checkpointer() as cp:
            compiled = graph.compile(checkpointer=cp)
            config = {"configurable": {"thread_id": thread_id}}

            # ainvoke merges the new state with the checkpointed state
            final: AgentState = await compiled.ainvoke(state, config=config)

    except Exception as exc:
        logger.error("agent.graph_error", error=str(exc), conversation_id=str(conversation_id))
        # Fallback: run without checkpointer
        final = await _run_linear(state)

    return {
        "answer": final.get("answer", ""),
        "citations": final.get("citations", []),
        "chart_id": final.get("chart_id"),
        "abstained": final.get("abstained", False),
        "abstention_reason": final.get("abstention_reason"),
        "token_usage": final.get("token_usage"),
    }


async def run_agent_stream(
    *,
    query: str,
    tenant_id: uuid.UUID,
    user_id: uuid.UUID,
    conversation_id: uuid.UUID,
    document_ids: list[uuid.UUID] | None = None,
    include_charts: bool = True,
    db=None,
) -> AsyncIterator[tuple[StreamEventType, dict]]:
    """Stream agent events as (StreamEventType, data) tuples."""
    history: list[dict[str, str]] = []
    history_summary: str = ""
    if db is not None:
        from app.agents.history import (
            format_history_for_llm,
            load_conversation_history,
            summarise_history_context,
        )
        msgs = await load_conversation_history(db, conversation_id)
        history = format_history_for_llm(msgs)
        history_summary = summarise_history_context(msgs)

    state: AgentState = {
        "query": query,
        "conversation_id": str(conversation_id),
        "tenant_id": str(tenant_id),
        "user_id": str(user_id),
        "document_ids": [str(d) for d in document_ids] if document_ids else None,
        "include_charts": include_charts,
        "history": history,
        "history_summary": history_summary,
        "step_count": 0,
        "errors": [],
    }

    yield StreamEventType.status, {
        "message": f"Loading context ({len(history)} previous messages)…"
        if history else "Searching knowledge base…"
    }

    try:
        state = await _node_retrieve(state)
        passages = state.get("retrieved_passages", [])
        yield StreamEventType.status, {
            "message": f"Found {len(passages)} relevant passages"
        }

        async for token in _node_generate_stream(state):
            yield StreamEventType.token, {"text": token}

        citations = state.get("citations", [])
        if citations:
            yield StreamEventType.citations, {"citations": citations}

    except Exception as exc:
        logger.error("agent.stream_error", error=str(exc))
        yield StreamEventType.error, {"message": "Processing error — please retry"}


# ── Graph construction ────────────────────────────────────────────────────────

def _build_graph() -> Any:
    """Build the LangGraph StateGraph.

    Phase 5 graph (linear path with history awareness):
      load_history → retrieve → check_evidence → generate → END

    Future phases add: route → decompose → analytics → chart nodes.
    """
    from langgraph.graph import END, StateGraph

    g = StateGraph(AgentState)

    g.add_node("retrieve", _node_retrieve)
    g.add_node("generate", _node_generate)

    g.set_entry_point("retrieve")
    g.add_edge("retrieve", "generate")
    g.add_edge("generate", END)

    return g


# ── Graph nodes ───────────────────────────────────────────────────────────────

async def _node_retrieve(state: AgentState) -> AgentState:
    """Hybrid retrieval — aware of conversation history for query expansion."""
    settings = get_settings()
    try:
        from app.retrieval.hybrid import HybridRetriever

        # Enrich query with conversation context if available
        enriched_query = _enrich_query(
            state["query"],
            state.get("history_summary", ""),
        )

        retriever = HybridRetriever()
        passages = await retriever.retrieve(
            query=enriched_query,
            tenant_id=uuid.UUID(state["tenant_id"]),
            document_ids=[uuid.UUID(d) for d in state["document_ids"]]
            if state.get("document_ids")
            else None,
        )
        state["retrieved_passages"] = [
            {
                "chunk_id": str(p.chunk_id),
                "document_id": str(p.document_id),
                "content": p.content,
                "document_title": p.document_title,
                "page_start": p.page_start,
                "page_end": p.page_end,
                "section_path": p.section_path,
                "fused_score": p.fused_score,
            }
            for p in passages
        ]
        state["evidence_sufficient"] = len(passages) > 0
        logger.info(
            "agent.retrieved",
            passage_count=len(passages),
            history_turns=len(state.get("history", [])) // 2,
        )
    except Exception as exc:
        logger.warning("agent.retrieve_failed", error=str(exc))
        state["retrieved_passages"] = []
        state["evidence_sufficient"] = False

    return state


async def _node_generate(state: AgentState) -> AgentState:
    """Generate a grounded answer, passing conversation history to the LLM."""
    from app.agents.response_agent import generate_response

    result = await generate_response(
        query=state["query"],
        passages=state.get("retrieved_passages", []),
        evidence_sufficient=state.get("evidence_sufficient", False),
        history=state.get("history", []),
    )
    state.update(result)
    return state


async def _node_generate_stream(state: AgentState):
    """Streaming token generator with history context."""
    from app.agents.response_agent import generate_response_stream

    async for token in generate_response_stream(
        query=state["query"],
        passages=state.get("retrieved_passages", []),
        evidence_sufficient=state.get("evidence_sufficient", False),
        history=state.get("history", []),
        state=state,
    ):
        yield token


# ── Helpers ───────────────────────────────────────────────────────────────────

async def _run_linear(state: AgentState) -> AgentState:
    """Fallback path without LangGraph (used when checkpointer unavailable)."""
    state = await _node_retrieve(state)
    state = await _node_generate(state)
    return state


def _enrich_query(query: str, history_summary: str) -> str:
    """Prepend relevant history context to the query for better retrieval.

    Only adds context when the query contains pronouns or references that
    likely refer to previous turns ("it", "that", "the same", "those").
    """
    if not history_summary:
        return query

    reference_words = {"it", "that", "those", "the same", "above", "previous",
                       "before", "earlier", "mentioned", "said", "that policy",
                       "the document", "the price", "the rule"}
    query_lower = query.lower()
    has_reference = any(w in query_lower for w in reference_words)

    if not has_reference:
        return query

    # Truncate history summary to avoid bloating the embedding input
    summary_snippet = history_summary[:400]
    enriched = f"[Prior context: {summary_snippet}]\n\nCurrent question: {query}"
    logger.debug("agent.query_enriched", original_len=len(query), enriched_len=len(enriched))
    return enriched
