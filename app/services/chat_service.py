"""Chat service — orchestrates the LangGraph agent with multi-turn memory.

Flow per request:
  1. Get or create Conversation record.
  2. Persist the user message.
  3. Pass DB session to run_agent() so it can load conversation history.
  4. LangGraph graph loads history, retrieves, generates with context.
  5. Persist assistant response + citations.
  6. Return ChatResponse.

Short-term memory is handled transparently inside run_agent():
  - History loaded from MessageORM (last N turns, token-budgeted).
  - LangGraph checkpointer persists graph state between requests
    so the agent can also refer to intermediate retrieval state.
"""
from __future__ import annotations

import time
import uuid
from datetime import datetime, timezone
from typing import AsyncIterator

from sqlalchemy.ext.asyncio import AsyncSession

from app.core.logging import get_logger
from app.domain.schemas import (
    ChatRequest,
    ChatResponse,
    CitationResponse,
    StreamEventType,
)
from app.storage.orm import ConversationORM, MessageORM

logger = get_logger(__name__)


class ChatService:
    def __init__(
        self,
        db: AsyncSession,
        tenant_id: uuid.UUID,
        user_id: uuid.UUID,
    ) -> None:
        self._db = db
        self._tenant_id = tenant_id
        self._user_id = user_id

    # ── Conversation management ───────────────────────────────────────────────

    async def _get_or_create_conversation(
        self,
        conversation_id: uuid.UUID | None,
        first_message: str,
    ) -> ConversationORM:
        from sqlalchemy import select

        if conversation_id:
            result = await self._db.execute(
                select(ConversationORM).where(
                    ConversationORM.id == conversation_id,
                    ConversationORM.tenant_id == self._tenant_id,
                    ConversationORM.user_id == self._user_id,
                )
            )
            conv = result.scalar_one_or_none()
            if conv:
                return conv

        now = datetime.now(timezone.utc)
        title = first_message[:60].strip() + ("…" if len(first_message) > 60 else "")
        conv = ConversationORM(
            id=uuid.uuid4(),
            tenant_id=self._tenant_id,
            user_id=self._user_id,
            title=title,
            created_at=now,
            updated_at=now,
        )
        self._db.add(conv)
        await self._db.flush()
        return conv

    # ── Synchronous chat ──────────────────────────────────────────────────────

    async def chat(self, request: ChatRequest) -> ChatResponse:
        t0 = time.monotonic()

        conv = await self._get_or_create_conversation(
            request.conversation_id, request.message
        )

        # Persist user message first so history loader can see it
        now = datetime.now(timezone.utc)
        user_msg = MessageORM(
            id=uuid.uuid4(),
            conversation_id=conv.id,
            role="user",
            content=request.message,
            created_at=now,
        )
        self._db.add(user_msg)
        await self._db.flush()

        # Run agent — pass db so it loads conversation history
        try:
            from app.agents.graph import run_agent

            result = await run_agent(
                query=request.message,
                tenant_id=self._tenant_id,
                user_id=self._user_id,
                conversation_id=conv.id,
                document_ids=request.document_ids,
                include_charts=request.include_charts,
                db=self._db,                   # ← history loading
            )
        except Exception as exc:
            logger.error(
                "chat.agent_error",
                conversation_id=str(conv.id),
                error=str(exc),
            )
            raise

        latency_ms = int((time.monotonic() - t0) * 1000)

        # Persist assistant response
        assistant_msg = MessageORM(
            id=uuid.uuid4(),
            conversation_id=conv.id,
            role="assistant",
            content=result.get("answer", ""),
            citations=result.get("citations", []),
            chart_id=result.get("chart_id"),
            token_usage=result.get("token_usage"),
            latency_ms=latency_ms,
            created_at=datetime.now(timezone.utc),
        )
        self._db.add(assistant_msg)
        await self._db.flush()

        citations = [CitationResponse(**c) for c in result.get("citations", [])]

        return ChatResponse(
            conversation_id=conv.id,
            message_id=assistant_msg.id,
            answer=result.get("answer", ""),
            citations=citations,
            chart_id=result.get("chart_id"),
            abstained=result.get("abstained", False),
            abstention_reason=result.get("abstention_reason"),
            token_usage=result.get("token_usage"),
            latency_ms=latency_ms,
        )

    # ── Streaming chat ────────────────────────────────────────────────────────

    async def chat_stream(
        self, request: ChatRequest
    ) -> AsyncIterator[tuple[StreamEventType, dict]]:
        conv = await self._get_or_create_conversation(
            request.conversation_id, request.message
        )

        now = datetime.now(timezone.utc)
        user_msg = MessageORM(
            id=uuid.uuid4(),
            conversation_id=conv.id,
            role="user",
            content=request.message,
            created_at=now,
        )
        self._db.add(user_msg)
        await self._db.flush()

        yield StreamEventType.status, {
            "message": "Loading conversation context…",
            "conversation_id": str(conv.id),
        }

        try:
            from app.agents.graph import run_agent_stream

            full_answer: list[str] = []
            result_meta: dict = {}

            async for event_type, data in run_agent_stream(
                query=request.message,
                tenant_id=self._tenant_id,
                user_id=self._user_id,
                conversation_id=conv.id,
                document_ids=request.document_ids,
                include_charts=request.include_charts,
                db=self._db,               # ← history loading
            ):
                if event_type == StreamEventType.token:
                    full_answer.append(data.get("text", ""))
                elif event_type in (StreamEventType.citations, StreamEventType.chart):
                    result_meta.update(data)
                yield event_type, data

            full_text = "".join(full_answer)
            assistant_msg = MessageORM(
                id=uuid.uuid4(),
                conversation_id=conv.id,
                role="assistant",
                content=full_text,
                citations=result_meta.get("citations", []),
                created_at=datetime.now(timezone.utc),
            )
            self._db.add(assistant_msg)
            await self._db.flush()

            yield StreamEventType.done, {"message_id": str(assistant_msg.id)}

        except Exception as exc:
            logger.error(
                "chat.stream_error",
                conversation_id=str(conv.id),
                error=str(exc),
            )
            yield StreamEventType.error, {
                "message": "An error occurred during processing"
            }
