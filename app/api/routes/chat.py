"""Chat endpoints — synchronous and streaming (SSE)."""
from __future__ import annotations

import json
import uuid
from datetime import datetime, timezone

from fastapi import APIRouter, HTTPException, status
from fastapi.responses import StreamingResponse
from sqlalchemy import select

from app.api.dependencies import AuthUser, DbSession
from app.core.logging import get_logger
from app.domain.schemas import (
    ChatRequest,
    ChatResponse,
    ConversationResponse,
    MessageResponse,
    PaginatedResponse,
    StreamEventType,
)
from app.storage.orm import ConversationORM, MessageORM

router = APIRouter(prefix="/api/v1", tags=["chat"])
logger = get_logger(__name__)


@router.post("/chat", response_model=ChatResponse)
async def chat(
    request: ChatRequest,
    db: DbSession,
    current_user: AuthUser,
) -> ChatResponse:
    """Submit a question and receive a grounded answer with citations.

    The agent workflow runs synchronously here.  For long questions use
    /chat/stream instead.
    """
    from app.services.chat_service import ChatService

    service = ChatService(db=db, tenant_id=current_user.tenant_id, user_id=current_user.user_id)
    return await service.chat(request)


@router.post("/chat/stream")
async def chat_stream(
    request: ChatRequest,
    db: DbSession,
    current_user: AuthUser,
) -> StreamingResponse:
    """Stream answer tokens and events as Server-Sent Events.

    Event types: status | token | citations | chart | done | error
    Each event is a JSON object: {"type": "<type>", "data": {...}}
    """
    from app.services.chat_service import ChatService

    service = ChatService(db=db, tenant_id=current_user.tenant_id, user_id=current_user.user_id)

    async def event_generator():
        try:
            async for event_type, data in service.chat_stream(request):
                payload = json.dumps({"type": event_type.value, "data": data})
                yield f"data: {payload}\n\n"
        except Exception as exc:
            logger.error("chat.stream_error", error=str(exc))
            payload = json.dumps({"type": StreamEventType.error.value, "data": {"message": "Stream failed"}})
            yield f"data: {payload}\n\n"

    return StreamingResponse(
        event_generator(),
        media_type="text/event-stream",
        headers={
            "Cache-Control": "no-cache",
            "X-Accel-Buffering": "no",
        },
    )


@router.get("/conversations/{conversation_id}", response_model=ConversationResponse)
async def get_conversation(
    conversation_id: uuid.UUID,
    db: DbSession,
    current_user: AuthUser,
) -> ConversationResponse:
    result = await db.execute(
        select(ConversationORM).where(
            ConversationORM.id == conversation_id,
            ConversationORM.tenant_id == current_user.tenant_id,
            ConversationORM.user_id == current_user.user_id,
        )
    )
    conv = result.scalar_one_or_none()
    if not conv:
        raise HTTPException(status_code=404, detail="Conversation not found")
    return ConversationResponse(
        id=conv.id,
        title=conv.title,
        created_at=conv.created_at,
        updated_at=conv.updated_at,
    )


@router.get("/conversations/{conversation_id}/messages", response_model=PaginatedResponse)
async def get_conversation_messages(
    conversation_id: uuid.UUID,
    db: DbSession,
    current_user: AuthUser,
    limit: int = 50,
    offset: int = 0,
) -> PaginatedResponse:
    # Verify conversation ownership
    conv_result = await db.execute(
        select(ConversationORM).where(
            ConversationORM.id == conversation_id,
            ConversationORM.tenant_id == current_user.tenant_id,
            ConversationORM.user_id == current_user.user_id,
        )
    )
    if not conv_result.scalar_one_or_none():
        raise HTTPException(status_code=404, detail="Conversation not found")

    result = await db.execute(
        select(MessageORM)
        .where(MessageORM.conversation_id == conversation_id)
        .order_by(MessageORM.created_at.asc())
        .limit(limit)
        .offset(offset)
    )
    messages = result.scalars().all()

    return PaginatedResponse(
        total=len(messages),
        limit=limit,
        offset=offset,
        items=[
            MessageResponse(
                id=m.id,
                role=m.role,
                content=m.content,
                created_at=m.created_at,
                citations=m.citations or [],
            )
            for m in messages
        ],
    )


@router.delete("/conversations/{conversation_id}", status_code=status.HTTP_204_NO_CONTENT, response_model=None)
async def delete_conversation(
    conversation_id: uuid.UUID,
    db: DbSession,
    current_user: AuthUser,
) -> None:
    result = await db.execute(
        select(ConversationORM).where(
            ConversationORM.id == conversation_id,
            ConversationORM.tenant_id == current_user.tenant_id,
            ConversationORM.user_id == current_user.user_id,
        )
    )
    conv = result.scalar_one_or_none()
    if not conv:
        raise HTTPException(status_code=404, detail="Conversation not found")
    await db.delete(conv)
    logger.info("chat.conversation_deleted", conversation_id=str(conversation_id))
