"""Conversation history utilities.

Loads past messages from the database and formats them into a structure
the LLM can use as multi-turn context.

Design constraints:
- Only the last N turns are loaded to stay within the context window.
- System messages are never included in history (they are injected fresh
  each turn by the response agent).
- Raw document passages from previous retrievals are NOT re-included —
  the agent re-retrieves as needed.  This keeps the prompt compact and
  ensures citations always point to currently-indexed content.
- Token budget is enforced: if history would exceed MAX_HISTORY_TOKENS,
  older turns are dropped until it fits.
"""
from __future__ import annotations

import uuid
from dataclasses import dataclass

from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from app.core.config import get_settings
from app.core.logging import get_logger
from app.storage.orm import MessageORM

logger = get_logger(__name__)

MAX_HISTORY_TURNS = 10          # user+assistant pairs to look back
MAX_HISTORY_TOKENS = 8_000      # rough token budget for history
_CHARS_PER_TOKEN = 4            # conservative estimate


@dataclass
class HistoryMessage:
    role: str           # "user" | "assistant"
    content: str
    message_id: str


async def load_conversation_history(
    db: AsyncSession,
    conversation_id: uuid.UUID,
    *,
    max_turns: int = MAX_HISTORY_TURNS,
) -> list[HistoryMessage]:
    """Return the last `max_turns` user+assistant message pairs.

    Messages are returned in chronological order (oldest first) so they
    can be prepended to the current-turn prompt in the correct sequence.
    """
    # Fetch 2×max_turns messages (user + assistant per turn) most recent first
    result = await db.execute(
        select(MessageORM)
        .where(MessageORM.conversation_id == conversation_id)
        .where(MessageORM.role.in_(["user", "assistant"]))
        .order_by(MessageORM.created_at.desc())
        .limit(max_turns * 2)
    )
    rows = result.scalars().all()

    # Reverse to chronological order
    rows = list(reversed(rows))

    # Build history list, enforcing token budget from the oldest end
    history: list[HistoryMessage] = []
    token_budget = MAX_HISTORY_TOKENS

    for row in rows:
        estimated_tokens = len(row.content) // _CHARS_PER_TOKEN
        if estimated_tokens > token_budget:
            # Truncate this message to fit
            keep_chars = token_budget * _CHARS_PER_TOKEN
            content = row.content[:keep_chars] + " …[truncated]"
            token_budget = 0
        else:
            content = row.content
            token_budget -= estimated_tokens

        history.append(
            HistoryMessage(
                role=row.role,
                content=content,
                message_id=str(row.id),
            )
        )

        if token_budget <= 0:
            break

    logger.debug(
        "history.loaded",
        conversation_id=str(conversation_id),
        message_count=len(history),
    )
    return history


def format_history_for_llm(history: list[HistoryMessage]) -> list[dict[str, str]]:
    """Convert history to OpenAI-style message dicts."""
    return [{"role": m.role, "content": m.content} for m in history]


def summarise_history_context(history: list[HistoryMessage]) -> str:
    """Return a short plain-text summary for injection into the agent state.

    Used when the graph needs to know what topics have been covered
    without re-sending the full message list.
    """
    if not history:
        return ""
    turns = []
    i = 0
    while i < len(history):
        if history[i].role == "user":
            user_msg = history[i].content[:200]
            asst_msg = ""
            if i + 1 < len(history) and history[i + 1].role == "assistant":
                asst_msg = history[i + 1].content[:200]
                i += 2
            else:
                i += 1
            turns.append(f"User: {user_msg}\nAssistant: {asst_msg}")
        else:
            i += 1
    return "\n\n".join(turns[-5:])   # last 5 turns as plain text
