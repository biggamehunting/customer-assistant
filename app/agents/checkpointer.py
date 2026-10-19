"""LangGraph checkpointer factory.

Production: AsyncPostgresSaver backed by the same PostgreSQL instance
            as the application.  Conversation state persists across
            requests, worker restarts, and deployments.

Development (no DB): MemorySaver — in-process only, lost on restart.
            Activated automatically when LANGGRAPH_CHECKPOINTER=memory
            or when the DB URL is not set.

Test: MemorySaver always used (no DB dependency in unit tests).

The checkpointer stores the full AgentState between turns, so the
agent can refer to what it retrieved and said in earlier messages of
the same conversation without reloading anything from the chat history
table.  Secrets and raw document text are deliberately excluded from
the state (see graph.py AgentState).
"""
from __future__ import annotations

import os
from contextlib import asynccontextmanager
from typing import AsyncGenerator

from langgraph.checkpoint.base import BaseCheckpointSaver
from langgraph.checkpoint.memory import MemorySaver

from app.core.config import get_settings
from app.core.logging import get_logger

logger = get_logger(__name__)

# Module-level cache so we create the saver once per process
_saver: BaseCheckpointSaver | None = None


def _use_memory_saver() -> bool:
    settings = get_settings()
    if settings.is_test:
        return True
    if os.getenv("LANGGRAPH_CHECKPOINTER", "").lower() == "memory":
        return True
    return False


@asynccontextmanager
async def get_checkpointer() -> AsyncGenerator[BaseCheckpointSaver, None]:
    """Async context manager that yields a ready-to-use checkpointer.

    Usage:
        async with get_checkpointer() as cp:
            graph = build_graph(checkpointer=cp)
            result = await graph.ainvoke(state, config={"configurable": {"thread_id": ...}})
    """
    global _saver

    if _use_memory_saver():
        if _saver is None or not isinstance(_saver, MemorySaver):
            _saver = MemorySaver()
            logger.info("checkpointer.memory_saver_created")
        yield _saver
        return

    # PostgreSQL checkpointer
    try:
        from langgraph.checkpoint.postgres.aio import AsyncPostgresSaver

        settings = get_settings()
        # langgraph-checkpoint-postgres needs a psycopg3 connection string
        # (no +asyncpg driver prefix — psycopg3 uses its own async support)
        pg_url = (
            str(settings.database_url)
            .replace("postgresql+asyncpg://", "postgresql://")
            .replace("postgresql+psycopg2://", "postgresql://")
        )

        async with AsyncPostgresSaver.from_conn_string(pg_url) as saver:
            if _saver is None:
                # Create the checkpoint tables once
                await saver.setup()
                logger.info("checkpointer.postgres_tables_created")
            _saver = saver
            yield saver

    except Exception as exc:
        logger.warning(
            "checkpointer.postgres_failed_falling_back",
            error=str(exc),
        )
        if _saver is None or not isinstance(_saver, MemorySaver):
            _saver = MemorySaver()
        yield _saver


def get_thread_id(conversation_id: str) -> str:
    """Map a conversation UUID to a LangGraph thread_id.

    One conversation = one thread.  The checkpointer persists all
    graph state under this key, keyed by (thread_id, checkpoint_id).
    """
    return f"conv:{conversation_id}"
