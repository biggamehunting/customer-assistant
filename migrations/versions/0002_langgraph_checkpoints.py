"""LangGraph checkpoint tables for short-term conversation memory.

The AsyncPostgresSaver creates these itself via saver.setup(), but having
them in Alembic means they are created deterministically on migrate and
destroyed cleanly on downgrade.

Revision ID: 0002
"""
from __future__ import annotations

import sqlalchemy as sa
from alembic import op
from sqlalchemy.dialects import postgresql

revision = "0002"
down_revision = "0001"
branch_labels = None
depends_on = None


def upgrade() -> None:
    # ── checkpoints ───────────────────────────────────────────────────────────
    # Stores the full serialised AgentState for each graph step.
    op.create_table(
        "checkpoints",
        sa.Column("thread_id", sa.Text, nullable=False),
        sa.Column("checkpoint_ns", sa.Text, nullable=False, server_default=""),
        sa.Column("checkpoint_id", sa.Text, nullable=False),
        sa.Column("parent_checkpoint_id", sa.Text),
        sa.Column("type", sa.Text),
        sa.Column("checkpoint", postgresql.JSONB, nullable=False),
        sa.Column("metadata", postgresql.JSONB, nullable=False, server_default="{}"),
        sa.PrimaryKeyConstraint("thread_id", "checkpoint_ns", "checkpoint_id",
                                name="pk_checkpoints"),
    )
    op.create_index(
        "ix_checkpoints_thread",
        "checkpoints",
        ["thread_id", "checkpoint_ns"],
    )

    # ── checkpoint_blobs ──────────────────────────────────────────────────────
    # Stores large binary values (embeddings, images) outside the JSONB column.
    op.create_table(
        "checkpoint_blobs",
        sa.Column("thread_id", sa.Text, nullable=False),
        sa.Column("checkpoint_ns", sa.Text, nullable=False, server_default=""),
        sa.Column("channel", sa.Text, nullable=False),
        sa.Column("version", sa.Text, nullable=False),
        sa.Column("type", sa.Text, nullable=False),
        sa.Column("blob", sa.LargeBinary),
        sa.PrimaryKeyConstraint("thread_id", "checkpoint_ns", "channel", "version",
                                name="pk_checkpoint_blobs"),
    )

    # ── checkpoint_writes ─────────────────────────────────────────────────────
    # Pending writes for in-flight graph steps (used for crash recovery).
    op.create_table(
        "checkpoint_writes",
        sa.Column("thread_id", sa.Text, nullable=False),
        sa.Column("checkpoint_ns", sa.Text, nullable=False, server_default=""),
        sa.Column("checkpoint_id", sa.Text, nullable=False),
        sa.Column("task_id", sa.Text, nullable=False),
        sa.Column("idx", sa.Integer, nullable=False),
        sa.Column("channel", sa.Text, nullable=False),
        sa.Column("type", sa.Text),
        sa.Column("blob", sa.LargeBinary, nullable=False),
        sa.PrimaryKeyConstraint(
            "thread_id", "checkpoint_ns", "checkpoint_id", "task_id", "idx",
            name="pk_checkpoint_writes",
        ),
    )


def downgrade() -> None:
    op.drop_table("checkpoint_writes")
    op.drop_table("checkpoint_blobs")
    op.drop_table("checkpoints")
