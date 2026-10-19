"""Initial schema — pgvector extension + all tables.

Revision ID: 0001
"""
from __future__ import annotations

import sqlalchemy as sa
from alembic import op
from pgvector.sqlalchemy import Vector
from sqlalchemy.dialects import postgresql

revision = "0001"
down_revision = None
branch_labels = None
depends_on = None


def upgrade() -> None:
    # ── pgvector extension ────────────────────────────────────────────────
    op.execute("CREATE EXTENSION IF NOT EXISTS vector")
    op.execute("CREATE EXTENSION IF NOT EXISTS pg_trgm")

    # ── tenants ───────────────────────────────────────────────────────────
    op.create_table(
        "tenants",
        sa.Column("id", postgresql.UUID(as_uuid=True), primary_key=True),
        sa.Column("name", sa.String(255), nullable=False),
        sa.Column("slug", sa.String(100), nullable=False, unique=True),
        sa.Column("is_active", sa.Boolean, default=True, nullable=False),
        sa.Column("max_documents", sa.Integer, default=10_000),
        sa.Column("max_storage_bytes", sa.BigInteger, default=50 * 1024 ** 3),
        sa.Column("created_at", sa.DateTime(timezone=True), nullable=False),
        sa.Column("updated_at", sa.DateTime(timezone=True), nullable=False),
    )

    # ── users ─────────────────────────────────────────────────────────────
    op.create_table(
        "users",
        sa.Column("id", postgresql.UUID(as_uuid=True), primary_key=True),
        sa.Column("tenant_id", postgresql.UUID(as_uuid=True),
                  sa.ForeignKey("tenants.id", ondelete="CASCADE"), nullable=False),
        sa.Column("email", sa.String(255), nullable=False),
        sa.Column("hashed_password", sa.String(255), nullable=False),
        sa.Column("full_name", sa.String(255)),
        sa.Column("role", sa.String(50), default="customer"),
        sa.Column("is_active", sa.Boolean, default=True),
        sa.Column("created_at", sa.DateTime(timezone=True), nullable=False),
    )
    op.create_unique_constraint("uq_user_tenant_email", "users", ["tenant_id", "email"])
    op.create_index("ix_users_tenant", "users", ["tenant_id"])

    # ── documents ─────────────────────────────────────────────────────────
    op.create_table(
        "documents",
        sa.Column("id", postgresql.UUID(as_uuid=True), primary_key=True),
        sa.Column("tenant_id", postgresql.UUID(as_uuid=True),
                  sa.ForeignKey("tenants.id", ondelete="CASCADE"), nullable=False),
        sa.Column("uploaded_by", postgresql.UUID(as_uuid=True), nullable=False),
        sa.Column("filename", sa.String(512), nullable=False),
        sa.Column("title", sa.String(512)),
        sa.Column("description", sa.Text),
        sa.Column("storage_uri", sa.String(2048), nullable=False),
        sa.Column("content_hash", sa.String(64), nullable=False),
        sa.Column("file_size_bytes", sa.BigInteger, nullable=False),
        sa.Column("mime_type", sa.String(100), default="application/pdf"),
        sa.Column("page_count", sa.Integer),
        sa.Column("chunk_count", sa.Integer),
        sa.Column("category", sa.String(100)),
        sa.Column("version", sa.String(50)),
        sa.Column("language", sa.String(10)),
        sa.Column("status", sa.String(50), nullable=False, default="pending"),
        sa.Column("parser_version", sa.String(50)),
        sa.Column("embedding_model", sa.String(100)),
        sa.Column("error_message", sa.Text),
        sa.Column("processing_started_at", sa.DateTime(timezone=True)),
        sa.Column("processing_completed_at", sa.DateTime(timezone=True)),
        sa.Column("created_at", sa.DateTime(timezone=True), nullable=False),
        sa.Column("updated_at", sa.DateTime(timezone=True), nullable=False),
    )
    op.create_index("ix_documents_tenant_status", "documents", ["tenant_id", "status"])
    op.create_index("ix_documents_content_hash", "documents", ["content_hash"])
    op.create_index("ix_documents_tenant_created", "documents", ["tenant_id", "created_at"])

    # ── chunks ────────────────────────────────────────────────────────────
    op.create_table(
        "chunks",
        sa.Column("id", postgresql.UUID(as_uuid=True), primary_key=True),
        sa.Column("document_id", postgresql.UUID(as_uuid=True),
                  sa.ForeignKey("documents.id", ondelete="CASCADE"), nullable=False),
        sa.Column("tenant_id", postgresql.UUID(as_uuid=True), nullable=False),
        sa.Column("content", sa.Text, nullable=False),
        sa.Column("content_type", sa.String(50), default="text"),
        sa.Column("content_hash", sa.String(64), nullable=False),
        sa.Column("page_start", sa.Integer, nullable=False),
        sa.Column("page_end", sa.Integer, nullable=False),
        sa.Column("chunk_index", sa.Integer, nullable=False),
        sa.Column("token_count", sa.Integer, nullable=False),
        sa.Column("section_path", postgresql.JSONB, default=list),
        sa.Column("metadata", postgresql.JSONB, default=dict),
        sa.Column("embedding_model", sa.String(100)),
        sa.Column("parser_version", sa.String(50)),
        sa.Column("created_at", sa.DateTime(timezone=True), nullable=False),
    )
    op.create_index("ix_chunks_document", "chunks", ["document_id"])
    op.create_index("ix_chunks_tenant", "chunks", ["tenant_id"])
    op.create_index("ix_chunks_content_hash", "chunks", ["content_hash"])
    op.create_index(
        "ix_chunks_fts", "chunks", ["content"],
        postgresql_using="gin",
        postgresql_ops={"content": "gin_trgm_ops"},
    )

    # ── chunk_embeddings ──────────────────────────────────────────────────
    op.create_table(
        "chunk_embeddings",
        sa.Column("chunk_id", postgresql.UUID(as_uuid=True),
                  sa.ForeignKey("chunks.id", ondelete="CASCADE"), primary_key=True),
        sa.Column("embedding_model", sa.String(100), nullable=False),
        sa.Column("embedding", Vector(1536), nullable=False),
        sa.Column("created_at", sa.DateTime(timezone=True), nullable=False),
    )
    op.create_index(
        "ix_embeddings_ivfflat",
        "chunk_embeddings",
        ["embedding"],
        postgresql_using="ivfflat",
        postgresql_ops={"embedding": "vector_cosine_ops"},
        postgresql_with={"lists": 100},
    )

    # ── extracted_tables ──────────────────────────────────────────────────
    op.create_table(
        "extracted_tables",
        sa.Column("id", postgresql.UUID(as_uuid=True), primary_key=True),
        sa.Column("document_id", postgresql.UUID(as_uuid=True),
                  sa.ForeignKey("documents.id", ondelete="CASCADE"), nullable=False),
        sa.Column("tenant_id", postgresql.UUID(as_uuid=True), nullable=False),
        sa.Column("page_start", sa.Integer, nullable=False),
        sa.Column("page_end", sa.Integer, nullable=False),
        sa.Column("caption", sa.Text),
        sa.Column("section_context", sa.Text),
        sa.Column("headers", postgresql.JSONB, nullable=False),
        sa.Column("rows", postgresql.JSONB, nullable=False),
        sa.Column("extraction_method", sa.String(50), nullable=False),
        sa.Column("confidence", sa.Float, default=1.0),
        sa.Column("created_at", sa.DateTime(timezone=True), nullable=False),
    )
    op.create_index("ix_tables_document", "extracted_tables", ["document_id"])

    # ── conversations ─────────────────────────────────────────────────────
    op.create_table(
        "conversations",
        sa.Column("id", postgresql.UUID(as_uuid=True), primary_key=True),
        sa.Column("tenant_id", postgresql.UUID(as_uuid=True), nullable=False),
        sa.Column("user_id", postgresql.UUID(as_uuid=True), nullable=False),
        sa.Column("title", sa.String(512)),
        sa.Column("is_active", sa.Boolean, default=True),
        sa.Column("created_at", sa.DateTime(timezone=True), nullable=False),
        sa.Column("updated_at", sa.DateTime(timezone=True), nullable=False),
    )
    op.create_index("ix_conversations_user_tenant", "conversations", ["user_id", "tenant_id"])

    # ── messages ──────────────────────────────────────────────────────────
    op.create_table(
        "messages",
        sa.Column("id", postgresql.UUID(as_uuid=True), primary_key=True),
        sa.Column("conversation_id", postgresql.UUID(as_uuid=True),
                  sa.ForeignKey("conversations.id", ondelete="CASCADE"), nullable=False),
        sa.Column("role", sa.String(20), nullable=False),
        sa.Column("content", sa.Text, nullable=False),
        sa.Column("citations", postgresql.JSONB, default=list),
        sa.Column("chart_id", postgresql.UUID(as_uuid=True)),
        sa.Column("token_usage", postgresql.JSONB),
        sa.Column("latency_ms", sa.Integer),
        sa.Column("created_at", sa.DateTime(timezone=True), nullable=False),
    )
    op.create_index("ix_messages_conversation", "messages", ["conversation_id"])

    # ── ingestion_jobs ────────────────────────────────────────────────────
    op.create_table(
        "ingestion_jobs",
        sa.Column("id", postgresql.UUID(as_uuid=True), primary_key=True),
        sa.Column("document_id", postgresql.UUID(as_uuid=True),
                  sa.ForeignKey("documents.id", ondelete="CASCADE"), nullable=False),
        sa.Column("tenant_id", postgresql.UUID(as_uuid=True), nullable=False),
        sa.Column("status", sa.String(50), nullable=False, default="pending"),
        sa.Column("attempt_count", sa.Integer, default=0),
        sa.Column("max_attempts", sa.Integer, default=3),
        sa.Column("error_message", sa.Text),
        sa.Column("started_at", sa.DateTime(timezone=True)),
        sa.Column("completed_at", sa.DateTime(timezone=True)),
        sa.Column("lease_expires_at", sa.DateTime(timezone=True)),
        sa.Column("created_at", sa.DateTime(timezone=True), nullable=False),
        sa.Column("updated_at", sa.DateTime(timezone=True), nullable=False),
    )
    op.create_index("ix_ingestion_jobs_status", "ingestion_jobs", ["status", "lease_expires_at"])
    op.create_index("ix_ingestion_jobs_document", "ingestion_jobs", ["document_id"])

    # ── chart_artifacts ───────────────────────────────────────────────────
    op.create_table(
        "chart_artifacts",
        sa.Column("id", postgresql.UUID(as_uuid=True), primary_key=True),
        sa.Column("conversation_id", postgresql.UUID(as_uuid=True), nullable=False),
        sa.Column("tenant_id", postgresql.UUID(as_uuid=True), nullable=False),
        sa.Column("chart_type", sa.String(50), nullable=False),
        sa.Column("title", sa.String(512), nullable=False),
        sa.Column("description", sa.Text),
        sa.Column("storage_uri", sa.String(2048), nullable=False),
        sa.Column("csv_uri", sa.String(2048)),
        sa.Column("source_citations", postgresql.JSONB, default=list),
        sa.Column("created_at", sa.DateTime(timezone=True), nullable=False),
    )
    op.create_index("ix_charts_conversation", "chart_artifacts", ["conversation_id", "tenant_id"])

    # ── audit_events ──────────────────────────────────────────────────────
    op.create_table(
        "audit_events",
        sa.Column("id", postgresql.UUID(as_uuid=True), primary_key=True),
        sa.Column("tenant_id", postgresql.UUID(as_uuid=True), nullable=False),
        sa.Column("user_id", postgresql.UUID(as_uuid=True)),
        sa.Column("event_type", sa.String(100), nullable=False),
        sa.Column("resource_type", sa.String(100)),
        sa.Column("resource_id", postgresql.UUID(as_uuid=True)),
        sa.Column("details", postgresql.JSONB, default=dict),
        sa.Column("ip_address", sa.String(50)),
        sa.Column("created_at", sa.DateTime(timezone=True), nullable=False),
    )
    op.create_index("ix_audit_tenant_created", "audit_events", ["tenant_id", "created_at"])
    op.create_index("ix_audit_event_type", "audit_events", ["event_type"])


def downgrade() -> None:
    for table in [
        "audit_events", "chart_artifacts", "ingestion_jobs", "messages",
        "conversations", "extracted_tables", "chunk_embeddings", "chunks",
        "documents", "users", "tenants",
    ]:
        op.drop_table(table)
    op.execute("DROP EXTENSION IF EXISTS vector")
    op.execute("DROP EXTENSION IF EXISTS pg_trgm")
