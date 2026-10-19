# Customer Assistant

A production-grade agentic RAG customer support assistant that ingests 5,000+ PDFs and answers natural-language questions with grounded, cited evidence.

## Features

- **Hybrid retrieval** — pgvector dense search + PostgreSQL full-text, fused with Reciprocal Rank Fusion
- **Grounded answers** — every claim is backed by a cited passage with document title and page number
- **Prompt injection defence** — document content is quarantined from system instructions; injection patterns are detected and removed
- **Abstention** — when evidence is insufficient, the system says so clearly rather than fabricating an answer
- **Mixed PDF handling** — digital text extraction (PyMuPDF) + page-level OCR routing (Tesseract dev / AWS Textract prod)
- **Table extraction** — structured table rows/columns preserved separately from prose
- **Async ingestion** — Celery workers with lease-based job claiming, retry logic, and stale-job recovery
- **Multi-tenant** — all data is scoped to tenant_id, enforced at the DB level before retrieval
- **Observability** — structured JSON logs, OpenTelemetry traces, Prometheus metrics

## Stack

| Layer | Technology |
|---|---|
| API | FastAPI + Pydantic v2 |
| Agent | LangGraph + LangChain |
| LLM | OpenAI GPT-4o (configurable) |
| Embeddings | OpenAI text-embedding-3-small |
| Vector store | PostgreSQL + pgvector |
| Background jobs | Celery + Redis |
| OCR | Tesseract (dev) / AWS Textract (prod) |
| PDF parsing | PyMuPDF |
| Auth | JWT (HS256) |
| Observability | structlog + OpenTelemetry |
| Containers | Docker + Docker Compose |
| CI | GitHub Actions |

## Quick start

```bash
cp .env.example .env        # add OPENAI_API_KEY
make docker-up              # starts postgres, redis, api, worker
curl http://localhost:8000/health/ready
open http://localhost:8000/docs
```

## Testing

```bash
make dev-install
make test          # all tests
make test-unit     # unit only (no DB needed)
```

## Ingestion

```bash
# Upload a single PDF
curl -X POST http://localhost:8000/api/v1/documents \
  -H "Authorization: Bearer $TOKEN" \
  -F "file=@document.pdf"

# Check processing status
curl http://localhost:8000/api/v1/documents/{document_id}/status \
  -H "Authorization: Bearer $TOKEN"
```

## Project structure

```
app/
├── main.py               FastAPI factory
├── api/routes/           HTTP route handlers
├── core/                 Config, logging, telemetry, security, exceptions
├── domain/               Models, interfaces, Pydantic schemas
├── agents/               LangGraph graph, response generation
├── ingestion/            Pipeline, OCR, chunker, embeddings, table extraction
├── retrieval/            Hybrid search, RRF fusion
├── storage/              ORM, database session, S3/local backends
├── workers/              Celery tasks
└── services/             ChatService, orchestration
migrations/               Alembic migrations
tests/
├── unit/                 Pure Python, no external dependencies
├── api/                  HTTP tests with mocked DB
├── security/             Auth, cross-tenant isolation, injection
└── ingestion/            Pipeline and worker tests
docs/
├── architecture.md       Detailed architecture and design decisions
└── local-development.md  Setup guide
```

## Phase status

| Phase | Status |
|---|---|
| 1 — Architecture & planning | ✅ Complete |
| 2 — Foundation (FastAPI, DB, Docker) | ✅ Complete |
| 3 — Ingestion pipeline | ✅ Implemented (needs live integration test) |
| 4 — Retrieval (hybrid, RRF, citations) | ✅ Implemented (needs pgvector integration test) |
| 5 — Full LangGraph routing | 🔄 Phase 2 linear path in place; full graph in Phase 5 |
| 6 — Tables, calculations, charts | 📋 Planned |
| 7 — Streaming frontend | 📋 Planned |
| 8 — Observability, evaluation | 📋 Planned |
| 9 — Terraform + AWS deployment | 📋 Planned |

## What is verified vs implemented

- **Implemented:** All Phase 1–2 source files, migrations, Docker Compose, CI workflow
- **Verified:** Unit tests pass without external dependencies; app starts and health endpoint responds
- **Needs live verification:** pgvector integration tests, full ingestion with real PDFs, streaming chat
- **Not deployed:** AWS infrastructure (Terraform config is Phase 9)
