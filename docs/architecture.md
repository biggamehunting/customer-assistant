# Architecture

## Overview

A production-grade agentic RAG customer assistant that ingests 5,000+ PDFs and answers customer questions with grounded, cited evidence.

```
                    CUSTOMER / ADMIN
                          │
                   Web Interface (Streamlit / React)
                          │
                    FastAPI (port 8000)
                          │
             ┌────────────┼─────────────┐
             │            │             │
          Auth/RBAC    Rate Limit   Request ID
             │
         LangGraph Agent Runtime
             │
    ┌────────┴────────┐
    │                 │
 Retrieval Node   Generation Node
    │                 │
 pgvector + FTS    OpenAI GPT-4o
    │
 Citation Validator
    │
 PostgreSQL (RDS in prod)


INGESTION PIPELINE (async, bounded memory)
    │
 Upload API / S3 PUT
    │
 SQS Queue (durable)
    │
 Celery Worker
    │
 ┌──┴──────────────────────────────────────┐
 │  1. Download PDF from S3                │
 │  2. Page classification (digital/OCR)  │
 │  3. Text extraction (PyMuPDF)           │
 │  4. OCR for scanned pages (Textract)   │
 │  5. Table extraction (PyMuPDF/Textract) │
 │  6. Structure-aware chunking            │
 │  7. OpenAI embedding (batched)          │
 │  8. PostgreSQL + pgvector indexing     │
 └─────────────────────────────────────────┘

OBSERVABILITY
  OpenTelemetry → CloudWatch / OTLP collector
  Structured JSON logs (structlog)
  Prometheus metrics endpoint
```

## Key Design Decisions

### 1. PostgreSQL + pgvector (not a dedicated vector DB)

**Decision:** Use pgvector in the same PostgreSQL instance rather than a separate vector database (Pinecone, Weaviate, etc.).

**Rationale:**
- Eliminates a separate infrastructure dependency.
- ACID transactions across document metadata and embeddings.
- Native hybrid search with `tsvector` (lexical) + `<=>` cosine (dense) in a single query.
- pgvector IVFFlat index is sufficient for 5,000–100,000 document chunks.
- Reconsider at 10M+ vectors where HNSW in a dedicated store may outperform.

### 2. Celery + Redis (not AWS Lambda or SQS direct)

**Decision:** Celery workers consuming from Redis queues, with SQS as the durable source in production.

**Rationale:**
- Celery provides bounded concurrency, retries, rate-limiting, and stale-job recovery out of the box.
- Workers can be scaled horizontally on ECS Fargate without code changes.
- SQS dead-letter queues capture permanently failed jobs without data loss.

### 3. Bounded memory ingestion

**Decision:** Process PDFs page-by-page, discard each page before moving to the next.

**Rationale:**
- A 2,000-page PDF loaded entirely into memory would exhaust worker RAM.
- Page-level OCR routing prevents sending every page to Textract (cost and latency).
- Chunker and embedder consume O(batch) memory, not O(document) memory.

### 4. Prompt injection defence via quarantine

**Decision:** Document content is placed in a clearly-labelled `[PASSAGE N]` block, separated from system instructions by a structural boundary. Retrieved content is never treated as executable instructions.

**Rationale:**
- PDF documents are untrusted third-party content.
- A regex-based sanitiser catches common injection patterns before content reaches the LLM context.
- Tool permissions are explicitly declared; the customer-facing agent cannot call ingestion or admin tools.

### 5. Citation validation before response

**Decision:** All `[N]` references in generated answers are checked against the actual retrieved passage list. Out-of-range citations are dropped with a log warning rather than passed to the user.

**Rationale:**
- LLMs hallucinate citation numbers. Passing invented citations to users destroys trust.
- Dropping invalid citations still allows the answer through if evidence is otherwise sufficient.

### 6. Abstention over hallucination

**Decision:** When retrieval returns empty or low-confidence results, the agent returns a clear "I couldn't find this in the knowledge base" response.

**Rationale:**
- For customer support, an admitted gap is safer than a confidently wrong answer.
- Abstention rate is tracked as a metric; high rates signal corpus gaps or retrieval quality issues.

### 7. Configurable OCR routing

**Decision:** Pages are classified individually. Only pages with fewer than 50 words extracted digitally are sent to OCR.

**Rationale:**
- Textract costs $0.0015 per page. Sending all 5,000 PDFs × average 20 pages = 100,000 pages × $0.0015 = $150 vs. only the scanned subset.
- Confidence threshold is configurable so operators can tune the routing decision.

## Component Boundaries

| Component | Responsibility | Does NOT do |
|---|---|---|
| FastAPI routes | HTTP I/O, auth checks, request/response mapping | Business logic, DB queries |
| ChatService | Conversation lifecycle, message persistence | Retrieval, generation |
| LangGraph agent | Workflow orchestration, state management | Direct DB access |
| HybridRetriever | Dense + lexical search, RRF fusion | LLM calls |
| ResponseAgent | Grounded generation, citation validation | Retrieval |
| IngestionPipeline | PDF → chunks + embeddings | Chat, retrieval |
| Celery worker | Async task execution, retries | HTTP |

## Data Flow: Chat Request

```
POST /api/v1/chat
  │
  ├── Auth middleware (JWT decode, tenant_id extraction)
  ├── Rate limit check
  ├── Request validation (Pydantic)
  │
  ▼
ChatService.chat()
  │
  ├── Get or create Conversation (DB)
  ├── Persist user Message (DB)
  │
  ▼
run_agent(query, tenant_id, ...)
  │
  ├── _node_retrieve()
  │   ├── Embed query (OpenAI)
  │   ├── Dense search (pgvector)
  │   ├── Lexical search (pg FTS)
  │   └── RRF fusion
  │
  └── _node_generate()
      ├── Build context from passages
      ├── Sanitise passage content (injection defence)
      ├── Call OpenAI GPT-4o
      ├── Extract + validate citations
      └── Return answer + citations
  │
  ▼
ChatService.chat() (continued)
  ├── Persist assistant Message with citations (DB)
  └── Return ChatResponse
```

## Data Flow: Document Ingestion

```
POST /api/v1/documents/upload
  │
  ├── File validation (MIME type, size)
  ├── Content hash (xxhash) → deduplication check
  ├── Storage.upload() → S3 / local
  ├── DocumentORM created (status=queued)
  ├── IngestionJobORM created (status=pending)
  └── Celery task enqueued
         │
         ▼
  Worker: process_document_task()
  │
  ├── Claim job (lease)
  ├── Download PDF bytes from storage
  ├── For each page:
  │   ├── Try digital text extraction (PyMuPDF)
  │   ├── If scanned → OCR (Tesseract / Textract)
  │   ├── Table extraction
  │   └── Chunk page text
  ├── Batch embed chunks (OpenAI, 100/batch)
  ├── Write chunks + embeddings to PostgreSQL
  ├── Write tables to extracted_tables
  └── Update document status=indexed
```

## Scaling Notes

- **Ingestion throughput:** limited by OpenAI embedding API rate limits. At tier 2 (1M tokens/min), a 800-token chunk requires ~1,250 chunks/min throughput. 5,000 PDFs × 20 pages × ~1 chunk/page = ~100,000 chunks → ~80 minutes at tier 2 with a single worker. Scale workers and batches for faster throughput.
- **Retrieval latency:** pgvector IVFFlat with 100 lists is tuned for ~100K vectors. Re-tune `lists` parameter as corpus grows (rule of thumb: `sqrt(n_vectors)`).
- **Read/write split:** consider read replicas for retrieval once write load from ingestion exceeds DB capacity.

## Future Phases

- Phase 5: Full LangGraph routing (query decomposition, multi-document, analytics, chart nodes)
- Phase 6: Structured table retrieval and Plotly chart generation
- Phase 7: Streaming frontend (Streamlit or React)
- Phase 8: OpenTelemetry + CloudWatch dashboards, evaluation pipeline
- Phase 9: Terraform + ECS Fargate deployment
