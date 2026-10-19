# Local Development Setup

## Prerequisites

- Docker + Docker Compose
- Python 3.12+
- `tesseract-ocr` (for local OCR; `brew install tesseract` on macOS, `apt-get install tesseract-ocr` on Linux)
- An OpenAI API key (optional for tests; required for live chat)

## Quick start

```bash
# 1. Clone and enter the repository
git clone <repo-url>
cd customer-assistant

# 2. Copy environment file
cp .env.example .env
# Edit .env and set OPENAI_API_KEY if you want live chat

# 3. Start all services
make docker-up
# This also runs alembic upgrade head automatically

# 4. Verify health
curl http://localhost:8000/health/ready

# 5. Open API docs
open http://localhost:8000/docs
```

## Running tests

Tests do not require a live OpenAI key or AWS credentials.

```bash
# Install dev dependencies
make dev-install

# Run all tests
make test

# Run specific suites
make test-unit
make test-api
make test-security
```

## Running the app without Docker

```bash
# Start PostgreSQL and Redis (Docker)
docker compose up postgres redis -d

# Install dependencies
make dev-install

# Apply migrations
make migrate

# Start API
make dev

# Start worker (separate terminal)
make worker
```

## Uploading a test document

```bash
# Get a token (replace with a real user in production)
TOKEN=$(python3 -c "
from app.core.security import create_access_token
import uuid
print(create_access_token(str(uuid.uuid4()), str(uuid.uuid4()), scopes=['customer', 'admin']))
")

# Upload a PDF
curl -X POST http://localhost:8000/api/v1/documents \
  -H "Authorization: Bearer $TOKEN" \
  -F "file=@/path/to/your/document.pdf"
```

## Environment variables

See `.env.example` for all options. The most important for local dev:

| Variable | Default | Notes |
|---|---|---|
| `OPENAI_API_KEY` | (empty) | Required for live LLM calls |
| `DATABASE_URL` | postgres://localhost | Set automatically in docker-compose |
| `STORAGE_BACKEND` | `local` | Use `s3` in production |
| `OCR_PROVIDER` | `tesseract` | Use `textract` in production |
| `LOG_LEVEL` | `INFO` | Set to `DEBUG` for verbose output |

## Pilot test recommendation

Before indexing all 5,000 documents, test with 30–50 representative PDFs covering:
- Normal digital text
- Scanned / image-only pages
- Pages with complex tables
- Mixed content (digital + scanned pages in same PDF)

Check extraction quality and citation accuracy before running the full batch.
