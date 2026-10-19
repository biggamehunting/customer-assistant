.PHONY: help install dev-install lint type-check test test-unit test-api test-security \
        migrate migrate-create docker-up docker-down docker-logs worker format clean

PYTHON := python3
PIP    := pip
APP    := app.main:app

help:
	@echo "Customer Assistant — Developer Commands"
	@echo ""
	@echo "  install          Install production dependencies"
	@echo "  dev-install      Install dev dependencies"
	@echo "  format           Auto-format code with ruff"
	@echo "  lint             Run ruff linter"
	@echo "  type-check       Run mypy type checker"
	@echo "  test             Run all tests with coverage"
	@echo "  test-unit        Run unit tests only"
	@echo "  test-api         Run API tests only"
	@echo "  test-security    Run security tests only"
	@echo "  migrate          Apply database migrations"
	@echo "  migrate-create   Create a new migration (MSG=...)"
	@echo "  docker-up        Start all services"
	@echo "  docker-down      Stop all services"
	@echo "  docker-logs      Tail API and worker logs"
	@echo "  worker           Start Celery worker locally"
	@echo "  clean            Remove build artifacts"

install:
	$(PIP) install --no-cache-dir .

dev-install:
	$(PIP) install --no-cache-dir ".[dev]"
	pre-commit install

format:
	ruff format app tests
	ruff check --fix app tests

lint:
	ruff check app tests

type-check:
	mypy app

test:
	pytest tests/ -v --tb=short

test-unit:
	pytest tests/unit/ -v --tb=short

test-api:
	pytest tests/api/ -v --tb=short

test-security:
	pytest tests/security/ -v --tb=short

test-ingestion:
	pytest tests/ingestion/ -v --tb=short

# ── Database ──────────────────────────────────────────────────────────────────

migrate:
	alembic upgrade head

migrate-create:
	@test -n "$(MSG)" || (echo "Usage: make migrate-create MSG='description'"; exit 1)
	alembic revision --autogenerate -m "$(MSG)"

migrate-down:
	alembic downgrade -1

# ── Docker ────────────────────────────────────────────────────────────────────

docker-up:
	docker compose up --build -d
	@echo "Waiting for services to be healthy..."
	@sleep 5
	docker compose exec api alembic upgrade head
	@echo ""
	@echo "API ready at http://localhost:8000"
	@echo "Docs at   http://localhost:8000/docs"

docker-down:
	docker compose down

docker-logs:
	docker compose logs -f api worker

docker-restart:
	docker compose restart api worker

docker-migrate:
	docker compose exec api alembic upgrade head

# ── Local dev ─────────────────────────────────────────────────────────────────

dev:
	uvicorn $(APP) --reload --host 0.0.0.0 --port 8000 --log-level info

worker:
	celery -A app.workers.ingestion_tasks.celery_app worker \
		--loglevel=info \
		--queues=ingestion,maintenance \
		--concurrency=2

clean:
	find . -type d -name __pycache__ -exec rm -rf {} + 2>/dev/null; true
	find . -name "*.pyc" -delete
	rm -rf .pytest_cache .mypy_cache .ruff_cache htmlcov .coverage
