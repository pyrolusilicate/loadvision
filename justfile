default:
    @just --list

# Install dependencies exactly from uv.lock
sync:
    uv sync --locked

# Format code and apply safe lint fixes
fmt:
    uv run ruff format
    uv run ruff check --fix

# Fast tests without external services
test *args:
    uv run pytest -m "not integration and not e2e and not data" {{args}}

# All checks: the same script runs in the pre-push hook and CI
check:
    scripts/check.sh

# Run checks before every push in this clone
hooks:
    git config core.hooksPath .githooks

# Service in Docker: Postgres, migrations and the API on port 8000
up:
    docker compose up -d --build api

# Local Postgres
db-up:
    docker compose up -d postgres

# Apply migrations to the database from .env
db-migrate:
    uv run --env-file .env alembic upgrade head

# Tests against the local Postgres; each run creates and drops its own databases
integration *args:
    uv run --env-file .env pytest -m integration {{args}}

db-down:
    docker compose down
