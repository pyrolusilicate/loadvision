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

# Service on this machine without Docker: on a Mac torch answers several times faster this way
serve:
    uv run --env-file .env uvicorn --factory loadvision.api.main:create_service_app \
        --host 127.0.0.1 --port 8000 --no-access-log

# Load test of the running service: p50, p95 and photos per second at a given concurrency
loadtest clients="4" requests="24":
    uv run python scripts/loadtest.py --clients {{clients}} --requests {{requests}} \
        --out runs/loadtest/clients-{{clients}}.json

# Retraining worker in Docker; its packages record the commit the image was built from
worker:
    GIT_COMMIT=$(git rev-parse HEAD) docker compose up -d --build worker

# Service in Docker on a machine with an NVIDIA card: the same images, built with the CUDA build
# of torch. The host needs nvidia-container-toolkit; see compose.gpu.yaml.
up-gpu:
    docker compose -f compose.yaml -f compose.gpu.yaml up -d --build api

# The retraining worker in Docker on a machine with an NVIDIA card
worker-gpu:
    GIT_COMMIT=$(git rev-parse HEAD) \
        docker compose -f compose.yaml -f compose.gpu.yaml up -d --build worker

# GPU server: torch for CUDA 12.6 instead of the CPU build. Every gpu- recipe passes these groups,
# because a plain `uv run` would bring the CPU build back; see docs/deploy-vm.md.
gpu := "--no-group cpu --group cuda"

# Install the project with torch for CUDA 12.6 (GPU server)
gpu-sync:
    uv sync {{gpu}}

# Prove that torch sees the GPU and computes on it (GPU server)
gpu-check:
    uv run {{gpu}} python -c "import torch; assert torch.cuda.is_available(), 'torch does not see the GPU'; x = torch.randn(512, 512, device='cuda'); print(torch.cuda.get_device_name(0), '| torch', torch.__version__, '| CUDA', torch.version.cuda, '| matmul ok:', bool((x @ x).isfinite().all()))"

# The service on the GPU server, outside Docker so that torch reaches the GPU
gpu-serve:
    uv run {{gpu}} --env-file .env uvicorn --factory loadvision.api.main:create_service_app --host 0.0.0.0 --port 8000 --no-access-log

# The retraining worker on the GPU server
gpu-worker:
    uv run {{gpu}} --env-file .env loadvision-worker

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
