#!/usr/bin/env bash
# Single source of project checks: used by `just check`, the pre-push hook and GitLab CI.
set -euo pipefail
cd "$(dirname "$0")/.."

uv run ruff check
uv run ruff format --check
uv run pyright
uv run lint-imports
uv run pytest -q -m "not integration and not e2e and not data"
