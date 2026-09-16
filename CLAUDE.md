# Loadvision

Hackathon MVP: estimate cargo load (0-100%) from a truck body photo, store it by shipment id, expose it via XML API, retrain on confirmed labels. Metric: MAE. Team of two: ML owner and platform owner.

## Commands

- `uv sync` install (every dependency group by default, so pyright and import-linter see all imports), `uv add <pkg>` add dependency (ask the team first)
- `scripts/check.sh` (or `just check`) = ruff check, ruff format --check, pyright, lint-imports, fast pytest. Single source of checks for the pre-push hook and CI; change checks only there
- Pre-push hook: `.githooks/pre-push`, enabled per clone with `git config core.hooksPath .githooks`
- GitLab CI: the hackathon GitLab has no runners; pipelines are created only when the CI/CD variable `RUN_CI=true` is set
- `just fmt` format and autofix
- `just test -k name` run selected fast tests
- `just db-up` / `just db-down` local Postgres (needs `.env` from `.env.example`)

## Layout and rules

- Layers, enforced by import-linter: `api | cli` -> `infrastructure | ml` -> `application` -> `domain`. Forbidden contracts add: domain imports no frameworks; application imports no web, database, XML or ML frameworks; shared modules (`errors`, `settings`, `observability`, `imaging`) import no project layers and no ML runtime.
- `ShipmentId` format lives only in `domain/shipment.py` (`SHIPMENT_ID_PATTERN`) and is repeated in the XSD; keep it out of configs.
- One file = one responsibility, roughly 50-300 lines. Split by the folder plan instead of growing a file.
- Protocols only where two implementations exist (real and fake for tests). Registries in `ml/registry.py` for backbones and heads.
- Shared contracts change only with both owners: `application/ports.py`, model package manifest, XSD, `errors.py`, observability event names, TrainingRunner, first migration.
- `imaging/preprocessing.py` is the only path from image bytes to model input, for both training and serving.
- Data lives in `data/train`, `data/test`; runs in `runs/`; service files in `storage/`. None of them are committed.
- `baseline/` is the organizers' reference code; do not edit or lint it.
- Pyright is strict for `domain` and `application`, standard elsewhere.

## Scope and dates

- Deadline: 22.09.2026 23:59. Priorities: Must (required by the task), Should (clear extra value), Could (only with spare time). Finish Must end to end before Should.
- Create files only when the work needs them; the folder plan says where they go.
- Fast retraining uses frozen encoders only. Stored folds never change for existing rows.
- Metrics source of truth: `runs/`. DB access: sync SQLAlchemy 2 with psycopg 3.

## References

- Architecture doc: https://claude.ai/artifact/CjXK6XQVUmFFA1AXi22SnU
- Miro board (diagrams, tasks, decision log): https://miro.com/app/board/uXjVHm5DNB8=/
