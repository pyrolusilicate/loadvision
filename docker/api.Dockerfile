# API image: the service and its migrations, without the ML training group.
#   docker compose build api    or    docker build -f docker/api.Dockerfile .
# UV_VERSION and PYTHON_VERSION follow .gitlab-ci.yml.
ARG UV_VERSION=0.12.14
ARG PYTHON_VERSION=3.12

FROM ghcr.io/astral-sh/uv:${UV_VERSION}-python${PYTHON_VERSION}-trixie-slim AS build
ENV UV_COMPILE_BYTECODE=1 \
    UV_LINK_MODE=copy \
    UV_PYTHON_DOWNLOADS=never
WORKDIR /app
# Dependencies go in a layer of their own, rebuilt only when the lock file changes.
COPY pyproject.toml uv.lock ./
RUN --mount=type=cache,target=/root/.cache/uv \
    uv sync --locked --no-default-groups --no-install-project
COPY README.md ./
COPY src ./src
RUN --mount=type=cache,target=/root/.cache/uv \
    uv sync --locked --no-default-groups --no-editable

FROM python:${PYTHON_VERSION}-slim-trixie
RUN useradd --create-home --uid 1000 loadvision
WORKDIR /app
COPY --from=build /app/.venv /app/.venv
COPY alembic.ini ./
COPY configs/app.yaml configs/app.yaml
ENV PATH="/app/.venv/bin:${PATH}" \
    PYTHONUNBUFFERED=1
USER loadvision
EXPOSE 8000
CMD ["uvicorn", "--factory", "loadvision.api.main:create_service_app", "--host", "0.0.0.0", "--port", "8000"]
