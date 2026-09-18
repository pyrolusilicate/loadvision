# Worker image: retrains on confirmed labels, with the ML group on the CPU build of torch.
#   docker compose build worker    or    just worker
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
    uv sync --locked --no-default-groups --group ml --no-install-project
COPY README.md ./
COPY src ./src
RUN --mount=type=cache,target=/root/.cache/uv \
    uv sync --locked --no-default-groups --group ml --no-editable

# The same official image as on Docker Hub, which Russian networks often cannot reach.
FROM mirror.gcr.io/library/python:${PYTHON_VERSION}-slim-trixie
# The image has no git checkout; a package it builds records this commit instead.
ARG GIT_COMMIT=""
# The cache folder belongs to the user, so the volume mounted over it is writable.
RUN useradd --create-home --uid 1000 loadvision \
    && mkdir -p /home/loadvision/.cache/huggingface \
    && chown -R loadvision /home/loadvision/.cache
WORKDIR /app
COPY --from=build /app/.venv /app/.venv
ENV PATH="/app/.venv/bin:${PATH}" \
    PYTHONUNBUFFERED=1 \
    LOADVISION_GIT_COMMIT=${GIT_COMMIT}
USER loadvision
# Settings, the experiment config and the folds file come from the mounted configs folder: the
# worker appends new photos to the folds file, and the change must outlive the container.
CMD ["loadvision-worker"]
