# API image: the service and its migrations, with the CPU build of torch to run a model package
# and without the libraries only training needs.
#   docker compose build api    or    docker build -f docker/api.Dockerfile .
# UV_VERSION and PYTHON_VERSION follow .gitlab-ci.yml.
ARG UV_VERSION=0.12.14
ARG PYTHON_VERSION=3.12

# Both stages start from Docker Hub's official image through Google's mirror, and uv comes from
# PyPI: Docker Hub does not answer from Russian networks, and one registry instead of two (the uv
# image lives on ghcr.io) halves the requests a flaky connection can fail.
FROM mirror.gcr.io/library/python:${PYTHON_VERSION}-slim-trixie AS build
ARG UV_VERSION
RUN pip install --no-cache-dir "uv==${UV_VERSION}"
ENV UV_COMPILE_BYTECODE=1 \
    UV_LINK_MODE=copy \
    UV_PYTHON_DOWNLOADS=never
WORKDIR /app
# Dependencies go in a layer of their own, rebuilt only when the lock file changes.
COPY pyproject.toml uv.lock ./
RUN --mount=type=cache,target=/root/.cache/uv \
    uv sync --locked --no-default-groups --group serve --no-install-project
COPY README.md ./
COPY src ./src
RUN --mount=type=cache,target=/root/.cache/uv \
    uv sync --locked --no-default-groups --group serve --no-editable

FROM mirror.gcr.io/library/python:${PYTHON_VERSION}-slim-trixie
RUN useradd --create-home --uid 1000 loadvision
WORKDIR /app
COPY --from=build /app/.venv /app/.venv
COPY alembic.ini ./
COPY configs/app.yaml configs/app.yaml
# Encoder weights come from the model package; a missing file fails loudly instead of a
# download.
ENV PATH="/app/.venv/bin:${PATH}" \
    PYTHONUNBUFFERED=1 \
    HF_HUB_OFFLINE=1
USER loadvision
EXPOSE 8000
# The service writes its own line for every request, with the request id; uvicorn's would repeat it.
CMD ["uvicorn", "--factory", "loadvision.api.main:create_service_app", "--host", "0.0.0.0", "--port", "8000", "--no-access-log"]
