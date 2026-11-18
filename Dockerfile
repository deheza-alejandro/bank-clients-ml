# syntax=docker/dockerfile:1

# See documentation in docs/development/docker.md

ARG PYTHON_VERSION=3.13
ARG UV_VERSION=0.12.1
ARG BUN_VERSION=1.4.0

FROM ghcr.io/astral-sh/uv:${UV_VERSION} AS uv
FROM oven/bun:${BUN_VERSION}-slim AS bun

FROM python:${PYTHON_VERSION}-slim-trixie AS python-base

ENV LANG=C.UTF-8 \
    LC_ALL=C.UTF-8 \
    PYTHONUNBUFFERED=1 \
    MPLBACKEND=Agg \
    UV_PROJECT_ENVIRONMENT=/opt/venv \
    UV_LINK_MODE=copy \
    UV_COMPILE_BYTECODE=1 \
    UV_PYTHON_DOWNLOADS=never \
    UV_NO_PROGRESS=1

COPY --from=uv /uv /uvx /bin/

WORKDIR /app

FROM python-base AS system-deps

RUN --mount=type=cache,target=/var/cache/apt,sharing=locked \
    --mount=type=cache,target=/var/lib/apt/lists,sharing=locked \
    apt-get update \
    && apt-get install -y --no-install-recommends \
        fonts-dejavu-core \
        libgomp1

FROM python-base AS py-deps

RUN --mount=type=cache,target=/root/.cache/uv,sharing=locked \
    --mount=type=bind,source=uv.lock,target=uv.lock \
    --mount=type=bind,source=pyproject.toml,target=pyproject.toml \
    --mount=type=bind,source=.python-version,target=.python-version \
    uv sync --locked --no-install-project --no-progress

FROM python-base AS js-deps

COPY --from=bun /usr/local/bin/bun /usr/local/bin/bun
RUN ln -s /usr/local/bin/bun /usr/local/bin/bunx

COPY package.json bun.lock ./

RUN --mount=type=cache,target=/root/.bun/install/cache,sharing=locked \
    bun install --frozen-lockfile

RUN --mount=type=bind,source=scripts/docker_entrypoint.py,target=/tmp/docker_entrypoint.py \
    python /tmp/docker_entrypoint.py --write-marker

FROM system-deps AS dev

ARG UID=1000
ARG GID=1000

ENV PATH="/opt/venv/bin:${PATH}"

RUN groupadd --gid "${GID}" dev \
    && useradd --uid "${UID}" --gid dev --create-home --shell /bin/bash dev \
    && mkdir -p /app \
    && chown dev:dev /app

COPY --from=bun /usr/local/bin/bun /usr/local/bin/bun
RUN ln -s /usr/local/bin/bun /usr/local/bin/bunx

COPY --from=py-deps --chown=dev:dev /opt/venv /opt/venv
COPY --from=js-deps --chown=dev:dev /app/node_modules /app/node_modules
COPY --chown=dev:dev . .

COPY scripts/docker_entrypoint.py /usr/local/bin/docker_entrypoint.py

USER dev

EXPOSE 2718

ENTRYPOINT ["python", "/usr/local/bin/docker_entrypoint.py"]
CMD ["uv", "run", "--no-sync", "marimo", "edit", "--headless", "--watch", "--no-token", "--skip-update-check", "--host", "0.0.0.0", "--port", "2718", "notebooks/notebook.py"]
