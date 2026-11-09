# syntax=docker/dockerfile:1

# Pinned tool versions. Keep these aligned with the project lockfiles.
ARG PYTHON_VERSION=3.13
ARG UV_VERSION=0.12.1
ARG BUN_VERSION=1.4.0

# Tool stages: copy only the binaries into the development image.
FROM ghcr.io/astral-sh/uv:${UV_VERSION} AS uv
FROM oven/bun:${BUN_VERSION}-slim AS bun

FROM python:${PYTHON_VERSION}-slim-bookworm AS base

ARG UID=1000
ARG GID=1000

ENV LANG=C.UTF-8 \
    LC_ALL=C.UTF-8 \
    PYTHONUNBUFFERED=1 \
    MPLBACKEND=Agg \
    UV_PROJECT_ENVIRONMENT=/opt/venv \
    UV_LINK_MODE=copy \
    UV_COMPILE_BYTECODE=1 \
    UV_PYTHON_DOWNLOADS=never \
    UV_NO_PROGRESS=1

# Runtime libraries:
# - libgomp1 is required by LightGBM.
# - fonts-dejavu-core keeps Matplotlib output reproducible.
# - git, passwd and bash support common development workflows.
RUN apt-get update \
    && apt-get install -y --no-install-recommends \
        bash \
        fonts-dejavu-core \
        git \
        libgomp1 \
        passwd \
    && rm -rf /var/lib/apt/lists/*

# Create an unprivileged development user.
RUN groupadd --gid "${GID}" dev \
    && useradd --uid "${UID}" --gid dev --create-home --shell /bin/bash dev \
    && mkdir -p /app \
    && chown dev:dev /app

COPY --from=uv /uv /uvx /bin/
COPY --from=bun /usr/local/bin/bun /usr/local/bin/bun
RUN ln -s /usr/local/bin/bun /usr/local/bin/bunx

RUN uv --version && bun --version

WORKDIR /app

FROM base AS deps

# Install Python dependencies outside /app so the host bind mount cannot hide
# the virtual environment. Dependencies are cached separately from the
# editable project so source changes do not reinstall everything.
COPY pyproject.toml uv.lock .python-version README.md ./

RUN --mount=type=cache,target=/root/.cache/uv,sharing=locked \
    uv sync --locked --no-install-project --no-progress

COPY src ./src

RUN --mount=type=cache,target=/root/.cache/uv,sharing=locked \
    uv sync --locked --no-progress

# Install the JavaScript tooling used by SVGO, cspell and the test suite.
COPY package.json bun.lock ./

RUN --mount=type=cache,target=/root/.bun/install/cache,sharing=locked \
    bun install --frozen-lockfile

FROM base AS dev

ENV PATH="/opt/venv/bin:${PATH}"

# Copy the application first, then the dependency trees. .dockerignore keeps
# the host node_modules and .venv out of the build context.
COPY --chown=dev:dev . .
COPY --from=deps --chown=dev:dev /app/node_modules /app/node_modules
COPY --from=deps --chown=dev:dev /opt/venv /opt/venv

USER dev

EXPOSE 2718

CMD ["uv", "run", "marimo", "edit", "--headless", "--no-token", "--skip-update-check", "--host", "0.0.0.0", "--port", "2718", "notebooks/notebook.py"]
