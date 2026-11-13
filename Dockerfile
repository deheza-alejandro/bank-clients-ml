# syntax=docker/dockerfile:1

ARG PYTHON_VERSION=3.13
ARG UV_VERSION=0.12.1
ARG BUN_VERSION=1.4.0

FROM ghcr.io/astral-sh/uv:${UV_VERSION} AS uv
FROM oven/bun:${BUN_VERSION}-slim AS bun

# Python + uv only. Keeping apt, the development user and Bun out of this stage
# lets the dependency stages start earlier and survive changes to them.
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

# System runtime libraries:
# - libgomp1 is required by LightGBM.
# - fonts-dejavu-core keeps Matplotlib output reproducible.
#
# The cache mounts keep apt indexes and .deb files out of the image.
FROM python-base AS system-deps

RUN --mount=type=cache,target=/var/cache/apt,sharing=locked \
    --mount=type=cache,target=/var/lib/apt/lists,sharing=locked \
    apt-get update \
    && apt-get install -y --no-install-recommends \
        fonts-dejavu-core \
        libgomp1

FROM python-base AS py-deps

# Install only the third-party dependencies in this stage. The project itself is
# intentionally not installed here: the ENTRYPOINT runs `uv sync --locked` on
# every container start, which installs it in about a second from the mounted
# source. 
#
# Trade-off: project packaging errors now surface when the container starts
# instead of during the build, so CI should install the project and run the
# tests against the image.
RUN --mount=type=cache,target=/root/.cache/uv,sharing=locked \
    --mount=type=bind,source=uv.lock,target=uv.lock \
    --mount=type=bind,source=pyproject.toml,target=pyproject.toml \
    --mount=type=bind,source=.python-version,target=.python-version \
    uv sync --locked --no-install-project --no-progress

FROM python-base AS js-deps

# Install the JavaScript tooling used by SVGO, cspell and the test suite.
# This stage inherits Python only for the marker helper; it does not wait for
# apt or for the development user.
COPY --from=bun /usr/local/bin/bun /usr/local/bin/bun
# The source image ships `bunx` as a symlink, so copying only the `bun` binary
# requires recreating it.
RUN ln -s /usr/local/bin/bun /usr/local/bin/bunx

COPY package.json bun.lock ./

RUN --mount=type=cache,target=/root/.bun/install/cache,sharing=locked \
    bun install --frozen-lockfile

# Write the marker with the same code used at runtime. A new volume copies this
# file from the image, so the first start does not need to run bun install.
RUN --mount=type=bind,source=scripts/docker_entrypoint.py,target=/tmp/docker_entrypoint.py \
    python /tmp/docker_entrypoint.py --write-marker

FROM system-deps AS dev

ARG UID=1000
ARG GID=1000

ENV PATH="/opt/venv/bin:${PATH}"

# Create the unprivileged development user only.
RUN groupadd --gid "${GID}" dev \
    && useradd --uid "${UID}" --gid dev --create-home --shell /bin/bash dev \
    && mkdir -p /app \
    && chown dev:dev /app

# Runtime copy of Bun; Bun is copied in both js-deps and dev because
# Docker stages do not share files: this stage needs it at runtime
# for the entrypoint and the test suite,
# while js-deps needs it for `bun install`. 
COPY --from=bun /usr/local/bin/bun /usr/local/bin/bun
RUN ln -s /usr/local/bin/bun /usr/local/bin/bunx

# Copy dependency trees first, then the application. The application copy
# is still needed to run the image without the bind mount.
# .dockerignore keeps the host node_modules and .venv out of the build context.
COPY --from=py-deps --chown=dev:dev /opt/venv /opt/venv
COPY --from=js-deps --chown=dev:dev /app/node_modules /app/node_modules
COPY --chown=dev:dev . .

# Copy the entrypoint outside /app so the bind mount cannot hide it.
COPY scripts/docker_entrypoint.py /usr/local/bin/docker_entrypoint.py

USER dev

EXPOSE 2718

ENTRYPOINT ["python", "/usr/local/bin/docker_entrypoint.py"]
CMD ["uv", "run", "--no-sync", "marimo", "edit", "--headless", "--no-token", "--skip-update-check", "--host", "0.0.0.0", "--port", "2718", "notebooks/notebook.py"]
