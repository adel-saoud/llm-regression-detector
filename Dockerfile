# syntax=docker/dockerfile:1.7

# ── Builder ─────────────────────────────────────────────────────────────────
FROM python:3.12-slim AS builder

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    UV_COMPILE_BYTECODE=1 \
    UV_LINK_MODE=copy

COPY --from=ghcr.io/astral-sh/uv:0.5 /uv /uvx /bin/

WORKDIR /app
COPY pyproject.toml uv.lock* ./
COPY src ./src

# Install only runtime deps (no dev/dashboard)
RUN --mount=type=cache,target=/root/.cache/uv \
    uv sync --no-dev --no-editable

# ── Runtime ─────────────────────────────────────────────────────────────────
FROM python:3.12-slim AS runtime

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PATH="/app/.venv/bin:$PATH"

# Non-root user
RUN groupadd --system --gid 1001 lrd && \
    useradd --system --uid 1001 --gid lrd --no-create-home --shell /usr/sbin/nologin lrd

WORKDIR /app
COPY --from=builder --chown=lrd:lrd /app/.venv /app/.venv
COPY --chown=lrd:lrd src ./src
COPY --chown=lrd:lrd prompts ./prompts
COPY --chown=lrd:lrd golden_dataset ./golden_dataset

USER lrd

ENTRYPOINT ["lrd"]
CMD ["--help"]
