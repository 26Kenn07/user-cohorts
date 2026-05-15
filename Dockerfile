FROM python:3.13-slim

RUN apt-get update && apt-get install -y --no-install-recommends \
        build-essential \
        curl \
        git \
    && rm -rf /var/lib/apt/lists/*

COPY --from=ghcr.io/astral-sh/uv:latest /uv /uvx /usr/local/bin/

WORKDIR /app

COPY pyproject.toml uv.lock ./
COPY db/ db/
COPY utils/ utils/
COPY models/ models/

RUN uv sync --frozen

ENV PATH="/app/.venv/bin:$PATH"

COPY . .
