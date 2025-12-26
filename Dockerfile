# syntax=docker/dockerfile:1.7
FROM python:3.12-slim AS builder

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=on \
    MPLBACKEND=Agg \
    UV_PROJECT_ENVIRONMENT=/opt/venv \
    PATH="/opt/venv/bin:/root/.local/bin:${PATH}"

RUN apt-get update && DEBIAN_FRONTEND=noninteractive apt-get install -y --no-install-recommends \
    build-essential \
    curl ca-certificates git \
    && rm -rf /var/lib/apt/lists/*

RUN curl -LsSf https://astral.sh/uv/install.sh | sh
ENV PATH="/root/.local/bin:${PATH}"

WORKDIR /app

COPY pyproject.toml uv.lock* ./
RUN uv sync --no-dev --frozen

COPY . /app
RUN uv sync --no-dev --frozen

FROM python:3.12-slim AS api

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    MPLBACKEND=Agg \
    PATH="/opt/venv/bin:${PATH}"

WORKDIR /app

COPY --from=builder /opt/venv /opt/venv

COPY . /app
RUN mkdir -p /app/data/processed
COPY data/processed/best_model_raw.pt /app/data/processed/best_model_raw.pt
COPY data/processed/best_model_pretrained.pt /app/data/processed/best_model_pretrained.pt

EXPOSE 8080

CMD ["uvicorn", "bin_it_right.api:api", "--host", "0.0.0.0", "--port", "8080"]

FROM api AS ml
EXPOSE 8888