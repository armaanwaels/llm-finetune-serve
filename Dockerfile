# GPU image for fine-tuning and serving. Installs the exact versions in uv.lock.
FROM nvidia/cuda:12.4.1-runtime-ubuntu22.04

ENV DEBIAN_FRONTEND=noninteractive PYTHONUNBUFFERED=1 UV_PYTHON=3.11 UV_LINK_MODE=copy
RUN apt-get update && apt-get install -y --no-install-recommends curl ca-certificates make \
    && rm -rf /var/lib/apt/lists/*
COPY --from=ghcr.io/astral-sh/uv:0.8 /uv /usr/local/bin/uv

WORKDIR /app
COPY pyproject.toml uv.lock ./
RUN uv sync --frozen --no-install-project --extra train --extra gpu
COPY . .
RUN uv sync --frozen --extra train --extra gpu

EXPOSE 8000
# Override MODEL with a mounted merged checkpoint, e.g. -v $PWD/outputs:/app/outputs
CMD ["make", "serve"]
