FROM nvidia/cuda:12.1.1-cudnn8-devel-ubuntu22.04

ENV DEBIAN_FRONTEND=noninteractive
ENV UV_LINK_MODE=copy
ENV UV_PYTHON=python3
ENV PIP_NO_CACHE_DIR=1

WORKDIR /workspace

RUN apt-get update && apt-get install -y --no-install-recommends \
    bash \
    build-essential \
    ca-certificates \
    curl \
    git \
    python3 \
    python3-dev \
    python3-pip \
    python3-venv \
    && rm -rf /var/lib/apt/lists/*

RUN curl -LsSf https://astral.sh/uv/install.sh | sh
ENV PATH="/root/.local/bin:${PATH}"

# Install dependencies first for better layer caching.
COPY pyproject.toml ./
RUN uv sync --no-dev

# Copy project source.
COPY . .

# Make venv python the default.
ENV PATH="/workspace/.venv/bin:${PATH}"

# Compile Deformable Attention CUDA ops.
RUN cd models/ops && python setup.py build install

CMD ["bash"]
