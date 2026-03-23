FROM nvidia/cuda:12.1.1-cudnn8-devel-ubuntu22.04
ARG BUILD_CUDA_OPS=1

ENV DEBIAN_FRONTEND=noninteractive
ENV UV_LINK_MODE=copy
ENV UV_PYTHON=3.12
ENV UV_PROJECT_ENVIRONMENT=/opt/venv
ENV VIRTUAL_ENV=/opt/venv
ENV PIP_NO_CACHE_DIR=1
ENV PATH="/opt/venv/bin:/root/.local/bin:${PATH}"

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

# Install dependencies first for better layer caching.
COPY pyproject.toml ./
RUN uv python install 3.12
RUN uv sync --no-dev --python 3.12

# Copy project source.
COPY . .

# Compile Deformable Attention CUDA ops (GPU build).
RUN if [ "$BUILD_CUDA_OPS" = "1" ]; then \
      cd models/ops && sh make.sh; \
    else \
      echo "Skipping CUDA ops build (BUILD_CUDA_OPS=${BUILD_CUDA_OPS})"; \
    fi

CMD ["bash"]
