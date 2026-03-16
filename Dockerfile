# === Stage 1: Build dependencies ===
FROM nvidia/cuda:12.8.1-devel-ubuntu22.04 AS builder

ENV DEBIAN_FRONTEND=noninteractive
ENV PYTHONDONTWRITEBYTECODE=1

# Install Python 3.12 and build essentials
RUN apt-get update && apt-get install -y --no-install-recommends \
    software-properties-common && \
    add-apt-repository ppa:deadsnakes/ppa && \
    apt-get update && apt-get install -y --no-install-recommends \
    python3.12 \
    python3.12-dev \
    python3.12-venv \
    build-essential \
    curl \
    && rm -rf /var/lib/apt/lists/*

# Install uv
COPY --from=ghcr.io/astral-sh/uv:latest /uv /usr/local/bin/uv

WORKDIR /app

# Copy dependency files first for layer caching
COPY pyproject.toml requirements.txt ./

# Create venv and install dependencies
RUN uv venv /app/.venv --python python3.12 && \
    . /app/.venv/bin/activate && \
    uv pip install -r requirements.txt

# === Stage 2: Runtime ===
FROM nvidia/cuda:12.8.1-runtime-ubuntu22.04

ENV DEBIAN_FRONTEND=noninteractive
ENV PYTHONDONTWRITEBYTECODE=1
ENV PYTHONUNBUFFERED=1

RUN apt-get update && apt-get install -y --no-install-recommends \
    software-properties-common && \
    add-apt-repository ppa:deadsnakes/ppa && \
    apt-get update && apt-get install -y --no-install-recommends \
    python3.12 \
    python3.12-venv \
    libfontconfig1 \
    libgl1 \
    libegl1 \
    libice6 \
    libsm6 \
    libxkbcommon-x11-0 \
    libxrender1 \
    libxcb-cursor0 \
    libxcb-render0 \
    libxcb-shape0 \
    libxcb-xfixes0 \
    libdbus-1-3 \
    libgomp1 \
    && rm -rf /var/lib/apt/lists/*

# Copy venv from builder
COPY --from=builder /app/.venv /app/.venv
# Numba JIT-compiles CUDA kernels at runtime and needs NVVM/libdevice.
COPY --from=builder /usr/local/cuda/nvvm /usr/local/cuda/nvvm

# Activate venv by prepending to PATH
ENV PATH="/app/.venv/bin:$PATH"
ENV VIRTUAL_ENV="/app/.venv"
ENV LD_LIBRARY_PATH="/usr/local/cuda/nvvm/lib64:/usr/local/cuda/lib64:${LD_LIBRARY_PATH}"
ENV NUMBA_CUDA_NVVM="/usr/local/cuda/nvvm/lib64/libnvvm.so"
ENV NUMBA_CUDA_LIBDEVICE="/usr/local/cuda/nvvm/libdevice"

WORKDIR /app

# Copy project source
COPY . .

# Install the project itself with viewer extras available in the container.
RUN /app/.venv/bin/python -m ensurepip --upgrade && \
    /app/.venv/bin/python -m pip install -e ".[viz]"

# Default: drop into bash for interactive use
CMD ["bash"]
