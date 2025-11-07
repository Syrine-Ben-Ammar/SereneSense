# syntax=docker/dockerfile:1.4

ARG CUDA_VERSION=12.2.0
ARG UBUNTU_VERSION=22.04
ARG PYTHON_VERSION=3.10

###############################################
# Builder image - installs dependencies
###############################################
FROM nvidia/cuda:${CUDA_VERSION}-cudnn8-devel-ubuntu${UBUNTU_VERSION} AS builder

ENV DEBIAN_FRONTEND=noninteractive \
    PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1

RUN apt-get update && \
    apt-get install -y --no-install-recommends \
        python${PYTHON_VERSION} \
        python3-pip \
        python3-venv \
        build-essential \
        git \
        libsndfile1 \
        ffmpeg \
        curl && \
    rm -rf /var/lib/apt/lists/*

RUN python${PYTHON_VERSION} -m venv /venv
ENV PATH="/venv/bin:${PATH}"

WORKDIR /workspace

COPY requirements.txt pyproject.toml setup.py README.md ./ 

RUN pip install --upgrade pip setuptools wheel && \
    pip install --no-cache-dir -r requirements.txt

COPY . .

RUN pip install --no-cache-dir -e .

###############################################
# Runtime image - minimal footprint
###############################################
FROM nvidia/cuda:${CUDA_VERSION}-cudnn8-runtime-ubuntu${UBUNTU_VERSION} AS runtime

ARG PYTHON_VERSION

ENV DEBIAN_FRONTEND=noninteractive \
    PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PATH="/venv/bin:${PATH}" \
    PYTHONPATH="/workspace"

RUN apt-get update && \
    apt-get install -y --no-install-recommends \
        python${PYTHON_VERSION} \
        python3-pip \
        python3-venv \
        libsndfile1 \
        ffmpeg \
        curl && \
    rm -rf /var/lib/apt/lists/*

COPY --from=builder /venv /venv

WORKDIR /workspace
COPY . .

RUN groupadd --gid 1000 seren && \
    useradd --uid 1000 --gid seren --create-home seren && \
    chown -R seren:seren /workspace

USER seren

VOLUME ["/workspace/data", "/workspace/models", "/workspace/logs"]

EXPOSE 8000

ENV SERENESENSE_ENV=production \
    SERENESENSE_DATA_DIR=/workspace/data \
    SERENESENSE_MODEL_DIR=/workspace/models \
    SERENESENSE_LOG_DIR=/workspace/logs

HEALTHCHECK --interval=30s --timeout=5s --retries=3 CMD curl -f http://localhost:8000/health || exit 1

ENTRYPOINT ["uvicorn", "src.core.deployment.api.fastapi_server:app", "--host", "0.0.0.0", "--port", "8000"]
