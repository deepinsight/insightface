FROM nvidia/cuda:12.8.0-cudnn-runtime-ubuntu22.04

ENV DEBIAN_FRONTEND=noninteractive \
    PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1

# System deps (opencv runtime libs + basic build tools)
RUN apt-get update && apt-get install -y --no-install-recommends \
    python3 python3-pip python3-dev python3-venv \
    git curl ca-certificates \
    build-essential pkg-config \
    libglib2.0-0 libsm6 libxext6 libxrender1 \
    && rm -rf /var/lib/apt/lists/*

RUN python3 -m pip install --upgrade pip setuptools wheel

# ---- PyTorch (CUDA 12.8 / cu128) ----
ARG TORCH_VERSION=2.10.0
ARG TORCHVISION_VERSION=0.25.0
ARG TORCHAUDIO_VERSION=2.10.0

# Use extra-index-url so deps can still come from PyPI
RUN python3 -m pip install \
    torch==${TORCH_VERSION} torchvision==${TORCHVISION_VERSION} torchaudio==${TORCHAUDIO_VERSION} \
    --extra-index-url https://download.pytorch.org/whl/cu128

# ---- Your packages ----
RUN python3 -m pip install \
    "numpy<1.24" \
    tensorboard \
    easydict \
    mxnet \
    onnx \
    scikit-learn \
    tqdm \
    wandb \
    "opencv-python-headless<4.10"

# Put code into the image
WORKDIR /workspace
COPY . /workspace

# If the repo has a requirements.txt, uncomment:
# RUN python3 -m pip install -r requirements.txt

CMD ["bash"]
