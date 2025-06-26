FROM nvidia/cuda:12.1.1-cudnn8-runtime-ubuntu22.04

# Set up basic environment
ENV DEBIAN_FRONTEND=noninteractive
RUN apt-get update && apt-get install -y \
    git wget curl zip unzip \
    build-essential ninja-build \
    libglib2.0-0 libsm6 libxext6 libxrender-dev \
    ffmpeg git-lfs \
    python3.10 python3.10-venv python3-pip \
    && apt-get clean

# Symlink python3.10 as default
RUN ln -s /usr/bin/python3.10 /usr/bin/python && ln -s /usr/bin/pip3 /usr/bin/pip

# Create and activate virtualenv
WORKDIR /workspace
RUN python -m venv venv
ENV PATH="/workspace/venv/bin:$PATH"

# Upgrade pip
RUN pip install --upgrade pip

# Clone CogACT
RUN git clone https://github.com/microsoft/CogACT.git
WORKDIR /workspace/CogACT

# Install Python dependencies
RUN pip install -e ".[train]" \
    && pip install packaging ninja

# Optional: Install Flash-Attention v2
RUN pip install "flash-attn==2.5.5" --no-build-isolation

# Set environment variables
ENV TOKENIZERS_PARALLELISM=false
ENV PYTHONUNBUFFERED=1
