FROM nvidia/cuda:12.1.1-cudnn8-devel-ubuntu22.04

# Set noninteractive frontend
ENV DEBIAN_FRONTEND=noninteractive

# Install system dependencies
RUN apt-get update && apt-get install -y \
    wget git curl zip unzip \
    build-essential ninja-build \
    libglib2.0-0 libsm6 libxext6 libxrender-dev \
    ffmpeg git-lfs vim tmux\
    && apt-get clean

# Install Miniconda
ENV CONDA_DIR=/opt/conda
ENV PATH=$CONDA_DIR/bin:$PATH

RUN wget --quiet https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh -O ~/miniconda.sh && \
    bash ~/miniconda.sh -b -p $CONDA_DIR && \
    rm ~/miniconda.sh && \
    conda clean -afy

# Create CogACT environment
RUN conda create -n cogact python=3.10 -y

# Activate env by default
SHELL ["conda", "run", "-n", "cogact", "/bin/bash", "-c"]

# Install PyTorch with CUDA 12.1 support
RUN conda install pytorch==2.2.2 torchvision==0.17.2 torchaudio==2.2.2 pytorch-cuda=12.1 -c pytorch -c nvidia -y

# Pin NumPy < 2.0 for compatibility
RUN pip install "numpy<2"

# Clone CogACT
WORKDIR /workspace
RUN git clone https://github.com/microsoft/CogACT.git
WORKDIR /workspace/CogACT

# Install CogACT + flash-attn
RUN pip install -e ".[train]" \
    && pip install packaging ninja flash-attn==2.5.5 --no-build-isolation

# Environment variables
ENV TOKENIZERS_PARALLELISM=false
ENV PYTHONUNBUFFERED=1

# Default to conda shell
ENTRYPOINT ["conda", "run", "-n", "cogact", "/bin/bash"]
