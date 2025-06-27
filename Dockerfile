# 1) CUDA 런타임 베이스 이미지
FROM --platform=linux/amd64 nvidia/cuda:12.1.1-cudnn8-runtime-ubuntu22.04

ARG ENV_NAME=coding
ARG PYTHON_VERSION=3.10

# 2) 시스템 패키지 설치
RUN apt-get update && \
    apt-get install -y --no-install-recommends \
      git build-essential cargo rustc wget bzip2 && \
    apt-get clean && rm -rf /var/lib/apt/lists/*

# 3) Miniconda 설치
ENV CONDA_DIR=/opt/conda
RUN wget --quiet https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh \
      -O /tmp/conda.sh && \
    bash /tmp/conda.sh -b -p $CONDA_DIR && \
    rm /tmp/conda.sh
ENV PATH=$CONDA_DIR/bin:$PATH

# 4) Conda env 생성
RUN conda create -y -n $ENV_NAME python=$PYTHON_VERSION && \
    conda clean -afy

# 5) SHELL 을 coding env 로 설정
SHELL ["conda", "run", "-n", "coding", "/bin/bash", "-c"]

WORKDIR /home/mixlab/tabular/shchoi
COPY . .

# 6) pip 업데이트 및 GPU용 PyTorch + 필수 패키지 설치
RUN pip install --upgrade pip && \
    pip install --no-cache-dir \
      torch torchvision torchaudio --extra-index-url https://download.pytorch.org/whl/cu121 && \
    pip install --no-cache-dir \
      hydra-core omegaconf wandb scikit-learn scipy

# 7) 로컬 패키지(프로젝트) 설치
RUN pip install -e .
