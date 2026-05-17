FROM python:3.10-slim

# Environment Variables
ENV DEBIAN_FRONTEND=noninteractive
ENV PYTHONUNBUFFERED=1

ENV PIP_NO_CACHE_DIR=1
ENV GRADIO_PORT=7860

# System dependencies
RUN apt-get update && apt-get install -y \
    git \
    gcc \
    g++ \
    make \
    cmake \
    wget \
    curl \
    libglib2.0-0 \
    libsm6 \
    libxext6 \
    libxrender-dev \
    libgomp1 \
    libgl1 \
    && rm -rf /var/lib/apt/lists/*

# Workdir
WORKDIR /app

# Upgrade pip
RUN pip install --upgrade pip setuptools wheel

# Copy requirements
COPY requirements.txt .

# CPU
# Install PyTorch CPU first
RUN pip install --no-cache-dir \
    torch torchvision \
    --index-url https://download.pytorch.org/whl/cpu

# Install PaddlePaddle CPU
RUN pip install paddlepaddle==3.2.0 \
    -i https://www.paddlepaddle.org.cn/packages/stable/cpu/

# GPU
# Install PyTorch for GPU
# RUN pip install torch torchvision --index-url https://download.pytorch.org/whl/cu118

# Install PaddlePaddle for GPU
# RUN pip install paddlepaddle-gpu==3.2.0 \
#     -i https://www.paddlepaddle.org.cn/packages/stable/cu118/

# Install remaining packages
RUN pip install -r requirements.txt

RUN pip install --no-build-isolation \
    'git+https://github.com/facebookresearch/detectron2.git'

RUN python -m pip install "paddleocr[all]"

# Clone repositories
RUN git clone https://github.com/facebookresearch/detectron2.git

RUN git clone https://github.com/facebookresearch/Mask2Former.git

RUN git clone https://github.com/PaddlePaddle/PaddleOCR.git

# Install PaddleOCR extra requirements
RUN pip install -r PaddleOCR/requirements.txt

# Copy project files
COPY . .

# Expose Gradio port
EXPOSE 7860

# Default command
CMD ["python", "run.py"]