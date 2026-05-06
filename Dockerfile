FROM python:3.12-slim

RUN apt-get update && apt-get install -y \
    build-essential \
    libgl1 \
    libglib2.0-0 \
    libsm6 \
    libxext6 \
    libxrender-dev \
    libgomp1 \
    wget \
    git \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

COPY requirements.txt .

# Install CPU-only PyTorch first. The default PyPI wheel bundles CUDA (~2.3 GB);
# the CPU wheel is ~250 MB. We install it explicitly so the second RUN (which
# installs the rest of requirements.txt) sees torch already satisfied and skips it.
RUN pip install --no-cache-dir \
    torch==2.8.0 torchvision \
    --index-url https://download.pytorch.org/whl/cpu

RUN pip install --no-cache-dir -r requirements.txt

COPY . .

RUN mkdir -p trajectories uploads

EXPOSE 8000

CMD ["sh", "-c", "uvicorn src.api.main:app --host 0.0.0.0 --port ${PORT:-8000} --workers 1"]