# CUDA‑enabled PyTorch base image
FROM pytorch/pytorch:2.2.1-cuda12.1-cudnn8-runtime

# 1️⃣  System packages you'll inevitably need
RUN apt-get update && \
    apt-get install -y --no-install-recommends \
    git build-essential vim curl && \
    rm -rf /var/lib/apt/lists/*

# 2️⃣  Copy Python dependency list and install it
WORKDIR /workspace
COPY docker/requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# 3️⃣  Expose common ports
EXPOSE 8888 6006

CMD ["bash"] 