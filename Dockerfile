# Base image: RunPod's official PyTorch image with CUDA 12.4
# Matches the cu124 wheels in requirements.txt
FROM runpod/pytorch:2.4.0-py3.11-cuda12.4.1-devel-ubuntu22.04

WORKDIR /workspace

# Install Python dependencies (PyTorch already present in base image)
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy project files
COPY . .

# Create expected directories
RUN mkdir -p data/raw data/processed models/checkpoints models/tokenizers logs

# Default: drop into bash so RunPod can run setup_runpod.sh or train_runpod.sh
CMD ["bash"]
