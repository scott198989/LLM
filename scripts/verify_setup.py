"""
Verify the training environment: CUDA, GPU, and all required libraries.
Run this after setup to confirm everything is working.
"""

import sys

def check(label, fn):
    try:
        result = fn()
        print(f"  [OK] {label}: {result}")
        return True
    except Exception as e:
        print(f"  [FAIL] {label}: {e}")
        return False

print("\n=== Python ===")
print(f"  Version: {sys.version}")

print("\n=== PyTorch + CUDA ===")
import torch
check("PyTorch version", lambda: torch.__version__)
check("CUDA available", lambda: torch.cuda.is_available())
check("CUDA version", lambda: torch.version.cuda)
check("cuDNN version", lambda: torch.backends.cudnn.version())
check("GPU count", lambda: torch.cuda.device_count())
if torch.cuda.is_available():
    check("GPU name", lambda: torch.cuda.get_device_name(0))
    check("GPU memory (GB)", lambda: f"{torch.cuda.get_device_properties(0).total_memory / 1e9:.2f}")
    # Quick tensor op on GPU
    check("GPU tensor op", lambda: (torch.randn(1000, 1000, device='cuda') @ torch.randn(1000, 1000, device='cuda')).shape)

print("\n=== Libraries ===")
import transformers
check("transformers", lambda: transformers.__version__)

import datasets
check("datasets", lambda: datasets.__version__)

import tokenizers
check("tokenizers", lambda: tokenizers.__version__)

import numpy
check("numpy", lambda: numpy.__version__)

import matplotlib
check("matplotlib", lambda: matplotlib.__version__)

import tqdm
check("tqdm", lambda: tqdm.__version__)

import accelerate
check("accelerate", lambda: accelerate.__version__)

print("\n=== Accelerate Config ===")
from accelerate import Accelerator
accelerator = Accelerator()
check("Accelerate device", lambda: str(accelerator.device))
check("Mixed precision", lambda: str(accelerator.mixed_precision))

print("\n=== Project Structure ===")
import os
dirs = ["data/raw", "data/processed", "models/checkpoints", "models/tokenizers", "scripts", "logs"]
root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
for d in dirs:
    path = os.path.join(root, d)
    status = "OK" if os.path.isdir(path) else "MISSING"
    print(f"  [{status}] {d}/")

print("\n=== Setup Complete ===\n")
