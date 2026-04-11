#!/bin/bash
# =============================================================================
# RunPod training launcher
# Auto-selects batch size, gradient accumulation, and model size
# based on detected GPU VRAM.  Extra flags are passed straight through.
#
# Usage:
#   bash scripts/train_runpod.sh                       # auto-configure
#   bash scripts/train_runpod.sh --compile             # + torch.compile (max-autotune)
#   bash scripts/train_runpod.sh --max_epochs 2        # override epochs
#   bash scripts/train_runpod.sh --no_grad_ckpt        # disable grad checkpointing
#
# GPU presets
#   H100 / H200  (80 GB) : batch=256  grad_accum=2  workers=16  no grad-ckpt
#   A100         (80 GB) : batch=256  grad_accum=2  workers=12
#   A100         (40 GB) : batch=128  grad_accum=4  workers=8
#   A6000        (48 GB) : batch=128  grad_accum=4  workers=8
#   RTX 4090     (24 GB) : batch=64   grad_accum=4  workers=6
#   RTX 3090     (24 GB) : batch=48   grad_accum=4  workers=6
#   ≤ 16 GB              : batch=16   grad_accum=8  workers=4
# =============================================================================

set -euo pipefail

WORKSPACE="/workspace"
cd "$WORKSPACE"

echo ""
echo "================================================================"
echo "  RunPod Training Launcher"
echo "================================================================"

# ---------------------------------------------------------------------------
# Detect GPU name and VRAM
# ---------------------------------------------------------------------------
GPU_NAME=$(nvidia-smi --query-gpu=name          --format=csv,noheader 2>/dev/null | head -1 || echo "unknown")
VRAM_MB=$(nvidia-smi  --query-gpu=memory.total  --format=csv,noheader,nounits 2>/dev/null | head -1 || echo "0")
VRAM_GB=$(( VRAM_MB / 1024 ))

echo "  GPU  : $GPU_NAME"
echo "  VRAM : ~${VRAM_GB} GB"
echo ""

# ---------------------------------------------------------------------------
# Select preset based on GPU name / VRAM
# ---------------------------------------------------------------------------
# Defaults (conservative)
BATCH=16; ACCUM=8; WORKERS=4; EXTRA_FLAGS=""

GPU_LOWER=$(echo "$GPU_NAME" | tr '[:upper:]' '[:lower:]')

if echo "$GPU_LOWER" | grep -qE "h100|h200"; then
    BATCH=256; ACCUM=2; WORKERS=16
    EXTRA_FLAGS="--no_grad_ckpt"
    PRESET="H100/H200 (80 GB)"

elif echo "$GPU_LOWER" | grep -q "a100" && (( VRAM_GB >= 70 )); then
    BATCH=256; ACCUM=2; WORKERS=12
    PRESET="A100 80 GB"

elif echo "$GPU_LOWER" | grep -q "a100"; then
    BATCH=128; ACCUM=4; WORKERS=8
    PRESET="A100 40 GB"

elif echo "$GPU_LOWER" | grep -qE "a6000|a5000"; then
    BATCH=128; ACCUM=4; WORKERS=8
    PRESET="A6000/A5000 (48 GB)"

elif echo "$GPU_LOWER" | grep -qE "4090|3090" || (( VRAM_GB >= 20 )); then
    BATCH=64; ACCUM=4; WORKERS=6
    PRESET="RTX 4090/3090 (24 GB)"

elif (( VRAM_GB >= 16 )); then
    BATCH=32; ACCUM=4; WORKERS=4
    PRESET="16 GB GPU"
fi

EFF_BATCH=$(( BATCH * ACCUM ))

echo "  Preset         : $PRESET"
echo "  Batch size     : $BATCH  (grad_accum=$ACCUM  →  effective=$EFF_BATCH)"
echo "  Workers        : $WORKERS"
echo "  Extra flags    : ${EXTRA_FLAGS:-none}"
echo ""
echo "  Pass --compile to enable torch.compile (max-autotune, ~30% faster)"
echo "  Pass --max_epochs N to override epoch count"
echo "================================================================"
echo ""

# ---------------------------------------------------------------------------
# Launch
# ---------------------------------------------------------------------------
python scripts/train.py         \
    --batch_size  "$BATCH"      \
    --grad_accum  "$ACCUM"      \
    --num_workers "$WORKERS"    \
    --max_epochs  3             \
    $EXTRA_FLAGS                \
    "$@"
