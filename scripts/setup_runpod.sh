#!/bin/bash
# =============================================================================
# RunPod pod setup script
# Run once after the pod starts (or paste into the RunPod "On-Start" command).
#
# Usage:
#   bash /workspace/scripts/setup_runpod.sh
#   bash /workspace/scripts/setup_runpod.sh --download   # also fetch datasets
# =============================================================================

set -euo pipefail

WORKSPACE="/workspace"
cd "$WORKSPACE"

echo "=== RunPod Setup ==="
echo "Workspace : $WORKSPACE"
nvidia-smi --query-gpu=name,memory.total --format=csv,noheader 2>/dev/null || echo "(nvidia-smi not available)"
echo ""

# ---------------------------------------------------------------------------
# 1. Install Python dependencies
# ---------------------------------------------------------------------------
echo "--- Installing Python dependencies ---"
pip install --no-cache-dir -r requirements.txt
echo ""

# ---------------------------------------------------------------------------
# 2. Ensure directory structure exists
# ---------------------------------------------------------------------------
mkdir -p data/raw data/processed models/checkpoints models/tokenizers logs
echo "Directories OK"
echo ""

# ---------------------------------------------------------------------------
# 3. Verify environment
# ---------------------------------------------------------------------------
echo "--- Verifying environment ---"
python scripts/verify_setup.py
echo ""

# ---------------------------------------------------------------------------
# 4. Optionally download datasets
# ---------------------------------------------------------------------------
if [[ "${1:-}" == "--download" ]]; then
    echo "--- Downloading datasets (alpaca + dolly + oasst + ultrachat) ---"
    python scripts/download_datasets.py --alpaca --dolly --oasst --dailydialog
    echo ""
    echo "--- Preprocessing ---"
    python scripts/preprocess.py --data_dir data/raw
    echo ""
fi

# ---------------------------------------------------------------------------
# Done
# ---------------------------------------------------------------------------
if [[ -f "data/processed/train_tokens.pt" ]]; then
    echo "=== Setup complete. Preprocessed data found. ==="
    echo "Launch training with:"
    echo "  bash scripts/train_runpod.sh"
else
    echo "=== Setup complete. No preprocessed data yet. ==="
    echo "Next steps:"
    echo "  1. python scripts/download_datasets.py --alpaca --dolly --oasst --dailydialog"
    echo "  2. python scripts/preprocess.py --data_dir data/raw"
    echo "  3. bash scripts/train_runpod.sh"
fi
