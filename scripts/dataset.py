"""
LM dataset and DataLoader factory.

Loads preprocessed token tensors from data/processed/ and wraps them
in PyTorch Dataset / DataLoader objects ready for train.py.

Can also be run standalone to inspect batches:
    python scripts/dataset.py
"""

import json
import os

import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset


class TokenDataset(Dataset):
    """
    Slices a flat token tensor into overlapping (x, y) pairs.

    x = tokens[i : i + block_size]
    y = tokens[i + 1 : i + block_size + 1]   (next-token targets)

    stride controls overlap:
      stride == block_size  → non-overlapping chunks (faster epoch, less data reuse)
      stride == 1           → maximum overlap (slowest, most thorough)
    """

    def __init__(self, tokens: torch.Tensor, block_size: int, stride: int | None = None):
        self.tokens     = tokens
        self.block_size = block_size
        self.stride     = stride if stride is not None else block_size

    def __len__(self):
        return max(0, (len(self.tokens) - self.block_size - 1) // self.stride + 1)

    def __getitem__(self, idx: int):
        start = idx * self.stride
        x = self.tokens[start     : start + self.block_size]
        y = self.tokens[start + 1 : start + self.block_size + 1]
        return x, y


def build_dataloaders(
    processed_dir: str        = "data/processed",
    batch_size:    int        = 16,
    num_workers:   int        = 0,
    pin_memory:    bool | None = None,   # None = auto (True when num_workers > 0)
    stride:        int | None  = None,
) -> tuple[DataLoader, DataLoader, dict]:
    """
    Load train.bin + val.bin, return (train_loader, val_loader, info).

    info dict contains vocab_size, block_size, etc. from tokenizer_info.json.
    """
    info_path = os.path.join(processed_dir, "tokenizer_info.json")
    if not os.path.exists(info_path):
        raise FileNotFoundError(
            f"{info_path} not found.\n"
            "Run:  python scripts/preprocess.py --data_dir data/raw"
        )
    with open(info_path) as f:
        info = json.load(f)

    block_size = info["block_size"]
    token_dtype = np.dtype(info.get("token_dtype", "uint16"))

    train_tokens = _load_token_file(
        processed_dir,
        preferred=info.get("train_file", "train.bin"),
        legacy="train_tokens.pt",
        dtype=token_dtype,
    )
    val_tokens = _load_token_file(
        processed_dir,
        preferred=info.get("val_file", "val.bin"),
        legacy="val_tokens.pt",
        dtype=token_dtype,
    )

    train_ds = TokenDataset(train_tokens, block_size, stride=stride)
    val_ds   = TokenDataset(val_tokens,   block_size, stride=block_size)  # val always non-overlapping

    if len(train_ds) == 0:
        raise ValueError(
            f"Training split is too small for block_size={block_size}. "
            "Re-run preprocess.py with more data."
        )
    if len(val_ds) == 0:
        raise ValueError(
            f"Validation split is too small for block_size={block_size}. "
            "Re-run preprocess.py or reduce the validation split only if you know why."
        )

    # pin_memory only helps when workers > 0 and a CUDA device is available
    _pin      = pin_memory if pin_memory is not None else (num_workers > 0)
    _persist  = num_workers > 0   # keep workers alive between epochs
    _prefetch = 2 if num_workers > 0 else None

    train_loader = DataLoader(
        train_ds,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=_pin,
        drop_last=True,
        persistent_workers=_persist,
        prefetch_factor=_prefetch,
    )
    val_loader = DataLoader(
        val_ds,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=_pin,
        drop_last=False,
        persistent_workers=_persist,
        prefetch_factor=_prefetch,
    )

    return train_loader, val_loader, info


def _load_token_file(processed_dir: str, preferred: str, legacy: str, dtype: np.dtype) -> torch.Tensor:
    """Load token IDs from the new .bin format, with legacy .pt fallback."""
    bin_path = os.path.join(processed_dir, preferred)
    if os.path.exists(bin_path):
        arr = np.fromfile(bin_path, dtype=dtype)
        return torch.from_numpy(arr.astype(np.int64))

    legacy_path = os.path.join(processed_dir, legacy)
    if os.path.exists(legacy_path):
        return torch.load(legacy_path, weights_only=True)

    raise FileNotFoundError(
        f"Neither {bin_path} nor {legacy_path} exists.\n"
        "Run:  python scripts/preprocess.py --data_dir data/raw"
    )


# ---------------------------------------------------------------------------
# Standalone inspection
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    import sys

    processed_dir = "data/processed"
    if not os.path.exists(os.path.join(processed_dir, "tokenizer_info.json")):
        print("No processed data found. Run preprocess.py first.")
        sys.exit(1)

    train_loader, val_loader, info = build_dataloaders(batch_size=4)

    print("\n=== Dataset Info ===")
    for k, v in info.items():
        print(f"  {k}: {v}")

    train_ds = train_loader.dataset
    val_ds   = val_loader.dataset
    print(f"\n  Train examples : {len(train_ds):,}")
    print(f"  Val   examples : {len(val_ds):,}")
    print(f"  Train batches  : {len(train_loader):,}  (batch_size=4)")
    print(f"  Val   batches  : {len(val_loader):,}")

    # Load tokenizer to decode
    try:
        from transformers import GPT2TokenizerFast
        tokenizer = GPT2TokenizerFast.from_pretrained("gpt2")
        tokenizer.add_special_tokens({"additional_special_tokens": ["<|sep|>"]})
        can_decode = True
    except Exception:
        can_decode = False

    print("\n=== Sample Batches ===")
    for split_name, loader in [("TRAIN", train_loader), ("VAL", val_loader)]:
        print(f"\n--- {split_name} batch ---")
        x, y = next(iter(loader))
        print(f"  x shape : {x.shape}  dtype={x.dtype}")
        print(f"  y shape : {y.shape}  dtype={y.dtype}")
        print(f"  x[0] first 20 token IDs : {x[0, :20].tolist()}")
        print(f"  y[0] first 20 token IDs : {y[0, :20].tolist()}")
        if can_decode:
            print(f"  x[0] decoded (first 200 chars):\n    {tokenizer.decode(x[0].tolist())[:200]}")

    print("\n=== Batch shapes look correct — ready for training ===\n")
