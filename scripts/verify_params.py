"""
HAVOC parameter-count verifier.

Builds the model from a config (default: configs/havoc-50m.json or the
canonical preset baked into config.py) and prints a layer-by-layer
parameter breakdown plus the total. Exits 0 if the unique parameter
count is within the configured tolerance of `expected_param_count`,
exits 1 otherwise — handy as a CI gate.

Usage:
    python scripts/verify_params.py
    python scripts/verify_params.py --config configs/havoc-50m.json
    python scripts/verify_params.py --target 50000000 --tolerance 0.05
"""

from __future__ import annotations

import argparse
import os
import sys

import torch.nn as nn

_HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, _HERE)

from config import HavocConfig, default_50m_config         # noqa: E402
from model  import HavocModel, count_params                # noqa: E402


def print_breakdown(model: nn.Module, cfg: HavocConfig) -> dict[str, int]:
    """Print per-leaf-module param counts and return aggregate stats."""
    sep = "-" * 78
    print("\n" + "=" * 78)
    print("  HAVOC parameter breakdown")
    print("=" * 78)
    print(f"  {'Module':<46} {'#Params':>14}  Shape")
    print(sep)

    seen_ids: set[int] = set()
    for name, module in model.named_modules():
        if list(module.children()):
            continue
        own = [p for p in module.parameters(recurse=False)]
        if not own:
            continue
        n = 0
        for p in own:
            if id(p) in seen_ids:
                continue
            seen_ids.add(id(p))
            n += p.numel()
        if n == 0:
            continue
        shape = ""
        if hasattr(module, "weight") and module.weight is not None:
            shape = str(list(module.weight.shape))
        disp = name if len(name) <= 46 else "..." + name[-43:]
        print(f"  {disp:<46} {n:>14,}  {shape}")
    print(sep)

    counts = count_params(model)
    print(f"  {'Total':<46} {counts['total']:>14,}")
    print(f"  {'Unique (deduplicated, weight-tied)':<46} {counts['unique']:>14,}")
    print(f"  {'Trainable':<46} {counts['trainable']:>14,}")
    print(sep)

    # Architecture summary
    print(f"  Layers / Heads / Hidden  : {cfg.num_layers} / {cfg.num_heads} / {cfg.hidden_size}")
    print(f"  Head dim                 : {cfg.head_dim}")
    print(f"  Intermediate (SwiGLU)    : {cfg.intermediate_size}")
    print(f"  Vocab                    : {cfg.vocab_size:,}")
    print(f"  Context length           : {cfg.max_seq_len}")
    print(f"  RoPE base                : {cfg.rope_base}")
    print(f"  Tied embeddings          : {cfg.tied_embeddings}")
    print("=" * 78 + "\n")
    return counts


def main() -> int:
    parser = argparse.ArgumentParser(description="Verify HAVOC parameter count.")
    parser.add_argument("--config", default=None,
                        help="Path to a HavocConfig JSON. Defaults to the preset in config.py.")
    parser.add_argument("--target", type=int, default=None,
                        help="Override the expected parameter count.")
    parser.add_argument("--tolerance", type=float, default=None,
                        help="Override the allowed fractional tolerance (e.g. 0.02 for 2 %%).")
    args = parser.parse_args()

    cfg = HavocConfig.from_json(args.config) if args.config else default_50m_config()
    target    = args.target    if args.target    is not None else cfg.expected_param_count
    tolerance = args.tolerance if args.tolerance is not None else cfg.param_count_tolerance

    model  = HavocModel(cfg)
    counts = print_breakdown(model, cfg)
    actual = counts["unique"]

    delta_abs = actual - target
    delta_pct = delta_abs / target if target else 0.0

    print(f"  Target        : {target:>14,}")
    print(f"  Actual unique : {actual:>14,}")
    print(f"  Delta         : {delta_abs:+,}  ({delta_pct*100:+.2f} %)")
    print(f"  Tolerance     : +/-{tolerance*100:.1f} %\n")

    ok = abs(delta_pct) <= tolerance
    print(("  PASS - within tolerance" if ok else "  FAIL - outside tolerance") + "\n")
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
