"""
HAVOC shoes-model pretraining loop.

Loads uint16 .bin shards produced by data/prepare_pretrain.py, samples
random block_size+1 windows across shards, and trains HavocGPT on
next-token prediction with bf16 mixed precision on a single GPU.

Defaults (single RTX 4090):
    1 epoch over ~1.7B tokens
    LR 3e-4 cosine -> 3e-5, 1% warmup
    AdamW (0.9, 0.95, wd=0.1)
    effective batch = 256K tokens via grad accumulation
    log every 10 steps, checkpoint every 1000

Usage:
    python train/pretrain.py \\
        --shards_dir data/shards \\
        --ckpt_dir   checkpoints \\
        --total_tokens 1_700_000_000

Resume:
    python train/pretrain.py --resume checkpoints/latest.pt
"""

from __future__ import annotations

import argparse
import csv
import math
import os
import re
import sys
import time
from glob import glob

import numpy as np
import torch
import torch.nn.functional as F

_HERE = os.path.dirname(os.path.abspath(__file__))
_ROOT = os.path.dirname(_HERE)
sys.path.insert(0, _ROOT)

from model.havoc import HavocConfig, HavocGPT   # noqa: E402


# ── Shard loader ───────────────────────────────────────────────────────────


class ShardSampler:
    """
    Memmaps every shard and samples random (x, y) windows of length block_size.

    Each shard is uint16. We pick a shard proportional to its length, then a
    uniform random offset within that shard. block_size + 1 tokens are read so
    we can build the (input, target) pair.
    """

    def __init__(self, shard_paths: list[str], block_size: int, seed: int):
        if not shard_paths:
            raise FileNotFoundError("no shards provided")
        self.block_size = block_size
        self.rng = np.random.default_rng(seed)
        self.maps: list[np.memmap] = []
        self.lens: list[int] = []
        for p in shard_paths:
            m = np.memmap(p, dtype=np.uint16, mode="r")
            if len(m) <= block_size + 1:
                continue
            self.maps.append(m)
            self.lens.append(len(m))
        if not self.maps:
            raise ValueError("all shards smaller than block_size+1")
        total = sum(self.lens)
        self.weights = np.asarray(self.lens, dtype=np.float64) / total
        self.total_tokens = total

    def sample(self, batch_size: int, device: str) -> tuple[torch.Tensor, torch.Tensor]:
        bs = self.block_size
        xs = np.empty((batch_size, bs), dtype=np.int64)
        ys = np.empty((batch_size, bs), dtype=np.int64)
        idx = self.rng.choice(len(self.maps), size=batch_size, p=self.weights)
        for i, si in enumerate(idx):
            m = self.maps[si]
            start = int(self.rng.integers(0, len(m) - bs - 1))
            xs[i] = m[start     : start + bs].astype(np.int64)
            ys[i] = m[start + 1 : start + bs + 1].astype(np.int64)
        x = torch.from_numpy(xs).to(device, non_blocking=True)
        y = torch.from_numpy(ys).to(device, non_blocking=True)
        return x, y


# ── LR schedule ────────────────────────────────────────────────────────────


def cosine_lr(step: int, warmup: int, total: int, max_lr: float, min_lr: float) -> float:
    if step < warmup:
        return max_lr * (step + 1) / max(warmup, 1)
    if step >= total:
        return min_lr
    progress = (step - warmup) / max(total - warmup, 1)
    return min_lr + 0.5 * (max_lr - min_lr) * (1.0 + math.cos(math.pi * progress))


# ── Checkpointing ──────────────────────────────────────────────────────────


def save_checkpoint(path: str, model: HavocGPT, optimizer, step: int,
                    tokens_seen: int, cfg: HavocConfig, args: argparse.Namespace) -> None:
    raw = model._orig_mod if hasattr(model, "_orig_mod") else model
    torch.save({
        "model":       raw.state_dict(),
        "optimizer":   optimizer.state_dict(),
        "step":        step,
        "tokens_seen": tokens_seen,
        "cfg":         cfg.__dict__,
        "args":        vars(args),
    }, path)


_STEP_CKPT_RE = re.compile(r"^step_(\d+)\.pt$")


def prune_old_checkpoints(ckpt_dir: str, keep_n: int) -> list[str]:
    """
    Keep only the `keep_n` most recent `step_*.pt` files (by the integer step
    parsed from the filename); remove the rest.

    Guarantees:
      - Only files matching `step_<digits>.pt` are considered, so `latest.pt`
        and any other file in `ckpt_dir` are never touched.
      - The most-recent N (highest step numbers) are retained, so the
        checkpoint just saved this step is always kept.
      - `keep_n <= 0` disables pruning entirely (no-op).

    Returns the list of removed paths.
    """
    if keep_n <= 0 or not os.path.isdir(ckpt_dir):
        return []
    entries: list[tuple[int, str]] = []
    for name in os.listdir(ckpt_dir):
        m = _STEP_CKPT_RE.match(name)
        if m:
            entries.append((int(m.group(1)), os.path.join(ckpt_dir, name)))
    if len(entries) <= keep_n:
        return []
    entries.sort(key=lambda x: x[0])           # ascending by step number
    to_remove = entries[:-keep_n]               # all but the keep_n most recent
    removed: list[str] = []
    for _, path in to_remove:
        try:
            os.remove(path)
            removed.append(path)
        except OSError as e:
            print(f"  [WARN] could not remove old ckpt {path}: {e}")
    return removed


# ── Training ───────────────────────────────────────────────────────────────


def main() -> int:
    p = argparse.ArgumentParser(description="HAVOC shoes-model pretraining.")
    # Data
    p.add_argument("--shards_dir", default="data/shards")
    p.add_argument("--ckpt_dir",   default="checkpoints")
    p.add_argument("--log_path",   default="data/train_log.csv")
    p.add_argument("--resume",     default=None)
    # Model
    p.add_argument("--block_size", type=int, default=1024)
    p.add_argument("--vocab_size", type=int, default=50257)
    p.add_argument("--n_layer",    type=int, default=12)
    p.add_argument("--n_head",     type=int, default=8)
    p.add_argument("--n_embd",     type=int, default=512)
    p.add_argument("--mlp_ratio",  type=int, default=4)
    p.add_argument("--dropout",    type=float, default=0.0)
    # Schedule
    p.add_argument("--total_tokens",     type=int,   default=1_700_000_000)
    p.add_argument("--batch_size",       type=int,   default=8,
                   help="Per-step micro-batch (sequences).")
    p.add_argument("--effective_tokens", type=int,   default=262_144,
                   help="Tokens per optimizer step (256K default).")
    p.add_argument("--lr",          type=float, default=3.0e-4)
    p.add_argument("--min_lr",      type=float, default=3.0e-5)
    p.add_argument("--warmup_frac", type=float, default=0.01)
    p.add_argument("--weight_decay",type=float, default=0.1)
    p.add_argument("--beta1",       type=float, default=0.9)
    p.add_argument("--beta2",       type=float, default=0.95)
    p.add_argument("--grad_clip",   type=float, default=1.0)
    # Cadence
    p.add_argument("--log_every",   type=int,   default=10)
    p.add_argument("--ckpt_every",  type=int,   default=1000)
    p.add_argument("--keep_n_ckpts",type=int,   default=3,
                   help="Keep only the N most recent step_*.pt files. "
                        "latest.pt is never deleted. 0 disables pruning.")
    p.add_argument("--eval_every",  type=int,   default=500)
    p.add_argument("--eval_iters",  type=int,   default=50)
    # Misc
    p.add_argument("--seed",        type=int,   default=1337)
    p.add_argument("--compile",     action="store_true")
    args = p.parse_args()

    os.makedirs(args.ckpt_dir, exist_ok=True)
    os.makedirs(os.path.dirname(args.log_path) or ".", exist_ok=True)
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    # ── Device ─────────────────────────────────────────────────────────────
    device = "cuda" if torch.cuda.is_available() else "cpu"
    if device == "cuda":
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32       = True
        torch.set_float32_matmul_precision("high")
    dtype = torch.bfloat16 if device == "cuda" and torch.cuda.is_bf16_supported() \
                           else torch.float16 if device == "cuda" \
                           else torch.float32
    print(f"  device  : {device}   dtype : {dtype}")
    if device == "cuda":
        props = torch.cuda.get_device_properties(0)
        print(f"  gpu     : {props.name}  ({props.total_memory/1024**3:.1f} GB)")

    # ── Shards ─────────────────────────────────────────────────────────────
    train_paths = sorted(glob(os.path.join(args.shards_dir, "train_*.bin")))
    val_paths   = sorted(glob(os.path.join(args.shards_dir, "val_*.bin")))
    if not train_paths:
        print(f"ERROR: no train_*.bin in {args.shards_dir}", file=sys.stderr)
        return 1
    train_sampler = ShardSampler(train_paths, args.block_size, seed=args.seed)
    val_sampler   = (ShardSampler(val_paths, args.block_size, seed=args.seed + 1)
                     if val_paths else None)
    print(f"  train   : {len(train_paths)} shards   {train_sampler.total_tokens:,} tokens")
    if val_sampler:
        print(f"  val     : {len(val_paths)} shards   {val_sampler.total_tokens:,} tokens")
    else:
        print("  val     : (none — skipping eval)")

    # ── Model ──────────────────────────────────────────────────────────────
    cfg = HavocConfig(
        block_size = args.block_size,
        vocab_size = args.vocab_size,
        n_layer    = args.n_layer,
        n_head     = args.n_head,
        n_embd     = args.n_embd,
        mlp_ratio  = args.mlp_ratio,
        dropout    = args.dropout,
    )
    model = HavocGPT(cfg).to(device)
    print(f"  model   : {model.num_params():,} params  "
          f"({model.num_params(non_embedding=True):,} non-embedding)")

    if args.compile and hasattr(torch, "compile"):
        print("  torch.compile(mode='default') ...")
        model = torch.compile(model)

    optimizer = (model._orig_mod if hasattr(model, "_orig_mod") else model)\
        .configure_optimizer(args.weight_decay, args.lr,
                             (args.beta1, args.beta2), device)

    # ── Step / schedule math ──────────────────────────────────────────────
    tokens_per_step = args.batch_size * args.block_size
    grad_accum      = max(1, args.effective_tokens // tokens_per_step)
    effective_bs    = grad_accum * args.batch_size
    tokens_per_optstep = effective_bs * args.block_size
    total_steps     = max(1, args.total_tokens // tokens_per_optstep)
    warmup_steps    = max(1, int(args.warmup_frac * total_steps))

    print(f"  schedule: micro_bs={args.batch_size}  grad_accum={grad_accum}  "
          f"eff_bs={effective_bs}  tokens/step={tokens_per_optstep:,}")
    print(f"  steps   : total={total_steps:,}   warmup={warmup_steps:,}   "
          f"lr={args.lr:.2e} -> {args.min_lr:.2e}")

    # ── Resume ─────────────────────────────────────────────────────────────
    start_step  = 0
    tokens_seen = 0
    if args.resume and os.path.isfile(args.resume):
        ck = torch.load(args.resume, map_location=device, weights_only=False)
        raw = model._orig_mod if hasattr(model, "_orig_mod") else model
        sd = {k.replace("_orig_mod.", ""): v for k, v in ck["model"].items()}
        raw.load_state_dict(sd, strict=False)
        optimizer.load_state_dict(ck["optimizer"])
        start_step  = int(ck.get("step", 0))
        tokens_seen = int(ck.get("tokens_seen", 0))
        print(f"  resumed : step={start_step:,}  tokens_seen={tokens_seen:,}  "
              f"from {args.resume}")
    elif args.resume:
        print(f"  [WARN] --resume {args.resume} not found, starting fresh")

    # ── Log ────────────────────────────────────────────────────────────────
    log_is_new = not os.path.isfile(args.log_path)
    log_f = open(args.log_path, "a", newline="", encoding="utf-8")
    log_w = csv.writer(log_f)
    if log_is_new:
        log_w.writerow(["step", "epoch_frac", "lr", "train_loss",
                        "val_loss", "tokens_seen", "tok_per_sec", "elapsed_s"])
        log_f.flush()

    # ── Eval ───────────────────────────────────────────────────────────────
    @torch.no_grad()
    def evaluate() -> float:
        if val_sampler is None:
            return float("nan")
        model.eval()
        losses = []
        for _ in range(args.eval_iters):
            x, y = val_sampler.sample(args.batch_size, device)
            with torch.amp.autocast(device, dtype=dtype, enabled=(device == "cuda")):
                _, loss = model(x, y)
            losses.append(loss.item())
        model.train()
        return float(np.mean(losses))

    # ── Train loop ─────────────────────────────────────────────────────────
    model.train()
    optimizer.zero_grad(set_to_none=True)
    t0 = time.time()
    micro_loss_accum = 0.0
    last_window_t = time.time()
    last_window_tokens = 0

    for step in range(start_step, total_steps):
        lr = cosine_lr(step, warmup_steps, total_steps, args.lr, args.min_lr)
        for g in optimizer.param_groups:
            g["lr"] = lr

        loss_sum = 0.0
        for micro in range(grad_accum):
            x, y = train_sampler.sample(args.batch_size, device)
            with torch.amp.autocast(device, dtype=dtype, enabled=(device == "cuda")):
                _, loss = model(x, y)
                loss = loss / grad_accum
            loss.backward()
            loss_sum += loss.item()
        train_loss = loss_sum   # already divided by grad_accum

        if args.grad_clip > 0:
            torch.nn.utils.clip_grad_norm_(model.parameters(), args.grad_clip)
        optimizer.step()
        optimizer.zero_grad(set_to_none=True)

        tokens_seen      += tokens_per_optstep
        last_window_tokens += tokens_per_optstep

        if (step + 1) % args.log_every == 0 or step == start_step:
            now = time.time()
            tok_per_s = last_window_tokens / max(now - last_window_t, 1e-6)
            last_window_t = now
            last_window_tokens = 0
            elapsed = now - t0
            epoch_frac = tokens_seen / max(args.total_tokens, 1)
            print(f"  step {step+1:>7,}/{total_steps:,}  "
                  f"loss={train_loss:.4f}  lr={lr:.2e}  "
                  f"toks/s={tok_per_s/1e3:.1f}k  "
                  f"seen={tokens_seen/1e6:.1f}M  "
                  f"({epoch_frac*100:.1f}%)")
            log_w.writerow([step + 1, f"{epoch_frac:.6f}", f"{lr:.6e}",
                            f"{train_loss:.6f}", "",
                            tokens_seen, int(tok_per_s), f"{elapsed:.1f}"])
            log_f.flush()

        if (step + 1) % args.eval_every == 0 and val_sampler is not None:
            val_loss = evaluate()
            print(f"  ---- step {step+1}: val_loss = {val_loss:.4f}")
            log_w.writerow([step + 1, "", "", "", f"{val_loss:.6f}",
                            tokens_seen, "", f"{time.time() - t0:.1f}"])
            log_f.flush()

        if (step + 1) % args.ckpt_every == 0 or (step + 1) == total_steps:
            latest = os.path.join(args.ckpt_dir, "latest.pt")
            tagged = os.path.join(args.ckpt_dir, f"step_{step+1:07d}.pt")
            save_checkpoint(latest, model, optimizer, step + 1, tokens_seen, cfg, args)
            save_checkpoint(tagged, model, optimizer, step + 1, tokens_seen, cfg, args)
            print(f"  ckpt    : {tagged}  (and updated latest.pt)")
            removed = prune_old_checkpoints(args.ckpt_dir, args.keep_n_ckpts)
            if removed:
                print(f"  pruned  : {len(removed)} old step_*.pt "
                      f"(keeping {args.keep_n_ckpts} most recent)")

    log_f.close()
    print(f"\nTraining done. Total elapsed: {(time.time() - t0)/60:.1f} min")
    return 0


if __name__ == "__main__":
    sys.exit(main())
