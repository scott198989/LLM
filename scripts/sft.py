"""
Supervised Fine-Tuning (SFT) for HAVOC.

Loads a pretrained checkpoint, then continues training on prompt/completion
pairs with loss masked over prompt tokens (only completion tokens contribute
to the cross-entropy). Reuses pretrain.py's infrastructure
(LR schedule, AdamW, mixed precision, checkpointing, early stopping, TB).

Data format (JSONL, one record per line):
    {"prompt": "What is gravity?", "completion": "A force that..."}
    {"instruction": "Summarise:", "input": "...", "output": "..."}
    {"system": "You are helpful.", "prompt": "...", "completion": "..."}

The chat template applied is:
    <|endoftext|>
    <|system|>{system}<|/system|>      (only when present)
    <|user|>{prompt}<|/user|>
    <|assistant|>{completion}<|/assistant|>
    <|endoftext|>

Loss is masked (-100) on every token up to and including the opening
<|assistant|> tag, so the model only learns to produce the assistant's
turn. Padding tokens are also -100.

Usage:
    python scripts/sft.py --base_ckpt models/checkpoints/best.pt \\
        --sft_data data/sft/sample.jsonl \\
        --tokenizer_dir models/tokenizers/havoc_bpe \\
        --max_epochs 2 --batch_size 8
"""

from __future__ import annotations

import argparse
import json
import math
import os
import random
import sys
import time

import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm

_HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, _HERE)

from config            import HavocConfig                                         # noqa: E402
from model             import HavocModel, count_params                            # noqa: E402
from tokenizer_havoc   import HavocTokenizer                                      # noqa: E402
from pretrain          import (configure_for_gpu, configure_optimizer, cosine_lr,  # noqa: E402
                               set_lr, EarlyStopping, save_checkpoint,
                               plot_loss_curves, HAS_TENSORBOARD, HAS_MATPLOTLIB)
if HAS_TENSORBOARD:
    from torch.utils.tensorboard import SummaryWriter

IGNORE_INDEX = -100


# ── SFT dataset ───────────────────────────────────────────────────────────


class SFTDataset(Dataset):
    """
    Reads a JSONL of prompt/completion records, tokenises each into a fixed-
    length (max_seq_len) tensor with loss masked on every position before
    the assistant turn.

    Returns (input_ids, label_ids) per item:
        input_ids[t]  - token at position t
        label_ids[t]  - target for predicting position t+1, or IGNORE_INDEX
    """

    def __init__(self,
                 path:         str,
                 tokenizer:    HavocTokenizer,
                 max_seq_len:  int,
                 ):
        self.tok        = tokenizer
        self.max_len    = max_seq_len
        self.records    = self._load(path)

    def _load(self, path: str) -> list[dict]:
        out: list[dict] = []
        with open(path, encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                try:
                    obj = json.loads(line)
                except json.JSONDecodeError:
                    continue
                prompt     = (obj.get("prompt")     or obj.get("input")
                              or obj.get("instruction") or "").strip()
                completion = (obj.get("completion") or obj.get("output")
                              or obj.get("response") or obj.get("text") or "").strip()
                system     = (obj.get("system") or "").strip()
                if not prompt or not completion:
                    continue
                out.append({"system": system, "prompt": prompt, "completion": completion})
        if not out:
            raise ValueError(f"No usable SFT records in {path}")
        return out

    def __len__(self) -> int:
        return len(self.records)

    def _build_ids(self, rec: dict) -> tuple[list[int], int]:
        """
        Returns (full_token_ids, prompt_len) where prompt_len is the count of
        tokens to mask (everything up to and including <|assistant|>).
        """
        tok = self.tok
        prompt_ids: list[int] = [tok.eos_token_id]

        if rec["system"]:
            prompt_ids.append(tok.system_token_id)
            prompt_ids.extend(tok.encode(rec["system"], add_special=False))
            prompt_ids.append(tok.end_system_token_id)

        prompt_ids.append(tok.user_token_id)
        prompt_ids.extend(tok.encode(rec["prompt"], add_special=False))
        prompt_ids.append(tok.end_user_token_id)
        prompt_ids.append(tok.assistant_token_id)

        completion_ids: list[int] = []
        completion_ids.extend(tok.encode(rec["completion"], add_special=False))
        completion_ids.append(tok.end_assistant_token_id)
        completion_ids.append(tok.eos_token_id)

        return prompt_ids + completion_ids, len(prompt_ids)

    def __getitem__(self, idx: int):
        rec = self.records[idx]
        full_ids, prompt_len = self._build_ids(rec)

        # Truncate to max_len + 1 so we can derive shifted (input, target) pairs
        full_ids = full_ids[: self.max_len + 1]
        if len(full_ids) < 2:
            full_ids = full_ids + [self.tok.eos_token_id] * (2 - len(full_ids))

        x = full_ids[:-1]                         # inputs
        y = list(full_ids[1:])                    # targets (next token)

        # Mask everything up to (and including) the start of the assistant turn:
        # the loss only counts positions that PREDICT a completion token.
        # Position t's target is full_ids[t+1]; we want target masked when
        # full_ids[t+1] is still a prompt token, i.e. (t+1) < prompt_len.
        for t in range(len(y)):
            if (t + 1) < prompt_len:
                y[t] = IGNORE_INDEX

        # Pad to max_len
        pad_id = self.tok.pad_token_id
        if len(x) < self.max_len:
            pad_n = self.max_len - len(x)
            x = x + [pad_id]         * pad_n
            y = y + [IGNORE_INDEX]   * pad_n

        return torch.tensor(x, dtype=torch.long), torch.tensor(y, dtype=torch.long)


def split_dataset(ds: SFTDataset, val_split: float, seed: int = 1337) -> tuple[Dataset, Dataset]:
    n = len(ds)
    if n < 4 or val_split <= 0:
        return ds, ds  # tiny — share data; the calling code can handle this
    val_n = max(1, int(n * val_split))
    indices = list(range(n))
    random.Random(seed).shuffle(indices)
    val_idx   = set(indices[:val_n])
    train_set = torch.utils.data.Subset(ds, [i for i in indices if i not in val_idx])
    val_set   = torch.utils.data.Subset(ds, list(val_idx))
    return train_set, val_set


# ── Eval ──────────────────────────────────────────────────────────────────


@torch.no_grad()
def evaluate_sft(model: HavocModel, val_loader, device: str, dtype) -> float:
    model.eval()
    losses = []
    for x, y in val_loader:
        x = x.to(device, non_blocking=True)
        y = y.to(device, non_blocking=True)
        with torch.amp.autocast(device, dtype=dtype, enabled=(device == "cuda")):
            logits, _ = model(x)
            loss = F.cross_entropy(
                logits.view(-1, logits.size(-1)),
                y.view(-1),
                ignore_index=IGNORE_INDEX,
            )
        losses.append(loss.item())
        if len(losses) >= 50:
            break
    return sum(losses) / max(len(losses), 1)


# ── Train loop ────────────────────────────────────────────────────────────


def sft(args: argparse.Namespace) -> None:
    # Load checkpoint to recover the trained config
    print(f"\nLoading base checkpoint: {args.base_ckpt}")
    ck    = torch.load(args.base_ckpt, map_location="cpu", weights_only=False)
    cfg   = HavocConfig(**{
        k: v for k, v in ck["cfg"].items()
        if k in {f.name for f in HavocConfig.__dataclass_fields__.values()}
    })

    # Apply SFT-specific overrides
    cfg.batch_size           = args.batch_size       or cfg.batch_size
    cfg.grad_accum_steps     = args.grad_accum       or cfg.grad_accum_steps
    cfg.max_epochs           = args.max_epochs       or 2
    cfg.lr                   = args.lr               or 2.0e-5
    cfg.min_lr               = args.min_lr           or 2.0e-6
    cfg.warmup_steps         = args.warmup_steps     or 50
    cfg.eval_interval        = args.eval_interval    or 100
    cfg.ckpt_interval        = args.ckpt_interval    or 250
    cfg.early_stop_patience  = args.patience         or 4

    print("\n" + "=" * 60 + "\n  HARDWARE\n" + "=" * 60)
    configure_for_gpu()
    device = "cuda" if torch.cuda.is_available() else "cpu"
    dtype  = torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16
    print(f"  Device : {device.upper()}")
    if device == "cuda":
        print(f"  GPU    : {torch.cuda.get_device_name(0)}")

    # ── Tokenizer ──────────────────────────────────────────────────────────
    tokenizer_dir = args.tokenizer_dir or "models/tokenizers/havoc_bpe"
    print(f"\n  Tokenizer : {tokenizer_dir}")
    tokenizer = HavocTokenizer.from_pretrained(tokenizer_dir)

    # ── Data ──────────────────────────────────────────────────────────────
    print("\n" + "=" * 60 + "\n  DATA\n" + "=" * 60)
    base_ds = SFTDataset(args.sft_data, tokenizer, max_seq_len=cfg.max_seq_len)
    train_ds, val_ds = split_dataset(base_ds, val_split=args.val_split)
    print(f"  Records : {len(base_ds):,}  (train={len(train_ds):,}  val={len(val_ds):,})")

    train_loader = DataLoader(train_ds, batch_size=cfg.batch_size, shuffle=True,
                              num_workers=cfg.num_workers, drop_last=True,
                              pin_memory=(device == "cuda"))
    val_loader   = DataLoader(val_ds,   batch_size=cfg.batch_size, shuffle=False,
                              num_workers=cfg.num_workers, drop_last=False,
                              pin_memory=(device == "cuda"))

    eff_bs = cfg.batch_size * cfg.grad_accum_steps
    steps_per_epoch = max(1, len(train_loader) // cfg.grad_accum_steps)
    total_steps     = steps_per_epoch * cfg.max_epochs
    print(f"  Batches : train={len(train_loader):,}   val={len(val_loader):,}")
    print(f"  Effective batch : {eff_bs}   total steps : {total_steps:,}")

    # ── Model ─────────────────────────────────────────────────────────────
    print("\n" + "=" * 60 + "\n  MODEL\n" + "=" * 60)
    model = HavocModel(cfg).to(device)
    sd    = {k.replace("_orig_mod.", ""): v for k, v in ck["model"].items()}
    missing, unexpected = model.load_state_dict(sd, strict=False)
    if missing:    print(f"  [WARN] missing keys    : {missing[:5]}")
    if unexpected: print(f"  [WARN] unexpected keys : {unexpected[:5]}")
    counts = count_params(model)
    print(f"  Params  : {counts['unique']:,}  ({counts['unique']/1e6:.1f} M)")
    print(f"  Layers / Heads / Hidden : {cfg.num_layers} / {cfg.num_heads} / {cfg.hidden_size}")

    optimizer = configure_optimizer(model, cfg)
    scaler    = torch.amp.GradScaler(device, enabled=(device == "cuda"))

    # ── Logging ───────────────────────────────────────────────────────────
    log_dir   = args.log_dir  or "logs"
    ckpt_dir  = args.ckpt_dir or "models/checkpoints"
    os.makedirs(log_dir, exist_ok=True)
    os.makedirs(ckpt_dir, exist_ok=True)
    tb_dir    = os.path.join(log_dir, "tensorboard_sft")
    writer    = SummaryWriter(log_dir=tb_dir) if HAS_TENSORBOARD else None
    log_path  = os.path.join(log_dir, "sft_log.jsonl")

    history    = {"train_steps": [], "train_losses": [], "val_steps": [], "val_losses": []}
    early_stop = EarlyStopping(cfg.early_stop_patience, cfg.early_stop_min_delta)
    best_val   = float("inf")
    opt_step   = 0

    # ── Train ─────────────────────────────────────────────────────────────
    print("\n" + "=" * 60 + "\n  SFT TRAINING\n" + "=" * 60)
    t0 = time.perf_counter()

    for epoch in range(cfg.max_epochs):
        model.train()
        optimizer.zero_grad(set_to_none=True)
        pbar = tqdm(train_loader, desc=f"Epoch {epoch+1}/{cfg.max_epochs}",
                    unit="batch", dynamic_ncols=True, colour="magenta")
        loss_accum = 0.0
        last_val   = float("nan")

        for batch_idx, (x, y) in enumerate(pbar):
            lr = cosine_lr(opt_step, cfg.warmup_steps, total_steps, cfg.lr, cfg.min_lr)
            set_lr(optimizer, lr)
            x = x.to(device, non_blocking=True)
            y = y.to(device, non_blocking=True)

            with torch.amp.autocast(device, dtype=dtype, enabled=(device == "cuda")):
                logits, _ = model(x)
                loss = F.cross_entropy(
                    logits.view(-1, logits.size(-1)), y.view(-1),
                    ignore_index=IGNORE_INDEX,
                )
                loss_scaled = loss / cfg.grad_accum_steps

            scaler.scale(loss_scaled).backward()
            loss_accum += loss.item()

            is_step = ((batch_idx + 1) % cfg.grad_accum_steps == 0
                       or batch_idx == len(train_loader) - 1)
            if is_step:
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(model.parameters(), cfg.grad_clip)
                scaler.step(optimizer); scaler.update()
                optimizer.zero_grad(set_to_none=True)

                avg_loss   = loss_accum / cfg.grad_accum_steps
                loss_accum = 0.0
                opt_step  += 1
                history["train_steps"].append(opt_step)
                history["train_losses"].append(avg_loss)
                if writer and opt_step % cfg.log_interval == 0:
                    writer.add_scalar("Loss/train_sft", avg_loss, opt_step)
                    writer.add_scalar("LR_sft",          lr,       opt_step)

                if opt_step % cfg.log_interval == 0:
                    with open(log_path, "a") as f:
                        f.write(json.dumps({
                            "step": opt_step, "epoch": epoch + 1,
                            "train_loss": round(avg_loss, 6),
                            "lr": round(lr, 8),
                        }) + "\n")
                pbar.set_postfix(
                    loss=f"{avg_loss:.4f}",
                    val=f"{last_val:.4f}" if not math.isnan(last_val) else "-",
                    lr=f"{lr:.2e}",
                    step=opt_step,
                )

                if opt_step % cfg.eval_interval == 0 and opt_step > 0 and len(val_ds) > 0:
                    val_loss = evaluate_sft(model, val_loader, device, dtype)
                    last_val = val_loss
                    history["val_steps"].append(opt_step)
                    history["val_losses"].append(val_loss)
                    if writer:
                        writer.add_scalar("Loss/val_sft", val_loss, opt_step)
                    tqdm.write(f"  [step {opt_step}]  val_loss={val_loss:.4f}  {early_stop.status()}")
                    model.train()

                if opt_step % cfg.ckpt_interval == 0 and opt_step > 0:
                    p = save_checkpoint(model, optimizer, cfg, ckpt_dir,
                                        epoch + 1, opt_step,
                                        last_val if not math.isnan(last_val) else float("inf"),
                                        tag=f"sft_step_{opt_step:07d}")
                    tqdm.write(f"  [step {opt_step}] ckpt -> {p}")

        pbar.close()

        # End of epoch
        val_loss = evaluate_sft(model, val_loader, device, dtype) if len(val_ds) > 0 else float("nan")
        if not math.isnan(val_loss):
            history["val_steps"].append(opt_step)
            history["val_losses"].append(val_loss)
            print(f"\n  Epoch {epoch+1} val loss : {val_loss:.4f}")
            if val_loss < best_val:
                best_val = val_loss
                p = save_checkpoint(model, optimizer, cfg, ckpt_dir,
                                    epoch + 1, opt_step, val_loss, tag="best_sft")
                print(f"  Best SFT ckpt -> {p}")
            if early_stop.step(val_loss):
                print(f"\n  *** EARLY STOPPING ***\n")
                break
        # Always save epoch checkpoint
        p = save_checkpoint(model, optimizer, cfg, ckpt_dir,
                            epoch + 1, opt_step, val_loss, tag=f"sft_epoch_{epoch+1:02d}")
        print(f"  Epoch SFT ckpt -> {p}\n")

    elapsed = time.perf_counter() - t0
    print("=" * 60)
    print(f"  SFT COMPLETE  -  {elapsed/60:.1f} min   best val={best_val:.4f}")
    print("=" * 60)

    plot_loss_curves(history["train_steps"], history["train_losses"],
                     history["val_steps"],   history["val_losses"],
                     save_path=os.path.join(log_dir, "sft_loss_curves.png"))
    if writer:
        writer.flush(); writer.close()
        print(f"  TensorBoard -> tensorboard --logdir {tb_dir}")
    print(f"  JSONL log    -> {log_path}\n")


# ── CLI ────────────────────────────────────────────────────────────────────


if __name__ == "__main__":
    p = argparse.ArgumentParser(description="HAVOC supervised fine-tuning.")
    p.add_argument("--base_ckpt",     required=True, help="Pretrained checkpoint to fine-tune from.")
    p.add_argument("--sft_data",      required=True, help="JSONL of prompt/completion records.")
    p.add_argument("--tokenizer_dir", default=None)
    p.add_argument("--ckpt_dir",      default=None)
    p.add_argument("--log_dir",       default=None)
    p.add_argument("--val_split",     type=float, default=0.05)
    p.add_argument("--batch_size",    type=int, default=None)
    p.add_argument("--grad_accum",    type=int, default=None)
    p.add_argument("--max_epochs",    type=int, default=None)
    p.add_argument("--lr",            type=float, default=None)
    p.add_argument("--min_lr",        type=float, default=None)
    p.add_argument("--warmup_steps",  type=int, default=None)
    p.add_argument("--eval_interval", type=int, default=None)
    p.add_argument("--ckpt_interval", type=int, default=None)
    p.add_argument("--patience",      type=int, default=None)
    sft(p.parse_args())
