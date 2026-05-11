"""
HAVOC pretraining — next-token prediction.

Replaces the old scripts/train.py. Model and config now live in dedicated
modules (model.py, config.py); this file owns only the training loop.

Preserved from the prior pipeline:
  - Cosine LR schedule with linear warm-up
  - AdamW (fused on CUDA), separate decay / no-decay groups
  - bf16/fp16 mixed precision via torch.amp
  - Gradient checkpointing (per HavocBlock)
  - Gradient accumulation
  - Step + epoch checkpointing, best/last tags
  - Resume from any checkpoint
  - TensorBoard, JSONL log, matplotlib loss-curve PNG
  - Early stopping on val-loss plateau
  - Flash attention via F.scaled_dot_product_attention
  - Optional torch.compile(max-autotune)
  - Auto-detects /workspace on RunPod
"""

from __future__ import annotations

import argparse
import json
import math
import os
import sys
import time

import torch
import torch.nn.functional as F
from tqdm import tqdm

_HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, _HERE)

from config           import HavocConfig                         # noqa: E402
from dataset          import build_dataloaders                   # noqa: E402
from model            import HavocModel, count_params             # noqa: E402

try:
    from torch.utils.tensorboard import SummaryWriter
    HAS_TENSORBOARD = True
except ImportError:
    HAS_TENSORBOARD = False

try:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    HAS_MATPLOTLIB = True
except ImportError:
    HAS_MATPLOTLIB = False


# ── Workspace detection (RunPod-friendly) ─────────────────────────────────
_WORKSPACE = (
    "/workspace"
    if os.path.isdir("/workspace")
    else os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
)


# ── Helpers ────────────────────────────────────────────────────────────────


def configure_for_gpu() -> None:
    """Apply global PyTorch settings that maximise CUDA throughput."""
    if not torch.cuda.is_available():
        return
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32       = True
    torch.backends.cudnn.benchmark        = True
    torch.set_float32_matmul_precision("high")
    torch.backends.cuda.matmul.allow_bf16_reduced_precision_reduction = True
    name = torch.cuda.get_device_name(0).lower()
    tag  = "H100/A100/H200" if any(x in name for x in ("h100", "a100", "h200")) else "GPU"
    print(f"  {tag} optimisations applied (TF32, cuDNN benchmark, BF16 reductions)")


def cosine_lr(step: int, warmup_steps: int, total_steps: int,
              max_lr: float, min_lr: float) -> float:
    if step < warmup_steps:
        return max_lr * (step + 1) / max(warmup_steps, 1)
    progress = (step - warmup_steps) / max(total_steps - warmup_steps, 1)
    progress = min(progress, 1.0)
    return min_lr + 0.5 * (max_lr - min_lr) * (1.0 + math.cos(math.pi * progress))


def set_lr(optimizer: torch.optim.Optimizer, lr: float) -> None:
    for g in optimizer.param_groups:
        g["lr"] = lr


class EarlyStopping:
    def __init__(self, patience: int = 4, min_delta: float = 1e-4):
        self.patience  = patience
        self.min_delta = min_delta
        self.best      = float("inf")
        self.counter   = 0
        self.triggered = False

    def step(self, val_loss: float) -> bool:
        if val_loss < self.best - self.min_delta:
            self.best    = val_loss
            self.counter = 0
        else:
            self.counter += 1
            if self.counter >= self.patience:
                self.triggered = True
        return self.triggered

    def status(self) -> str:
        return f"patience {self.counter}/{self.patience}  best={self.best:.4f}"


def save_checkpoint(model, optimizer, cfg: HavocConfig, ckpt_dir: str,
                    epoch: int, step: int, val_loss: float, tag: str) -> str:
    os.makedirs(ckpt_dir, exist_ok=True)
    raw  = model._orig_mod if hasattr(model, "_orig_mod") else model
    path = os.path.join(ckpt_dir, f"{tag}.pt")
    torch.save({
        "model":     raw.state_dict(),
        "optimizer": optimizer.state_dict(),
        "cfg":       cfg.to_dict(),
        "epoch":     epoch,
        "step":      step,
        "val_loss":  val_loss,
    }, path)
    return path


def plot_loss_curves(train_steps, train_losses, val_steps, val_losses, save_path):
    if not HAS_MATPLOTLIB:
        print("  matplotlib not installed - skipping loss curve plot")
        return
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    fig.suptitle("HAVOC pretrain loss", fontsize=14, fontweight="bold")

    ax = axes[0]
    ax.plot(train_steps, train_losses, color="#aec6e8", linewidth=0.7,
            alpha=0.6, label="Train (raw)")
    if len(train_losses) > 10:
        ema, alpha, s = [], 0.05, train_losses[0]
        for v in train_losses:
            s = alpha * v + (1 - alpha) * s
            ema.append(s)
        ax.plot(train_steps, ema, color="#1f77b4", linewidth=1.8,
                label="Train (EMA a=0.05)")
    ax.set_xlabel("Optimizer step"); ax.set_ylabel("Cross-entropy loss")
    ax.set_title("Training loss"); ax.legend(); ax.grid(True, alpha=0.3)

    ax = axes[1]
    ax.plot(val_steps, val_losses, color="#ff7f0e", linewidth=2.0,
            marker="o", markersize=4, label="Val")
    if val_losses:
        i = val_losses.index(min(val_losses))
        ax.scatter([val_steps[i]], [val_losses[i]],
                   color="#d62728", s=80, zorder=5,
                   label=f"Best = {val_losses[i]:.4f}")
    ax.set_xlabel("Optimizer step"); ax.set_ylabel("Cross-entropy loss")
    ax.set_title("Validation loss"); ax.legend(); ax.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  Loss curve saved -> {save_path}")


@torch.no_grad()
def evaluate(model: HavocModel, val_loader, device: str, dtype) -> float:
    model.eval()
    losses = []
    for x, y in val_loader:
        x = x.to(device, non_blocking=True)
        y = y.to(device, non_blocking=True)
        with torch.amp.autocast(device, dtype=dtype, enabled=(device == "cuda")):
            _, loss = model(x, y)
        losses.append(loss.item())
        if len(losses) >= 100:
            break
    return sum(losses) / max(len(losses), 1)


def configure_optimizer(model: HavocModel, cfg: HavocConfig) -> torch.optim.AdamW:
    decay   = [p for _n, p in model.named_parameters() if p.dim() >= 2]
    nodecay = [p for _n, p in model.named_parameters() if p.dim() <  2]
    groups  = [
        {"params": decay,   "weight_decay": cfg.weight_decay},
        {"params": nodecay, "weight_decay": 0.0},
    ]
    print(f"  AdamW : {sum(p.numel() for p in decay):,} params w/ decay  |  "
          f"{sum(p.numel() for p in nodecay):,} without")
    use_fused = (torch.cuda.is_available() and
                 "fused" in torch.optim.AdamW.__init__.__code__.co_varnames)
    kwargs = {"fused": True} if use_fused else {}
    return torch.optim.AdamW(groups, lr=cfg.lr, betas=tuple(cfg.adam_betas), **kwargs)


def model_summary(model: HavocModel, cfg: HavocConfig, eff_bs: int, dtype, device: str) -> None:
    counts = count_params(model)
    flash  = hasattr(F, "scaled_dot_product_attention")
    mem_mb = counts["unique"] * 2 / 1024 ** 2
    sep    = "=" * 72
    print(f"\n{sep}\n  HAVOC MODEL SUMMARY\n{sep}")
    print(f"  Layers / Heads / Hidden  : {cfg.num_layers} / {cfg.num_heads} / {cfg.hidden_size}")
    print(f"  Head dim                 : {cfg.head_dim}")
    print(f"  Intermediate (SwiGLU)    : {cfg.intermediate_size}")
    print(f"  Context length           : {cfg.max_seq_len}")
    print(f"  Vocab                    : {cfg.vocab_size:,}")
    print(f"  Pos. encoding            : RoPE (base={cfg.rope_base})")
    print(f"  Norm                     : RMSNorm (eps={cfg.rms_norm_eps})")
    print(f"  Tied embeddings          : {cfg.tied_embeddings}")
    print(f"  Total / unique           : {counts['total']:,} / {counts['unique']:,}")
    print(f"  Param mem (bf16)         : {mem_mb:.0f} MB")
    print(f"  Optimizer                : AdamW  wd={cfg.weight_decay}  betas={cfg.adam_betas}")
    print(f"  LR schedule              : warm-up {cfg.warmup_steps} -> cosine")
    print(f"  Peak / floor LR          : {cfg.lr} / {cfg.min_lr}")
    print(f"  Effective batch          : {eff_bs}")
    print(f"  Mixed precision          : {dtype}  ({device})")
    print(f"  Gradient checkpointing   : {cfg.gradient_checkpointing}")
    print(f"  Flash Attention          : {'yes' if flash else 'no'}")
    print(f"  TensorBoard              : {'yes' if HAS_TENSORBOARD else 'no'}")
    print(f"{sep}\n")


def _print_sample(label: str, ids: torch.Tensor, tokenizer) -> None:
    if tokenizer is not None:
        text = tokenizer.decode(ids[0].tolist(), skip_special_tokens=False)
    else:
        text = str(ids[0].tolist()[:40]) + " ..."
    # Re-encode through ASCII to strip any console-unprintable chars (Windows cp1252).
    try:
        sys.stdout.buffer.write(
            f"\n--- {label} ---\n{text[:500]}\n{'-' * (len(label) + 6)}\n".encode("utf-8", "replace")
        )
        sys.stdout.flush()
    except Exception:
        safe = text[:500].encode("ascii", "replace").decode("ascii")
        print(f"\n--- {label} ---\n{safe}\n{'-' * (len(label) + 6)}")


# ── Training loop ─────────────────────────────────────────────────────────


def train(args: argparse.Namespace) -> None:
    cfg = HavocConfig.from_json(args.config) if args.config else HavocConfig()

    # CLI overrides (matches arguments below)
    cfg.num_layers              = args.num_layers              or cfg.num_layers
    cfg.num_heads               = args.num_heads               or cfg.num_heads
    cfg.hidden_size             = args.hidden_size             or cfg.hidden_size
    cfg.intermediate_size       = args.intermediate_size       or cfg.intermediate_size
    cfg.max_seq_len             = args.max_seq_len             or cfg.max_seq_len
    cfg.batch_size              = args.batch_size              or cfg.batch_size
    cfg.grad_accum_steps        = args.grad_accum              or cfg.grad_accum_steps
    cfg.max_epochs              = args.max_epochs              or cfg.max_epochs
    cfg.lr                      = args.lr                      or cfg.lr
    cfg.min_lr                  = args.min_lr                  or cfg.min_lr
    cfg.warmup_steps            = args.warmup_steps            or cfg.warmup_steps
    cfg.eval_interval           = args.eval_interval           or cfg.eval_interval
    cfg.ckpt_interval           = args.ckpt_interval           or cfg.ckpt_interval
    cfg.log_interval            = args.log_interval            or cfg.log_interval
    cfg.early_stop_patience     = args.patience                or cfg.early_stop_patience
    cfg.num_workers             = args.num_workers             or cfg.num_workers
    if args.no_grad_ckpt:
        cfg.gradient_checkpointing = False

    processed_dir = args.processed_dir or os.path.join(_WORKSPACE, "data/processed")
    ckpt_dir      = args.ckpt_dir      or os.path.join(_WORKSPACE, "models/checkpoints")
    log_dir       = args.log_dir       or os.path.join(_WORKSPACE, "logs")
    tokenizer_dir = args.tokenizer_dir or os.path.join(_WORKSPACE, "models/tokenizers/havoc_bpe")

    os.makedirs(ckpt_dir, exist_ok=True)
    os.makedirs(log_dir,  exist_ok=True)

    # ── Hardware ──────────────────────────────────────────────────────────
    print("\n" + "=" * 60 + "\n  HARDWARE\n" + "=" * 60)
    configure_for_gpu()
    device = "cuda" if torch.cuda.is_available() else "cpu"
    dtype  = torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16
    print(f"  Device : {device.upper()}")
    if device == "cuda":
        props = torch.cuda.get_device_properties(0)
        print(f"  GPU    : {props.name}  ({props.total_memory / 1024**3:.1f} GB VRAM)")
        print(f"  CUDA   : {torch.version.cuda}")
        torch.cuda.synchronize()

    # ── Data ──────────────────────────────────────────────────────────────
    print("\n" + "=" * 60 + "\n  DATA\n" + "=" * 60)
    train_loader, val_loader, info = build_dataloaders(
        processed_dir = processed_dir,
        batch_size    = cfg.batch_size,
        num_workers   = cfg.num_workers,
    )
    cfg.vocab_size  = info["vocab_size"]
    cfg.max_seq_len = info["block_size"]

    eot_id        = info.get("eot_token_id")
    think_id      = info.get("think_token_id")
    end_think_id  = info.get("end_think_token_id")

    eff_bs = cfg.batch_size * cfg.grad_accum_steps
    steps_per_epoch = len(train_loader) // cfg.grad_accum_steps
    total_steps     = steps_per_epoch * cfg.max_epochs

    print(f"  Vocab         : {cfg.vocab_size:,}   block_size : {cfg.max_seq_len}")
    print(f"  Train batches : {len(train_loader):,}    Val batches : {len(val_loader):,}")
    print(f"  Effective batch : {eff_bs}    Steps/epoch : {steps_per_epoch:,}    "
          f"Total steps : {total_steps:,}")

    # ── Model ─────────────────────────────────────────────────────────────
    model = HavocModel(cfg).to(device)
    model_summary(model, cfg, eff_bs, dtype, device)

    if args.compile and hasattr(torch, "compile"):
        print("  torch.compile(mode='max-autotune') - tracing ...")
        model = torch.compile(model, mode="max-autotune")

    optimizer = configure_optimizer(model, cfg)
    scaler    = torch.amp.GradScaler(device, enabled=(device == "cuda"))

    # ── Resume ────────────────────────────────────────────────────────────
    resume_epoch = 0
    if args.resume and os.path.isfile(args.resume):
        print(f"\n  Resuming from: {args.resume}")
        ck = torch.load(args.resume, map_location=device, weights_only=False)
        raw = model._orig_mod if hasattr(model, "_orig_mod") else model
        sd  = {k.replace("_orig_mod.", ""): v for k, v in ck["model"].items()}
        raw.load_state_dict(sd, strict=False)
        optimizer.load_state_dict(ck["optimizer"])
        resume_epoch = ck.get("epoch", 0)
        print(f"  Resumed epoch={resume_epoch}  step={ck.get('step', 0)}  "
              f"val_loss={ck.get('val_loss', float('nan')):.4f}")
    elif args.resume:
        print(f"  [WARN] --resume path not found: {args.resume}")

    # ── Logging ───────────────────────────────────────────────────────────
    tb_dir   = os.path.join(log_dir, "tensorboard")
    writer   = SummaryWriter(log_dir=tb_dir) if HAS_TENSORBOARD else None
    log_path = os.path.join(log_dir, "pretrain_log.jsonl")
    if writer:
        print(f"  TensorBoard -> tensorboard --logdir {tb_dir}")

    # ── Tokenizer (for sample decoding only - optional) ──────────────────
    tokenizer = None
    try:
        from tokenizer_havoc import HavocTokenizer
        if os.path.isfile(os.path.join(tokenizer_dir, "tokenizer.json")):
            tokenizer = HavocTokenizer.from_pretrained(tokenizer_dir)
    except Exception as exc:
        print(f"  [WARN] could not load HavocTokenizer ({exc}) - samples will show ids")

    history = {"train_steps": [], "train_losses": [], "val_steps": [], "val_losses": []}
    early_stop = EarlyStopping(cfg.early_stop_patience, cfg.early_stop_min_delta)
    best_val   = float("inf")
    opt_step   = 0
    stop_reason = "max epochs reached"

    # ── Training ──────────────────────────────────────────────────────────
    print("\n" + "=" * 60 + "\n  TRAINING\n" + "=" * 60)
    t0 = time.perf_counter()

    for epoch in range(resume_epoch, cfg.max_epochs):
        model.train()
        optimizer.zero_grad(set_to_none=True)

        pbar = tqdm(train_loader, desc=f"Epoch {epoch+1}/{cfg.max_epochs}",
                    unit="batch", dynamic_ncols=True, colour="cyan")
        loss_accum  = 0.0
        last_val    = float("nan")
        tokens_seen = 0

        for batch_idx, (x, y) in enumerate(pbar):
            lr = cosine_lr(opt_step, cfg.warmup_steps, total_steps, cfg.lr, cfg.min_lr)
            set_lr(optimizer, lr)
            x = x.to(device, non_blocking=True)
            y = y.to(device, non_blocking=True)

            with torch.amp.autocast(device, dtype=dtype, enabled=(device == "cuda")):
                _, loss = model(x, y)
                loss_scaled = loss / cfg.grad_accum_steps

            scaler.scale(loss_scaled).backward()
            loss_accum  += loss.item()
            tokens_seen += x.numel()

            is_step = ((batch_idx + 1) % cfg.grad_accum_steps == 0
                       or batch_idx == len(train_loader) - 1)

            if is_step:
                scaler.unscale_(optimizer)
                grad_norm = torch.nn.utils.clip_grad_norm_(
                    model.parameters(), cfg.grad_clip)
                scaler.step(optimizer)
                scaler.update()
                optimizer.zero_grad(set_to_none=True)

                avg_loss   = loss_accum / cfg.grad_accum_steps
                loss_accum = 0.0
                opt_step  += 1

                if device == "cuda":
                    mem_gb = torch.cuda.memory_reserved() / 1024 ** 3
                    mem_pct = (torch.cuda.memory_reserved() /
                               torch.cuda.get_device_properties(0).total_memory * 100)
                else:
                    mem_gb, mem_pct = 0.0, 0.0

                history["train_steps"].append(opt_step)
                history["train_losses"].append(avg_loss)

                if writer and opt_step % cfg.log_interval == 0:
                    writer.add_scalar("Loss/train", avg_loss, opt_step)
                    writer.add_scalar("LR",         lr,       opt_step)
                    writer.add_scalar("GradNorm",   grad_norm, opt_step)
                    if device == "cuda":
                        writer.add_scalar("GPU/mem_GB",  mem_gb,  opt_step)
                        writer.add_scalar("GPU/mem_pct", mem_pct, opt_step)

                if opt_step % cfg.log_interval == 0:
                    elapsed = time.perf_counter() - t0
                    with open(log_path, "a") as f:
                        f.write(json.dumps({
                            "step": opt_step, "epoch": epoch + 1,
                            "train_loss": round(avg_loss, 6),
                            "lr": round(lr, 8),
                            "tok_per_sec": int(tokens_seen / max(elapsed, 1e-6)),
                        }) + "\n")

                pbar.set_postfix(
                    loss=f"{avg_loss:.4f}",
                    val=f"{last_val:.4f}" if not math.isnan(last_val) else "-",
                    lr=f"{lr:.2e}",
                    mem=f"{mem_pct:.0f}%",
                    step=opt_step,
                )

                if opt_step % cfg.eval_interval == 0 and opt_step > 0:
                    val_loss = evaluate(model, val_loader, device, dtype)
                    last_val = val_loss
                    history["val_steps"].append(opt_step)
                    history["val_losses"].append(val_loss)
                    if writer:
                        writer.add_scalar("Loss/val", val_loss, opt_step)
                    with open(log_path, "a") as f:
                        f.write(json.dumps({
                            "step": opt_step, "epoch": epoch + 1,
                            "val_loss": round(val_loss, 6),
                        }) + "\n")
                    marker = "*NEW BEST*" if val_loss < best_val else ""
                    tqdm.write(f"  [step {opt_step:>6} | epoch {epoch+1}]  "
                               f"val_loss={val_loss:.4f}  {early_stop.status()}  {marker}")
                    model.train()

                if opt_step % cfg.ckpt_interval == 0 and opt_step > 0:
                    p = save_checkpoint(model, optimizer, cfg, ckpt_dir,
                                        epoch + 1, opt_step,
                                        last_val if not math.isnan(last_val) else float("inf"),
                                        tag=f"step_{opt_step:07d}")
                    tqdm.write(f"  [step {opt_step}] checkpoint -> {p}")

        pbar.close()

        # End-of-epoch
        val_loss = evaluate(model, val_loader, device, dtype)
        history["val_steps"].append(opt_step)
        history["val_losses"].append(val_loss)
        if writer:
            writer.add_scalar("Loss/val_epoch", val_loss, epoch + 1)

        elapsed = time.perf_counter() - t0
        is_best = val_loss < best_val
        print(f"\n{'-'*60}")
        print(f"  Epoch {epoch+1}/{cfg.max_epochs} complete")
        print(f"  Val loss     : {val_loss:.4f}  "
              f"{'<- NEW BEST' if is_best else f'(best={best_val:.4f})'}")
        print(f"  Elapsed      : {elapsed/60:.1f} min")
        print(f"  Throughput   : {tokens_seen/max(elapsed,1e-6):,.0f} tokens/sec")

        if is_best:
            best_val = val_loss
            p = save_checkpoint(model, optimizer, cfg, ckpt_dir,
                                epoch + 1, opt_step, val_loss, tag="best")
            print(f"  Best ckpt    : {p}")
        p = save_checkpoint(model, optimizer, cfg, ckpt_dir,
                            epoch + 1, opt_step, val_loss,
                            tag=f"epoch_{epoch+1:02d}")
        print(f"  Epoch ckpt   : {p}")

        # Generation samples
        model.eval()
        if eot_id is not None:
            ctx = torch.tensor([[eot_id]], dtype=torch.long, device=device)
            sample = model.generate(ctx, max_new_tokens=80, temperature=0.8, top_k=40)
            _print_sample("Standard generation", sample, tokenizer)
            if think_id is not None and end_think_id is not None:
                ctx2 = torch.tensor([[eot_id, think_id]], dtype=torch.long, device=device)
                cot = model.generate(ctx2, max_new_tokens=80, temperature=0.8,
                                     top_k=40, stop_token_id=end_think_id)
                _print_sample("CoT-style generation", cot, tokenizer)

        if early_stop.step(val_loss):
            stop_reason = (f"early stopping - val loss did not improve by "
                           f"{cfg.early_stop_min_delta} for "
                           f"{cfg.early_stop_patience} evaluations")
            print(f"\n  *** {stop_reason.upper()} ***\n")
            break
        print()

    total_time = time.perf_counter() - t0
    print("=" * 60)
    print(f"  TRAINING COMPLETE  ({stop_reason})")
    print(f"  Total time  : {total_time/60:.1f} min")
    print(f"  Best val    : {best_val:.4f}")
    print(f"  Total steps : {opt_step:,}")
    print("=" * 60)

    plot_loss_curves(history["train_steps"], history["train_losses"],
                     history["val_steps"],   history["val_losses"],
                     save_path=os.path.join(log_dir, "loss_curves.png"))

    if writer:
        writer.flush(); writer.close()
        print(f"  TensorBoard logs -> tensorboard --logdir {tb_dir}")
    print(f"  JSONL log    -> {log_path}\n")


# ── CLI ────────────────────────────────────────────────────────────────────


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="HAVOC pretraining (next-token prediction).",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    # Config + paths
    parser.add_argument("--config",        default=None, help="HavocConfig JSON.")
    parser.add_argument("--processed_dir", default=None)
    parser.add_argument("--ckpt_dir",      default=None)
    parser.add_argument("--log_dir",       default=None)
    parser.add_argument("--tokenizer_dir", default=None)
    parser.add_argument("--resume",        default=None)
    # Architecture overrides
    parser.add_argument("--num_layers",        type=int, default=None)
    parser.add_argument("--num_heads",         type=int, default=None)
    parser.add_argument("--hidden_size",       type=int, default=None)
    parser.add_argument("--intermediate_size", type=int, default=None)
    parser.add_argument("--max_seq_len",       type=int, default=None)
    # Schedule overrides
    parser.add_argument("--batch_size",    type=int,   default=None)
    parser.add_argument("--grad_accum",    type=int,   default=None)
    parser.add_argument("--max_epochs",    type=int,   default=None)
    parser.add_argument("--lr",            type=float, default=None)
    parser.add_argument("--min_lr",        type=float, default=None)
    parser.add_argument("--warmup_steps",  type=int,   default=None)
    parser.add_argument("--eval_interval", type=int,   default=None)
    parser.add_argument("--ckpt_interval", type=int,   default=None)
    parser.add_argument("--log_interval",  type=int,   default=None)
    parser.add_argument("--patience",      type=int,   default=None)
    parser.add_argument("--num_workers",   type=int,   default=None)
    # Hardware
    parser.add_argument("--no_grad_ckpt",  action="store_true",
                        help="Disable gradient checkpointing.")
    parser.add_argument("--compile",       action="store_true",
                        help="torch.compile(mode='max-autotune').")
    train(parser.parse_args())
