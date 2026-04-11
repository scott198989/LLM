"""
GPT-style Transformer language model — next-token prediction.

Architecture
  - Token embedding + positional embedding
  - 12 × TransformerBlock (MultiHeadAttention + FeedForward + LayerNorm)
  - Final LayerNorm + tied output head
  - 12 layers / 768 hidden dims / 12 attention heads  (~117 M params)

Training features
  - Cosine LR schedule with linear warm-up
  - AdamW optimizer with decay / no-decay param groups  (fused kernel on CUDA)
  - Mixed precision  (bfloat16 via torch.amp — native on H100)
  - Gradient checkpointing  (recompute activations; halves VRAM usage)
  - Gradient accumulation  (simulate large effective batch sizes)
  - Step-level checkpointing every N steps + epoch-end best/last saves
  - TensorBoard logging  (loss, val-loss, LR, GPU memory, tokens/sec)
  - Matplotlib loss-curve PNG  (saved to logs/ at the end of training)
  - Early stopping  (stops when val loss plateaus for K evaluations)
  - Flash Attention via scaled_dot_product_attention  (PyTorch 2+)
  - Optional torch.compile  (max-autotune mode for H100)

H100 optimisations
  - TF32 for float32 matmuls  (3× faster than FP32, same accuracy)
  - BF16 mixed precision  (H100 has dedicated BF16 tensor cores)
  - Fused AdamW kernel
  - Non-blocking CUDA transfers
  - torch.compile(mode="max-autotune")
  - cuDNN auto-tuner

Chain of Thought
  - generate_cot(): structures output as  <|think|> … reasoning … <|/think|>  answer

RunPod
  - Paths auto-resolve to /workspace on a RunPod pod
  - train_runpod.sh selects model size & batch based on detected VRAM

Usage:
    python scripts/train.py
    python scripts/train.py --batch_size 256 --compile --max_epochs 3
    python scripts/train.py --grad_accum 4 --ckpt_interval 500 --no_grad_ckpt
"""

import argparse
import json
import math
import os
import time

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.utils.checkpoint as grad_ckpt
from tqdm import tqdm

from dataset import build_dataloaders

# ── Optional: TensorBoard ─────────────────────────────────────────────────────
try:
    from torch.utils.tensorboard import SummaryWriter
    HAS_TENSORBOARD = True
except ImportError:
    HAS_TENSORBOARD = False

# ── Optional: Matplotlib ──────────────────────────────────────────────────────
try:
    import matplotlib
    matplotlib.use("Agg")           # non-interactive; works on headless RunPod pods
    import matplotlib.pyplot as plt
    HAS_MATPLOTLIB = True
except ImportError:
    HAS_MATPLOTLIB = False


# ---------------------------------------------------------------------------
# RunPod workspace detection
# ---------------------------------------------------------------------------

_WORKSPACE = (
    "/workspace"
    if os.path.isdir("/workspace")
    else os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
)


# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------

class Config:
    """
    Central configuration.  All fields are overridable at the command line.

    H100 recommended starting point:
        --batch_size 256 --grad_accum 2 --compile --no_grad_ckpt --max_epochs 3
    """

    # ── Model architecture ──────────────────────────────────────────────────
    vocab_size   = 50265        # GPT-2 + <|sep|> + <|think|> + <|/think|> + 5 ChatML role tokens
    n_layer      = 12
    n_head       = 12
    n_embd       = 768
    block_size   = 1024
    dropout      = 0.1

    # ── Training ────────────────────────────────────────────────────────────
    batch_size              = 32        # per-device batch; H100 can handle 256+
    grad_accum_steps        = 1         # effective_batch = batch_size × grad_accum
    max_epochs              = 3         # 2-3 epochs; early stopping kicks in if needed
    lr                      = 3e-4      # peak LR (cosine schedule)
    min_lr                  = 3e-5      # floor LR at end of cosine decay
    warmup_steps            = 150       # linear warm-up before cosine decay
    weight_decay            = 0.1
    adam_betas              = (0.9, 0.95)
    grad_clip               = 1.0

    # ── Scheduling / logging ─────────────────────────────────────────────────
    eval_interval           = 200       # validate every N *optimizer* steps
    log_interval            = 20        # TensorBoard scalar every N steps
    ckpt_interval           = 500       # mid-run checkpoint every N steps
    num_workers             = 8         # DataLoader workers (H100 pods have 16+ cores)

    # ── Early stopping ───────────────────────────────────────────────────────
    early_stop_patience     = 4         # eval periods with no improvement → stop
    early_stop_min_delta    = 1e-4      # minimum improvement that counts

    # ── Hardware ─────────────────────────────────────────────────────────────
    gradient_checkpointing  = True      # set False on H100 if VRAM is plentiful
    device = "cuda" if torch.cuda.is_available() else "cpu"
    dtype  = torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16

    # ── Paths ─────────────────────────────────────────────────────────────────
    processed_dir = os.path.join(_WORKSPACE, "data/processed")
    ckpt_dir      = os.path.join(_WORKSPACE, "models/checkpoints")
    log_dir       = os.path.join(_WORKSPACE, "logs")


cfg = Config()


# ---------------------------------------------------------------------------
# Model components  (architecture unchanged from previous version)
# ---------------------------------------------------------------------------

class MultiHeadAttention(nn.Module):
    """
    Multi-Head Causal Self-Attention.

    Fused Q/K/V projection → split into n_head heads → Flash Attention
    (scaled_dot_product_attention, causal mask, PyTorch 2+) → output proj.
    """

    def __init__(self, cfg: Config):
        super().__init__()
        assert cfg.n_embd % cfg.n_head == 0
        self.n_head   = cfg.n_head
        self.n_embd   = cfg.n_embd
        self.head_dim = cfg.n_embd // cfg.n_head
        self._dropout = cfg.dropout

        self.c_attn    = nn.Linear(cfg.n_embd, 3 * cfg.n_embd, bias=True)
        self.c_proj    = nn.Linear(cfg.n_embd, cfg.n_embd,     bias=True)
        self.resid_drop = nn.Dropout(cfg.dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B, T, C = x.shape
        q, k, v = self.c_attn(x).split(self.n_embd, dim=2)

        def _heads(t):
            return t.view(B, T, self.n_head, self.head_dim).transpose(1, 2)

        q, k, v  = _heads(q), _heads(k), _heads(v)
        drop_p   = self._dropout if self.training else 0.0
        out      = F.scaled_dot_product_attention(q, k, v,
                                                  dropout_p=drop_p,
                                                  is_causal=True)
        out = out.transpose(1, 2).contiguous().view(B, T, C)
        return self.resid_drop(self.c_proj(out))


class FeedForward(nn.Module):
    """Position-wise FFN: Linear → GELU → Linear → Dropout  (4× expansion)."""

    def __init__(self, cfg: Config):
        super().__init__()
        self.fc1  = nn.Linear(cfg.n_embd, 4 * cfg.n_embd, bias=True)
        self.act  = nn.GELU()
        self.fc2  = nn.Linear(4 * cfg.n_embd, cfg.n_embd, bias=True)
        self.drop = nn.Dropout(cfg.dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.drop(self.fc2(self.act(self.fc1(x))))


class TransformerBlock(nn.Module):
    """
    Pre-norm Transformer block:
        x = x + Attention( LayerNorm(x) )
        x = x + FeedForward( LayerNorm(x) )

    Gradient checkpointing recomputes activations during the backward pass
    to trade ~30% extra compute for ~50% less peak VRAM.
    """

    def __init__(self, cfg: Config):
        super().__init__()
        self.ln1  = nn.LayerNorm(cfg.n_embd)
        self.attn = MultiHeadAttention(cfg)
        self.ln2  = nn.LayerNorm(cfg.n_embd)
        self.ff   = FeedForward(cfg)
        self._use_checkpoint = cfg.gradient_checkpointing

    def _forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x + self.attn(self.ln1(x))
        x = x + self.ff(self.ln2(x))
        return x

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if self._use_checkpoint and self.training:
            return grad_ckpt.checkpoint(self._forward, x, use_reentrant=False)
        return self._forward(x)


class GPT(nn.Module):
    """
    GPT-style decoder-only Transformer for next-token prediction.

    Embeddings  : tok_emb (vocab × n_embd) + pos_emb (block_size × n_embd)
    Core        : n_layer × TransformerBlock
    Output      : LayerNorm + linear head (weight-tied to tok_emb)
    CoT         : generate_cot() enforces <|think|> … <|/think|> structure
    """

    def __init__(self, cfg: Config):
        super().__init__()
        self.cfg = cfg

        self.tok_emb = nn.Embedding(cfg.vocab_size, cfg.n_embd)
        self.pos_emb = nn.Embedding(cfg.block_size, cfg.n_embd)
        self.drop    = nn.Dropout(cfg.dropout)
        self.blocks  = nn.ModuleList([TransformerBlock(cfg) for _ in range(cfg.n_layer)])
        self.ln_f    = nn.LayerNorm(cfg.n_embd)
        self.head    = nn.Linear(cfg.n_embd, cfg.vocab_size, bias=False)

        # Weight tying: output head shares weights with token embedding
        self.tok_emb.weight = self.head.weight

        self.apply(self._init_weights)
        # Scale residual projections per GPT-2 paper §2.3
        for name, p in self.named_parameters():
            if name.endswith(("c_proj.weight", "fc2.weight")):
                nn.init.normal_(p, mean=0.0, std=0.02 / (2 * cfg.n_layer) ** 0.5)

    @staticmethod
    def _init_weights(m: nn.Module) -> None:
        if isinstance(m, nn.Linear):
            nn.init.normal_(m.weight, std=0.02)
            if m.bias is not None:
                nn.init.zeros_(m.bias)
        elif isinstance(m, nn.Embedding):
            nn.init.normal_(m.weight, std=0.02)
        elif isinstance(m, nn.LayerNorm):
            nn.init.ones_(m.weight)
            nn.init.zeros_(m.bias)

    def forward(self, idx: torch.Tensor,
                targets: torch.Tensor | None = None,
                ) -> tuple[torch.Tensor, torch.Tensor | None]:
        B, T = idx.shape
        assert T <= self.cfg.block_size
        tok = self.tok_emb(idx)
        pos = self.pos_emb(torch.arange(T, device=idx.device))
        x   = self.drop(tok + pos)
        for block in self.blocks:
            x = block(x)
        x      = self.ln_f(x)
        logits = self.head(x)
        loss   = (F.cross_entropy(logits.view(-1, logits.size(-1)), targets.view(-1))
                  if targets is not None else None)
        return logits, loss

    @torch.no_grad()
    def generate(self, idx, max_new_tokens, temperature=1.0,
                 top_k=None, stop_token_id=None):
        for _ in range(max_new_tokens):
            idx_c   = idx[:, -self.cfg.block_size:]
            logits, _ = self(idx_c)
            logits  = logits[:, -1, :] / temperature
            if top_k is not None:
                v, _ = torch.topk(logits, min(top_k, logits.size(-1)))
                logits[logits < v[:, [-1]]] = -float("inf")
            next_tok = torch.multinomial(F.softmax(logits, dim=-1), 1)
            idx      = torch.cat([idx, next_tok], dim=1)
            if stop_token_id is not None and next_tok.item() == stop_token_id:
                break
        return idx

    @torch.no_grad()
    def generate_cot(self, idx, think_token_id, end_think_token_id,
                     max_think_tokens=256, max_answer_tokens=200,
                     temperature=0.8, top_k=40):
        """Chain-of-Thought: prompt → <|think|> reasoning <|/think|> answer."""
        idx = torch.cat([idx,
                         torch.tensor([[think_token_id]],
                                      dtype=torch.long, device=idx.device)], dim=1)
        idx = self.generate(idx, max_think_tokens, temperature=temperature,
                            top_k=top_k, stop_token_id=end_think_token_id)
        if idx[0, -1].item() != end_think_token_id:
            idx = torch.cat([idx,
                             torch.tensor([[end_think_token_id]],
                                          dtype=torch.long, device=idx.device)], dim=1)
        return self.generate(idx, max_answer_tokens,
                             temperature=temperature, top_k=top_k)


# ---------------------------------------------------------------------------
# Model summary
# ---------------------------------------------------------------------------

def model_summary(model: GPT, cfg: Config) -> None:
    sep = "=" * 72
    print(f"\n{sep}")
    print("  MODEL SUMMARY — GPT Transformer (next-token prediction)")
    print(sep)
    print(f"  {'Layer':<42} {'#Params':>10}  Shape")
    print("-" * 72)
    for name, module in model.named_modules():
        if list(module.children()):
            continue
        n = sum(p.numel() for p in module.parameters())
        if n == 0:
            continue
        shape = (str(list(module.weight.shape))
                 if hasattr(module, "weight") and module.weight is not None else "")
        disp  = name if len(name) <= 42 else "…" + name[-41:]
        print(f"  {disp:<42} {n:>10,}  {shape}")
    print("-" * 72)
    total    = sum(p.numel() for p in model.parameters())
    unique   = sum(p.numel() for p in set(model.parameters()))
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    param_mb = unique * 2 / 1024 ** 2        # fp16/bf16 bytes
    flash    = hasattr(F, "scaled_dot_product_attention")
    eff_bs   = cfg.batch_size * cfg.grad_accum_steps
    print(f"\n  Architecture")
    print(f"    Layers / Heads / Hidden  : {cfg.n_layer} / {cfg.n_head} / {cfg.n_embd}")
    print(f"    Head dim                 : {cfg.n_embd // cfg.n_head}")
    print(f"    Context length           : {cfg.block_size}")
    print(f"    Vocab size               : {cfg.vocab_size:,}")
    print(f"    FFN expansion            : 4×  ({cfg.n_embd} → {4*cfg.n_embd})")
    print(f"\n  Parameters")
    print(f"    Total (with tied)        : {total:>12,}  ({total/1e6:.1f} M)")
    print(f"    Unique (deduplicated)    : {unique:>12,}  ({unique/1e6:.1f} M)")
    print(f"    Trainable                : {trainable:>12,}  ({trainable/1e6:.1f} M)")
    print(f"    Param memory (bf16)      : {param_mb:.0f} MB")
    print(f"\n  Training")
    print(f"    Objective                : next-token prediction (causal LM)")
    print(f"    Optimizer                : AdamW  wd={cfg.weight_decay}  betas={cfg.adam_betas}")
    print(f"    LR schedule              : linear warm-up ({cfg.warmup_steps} steps) → cosine")
    print(f"    Peak / floor LR          : {cfg.lr} / {cfg.min_lr}")
    print(f"    Batch / grad-accum / eff : {cfg.batch_size} / {cfg.grad_accum_steps} / {eff_bs}")
    print(f"    Mixed precision          : {cfg.dtype}  ({cfg.device})")
    print(f"    Gradient checkpointing   : {cfg.gradient_checkpointing}")
    print(f"    Flash Attention          : {'Yes (PyTorch 2+)' if flash else 'No'}")
    print(f"    TensorBoard              : {'Yes' if HAS_TENSORBOARD else 'No — pip install tensorboard'}")
    print(f"    Matplotlib curves        : {'Yes' if HAS_MATPLOTLIB else 'No — pip install matplotlib'}")
    print(f"    Early stop patience      : {cfg.early_stop_patience} eval periods")
    print(f"\n  Chain of Thought")
    print(f"    Tokens                   : <|think|>  <|/think|>")
    print(f"    Method                   : model.generate_cot(idx, think_id, end_think_id)")
    print(sep + "\n")


# ---------------------------------------------------------------------------
# H100 optimisations
# ---------------------------------------------------------------------------

def configure_for_h100() -> None:
    """
    Apply global PyTorch settings that maximise H100 throughput.

    TF32: H100 TensorCores run float32 matmuls in TF32 mode — same dynamic
          range as FP32, ~8× the raw FLOPs.
    BF16: native H100 BF16 Tensor Cores are used by torch.amp.autocast.
    benchmark: lets cuDNN profile kernel variants for the first batch and
               pick the fastest one — pays off when shapes are fixed.
    """
    if not torch.cuda.is_available():
        return
    torch.backends.cuda.matmul.allow_tf32             = True
    torch.backends.cudnn.allow_tf32                   = True
    torch.backends.cudnn.benchmark                    = True
    torch.set_float32_matmul_precision("high")        # TF32 for float32 ops
    # BF16 reductions — marginal extra throughput on H100
    torch.backends.cuda.matmul.allow_bf16_reduced_precision_reduction = True
    gpu_name = torch.cuda.get_device_name(0).lower()
    if "h100" in gpu_name or "h200" in gpu_name or "a100" in gpu_name:
        print(f"  H100/A100 detected — TF32, cuDNN benchmark, BF16 reductions enabled")
    else:
        print(f"  GPU optimisations applied  (TF32 + cuDNN benchmark)")


# ---------------------------------------------------------------------------
# LR schedule — linear warm-up + cosine decay
# ---------------------------------------------------------------------------

def cosine_lr(step: int, warmup_steps: int, total_steps: int,
              max_lr: float, min_lr: float) -> float:
    """
    Returns the learning rate for a given step.

    Phase 1 (step < warmup_steps)  : linear ramp  0 → max_lr
    Phase 2 (step ≥ warmup_steps)  : cosine decay max_lr → min_lr
    """
    if step < warmup_steps:
        return max_lr * (step + 1) / max(warmup_steps, 1)
    progress = (step - warmup_steps) / max(total_steps - warmup_steps, 1)
    progress = min(progress, 1.0)
    return min_lr + 0.5 * (max_lr - min_lr) * (1.0 + math.cos(math.pi * progress))


def set_lr(optimizer: torch.optim.Optimizer, lr: float) -> None:
    for g in optimizer.param_groups:
        g["lr"] = lr


# ---------------------------------------------------------------------------
# Early stopping
# ---------------------------------------------------------------------------

class EarlyStopping:
    """
    Tracks validation loss and signals when training should stop.

    Stops when `patience` consecutive evaluation periods pass without
    improving by at least `min_delta`.
    """

    def __init__(self, patience: int = 4, min_delta: float = 1e-4):
        self.patience  = patience
        self.min_delta = min_delta
        self.best      = float("inf")
        self.counter   = 0
        self.triggered = False

    def step(self, val_loss: float) -> bool:
        """Returns True when training should stop."""
        if val_loss < self.best - self.min_delta:
            self.best    = val_loss
            self.counter = 0
        else:
            self.counter += 1
            if self.counter >= self.patience:
                self.triggered = True
        return self.triggered

    def status(self) -> str:
        return (f"patience {self.counter}/{self.patience}  "
                f"best={self.best:.4f}")


# ---------------------------------------------------------------------------
# Checkpointing
# ---------------------------------------------------------------------------

def save_checkpoint(model, optimizer, cfg: Config, epoch: int,
                    step: int, val_loss: float, tag: str) -> str:
    """Save a checkpoint and return its path."""
    os.makedirs(cfg.ckpt_dir, exist_ok=True)
    path = os.path.join(cfg.ckpt_dir, f"{tag}.pt")
    # If model was compiled, unwrap the original module for saving
    raw = model._orig_mod if hasattr(model, "_orig_mod") else model
    torch.save({
        "model":     raw.state_dict(),
        "optimizer": optimizer.state_dict(),
        "cfg":       vars(cfg),
        "epoch":     epoch,
        "step":      step,
        "val_loss":  val_loss,
    }, path)
    return path


# ---------------------------------------------------------------------------
# Loss visualisation
# ---------------------------------------------------------------------------

def plot_loss_curves(train_steps: list, train_losses: list,
                     val_steps: list,   val_losses: list,
                     save_path: str) -> None:
    """
    Plot training and validation loss curves and save to `save_path`.
    Also draws a smoothed EMA trend line over the raw training loss.
    """
    if not HAS_MATPLOTLIB:
        print("  matplotlib not installed — skipping loss curve plot")
        return

    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    fig.suptitle("Training Loss Curves", fontsize=14, fontweight="bold")

    # ── Left: raw + smoothed train loss ──────────────────────────────────────
    ax = axes[0]
    ax.plot(train_steps, train_losses, color="#aec6e8", linewidth=0.7,
            alpha=0.6, label="Train loss (raw)")

    # EMA smoothing
    if len(train_losses) > 10:
        ema, alpha = [], 0.05
        s = train_losses[0]
        for v in train_losses:
            s = alpha * v + (1 - alpha) * s
            ema.append(s)
        ax.plot(train_steps, ema, color="#1f77b4", linewidth=1.8,
                label="Train loss (EMA α=0.05)")

    ax.set_xlabel("Optimizer step")
    ax.set_ylabel("Cross-entropy loss")
    ax.set_title("Training Loss")
    ax.legend()
    ax.grid(True, alpha=0.3)

    # ── Right: validation loss ────────────────────────────────────────────────
    ax = axes[1]
    ax.plot(val_steps, val_losses, color="#ff7f0e", linewidth=2.0,
            marker="o", markersize=4, label="Val loss")

    # Mark the best point
    if val_losses:
        best_idx = val_losses.index(min(val_losses))
        ax.scatter([val_steps[best_idx]], [val_losses[best_idx]],
                   color="#d62728", s=80, zorder=5,
                   label=f"Best = {val_losses[best_idx]:.4f}")

    ax.set_xlabel("Optimizer step")
    ax.set_ylabel("Cross-entropy loss")
    ax.set_title("Validation Loss")
    ax.legend()
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  Loss curve saved → {save_path}")


# ---------------------------------------------------------------------------
# Evaluation
# ---------------------------------------------------------------------------

@torch.no_grad()
def evaluate(model: GPT, val_loader, cfg: Config) -> float:
    """
    Run the model over validation batches and return mean cross-entropy loss.
    Capped at 100 batches so it stays fast during training.
    """
    model.eval()
    losses = []
    for x, y in val_loader:
        x = x.to(cfg.device, non_blocking=True)
        y = y.to(cfg.device, non_blocking=True)
        with torch.amp.autocast(cfg.device, dtype=cfg.dtype,
                                enabled=(cfg.device == "cuda")):
            _, loss = model(x, y)
        losses.append(loss.item())
        if len(losses) >= 100:
            break
    return sum(losses) / len(losses)


# ---------------------------------------------------------------------------
# Optimizer
# ---------------------------------------------------------------------------

def configure_optimizer(model: GPT, cfg: Config) -> torch.optim.AdamW:
    """
    AdamW with separate weight-decay groups.
    2-D+ params (weights) → decay   |   1-D params (biases, LN) → no decay
    """
    decay   = [p for n, p in model.named_parameters() if p.dim() >= 2]
    nodecay = [p for n, p in model.named_parameters() if p.dim() <  2]
    groups  = [
        {"params": decay,   "weight_decay": cfg.weight_decay},
        {"params": nodecay, "weight_decay": 0.0},
    ]
    print(f"  AdamW : {sum(p.numel() for p in decay):,} params w/ decay  |  "
          f"{sum(p.numel() for p in nodecay):,} without")
    use_fused = (torch.cuda.is_available() and
                 "fused" in torch.optim.AdamW.__init__.__code__.co_varnames)
    kwargs = {"fused": True} if use_fused else {}
    return torch.optim.AdamW(groups, lr=cfg.lr, betas=cfg.adam_betas, **kwargs)


# ---------------------------------------------------------------------------
# Training loop
# ---------------------------------------------------------------------------

def train(args: argparse.Namespace) -> None:   # noqa: C901  (intentionally long)
    os.makedirs(cfg.ckpt_dir, exist_ok=True)
    os.makedirs(cfg.log_dir,  exist_ok=True)

    # ── H100 / GPU setup ──────────────────────────────────────────────────────
    print("\n" + "=" * 60)
    print("  HARDWARE")
    print("=" * 60)
    configure_for_h100()
    print(f"  Device : {cfg.device.upper()}")
    if cfg.device == "cuda":
        props   = torch.cuda.get_device_properties(0)
        vram_gb = props.total_memory / 1024 ** 3
        print(f"  GPU    : {props.name}  ({vram_gb:.1f} GB VRAM)")
        print(f"  CUDA   : {torch.version.cuda}")
        # Pre-allocate a small tensor to warm up the CUDA context
        _ = torch.zeros(1, device=cfg.device)
        torch.cuda.synchronize()

    # ── Data ──────────────────────────────────────────────────────────────────
    print("\n" + "=" * 60)
    print("  DATA")
    print("=" * 60)
    train_loader, val_loader, info = build_dataloaders(
        processed_dir=cfg.processed_dir,
        batch_size=cfg.batch_size,
        num_workers=cfg.num_workers,
    )
    cfg.vocab_size = info["vocab_size"]
    cfg.block_size = info["block_size"]
    think_id     = info.get("think_token_id")
    end_think_id = info.get("end_think_token_id")
    eot_id       = info.get("eot_token_id", 50256)

    steps_per_epoch = len(train_loader) // cfg.grad_accum_steps
    total_steps     = steps_per_epoch * cfg.max_epochs

    print(f"  Vocab         : {cfg.vocab_size:,}   block_size : {cfg.block_size}")
    print(f"  Train batches : {len(train_loader):,}    Val batches : {len(val_loader):,}")
    print(f"  Effective batch size : {cfg.batch_size * cfg.grad_accum_steps:,}  "
          f"({cfg.batch_size} × {cfg.grad_accum_steps} grad-accum)")
    print(f"  Steps/epoch : {steps_per_epoch:,}    Total steps : {total_steps:,}")

    # ── Data quality warnings ─────────────────────────────────────────────────
    if cfg.block_size < 512:
        print(f"\n  [WARN] block_size={cfg.block_size} — re-run preprocess.py with "
              "--block_size 1024 for full context length")
    total_toks = info.get("train_tokens", 0) + info.get("val_tokens", 0)
    if total_toks < 1_000_000:
        print(f"\n  [WARN] Only {total_toks:,} tokens in dataset.  A 124 M-param model "
              "needs 10 M+ tokens for meaningful training.\n"
              "         Run:  python scripts/preprocess.py --gutenberg 1342 84 11 2701\n"
              "         (Pride & Prejudice, Frankenstein, Alice, Moby-Dick)\n"
              "         Or add your own .jsonl / .txt / .docx files to data/raw/")

    # ── Model ─────────────────────────────────────────────────────────────────
    model = GPT(cfg).to(cfg.device)
    model_summary(model, cfg)

    if args.compile:
        if hasattr(torch, "compile"):
            mode = "max-autotune"   # best mode for H100
            print(f"  torch.compile(mode='{mode}') — tracing … (first batch will be slow)")
            model = torch.compile(model, mode=mode)
        else:
            print("  torch.compile not available (PyTorch < 2.0) — skipping")

    # ── Optimizer + scaler ────────────────────────────────────────────────────
    optimizer = configure_optimizer(model, cfg)
    scaler    = torch.amp.GradScaler(cfg.device, enabled=(cfg.device == "cuda"))

    # ── Resume from checkpoint ────────────────────────────────────────────────
    resume_epoch = 0
    if getattr(args, "resume", None) and os.path.isfile(args.resume):
        print(f"\n  Resuming from: {args.resume}")
        ckpt_r = torch.load(args.resume, map_location=cfg.device, weights_only=False)
        raw    = model._orig_mod if hasattr(model, "_orig_mod") else model
        sd     = {k.replace("_orig_mod.", ""): v for k, v in ckpt_r["model"].items()}
        raw.load_state_dict(sd, strict=False)
        optimizer.load_state_dict(ckpt_r["optimizer"])
        resume_epoch = ckpt_r.get("epoch", 0)
        print(f"  Resumed epoch={resume_epoch}  "
              f"step={ckpt_r.get('step', 0)}  "
              f"val_loss={ckpt_r.get('val_loss', float('nan')):.4f}")
    elif getattr(args, "resume", None):
        print(f"  [WARN] --resume path not found: {args.resume} — starting fresh")

    # ── TensorBoard ───────────────────────────────────────────────────────────
    tb_dir = os.path.join(cfg.log_dir, "tensorboard")
    if HAS_TENSORBOARD:
        writer = SummaryWriter(log_dir=tb_dir)
        print(f"  TensorBoard → tensorboard --logdir {tb_dir}")
    else:
        writer = None
        print("  TensorBoard not installed — pip install tensorboard")

    # ── Tokenizer (for generation samples) ───────────────────────────────────
    try:
        from transformers import GPT2TokenizerFast
        tokenizer = GPT2TokenizerFast.from_pretrained("gpt2")
        special = [
            "<|sep|>", "<|think|>", "<|/think|>",
            "<|system|>", "<|/system|>", "<|user|>", "<|/user|>", "<|assistant|>",
        ]
        tokenizer.add_special_tokens({"additional_special_tokens": special})
    except Exception:
        tokenizer = None

    # ── History for loss curves ───────────────────────────────────────────────
    history = {
        "train_steps": [], "train_losses": [],
        "val_steps":   [], "val_losses":   [],
    }
    log_path     = os.path.join(cfg.log_dir, "train_log.jsonl")
    early_stop   = EarlyStopping(cfg.early_stop_patience, cfg.early_stop_min_delta)
    best_val     = float("inf")
    opt_step     = 0          # optimizer steps (after grad accum)
    train_step   = 0          # raw batch steps
    stop_reason  = "max epochs reached"

    # ── Training ──────────────────────────────────────────────────────────────
    print("\n" + "=" * 60)
    print("  TRAINING")
    print("=" * 60)
    t0 = time.perf_counter()

    for epoch in range(resume_epoch, cfg.max_epochs):
        model.train()
        optimizer.zero_grad(set_to_none=True)

        # ── tqdm progress bar ─────────────────────────────────────────────────
        pbar = tqdm(
            train_loader,
            desc=f"Epoch {epoch+1}/{cfg.max_epochs}",
            unit="batch",
            dynamic_ncols=True,
            colour="cyan",
        )

        loss_accum  = 0.0          # accumulated loss over grad_accum_steps
        last_val    = float("nan")
        tokens_seen = 0

        for batch_idx, (x, y) in enumerate(pbar):
            # ── LR schedule ───────────────────────────────────────────────────
            lr = cosine_lr(opt_step, cfg.warmup_steps, total_steps,
                           cfg.lr, cfg.min_lr)
            set_lr(optimizer, lr)

            # ── Non-blocking GPU transfer ──────────────────────────────────────
            x = x.to(cfg.device, non_blocking=True)
            y = y.to(cfg.device, non_blocking=True)

            # ── Forward + loss ─────────────────────────────────────────────────
            with torch.amp.autocast(cfg.device, dtype=cfg.dtype,
                                    enabled=(cfg.device == "cuda")):
                _, loss = model(x, y)
                # Scale loss for gradient accumulation
                loss_scaled = loss / cfg.grad_accum_steps

            # ── Backward ──────────────────────────────────────────────────────
            scaler.scale(loss_scaled).backward()
            loss_accum += loss.item()
            tokens_seen += x.numel()

            is_accum_step = ((batch_idx + 1) % cfg.grad_accum_steps == 0
                             or batch_idx == len(train_loader) - 1)

            if is_accum_step:
                # ── Gradient clipping + optimizer step ────────────────────────
                scaler.unscale_(optimizer)
                grad_norm = torch.nn.utils.clip_grad_norm_(
                    model.parameters(), cfg.grad_clip
                )
                scaler.step(optimizer)
                scaler.update()
                optimizer.zero_grad(set_to_none=True)

                avg_loss = loss_accum / cfg.grad_accum_steps
                loss_accum = 0.0
                opt_step  += 1

                # ── GPU memory usage ──────────────────────────────────────────
                if cfg.device == "cuda":
                    mem_gb = torch.cuda.memory_reserved() / 1024 ** 3
                    mem_pct = (torch.cuda.memory_reserved() /
                               torch.cuda.get_device_properties(0).total_memory * 100)
                else:
                    mem_gb, mem_pct = 0.0, 0.0

                # ── History ───────────────────────────────────────────────────
                history["train_steps"].append(opt_step)
                history["train_losses"].append(avg_loss)

                # ── TensorBoard ───────────────────────────────────────────────
                if writer and opt_step % cfg.log_interval == 0:
                    writer.add_scalar("Loss/train",     avg_loss,   opt_step)
                    writer.add_scalar("LR",             lr,         opt_step)
                    writer.add_scalar("GradNorm",       grad_norm,  opt_step)
                    if cfg.device == "cuda":
                        writer.add_scalar("GPU/mem_GB",  mem_gb,    opt_step)
                        writer.add_scalar("GPU/mem_pct", mem_pct,   opt_step)

                # ── JSONL log ─────────────────────────────────────────────────
                if opt_step % cfg.log_interval == 0:
                    elapsed = time.perf_counter() - t0
                    tok_per_sec = tokens_seen / max(elapsed, 1e-6)
                    with open(log_path, "a") as f:
                        f.write(json.dumps({
                            "step": opt_step, "epoch": epoch + 1,
                            "train_loss": round(avg_loss, 6),
                            "lr": round(lr, 8),
                            "tok_per_sec": int(tok_per_sec),
                        }) + "\n")

                # ── tqdm postfix ──────────────────────────────────────────────
                pbar.set_postfix(
                    loss   = f"{avg_loss:.4f}",
                    val    = f"{last_val:.4f}" if not math.isnan(last_val) else "—",
                    lr     = f"{lr:.2e}",
                    mem    = f"{mem_pct:.0f}%",
                    step   = opt_step,
                )

                # ── Mid-epoch validation ──────────────────────────────────────
                if opt_step % cfg.eval_interval == 0 and opt_step > 0:
                    val_loss = evaluate(model, val_loader, cfg)
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

                    marker = "★ NEW BEST" if val_loss < best_val else ""
                    tqdm.write(
                        f"  [step {opt_step:>6} | epoch {epoch+1}]  "
                        f"val_loss={val_loss:.4f}  {early_stop.status()}  {marker}"
                    )
                    model.train()

                # ── Step checkpoint ────────────────────────────────────────────
                if opt_step % cfg.ckpt_interval == 0 and opt_step > 0:
                    path = save_checkpoint(model, optimizer, cfg,
                                           epoch + 1, opt_step,
                                           last_val if not math.isnan(last_val) else float("inf"),
                                           tag=f"step_{opt_step:07d}")
                    tqdm.write(f"  [step {opt_step}] checkpoint → {path}")

            train_step += 1

        pbar.close()

        # ── Epoch-end evaluation ───────────────────────────────────────────────
        val_loss = evaluate(model, val_loader, cfg)
        history["val_steps"].append(opt_step)
        history["val_losses"].append(val_loss)
        if writer:
            writer.add_scalar("Loss/val_epoch", val_loss, epoch + 1)

        elapsed  = time.perf_counter() - t0
        tok_s    = tokens_seen / max(elapsed, 1e-6)
        is_best  = val_loss < best_val

        print(f"\n{'─'*60}")
        print(f"  Epoch {epoch+1}/{cfg.max_epochs} complete")
        print(f"  Val loss     : {val_loss:.4f}  {'← NEW BEST' if is_best else f'(best={best_val:.4f})'}")
        print(f"  Elapsed      : {elapsed/60:.1f} min")
        print(f"  Throughput   : {tok_s:,.0f} tokens/sec")

        if is_best:
            best_val = val_loss
            path = save_checkpoint(model, optimizer, cfg,
                                   epoch + 1, opt_step, val_loss, tag="best")
            print(f"  Best ckpt    : {path}")

        path = save_checkpoint(model, optimizer, cfg,
                               epoch + 1, opt_step, val_loss,
                               tag=f"epoch_{epoch+1:02d}")
        print(f"  Epoch ckpt   : {path}")

        # ── Generation samples ─────────────────────────────────────────────────
        model.eval()
        ctx = torch.tensor([[eot_id]], dtype=torch.long, device=cfg.device)
        sample = model.generate(ctx, max_new_tokens=150, temperature=0.8, top_k=40)
        _print_sample("Standard generation", sample, tokenizer)

        if think_id is not None and end_think_id is not None:
            cot = model.generate_cot(ctx, think_id, end_think_id,
                                     max_think_tokens=150, max_answer_tokens=100)
            _print_sample("Chain-of-Thought generation", cot, tokenizer)

        # ── Early stopping check ───────────────────────────────────────────────
        if early_stop.step(val_loss):
            stop_reason = (f"early stopping — val loss did not improve by "
                           f"{cfg.early_stop_min_delta} for "
                           f"{cfg.early_stop_patience} evaluations")
            print(f"\n  *** {stop_reason.upper()} ***\n")
            break

        print()

    # ── Training complete ──────────────────────────────────────────────────────
    total_time = time.perf_counter() - t0
    print("=" * 60)
    print(f"  TRAINING COMPLETE  ({stop_reason})")
    print(f"  Total time  : {total_time/60:.1f} min")
    print(f"  Best val    : {best_val:.4f}")
    print(f"  Total steps : {opt_step:,}")
    print("=" * 60)

    # ── Loss curves ────────────────────────────────────────────────────────────
    curve_path = os.path.join(cfg.log_dir, "loss_curves.png")
    plot_loss_curves(
        history["train_steps"], history["train_losses"],
        history["val_steps"],   history["val_losses"],
        save_path=curve_path,
    )

    if writer:
        writer.flush()
        writer.close()
        print(f"  TensorBoard logs → tensorboard --logdir {tb_dir}")

    print(f"  JSONL log    → {log_path}\n")


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _print_sample(label: str, token_ids: torch.Tensor, tokenizer) -> None:
    if tokenizer is not None:
        text = tokenizer.decode(token_ids[0].tolist(), skip_special_tokens=False)
    else:
        text = str(token_ids[0].tolist()[:40]) + " ..."
    print(f"\n--- {label} ---\n{text[:500]}\n{'─'*(len(label)+6)}")


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Train a GPT Transformer (H100-optimised, RunPod-ready, CoT-capable)",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )

    # Resume
    parser.add_argument("--resume", default=None, metavar="CKPT",
                        help="Path to checkpoint to resume training from "
                             "(e.g. models/checkpoints/step_0005000.pt)")

    # Data
    parser.add_argument("--processed_dir", default=cfg.processed_dir)

    # Model
    parser.add_argument("--n_layer",   type=int,   default=cfg.n_layer)
    parser.add_argument("--n_head",    type=int,   default=cfg.n_head)
    parser.add_argument("--n_embd",    type=int,   default=cfg.n_embd)

    # Training schedule
    parser.add_argument("--batch_size",    type=int,   default=cfg.batch_size,
                        help="Per-device batch size (H100: 256+)")
    parser.add_argument("--grad_accum",    type=int,   default=cfg.grad_accum_steps,
                        help="Gradient accumulation steps (effective_batch = batch × accum)")
    parser.add_argument("--max_epochs",    type=int,   default=cfg.max_epochs)
    parser.add_argument("--lr",            type=float, default=cfg.lr)
    parser.add_argument("--min_lr",        type=float, default=cfg.min_lr)
    parser.add_argument("--warmup_steps",  type=int,   default=cfg.warmup_steps)

    # Intervals
    parser.add_argument("--eval_interval", type=int,   default=cfg.eval_interval,
                        help="Validate every N optimizer steps")
    parser.add_argument("--ckpt_interval", type=int,   default=cfg.ckpt_interval,
                        help="Step checkpoint every N optimizer steps")
    parser.add_argument("--log_interval",  type=int,   default=cfg.log_interval)

    # Early stopping
    parser.add_argument("--patience",      type=int,   default=cfg.early_stop_patience)

    # Hardware / speed
    parser.add_argument("--num_workers",   type=int,   default=cfg.num_workers)
    parser.add_argument("--no_grad_ckpt",  action="store_true",
                        help="Disable gradient checkpointing (uses more VRAM, faster)")
    parser.add_argument("--compile",       action="store_true",
                        help="torch.compile(mode='max-autotune') — ~30%% faster on H100")

    args = parser.parse_args()

    # Apply overrides
    cfg.processed_dir          = args.processed_dir
    cfg.n_layer                = args.n_layer
    cfg.n_head                 = args.n_head
    cfg.n_embd                 = args.n_embd
    cfg.batch_size             = args.batch_size
    cfg.grad_accum_steps       = args.grad_accum
    cfg.max_epochs             = args.max_epochs
    cfg.lr                     = args.lr
    cfg.min_lr                 = args.min_lr
    cfg.warmup_steps           = args.warmup_steps
    cfg.eval_interval          = args.eval_interval
    cfg.ckpt_interval          = args.ckpt_interval
    cfg.log_interval           = args.log_interval
    cfg.early_stop_patience    = args.patience
    cfg.num_workers            = args.num_workers
    cfg.gradient_checkpointing = not args.no_grad_ckpt
    # resume is consumed directly from args inside train()

    train(args)
