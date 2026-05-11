"""
HAVOC — a from-scratch decoder-only transformer.

Modern internals:
  - Rotary position embeddings (RoPE), per-head, applied inside attention
  - RMSNorm (no bias, no mean-centering)
  - SwiGLU FFN  (gate, up, down)
  - Causal multi-head self-attention via F.scaled_dot_product_attention
  - Tied input/output embeddings
  - Optional gradient checkpointing per block

Target size: ~49.3 M params at the canonical 50 M preset.
Model definition is independent of training; pretrain.py and sft.py drive
training, inference.py / refinement.py / orchestrator.py drive inference.
"""

from __future__ import annotations

import math

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.utils.checkpoint as grad_ckpt

from config import HavocConfig


# ───────────────────────────── building blocks ────────────────────────────


class RMSNorm(nn.Module):
    """Root-mean-square layer norm — same shape as LayerNorm but no bias."""

    def __init__(self, dim: int, eps: float = 1.0e-6):
        super().__init__()
        self.scale = nn.Parameter(torch.ones(dim))
        self.eps = eps

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        rms = x.pow(2).mean(dim=-1, keepdim=True)
        x = x * torch.rsqrt(rms + self.eps)
        return x * self.scale


class RoPE(nn.Module):
    """
    Rotary Position Embeddings (Llama-style — split halves, not interleaved).

    Builds cos/sin tables for `max_seq_len`. forward(q, k) returns rotated
    (q, k) of identical shape. Caches are buffers so they move with .to(device).
    """

    def __init__(self, head_dim: int, max_seq_len: int, base: float = 10000.0):
        super().__init__()
        if head_dim % 2 != 0:
            raise ValueError(f"RoPE requires even head_dim, got {head_dim}")
        inv_freq = 1.0 / (base ** (torch.arange(0, head_dim, 2).float() / head_dim))
        self.register_buffer("inv_freq", inv_freq, persistent=False)
        self._build_cache(max_seq_len)

    def _build_cache(self, seq_len: int) -> None:
        t = torch.arange(seq_len, device=self.inv_freq.device).float()
        freqs = torch.outer(t, self.inv_freq)              # (T, D/2)
        self.register_buffer("cos_cached", freqs.cos(), persistent=False)
        self.register_buffer("sin_cached", freqs.sin(), persistent=False)

    @staticmethod
    def _rotate(x: torch.Tensor, cos: torch.Tensor, sin: torch.Tensor) -> torch.Tensor:
        # x: (B, H, T, D)   cos/sin: (T, D/2) → broadcast to (1, 1, T, D/2)
        # NOTE: do NOT name this `_apply` — collides with nn.Module._apply
        x1, x2 = x.chunk(2, dim=-1)
        cos = cos.unsqueeze(0).unsqueeze(0)
        sin = sin.unsqueeze(0).unsqueeze(0)
        return torch.cat([x1 * cos - x2 * sin, x2 * cos + x1 * sin], dim=-1)

    def forward(self, q: torch.Tensor, k: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        T = q.shape[-2]
        if T > self.cos_cached.shape[0]:
            self._build_cache(T)
        cos = self.cos_cached[:T].to(q.dtype)
        sin = self.sin_cached[:T].to(q.dtype)
        return self._rotate(q, cos, sin), self._rotate(k, cos, sin)


class SwiGLU(nn.Module):
    """Three-matrix gated-MLP: down(silu(gate(x)) * up(x))."""

    def __init__(self, d_model: int, d_ff: int, dropout: float = 0.0):
        super().__init__()
        self.gate = nn.Linear(d_model, d_ff,    bias=False)
        self.up   = nn.Linear(d_model, d_ff,    bias=False)
        self.down = nn.Linear(d_ff,    d_model, bias=False)
        self.drop = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.drop(self.down(F.silu(self.gate(x)) * self.up(x)))


class HavocAttention(nn.Module):
    """Causal multi-head self-attention with RoPE, fused QKV projection."""

    def __init__(self, cfg: HavocConfig):
        super().__init__()
        if cfg.hidden_size % cfg.num_heads != 0:
            raise ValueError("hidden_size must be divisible by num_heads")
        self.n_head    = cfg.num_heads
        self.head_dim  = cfg.hidden_size // cfg.num_heads
        self.hidden    = cfg.hidden_size
        self._dropout  = cfg.dropout

        self.qkv  = nn.Linear(cfg.hidden_size, 3 * cfg.hidden_size, bias=False)
        self.proj = nn.Linear(cfg.hidden_size, cfg.hidden_size,     bias=False)
        self.rope = RoPE(self.head_dim, cfg.max_seq_len, base=cfg.rope_base)
        self.resid_drop = nn.Dropout(cfg.dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B, T, C = x.shape
        q, k, v = self.qkv(x).chunk(3, dim=-1)

        def split(t: torch.Tensor) -> torch.Tensor:
            return t.view(B, T, self.n_head, self.head_dim).transpose(1, 2)

        q, k, v = split(q), split(k), split(v)
        q, k    = self.rope(q, k)

        drop_p  = self._dropout if self.training else 0.0
        out     = F.scaled_dot_product_attention(q, k, v,
                                                 dropout_p=drop_p,
                                                 is_causal=True)
        out     = out.transpose(1, 2).contiguous().view(B, T, C)
        return self.resid_drop(self.proj(out))


class HavocBlock(nn.Module):
    """Pre-norm transformer block: RMSNorm → Attn → res ; RMSNorm → SwiGLU → res."""

    def __init__(self, cfg: HavocConfig):
        super().__init__()
        self.norm1 = RMSNorm(cfg.hidden_size, eps=cfg.rms_norm_eps)
        self.attn  = HavocAttention(cfg)
        self.norm2 = RMSNorm(cfg.hidden_size, eps=cfg.rms_norm_eps)
        self.mlp   = SwiGLU(cfg.hidden_size, cfg.intermediate_size, cfg.dropout)
        self._use_checkpoint = cfg.gradient_checkpointing

    def _forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x + self.attn(self.norm1(x))
        x = x + self.mlp(self.norm2(x))
        return x

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if self._use_checkpoint and self.training:
            return grad_ckpt.checkpoint(self._forward, x, use_reentrant=False)
        return self._forward(x)


# ─────────────────────────────── full model ───────────────────────────────


class HavocModel(nn.Module):
    """
    Decoder-only transformer LM.

    Embeddings:        tok_emb (vocab × d_model)              ── tied with lm_head
    Body:              num_layers × HavocBlock
    Head:              RMSNorm → Linear (no bias)             ── shares weight w/ tok_emb
    Positional info:   RoPE inside each attention head        ── no learned pos_emb
    """

    def __init__(self, cfg: HavocConfig):
        super().__init__()
        self.cfg = cfg

        self.tok_emb  = nn.Embedding(cfg.vocab_size, cfg.hidden_size)
        self.drop     = nn.Dropout(cfg.dropout)
        self.blocks   = nn.ModuleList([HavocBlock(cfg) for _ in range(cfg.num_layers)])
        self.norm_f   = RMSNorm(cfg.hidden_size, eps=cfg.rms_norm_eps)
        self.lm_head  = nn.Linear(cfg.hidden_size, cfg.vocab_size, bias=False)

        if cfg.tied_embeddings:
            self.lm_head.weight = self.tok_emb.weight

        self.apply(self._init_weights)
        # GPT-2-style scaled init for residual projections (depth ≈ 2·n_layer)
        scale = 0.02 / math.sqrt(2 * cfg.num_layers)
        for name, p in self.named_parameters():
            if name.endswith(("attn.proj.weight", "mlp.down.weight")):
                nn.init.normal_(p, mean=0.0, std=scale)

    @staticmethod
    def _init_weights(m: nn.Module) -> None:
        if isinstance(m, nn.Linear):
            nn.init.normal_(m.weight, std=0.02)
            if m.bias is not None:
                nn.init.zeros_(m.bias)
        elif isinstance(m, nn.Embedding):
            nn.init.normal_(m.weight, std=0.02)

    # ── forward ────────────────────────────────────────────────────────────

    def forward(self,
                idx: torch.Tensor,
                targets: torch.Tensor | None = None,
                ) -> tuple[torch.Tensor, torch.Tensor | None]:
        B, T = idx.shape
        if T > self.cfg.max_seq_len:
            raise ValueError(f"sequence length {T} exceeds max_seq_len {self.cfg.max_seq_len}")
        x = self.drop(self.tok_emb(idx))
        for block in self.blocks:
            x = block(x)
        x      = self.norm_f(x)
        logits = self.lm_head(x)
        loss   = None
        if targets is not None:
            loss = F.cross_entropy(logits.view(-1, logits.size(-1)),
                                   targets.view(-1))
        return logits, loss

    # ── basic generation (used by inference.py / refinement scaffolds) ─────

    @torch.no_grad()
    def generate(self,
                 idx: torch.Tensor,
                 max_new_tokens: int,
                 temperature: float = 1.0,
                 top_k: int | None = None,
                 stop_token_id: int | None = None,
                 ) -> torch.Tensor:
        for _ in range(max_new_tokens):
            ctx        = idx[:, -self.cfg.max_seq_len:]
            logits, _  = self(ctx)
            logits     = logits[:, -1, :] / max(temperature, 1.0e-6)
            if top_k is not None:
                v, _ = torch.topk(logits, min(top_k, logits.size(-1)))
                logits[logits < v[:, [-1]]] = -float("inf")
            probs    = F.softmax(logits, dim=-1)
            next_tok = torch.multinomial(probs, num_samples=1)
            idx      = torch.cat([idx, next_tok], dim=1)
            if stop_token_id is not None and next_tok.item() == stop_token_id:
                break
        return idx


# ──────────────────────── helpers ────────────────────────


def count_params(model: nn.Module) -> dict[str, int]:
    """Returns total / unique / trainable parameter counts."""
    total     = sum(p.numel() for p in model.parameters())
    unique    = sum(p.numel() for p in {id(p): p for p in model.parameters()}.values())
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    return {"total": total, "unique": unique, "trainable": trainable}


def build_model(cfg: HavocConfig) -> HavocModel:
    """Convenience constructor that mirrors HavocModel(cfg) for symmetry."""
    return HavocModel(cfg)
