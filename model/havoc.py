"""
HAVOC shoes-model: nanoGPT-style decoder-only transformer.

Configured for ~49-64M params (49M non-embedding) at:
    n_layer=12  n_head=8  n_embd=512  block_size=1024  vocab=50257 (GPT-2 BPE)

Uses F.scaled_dot_product_attention (Flash) when available, tied output
embeddings, learned positional embeddings, and GELU MLPs. Nothing exotic.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

import torch
import torch.nn as nn
import torch.nn.functional as F


@dataclass
class HavocConfig:
    block_size:    int   = 1024
    vocab_size:    int   = 50257   # GPT-2 BPE
    n_layer:       int   = 12
    n_head:        int   = 8
    n_embd:        int   = 512
    mlp_ratio:     int   = 4       # MLP hidden = mlp_ratio * n_embd
    dropout:       float = 0.0
    bias:          bool  = True    # GPT-2 uses biases in Linear / LN
    tie_embeddings: bool = True


# ── Submodules ─────────────────────────────────────────────────────────────


class LayerNorm(nn.Module):
    """LayerNorm with optional bias (matches nanoGPT)."""
    def __init__(self, ndim: int, bias: bool):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(ndim))
        self.bias   = nn.Parameter(torch.zeros(ndim)) if bias else None

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return F.layer_norm(x, self.weight.shape, self.weight, self.bias, 1e-5)


class CausalSelfAttention(nn.Module):
    def __init__(self, cfg: HavocConfig):
        super().__init__()
        assert cfg.n_embd % cfg.n_head == 0
        self.n_head = cfg.n_head
        self.n_embd = cfg.n_embd
        self.head_dim = cfg.n_embd // cfg.n_head
        self.dropout = cfg.dropout

        self.c_attn = nn.Linear(cfg.n_embd, 3 * cfg.n_embd, bias=cfg.bias)
        self.c_proj = nn.Linear(cfg.n_embd, cfg.n_embd,     bias=cfg.bias)

        self.attn_dropout  = nn.Dropout(cfg.dropout)
        self.resid_dropout = nn.Dropout(cfg.dropout)

        self._flash = hasattr(F, "scaled_dot_product_attention")
        if not self._flash:
            self.register_buffer(
                "bias_mask",
                torch.tril(torch.ones(cfg.block_size, cfg.block_size))
                     .view(1, 1, cfg.block_size, cfg.block_size),
                persistent=False,
            )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B, T, C = x.shape
        q, k, v = self.c_attn(x).split(self.n_embd, dim=2)
        q = q.view(B, T, self.n_head, self.head_dim).transpose(1, 2)
        k = k.view(B, T, self.n_head, self.head_dim).transpose(1, 2)
        v = v.view(B, T, self.n_head, self.head_dim).transpose(1, 2)

        if self._flash:
            y = F.scaled_dot_product_attention(
                q, k, v,
                attn_mask=None,
                dropout_p=self.dropout if self.training else 0.0,
                is_causal=True,
            )
        else:
            att = (q @ k.transpose(-2, -1)) * (1.0 / math.sqrt(self.head_dim))
            att = att.masked_fill(self.bias_mask[:, :, :T, :T] == 0, float("-inf"))
            att = F.softmax(att, dim=-1)
            att = self.attn_dropout(att)
            y = att @ v

        y = y.transpose(1, 2).contiguous().view(B, T, C)
        return self.resid_dropout(self.c_proj(y))


class MLP(nn.Module):
    def __init__(self, cfg: HavocConfig):
        super().__init__()
        hidden = cfg.mlp_ratio * cfg.n_embd
        self.c_fc   = nn.Linear(cfg.n_embd, hidden, bias=cfg.bias)
        self.c_proj = nn.Linear(hidden, cfg.n_embd, bias=cfg.bias)
        self.dropout = nn.Dropout(cfg.dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.dropout(self.c_proj(F.gelu(self.c_fc(x))))


class Block(nn.Module):
    def __init__(self, cfg: HavocConfig):
        super().__init__()
        self.ln_1 = LayerNorm(cfg.n_embd, bias=cfg.bias)
        self.attn = CausalSelfAttention(cfg)
        self.ln_2 = LayerNorm(cfg.n_embd, bias=cfg.bias)
        self.mlp  = MLP(cfg)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x + self.attn(self.ln_1(x))
        x = x + self.mlp(self.ln_2(x))
        return x


# ── Model ──────────────────────────────────────────────────────────────────


class HavocGPT(nn.Module):
    def __init__(self, cfg: HavocConfig):
        super().__init__()
        self.cfg = cfg

        self.tok_emb = nn.Embedding(cfg.vocab_size, cfg.n_embd)
        self.pos_emb = nn.Embedding(cfg.block_size, cfg.n_embd)
        self.drop    = nn.Dropout(cfg.dropout)
        self.blocks  = nn.ModuleList([Block(cfg) for _ in range(cfg.n_layer)])
        self.ln_f    = LayerNorm(cfg.n_embd, bias=cfg.bias)
        self.lm_head = nn.Linear(cfg.n_embd, cfg.vocab_size, bias=False)

        if cfg.tie_embeddings:
            self.lm_head.weight = self.tok_emb.weight

        self.apply(self._init_weights)
        # GPT-2 paper scaling on residual projections
        for n, p in self.named_parameters():
            if n.endswith("c_proj.weight"):
                nn.init.normal_(p, mean=0.0, std=0.02 / math.sqrt(2 * cfg.n_layer))

    @staticmethod
    def _init_weights(module: nn.Module) -> None:
        if isinstance(module, nn.Linear):
            nn.init.normal_(module.weight, mean=0.0, std=0.02)
            if module.bias is not None:
                nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            nn.init.normal_(module.weight, mean=0.0, std=0.02)

    def num_params(self, non_embedding: bool = False) -> int:
        n = sum(p.numel() for p in self.parameters())
        if non_embedding:
            n -= self.pos_emb.weight.numel()
            if not self.cfg.tie_embeddings:
                n -= self.tok_emb.weight.numel()
            # Tied tok_emb/lm_head: count once as part of "embedding"
            if self.cfg.tie_embeddings:
                n -= self.tok_emb.weight.numel()
        return n

    def forward(self,
                idx: torch.Tensor,
                targets: torch.Tensor | None = None
                ) -> tuple[torch.Tensor, torch.Tensor | None]:
        B, T = idx.shape
        assert T <= self.cfg.block_size, f"sequence length {T} > block_size {self.cfg.block_size}"

        pos = torch.arange(0, T, dtype=torch.long, device=idx.device)
        x = self.drop(self.tok_emb(idx) + self.pos_emb(pos))
        for block in self.blocks:
            x = block(x)
        x = self.ln_f(x)

        if targets is not None:
            logits = self.lm_head(x)
            loss = F.cross_entropy(
                logits.view(-1, logits.size(-1)),
                targets.view(-1),
                ignore_index=-100,
            )
            return logits, loss

        # Inference: only need logits on the final position
        logits = self.lm_head(x[:, [-1], :])
        return logits, None

    def configure_optimizer(self,
                            weight_decay: float,
                            lr: float,
                            betas: tuple[float, float],
                            device_type: str
                            ) -> torch.optim.Optimizer:
        """
        AdamW with weight decay on Linear/Embedding weights only (no biases / LNs).
        Uses the fused implementation when CUDA + supported.
        """
        params = [(n, p) for n, p in self.named_parameters() if p.requires_grad]
        decay  = [p for n, p in params if p.dim() >= 2]
        nodecay = [p for n, p in params if p.dim() <  2]
        groups = [
            {"params": decay,   "weight_decay": weight_decay},
            {"params": nodecay, "weight_decay": 0.0},
        ]
        use_fused = (device_type == "cuda" and
                     "fused" in torch.optim.AdamW.__init__.__code__.co_varnames)
        kwargs = {"fused": True} if use_fused else {}
        return torch.optim.AdamW(groups, lr=lr, betas=betas, **kwargs)

    @torch.no_grad()
    def generate(self,
                 idx: torch.Tensor,
                 max_new_tokens: int,
                 temperature: float = 1.0,
                 top_k: int | None = None
                 ) -> torch.Tensor:
        self.eval()
        for _ in range(max_new_tokens):
            idx_cond = idx if idx.size(1) <= self.cfg.block_size \
                          else idx[:, -self.cfg.block_size:]
            logits, _ = self(idx_cond)
            logits = logits[:, -1, :] / max(temperature, 1e-6)
            if top_k is not None:
                v, _ = torch.topk(logits, min(top_k, logits.size(-1)))
                logits[logits < v[:, [-1]]] = float("-inf")
            probs = F.softmax(logits, dim=-1)
            nxt = torch.multinomial(probs, num_samples=1)
            idx = torch.cat([idx, nxt], dim=1)
        return idx
