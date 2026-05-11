"""
HAVOC model + training + refinement configuration.

A single dataclass holds every knob so that pretrain.py, sft.py, inference.py,
and verify_params.py read from the same source of truth. Configs persist to
JSON and round-trip via from_json / to_json.
"""

from __future__ import annotations

import json
import os
from dataclasses import asdict, dataclass, field, fields
from typing import Any


# Special tokens used by HavocTokenizer. Order is significant — these IDs
# are assigned 0..len-1 at training time so the tokenizer file is
# deterministic across runs.
SPECIAL_TOKENS: tuple[str, ...] = (
    "<|endoftext|>",
    "<|pad|>",
    "<|sep|>",
    "<|user|>",
    "<|/user|>",
    "<|assistant|>",
    "<|/assistant|>",
    "<|system|>",
    "<|/system|>",
    "<|think|>",
    "<|/think|>",
    "<|tool|>",
    "<|/tool|>",
)


@dataclass
class HavocConfig:
    # ── Model architecture ─────────────────────────────────────────────────
    vocab_size: int = 16384
    hidden_size: int = 512
    num_layers: int = 12
    num_heads: int = 8
    intermediate_size: int = 1536          # SwiGLU 3 × expansion
    max_seq_len: int = 2048
    dropout: float = 0.1
    rope_base: float = 10000.0
    tied_embeddings: bool = True
    rms_norm_eps: float = 1.0e-6

    # ── Training defaults ──────────────────────────────────────────────────
    gradient_checkpointing: bool = True
    batch_size: int = 32
    grad_accum_steps: int = 1
    max_epochs: int = 3
    lr: float = 3.0e-4
    min_lr: float = 3.0e-5
    warmup_steps: int = 150
    weight_decay: float = 0.1
    grad_clip: float = 1.0
    eval_interval: int = 200
    log_interval: int = 20
    ckpt_interval: int = 500
    early_stop_patience: int = 4
    early_stop_min_delta: float = 1.0e-4
    num_workers: int = 8

    # ── Refinement defaults (inference-time, never affects model weights) ──
    enable_refinement: bool = False
    refinement_max_passes: int = 10
    refinement_confidence_threshold: float = 0.85
    refinement_similarity_threshold: float = 0.9

    # ── Orchestration defaults (also inference-time) ───────────────────────
    enable_orchestration: bool = False
    enable_retrieval: bool = True
    enable_tools: bool = True
    knowledge_dir: str = "data/knowledge"
    retrieval_top_k: int = 4

    # ── Special token IDs (filled in by tokenizer load) ────────────────────
    pad_token_id: int = 1                  # matches SPECIAL_TOKENS index
    eos_token_id: int = 0                  # <|endoftext|>
    sep_token_id: int = 2

    # ── Param-count validation target ──────────────────────────────────────
    expected_param_count: int = 49_295_872
    param_count_tolerance: float = 0.02    # ±2 %

    # ── Adam betas: dataclass needs default_factory for tuple defaults ─────
    adam_betas: tuple[float, float] = field(default_factory=lambda: (0.9, 0.95))

    # ── Derived properties ─────────────────────────────────────────────────
    @property
    def head_dim(self) -> int:
        return self.hidden_size // self.num_heads

    # ── (de)serialisation ──────────────────────────────────────────────────
    @classmethod
    def from_json(cls, path: str) -> "HavocConfig":
        with open(path, encoding="utf-8") as f:
            data: dict[str, Any] = json.load(f)
        # Drop unknown keys defensively so older configs still load.
        valid = {f.name for f in fields(cls)}
        clean = {k: v for k, v in data.items() if k in valid}
        # JSON has no tuple type; restore adam_betas if present
        if isinstance(clean.get("adam_betas"), list):
            clean["adam_betas"] = tuple(clean["adam_betas"])
        return cls(**clean)

    def to_json(self, path: str) -> None:
        os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
        with open(path, "w", encoding="utf-8") as f:
            json.dump(asdict(self), f, indent=2, sort_keys=False)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def default_50m_config() -> HavocConfig:
    """The canonical ~49.3M HAVOC preset."""
    return HavocConfig()
