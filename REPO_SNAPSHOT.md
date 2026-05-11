# REPO_SNAPSHOT.md

Read-only inventory of the HAVOC repo at `c:\Users\Scott\OneDrive\Desktop\LLM`,
generated 2026-05-11 for review by another Claude instance. No files were
modified to produce this snapshot.

---

## 1. REPO STRUCTURE

Three-level tree (excluding `.git`, `__pycache__`, `node_modules`, `venv`,
`data/shards`, `checkpoints/`, `.ipynb_checkpoints/`, and `*.pyc`):

```
./
  .claude/
    scheduled_tasks.lock
    settings.local.json
  chat_ui/
    templates/
      index.html
    app.py
  configs/
    havoc-50m.json
  data/
    processed/
      tokenizer_info.json
      train_tokens.pt
      val_tokens.pt
    processed_smoke/
      tokenizer_info.json
      train.bin
      val.bin
    processed_smoke_run/
      tokenizer_info.json
      train.bin
      val.bin
    processed_tiny_v0/
      train.bin
      val.bin
    processed_v0/
      tokenizer_info.json
      train.bin
      val.bin
    raw/
      gutenberg/
      alpaca.jsonl
      dolly.jsonl
      gutenberg.jsonl
      oasst.jsonl
      ultrachat.jsonl
    sft_v0/
      all.jsonl
      train.jsonl
      val.jsonl
    smoke_raw/
      conversations.jsonl
    prepare_pretrain.py
  logs/
    tiny_pretrain/
      pretrain_log.jsonl
    tiny_sft/
      sft_log.jsonl
    v0_pretrain/
      tensorboard/
      loss_curves.png
      pretrain_log.jsonl
    v0_sft/
      tensorboard_sft/
      sft_log.jsonl
      sft_loss_curves.png
    gui_settings.json
  model/
    __init__.py
    havoc.py
  models/
    tokenizers/
      havoc_bpe_full/
      havoc_smoke/
      havoc_v0/
  notebooks/
    runpod_pretrain.ipynb
  scripts/
    agents/
      __init__.py
      critic.py
      retrieval.py
    tools/
      __init__.py
      calculator.py
      file_reader.py
      json_validator.py
      python_exec.py
      regex_utility.py
      router.py
      text_parser.py
      unit_converter.py
    build_v0_dataset.py
    config.py
    dataset.py
    download_datasets.py
    eval_tiny.py
    eval_v0.py
    gui_app.py
    inference.py
    model.py
    orchestrator.py
    preprocess.py
    pretrain.py
    refinement.py
    setup_runpod.sh
    sft.py
    tokenizer_havoc.py
    train_runpod.sh
    train_tokenizer.py
    verifier.py
    verify_params.py
    verify_setup.py
  train/
    __init__.py
    pretrain.py
  -ScottsLaptop.gitignore
  .dockerignore
  .gitignore
  CHANGELOG.md
  conversations_audit.md
  Dockerfile
  README.md
  REPO_SNAPSHOT.md
  requirements.txt
  system_prompt.txt
```

Notes:
- Two parallel transformer stacks coexist in the repo:
  - **Legacy v0** under `scripts/` (RoPE/RMSNorm/SwiGLU HavocModel, custom HavocTokenizer, 16K vocab — described in README).
  - **Shoes model** under `model/` + `train/` + `data/prepare_pretrain.py` (nanoGPT-style HavocGPT, GPT-2 BPE, 50257 vocab — new in this session).
- `data/shards/` and `checkpoints/` are not yet on disk; they will be created by `data/prepare_pretrain.py` and `train/pretrain.py` respectively.

---

## 2. FILE CONTENTS

### data/prepare_pretrain.py

```python
"""
HAVOC shoes-model pretrain data prep.

Reads raw corpora (Parquet + OASST2 .jsonl.gz), tokenizes with the GPT-2
BPE via tiktoken, mixes at document level to hit per-source token budgets,
shuffles, and writes uint16 binary memmap shards to data/shards/ in the
nanoGPT format.

Target mix (~1.7B tokens, 1 epoch):
    fineweb-edu-dedup : ~1.0B tokens   (Parquet under <raw>/smollm/fineweb-edu-dedup/)
    cosmopedia-v2     : ~200M tokens   (Parquet under <raw>/smollm/cosmopedia-v2/)
    TinyStoriesV2-GPT4: ~300M tokens   (Parquet under <raw>/tinystories/)
    OASST2 (en)       : ~200M tokens   (jsonl.gz under <raw>/oasst2/)

Each shard contains contiguous uint16 token IDs. A meta.json sidecar records
total counts and per-source breakdown.

Usage (defaults assume local Windows path):
    python data/prepare_pretrain.py \
        --raw_dir   C:/havoc_data \
        --out_dir   data/shards \
        --tokens_per_shard 100_000_000

On RunPod:
    python data/prepare_pretrain.py \
        --raw_dir   /workspace/havoc_data/raw \
        --out_dir   /workspace/havoc/data/shards
"""

from __future__ import annotations

import argparse
import glob
import gzip
import json
import math
import os
import random
import sys
import time
from typing import Iterable

import numpy as np
from tqdm import tqdm


# ── Source readers ─────────────────────────────────────────────────────────


def _iter_parquet_text(parquet_paths: list[str],
                       text_keys: tuple[str, ...] = ("text", "content")) -> Iterable[str]:
    """Yield non-empty text fields from a list of Parquet files."""
    import pyarrow.parquet as pq
    for path in parquet_paths:
        try:
            pf = pq.ParquetFile(path)
        except Exception as e:
            print(f"  [WARN] could not open {path}: {e}")
            continue
        # Detect which key is present in this file
        cols = pf.schema_arrow.names
        key = next((k for k in text_keys if k in cols), None)
        if key is None:
            print(f"  [WARN] {os.path.basename(path)}: no text column "
                  f"(have {cols[:5]}...), skipping")
            continue
        for batch in pf.iter_batches(batch_size=1024, columns=[key]):
            col = batch.column(0)
            for v in col.to_pylist():
                if isinstance(v, str) and v.strip():
                    yield v


def _list_parquets(root: str) -> list[str]:
    if not os.path.isdir(root):
        return []
    out = sorted(glob.glob(os.path.join(root, "**", "*.parquet"), recursive=True))
    return out


def read_fineweb_edu(raw_dir: str) -> Iterable[str]:
    root = os.path.join(raw_dir, "smollm", "fineweb-edu-dedup")
    files = _list_parquets(root)
    print(f"  fineweb-edu-dedup  : {len(files)} parquet files under {root}")
    yield from _iter_parquet_text(files)


def read_cosmopedia_v2(raw_dir: str) -> Iterable[str]:
    root = os.path.join(raw_dir, "smollm", "cosmopedia-v2")
    files = _list_parquets(root)
    print(f"  cosmopedia-v2      : {len(files)} parquet files under {root}")
    yield from _iter_parquet_text(files)


def read_tinystories(raw_dir: str) -> Iterable[str]:
    """
    Reads TinyStoriesV2-GPT4. Prefers V2-GPT4 parquet files; falls back to any
    parquet under <raw>/tinystories/ if no V2 file is named explicitly.
    """
    root = os.path.join(raw_dir, "tinystories")
    if not os.path.isdir(root):
        return
    all_pq = _list_parquets(root)
    v2 = [p for p in all_pq if "v2" in os.path.basename(p).lower()
          or "v2" in os.path.dirname(p).lower()]
    files = v2 or all_pq
    print(f"  tinystories (V2)   : {len(files)} parquet files under {root}")
    yield from _iter_parquet_text(files)


def read_oasst2(raw_dir: str,
                lang: str = "en",
                min_quality: float = 0.5) -> Iterable[str]:
    """
    Walk every conversation tree in the OASST2 `ready` dump, flatten each
    root->leaf branch into a multi-turn dialogue text block.

    Filter rules:
      - drop nodes where lang != `lang`
      - drop nodes where labels.spam.value > 0.5 (when present)
      - drop nodes where labels.quality.value < min_quality (when present)
    """
    root_dir = os.path.join(raw_dir, "oasst2")
    if not os.path.isdir(root_dir):
        return
    gz_files = sorted(glob.glob(os.path.join(root_dir, "*ready*.jsonl.gz"))) \
            or sorted(glob.glob(os.path.join(root_dir, "*.jsonl.gz")))
    if not gz_files:
        print(f"  oasst2             : no .jsonl.gz files under {root_dir}")
        return
    print(f"  oasst2             : {len(gz_files)} jsonl.gz files under {root_dir}")

    def _label(node: dict, name: str) -> float | None:
        labels = node.get("labels") or {}
        v = labels.get(name)
        if isinstance(v, dict):
            v = v.get("value")
        if isinstance(v, (int, float)):
            return float(v)
        return None

    def _accept(node: dict) -> bool:
        if node.get("lang", lang) != lang:
            return False
        spam = _label(node, "spam")
        if spam is not None and spam > 0.5:
            return False
        quality = _label(node, "quality")
        if quality is not None and quality < min_quality:
            return False
        if not (node.get("text") or "").strip():
            return False
        return True

    def _walk(node: dict, history: list[tuple[str, str]]) -> Iterable[list[tuple[str, str]]]:
        if not _accept(node):
            return
        role = node.get("role", "")
        text = node["text"].strip()
        new_hist = history + [(role, text)]
        replies = node.get("replies") or []
        if not replies:
            yield new_hist
            return
        for r in replies:
            yield from _walk(r, new_hist)

    for path in gz_files:
        with gzip.open(path, "rt", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                try:
                    tree = json.loads(line)
                except json.JSONDecodeError:
                    continue
                root_node = tree.get("prompt") or tree
                for turns in _walk(root_node, []):
                    parts = []
                    for role, text in turns:
                        speaker = "User" if role == "prompter" else "Assistant"
                        parts.append(f"{speaker}: {text}")
                    yield "\n\n".join(parts)


# ── Tokenization + shard writer ────────────────────────────────────────────


_DTYPE = np.uint16  # GPT-2 vocab is 50257 -> fits in uint16


class ShardWriter:
    """
    Writes a stream of token IDs to fixed-size .bin shards on disk.

    Layout (nanoGPT-style):
        data/shards/train_00000.bin   (uint16, tokens_per_shard tokens)
        data/shards/train_00001.bin
        ...
        data/shards/val_00000.bin
        data/shards/meta.json
    """

    def __init__(self, out_dir: str, split: str, tokens_per_shard: int):
        self.out_dir          = out_dir
        self.split            = split
        self.tokens_per_shard = tokens_per_shard
        os.makedirs(out_dir, exist_ok=True)
        self._buf: list[int]  = []
        self._shard_idx       = 0
        self._total           = 0
        self.shards: list[str] = []

    def add(self, ids: list[int]) -> None:
        self._buf.extend(ids)
        while len(self._buf) >= self.tokens_per_shard:
            chunk = self._buf[: self.tokens_per_shard]
            self._buf = self._buf[self.tokens_per_shard:]
            self._flush(chunk)

    def close(self) -> None:
        if self._buf:
            self._flush(self._buf)
            self._buf = []

    def _flush(self, chunk: list[int]) -> None:
        path = os.path.join(self.out_dir, f"{self.split}_{self._shard_idx:05d}.bin")
        arr = np.asarray(chunk, dtype=_DTYPE)
        arr.tofile(path)
        self.shards.append(path)
        self._shard_idx += 1
        self._total     += len(chunk)

    @property
    def total_tokens(self) -> int:
        return self._total + len(self._buf)


def _build_encoder():
    """Return (encode_fn, eot_id, vocab_size). Tries tiktoken, falls back to HF."""
    try:
        import tiktoken
        enc = tiktoken.get_encoding("gpt2")
        eot = enc.eot_token  # 50256
        vocab = enc.n_vocab  # 50257
        def encode(text: str) -> list[int]:
            return enc.encode_ordinary(text)
        return encode, eot, vocab
    except Exception as e:
        print(f"  [INFO] tiktoken unavailable ({e}); falling back to transformers gpt2")
        from transformers import GPT2TokenizerFast
        hf = GPT2TokenizerFast.from_pretrained("gpt2")
        eot = hf.eos_token_id  # 50256
        vocab = hf.vocab_size  # 50257 with added specials? gpt2 base = 50257
        def encode(text: str) -> list[int]:
            return hf.encode(text, add_special_tokens=False)
        return encode, eot, vocab


# ── Main ───────────────────────────────────────────────────────────────────


def main() -> int:
    p = argparse.ArgumentParser(description="HAVOC shoes-model pretrain data prep.")
    p.add_argument("--raw_dir", default=r"C:\havoc_data",
                   help="Root of raw corpora (contains smollm/, tinystories/, oasst2/).")
    p.add_argument("--out_dir", default="data/shards")

    # Per-source token budgets (None = no cap, take everything we encode)
    p.add_argument("--budget_fineweb_edu",   type=int, default=1_000_000_000)
    p.add_argument("--budget_cosmopedia_v2", type=int, default=200_000_000)
    p.add_argument("--budget_tinystories",   type=int, default=300_000_000)
    p.add_argument("--budget_oasst2",        type=int, default=200_000_000)

    p.add_argument("--val_tokens",       type=int, default=2_000_000,
                   help="Tokens to hold out for val.bin.")
    p.add_argument("--tokens_per_shard", type=int, default=100_000_000)
    p.add_argument("--shuffle_buffer",   type=int, default=4096,
                   help="Document-level shuffle buffer size.")
    p.add_argument("--oasst2_min_quality", type=float, default=0.5)
    p.add_argument("--oasst2_lang",        default="en")
    p.add_argument("--seed",               type=int, default=1337)
    args = p.parse_args()

    rng = random.Random(args.seed)
    os.makedirs(args.out_dir, exist_ok=True)

    # ── Tokenizer ──────────────────────────────────────────────────────────
    print("=== Tokenizer ===")
    encode, eot_id, vocab_size = _build_encoder()
    print(f"  GPT-2 BPE   vocab_size = {vocab_size}   eot_id = {eot_id}")
    assert vocab_size <= np.iinfo(_DTYPE).max + 1, "vocab does not fit in uint16"

    # ── Sources ────────────────────────────────────────────────────────────
    sources = [
        ("fineweb_edu",   args.budget_fineweb_edu,   read_fineweb_edu(args.raw_dir)),
        ("cosmopedia_v2", args.budget_cosmopedia_v2, read_cosmopedia_v2(args.raw_dir)),
        ("tinystories",   args.budget_tinystories,   read_tinystories(args.raw_dir)),
        ("oasst2",        args.budget_oasst2,
         read_oasst2(args.raw_dir,
                     lang=args.oasst2_lang,
                     min_quality=args.oasst2_min_quality)),
    ]

    # ── Tokenize each source to its budget, collect documents ─────────────
    docs: list[tuple[str, list[int]]] = []
    per_source: dict[str, int] = {}
    print("\n=== Tokenizing sources ===")
    t0 = time.time()
    for name, budget, stream in sources:
        if budget <= 0:
            print(f"  {name:<14}: budget = 0, skipped")
            per_source[name] = 0
            continue
        used = 0
        n_docs = 0
        pbar = tqdm(desc=f"  {name}", unit="tok", total=budget, dynamic_ncols=True)
        for text in stream:
            ids = encode(text)
            ids.append(eot_id)
            docs.append((name, ids))
            used += len(ids)
            n_docs += 1
            pbar.update(len(ids))
            if used >= budget:
                break
        pbar.close()
        per_source[name] = used
        print(f"    -> {name}: {used:,} tokens over {n_docs:,} docs")

    elapsed = time.time() - t0
    print(f"  Tokenization done in {elapsed/60:.1f} min")

    if not docs:
        print("ERROR: no documents produced; check --raw_dir paths.", file=sys.stderr)
        return 1

    # ── Document-level shuffle ─────────────────────────────────────────────
    print("\n=== Shuffling documents ===")
    rng.shuffle(docs)
    print(f"  shuffled {len(docs):,} documents")

    # ── Split: first val_tokens go to val.bin, rest to train shards ───────
    print("\n=== Writing shards ===")
    val_writer   = ShardWriter(args.out_dir, "val",   tokens_per_shard=args.val_tokens)
    train_writer = ShardWriter(args.out_dir, "train", tokens_per_shard=args.tokens_per_shard)

    val_taken = 0
    for name, ids in docs:
        if val_taken < args.val_tokens:
            need = args.val_tokens - val_taken
            if len(ids) <= need:
                val_writer.add(ids)
                val_taken += len(ids)
            else:
                val_writer.add(ids[:need])
                val_taken += need
                train_writer.add(ids[need:])
        else:
            train_writer.add(ids)

    val_writer.close()
    train_writer.close()

    print(f"  train shards : {len(train_writer.shards)}   "
          f"tokens : {train_writer.total_tokens:,}")
    print(f"  val   shards : {len(val_writer.shards)}     "
          f"tokens : {val_writer.total_tokens:,}")

    # ── Sidecar meta ───────────────────────────────────────────────────────
    meta = {
        "vocab_size":       vocab_size,
        "eot_id":           eot_id,
        "dtype":            "uint16",
        "tokens_per_shard": args.tokens_per_shard,
        "train_shards":     [os.path.basename(s) for s in train_writer.shards],
        "val_shards":       [os.path.basename(s) for s in val_writer.shards],
        "train_tokens":     train_writer.total_tokens,
        "val_tokens":       val_writer.total_tokens,
        "per_source_tokens": per_source,
        "seed":             args.seed,
    }
    meta_path = os.path.join(args.out_dir, "meta.json")
    with open(meta_path, "w") as f:
        json.dump(meta, f, indent=2)
    print(f"  meta -> {meta_path}")

    # ── Report ─────────────────────────────────────────────────────────────
    sep = "=" * 72
    total = sum(per_source.values())
    print(f"\n{sep}\n  HAVOC SHOES-MODEL PRETRAIN PREP REPORT\n{sep}")
    print(f"  raw_dir : {args.raw_dir}")
    print(f"  out_dir : {os.path.abspath(args.out_dir)}")
    print(f"  vocab   : {vocab_size}   (GPT-2 BPE, uint16 shards)")
    print(f"\n  Per-source tokens:")
    for k, v in per_source.items():
        pct = (100 * v / total) if total else 0.0
        print(f"    {k:<14} : {v:>14,}   ({pct:5.1f}%)")
    print(f"\n  Total   : {total:,} tokens")
    print(f"  Train   : {train_writer.total_tokens:,} tokens "
          f"in {len(train_writer.shards)} shards")
    print(f"  Val     : {val_writer.total_tokens:,} tokens "
          f"in {len(val_writer.shards)} shards")
    print(f"{sep}\n")
    return 0


if __name__ == "__main__":
    sys.exit(main())
```

### model/havoc.py

```python
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
```

### model/tokenizer.py

FILE NOT YET CREATED

Note: the shoes-model uses the GPT-2 BPE via `tiktoken` (loaded inside
`data/prepare_pretrain.py::_build_encoder`). No project-owned tokenizer
module exists under `model/`. The legacy v0 stack has a separate
`scripts/tokenizer_havoc.py` (see section 3).

### model/config.py

FILE NOT YET CREATED

Note: the `HavocConfig` dataclass for the shoes-model lives inline at the
top of `model/havoc.py`. A standalone `model/config.py` does not exist.

### train/pretrain.py

```python
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
    python train/pretrain.py \
        --shards_dir data/shards \
        --ckpt_dir   checkpoints \
        --total_tokens 1_700_000_000

Resume:
    python train/pretrain.py --resume checkpoints/latest.pt
"""

from __future__ import annotations

import argparse
import csv
import math
import os
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
        train_loss = loss_sum

        if args.grad_clip > 0:
            torch.nn.utils.clip_grad_norm_(model.parameters(), args.grad_clip)
        optimizer.step()
        optimizer.zero_grad(set_to_none=True)

        tokens_seen        += tokens_per_optstep
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

    log_f.close()
    print(f"\nTraining done. Total elapsed: {(time.time() - t0)/60:.1f} min")
    return 0


if __name__ == "__main__":
    sys.exit(main())
```

### train/train_config.py

FILE NOT YET CREATED

Note: training hyperparameters are exposed as CLI flags on
`train/pretrain.py` (see the `argparse` block in §2 above). No standalone
`train_config.py` module exists.

### requirements.txt

```text
# PyTorch is pre-installed on RunPod images (CUDA 12.4+).
# For local install on a CUDA-13 driver:
#   pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu130
# For CUDA 12.4:
#   pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu124

# ── Core training ──────────────────────────────────────────────────────────
tokenizers>=0.19.0
tiktoken>=0.7.0               # GPT-2 BPE for shoes-model pretrain
pyarrow>=15.0.0               # Parquet readers for fineweb/cosmopedia/tinystories
numpy>=1.26.0
tqdm>=4.66.0
accelerate>=0.30.0
requests>=2.31.0
python-docx>=1.1.0
matplotlib>=3.8.0
tensorboard>=2.14.0

# ── Inference / web UI ─────────────────────────────────────────────────────
fastapi>=0.110.0
uvicorn>=0.27.0
pydantic>=2.0
customtkinter>=5.2.0          # scripts/gui_app.py (optional desktop UI)

# ── Orchestration: retrieval, tools, deterministic verifier ───────────────
rank_bm25>=0.2.2              # BM25 retrieval over data/knowledge/
pint>=0.23                    # unit conversions
jsonschema>=4.21.0            # JSON validator + verifier schema checks
RestrictedPython>=7.1         # python_exec sandbox (falls back to safe-builtins)
sympy>=1.12                   # not required, handy for symbolic math

# ── Optional / ancillary ───────────────────────────────────────────────────
# HAVOC core does NOT depend on transformers/datasets — they remain
# available for benchmark / external dataset utilities.
transformers>=4.40.0
datasets>=2.19.0
```

### .gitignore

```text
# ── Python ────────────────────────────────────────────────────────────────
venv/
.venv/
__pycache__/
*.pyc
*.pyo
*.egg-info/
dist/
build/

# ── Model artifacts (regeneratable, large) ───────────────────────────────
*.pt
*.pth
*.bin
*.safetensors
checkpoints/
checkpoints/*
models/checkpoints/
models/checkpoints/*
models/tokenizers/
models/tokenizers/*

# ── Pretrain shards / processed data ─────────────────────────────────────
data/shards/
data/shards/*
data/raw/*
data/processed/*
data/processed_v0/
data/processed_v0/*
data/processed_tiny_v0/
data/processed_tiny_v0/*
data/processed_smoke/
data/processed_smoke_run/
data/smoke_raw/
data/sft/
data/sft_v0/
data/knowledge/
data/train_log.csv

# Keep directory placeholders (commit empty .gitkeep markers)
!data/raw/.gitkeep
!data/sft/.gitkeep
!data/knowledge/.gitkeep
!data/shards/.gitkeep
!checkpoints/.gitkeep
!models/checkpoints/.gitkeep
!models/tokenizers/.gitkeep

# ── Logs / runs ───────────────────────────────────────────────────────────
logs/*
runs/
tensorboard/
wandb/
*.log

# ── Notebook checkpoints ──────────────────────────────────────────────────
.ipynb_checkpoints/

# ── HuggingFace cache ─────────────────────────────────────────────────────
.cache/
.huggingface/

# ── Env / secrets ─────────────────────────────────────────────────────────
.env
.env.*
*.key
*.pem

# ── IDE / OS ──────────────────────────────────────────────────────────────
.vscode/
.idea/
.DS_Store
Thumbs.db
```

### README.md

```markdown
# HAVOC

A from-scratch ~49M-parameter decoder-only transformer with iterative
self-refinement and a lightweight agent + tool orchestration layer.

The point isn't raw scale — at 49M params HAVOC will never match a 7B
model on knowledge-heavy tasks. The point is to make a *small* model
punch above its weight by combining:

* **A modern, clean core** — RoPE, RMSNorm, SwiGLU, tied embeddings,
  flash attention, gradient checkpointing.
* **Iterative self-refinement at inference time** — up to 10 visible
  passes with calibrated confidence (capped at 95%, never claims 100%);
  early-stops when stable.
* **An additive agent / tool orchestration layer** — retrieval (RAG),
  a critic role on the same model, a tool router (calculator, Python
  sandbox, JSON validator, file reader, text parser, unit converter,
  regex), and a *deterministic* verifier that cross-checks the output.

There are no Qwen, Llama, or other pretrained weights anywhere in this
repo. The tokenizer is trained from scratch too.

## Architecture

| Field | Value |
|---|---|
| `vocab_size` | 16384 |
| `hidden_size` | 512 |
| `num_layers` | 12 |
| `num_heads` | 8 (head_dim 64) |
| `intermediate_size` (SwiGLU) | 1536 |
| `max_seq_len` | 2048 |
| `dropout` | 0.1 |
| Position encoding | RoPE (base 10000) |
| Norm | RMSNorm |
| Activation | SwiGLU |
| Tied embeddings | yes |
| **Total params** | **49,295,872** |

(...the README continues with parameter-math, layout, pipeline,
refinement, orchestration, configuration, resuming, RunPod, and "what
this project is *not*" sections. Full file is ~243 lines; the layout
in the README describes the *legacy v0* pipeline under `scripts/`,
not the new `model/` + `train/` + `data/prepare_pretrain.py`
shoes-model stack added in this session.)
```

The full README.md is on disk at the repo root; the abbreviated block
above is its first ~30 lines plus a pointer. The legacy-v0 architecture
in the README (vocab=16384, RoPE/RMSNorm/SwiGLU) **does not match** the
new shoes-model in `model/havoc.py` (vocab=50257 GPT-2 BPE, learned
positional embeddings, LayerNorm, GELU). The README has not been updated
to reflect the new stack.

---

## 3. EXISTING TOKENIZER CHECK

**JSON tokenizer artifacts on disk (project-owned, excluding venv):**

- `data/processed/tokenizer_info.json`
- `data/processed_smoke/tokenizer_info.json`
- `data/processed_smoke_run/tokenizer_info.json`
- `data/processed_v0/tokenizer_info.json`
- `models/tokenizers/havoc_smoke/tokenizer.json`
- `models/tokenizers/havoc_v0/tokenizer.json`
- `models/tokenizers/havoc_bpe_full/tokenizer.json`

These all belong to the **legacy v0 HavocTokenizer** (custom byte-level
BPE, vocab 16384, special-token IDs filled in at training time). They
are NOT used by the new shoes-model, which goes through `tiktoken`'s
GPT-2 encoder at runtime.

**No `vocab.json` or `merges.txt` exists at the project level** — only
copies inside `venv/Lib/site-packages/` (irrelevant).

**Python files that reference "tokenizer"** (`grep -i tokenizer *.py`,
excluding venv and `__pycache__`):

| File | Matches |
|---|---|
| `scripts/preprocess.py` | 43 |
| `scripts/tokenizer_havoc.py` | 31 |
| `scripts/build_v0_dataset.py` | 24 |
| `scripts/inference.py` | 22 |
| `scripts/pretrain.py` | 13 |
| `scripts/train_tokenizer.py` | 11 |
| `scripts/sft.py` | 10 |
| `scripts/dataset.py` | 8 |
| `data/prepare_pretrain.py` | 4 |
| `scripts/config.py` | 3 |
| `scripts/verify_setup.py` | 3 |
| `scripts/eval_tiny.py` | 2 |
| `scripts/eval_v0.py` | 2 |
| `chat_ui/app.py` | 2 |
| **Total** | **178 occurrences across 14 files** |

The new shoes-model touches the tokenizer in exactly one place:
`data/prepare_pretrain.py:_build_encoder()` (lines 240-258), which loads
`tiktoken.get_encoding("gpt2")` with a `transformers.GPT2TokenizerFast`
fallback. All other 174 occurrences belong to the legacy v0 stack under
`scripts/`.

---

## 4. DATA DIRECTORY CHECK

`C:\havoc_data` exists. Top-level inventory (read-only `Get-ChildItem`,
size summed recursively):

| Folder | Size (GB) |
|---|---|
| `oasst2` | 0.05 |
| `smollm` | 7.92 |
| `tinystories` | 7.10 |
| **Total** | **~15.07 GB** |

`smollm` is the largest because it contains both `fineweb-edu-dedup` and
`cosmopedia-v2` subdirectories (per `data/prepare_pretrain.py`'s
expected layout). The `oasst2` folder is tiny because OASST2 ships as a
single gzipped `.jsonl.gz`. No files were opened or read during this
check.

---

## 5. PYTHON ENVIRONMENT

`pip list` from the active interpreter:

```
Package            Version
------------------ ---------
annotated-doc      0.0.4
anyio              4.13.0
certifi            2026.2.25
charset-normalizer 3.4.7
click              8.3.3
colorama           0.4.6
filelock           3.29.0
fsspec             2026.4.0
h11                0.16.0
hf-xet             1.5.0
httpcore           1.0.9
httpx              0.28.1
huggingface_hub    1.14.0
idna               3.11
markdown-it-py     4.2.0
mdurl              0.1.2
numpy              2.4.4
packaging          26.2
pip                25.3
Pygments           2.20.0
PyYAML             6.0.3
regex              2026.4.4
requests           2.33.1
rich               15.0.0
shellingham        1.5.4
tiktoken           0.12.0
tqdm               4.67.3
typer              0.25.1
typing_extensions  4.15.0
urllib3            2.6.3
```

Notable gaps for `requirements.txt`: **torch, transformers, datasets,
pyarrow, tokenizers, accelerate, matplotlib, tensorboard, fastapi,
uvicorn, pydantic, customtkinter, rank_bm25, pint, jsonschema,
RestrictedPython, sympy, python-docx** are NOT installed in this shell's
Python. `tiktoken` IS available (0.12.0). The repo's `venv/` directory
exists (visible in the tree) and likely contains these — this `pip list`
was run against whatever Python is on PATH, not necessarily the venv.

---

## 6. GPU CHECK

`nvidia-smi` output (local machine):

```
Mon May 11 01:52:03 2026
+-----------------------------------------------------------------------------------------+
| NVIDIA-SMI 581.95                 Driver Version: 581.95         CUDA Version: 13.0     |
+-----------------------------------------+------------------------+----------------------+
| GPU  Name                  Driver-Model | Bus-Id          Disp.A | Volatile Uncorr. ECC |
| Fan  Temp   Perf          Pwr:Usage/Cap |           Memory-Usage | GPU-Util  Compute M. |
|=========================================+========================+======================|
|   0  NVIDIA GeForce RTX 2050      WDDM  |   00000000:01:00.0 Off |                  N/A |
| N/A   55C    P8              2W /   30W |       0MiB /   4096MiB |      0%      Default |
+-----------------------------------------+------------------------+----------------------+

Processes:
  GPU 0:  WsaClient.exe   (background)
  GPU 0:  ollama.exe      (background)
```

**Local GPU is an RTX 2050 with 4 GB VRAM.** This is far below what the
shoes-model training defaults (block_size=1024, batch_size=8) will
comfortably fit — pretraining is intended to run on the **RunPod RTX
4090** per the user's brief. Local can still be used for smoke tests
with reduced `--batch_size` and `--block_size`.

---

## Snapshot summary

- New shoes-model stack created this session: `data/prepare_pretrain.py`,
  `model/havoc.py`, `train/pretrain.py`, `notebooks/runpod_pretrain.ipynb`.
- `requirements.txt` was extended (added `tiktoken`, `pyarrow`); the new
  stack does NOT add any other dependencies beyond what was already listed.
- `.gitignore` was overwritten with a refreshed ML-project default that
  preserves all entries from the previous version plus adds
  `data/shards/`, `checkpoints/`, `data/train_log.csv`, `wandb/`, `runs/`,
  and `.huggingface/`.
- Two transformer stacks coexist (legacy v0 under `scripts/` vs new
  shoes-model under `model/`+`train/`). The README still describes only
  the legacy v0 stack.
- Raw data is staged at `C:\havoc_data` (~15 GB across smollm/, tinystories/,
  oasst2/) and ready for `data/prepare_pretrain.py`.
- Pretraining is intended for RunPod; the local GPU (RTX 2050, 4 GB) is
  too small for the default config.
