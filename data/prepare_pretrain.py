"""
HAVOC shoes-model pretrain data prep.

Reads raw corpora (Parquet + OASST2 .jsonl.gz), tokenizes with the GPT-2
BPE via tiktoken, mixes at document level to hit per-source token budgets,
shuffles, and writes uint16 binary memmap shards to data/shards/ in the
nanoGPT format.

Target mix (~2.7B tokens, 1 epoch):
    fineweb-edu-dedup : ~1.8B tokens   (Parquet under <raw>/smollm/fineweb-edu-dedup/)
    cosmopedia-v2     : ~400M tokens   (Parquet under <raw>/smollm/cosmopedia-v2/)
    TinyStoriesV2-GPT4: ~300M tokens   (Parquet under <raw>/tinystories/)
    OASST2 (en)       : ~200M tokens   (jsonl.gz under <raw>/oasst2/, or fewer if the
                                        source has less after en+quality filtering)

Each shard contains contiguous uint16 token IDs. A meta.json sidecar records
total counts and per-source breakdown.

Usage (defaults assume local Windows path):
    python data/prepare_pretrain.py \\
        --raw_dir   C:/havoc_data \\
        --out_dir   data/shards \\
        --tokens_per_shard 100_000_000

On RunPod:
    python data/prepare_pretrain.py \\
        --raw_dir   /workspace/havoc_data/raw \\
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
    p.add_argument("--budget_fineweb_edu",   type=int, default=1_800_000_000)
    p.add_argument("--budget_cosmopedia_v2", type=int, default=400_000_000)
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
    # Each document is (source_name, list[int]) — IDs already include trailing EOT.
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
                # Spill remainder into train
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
