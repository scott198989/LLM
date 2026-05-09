"""
HAVOC v0 pretraining dataset assembler.

Walks a local staging directory + pulls Cosmopedia-100k from HuggingFace,
classifies each source into one of four buckets, trains a HavocTokenizer
on the combined corpus, then writes train.bin / val.bin sized to a
configurable target with the requested per-bucket sampling proportions.

Default buckets and mix:
    cosmopedia      50%   (HuggingFaceTB/cosmopedia-100k, split=train)
    academic        25%   (Academic Corpus/_cleaned/corpus.jsonl)
    conversational  15%   (Prompt Completion Pairs/D_Conversations*.jsonl)
    stem            10%   (Prompt Completion Pairs/D_*.jsonl, excluding Conversations)

Skipped automatically:
    python.jsonl, bash_shell.jsonl, git_commands.jsonl  (unfinished)
    OpenWebMath                                         (excluded by request)

Each D_*.jsonl row is converted to plain chat text:
    User: {prompt}
    Assistant: {completion}

The original D_*.jsonl files are NEVER modified.

Usage (default paths):
    python scripts/build_v0_dataset.py

Override paths:
    python scripts/build_v0_dataset.py \\
        --staging_dir "C:\\Users\\Scott\\OneDrive\\Desktop\\HAVOC train data" \\
        --out_dir data/processed_v0 \\
        --tokenizer_dir models/tokenizers/havoc_v0 \\
        --total_tokens 40_000_000

Faster smoke test (limits cosmopedia, smaller target):
    python scripts/build_v0_dataset.py --max_cosmopedia 5000 --total_tokens 5_000_000
"""

from __future__ import annotations

import argparse
import json
import math
import os
import random
import sys
from collections import defaultdict
from typing import Iterable

import numpy as np
from tqdm import tqdm

_HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, _HERE)

from tokenizer_havoc import HavocTokenizer            # noqa: E402


# ── classification rules ─────────────────────────────────────────────────


_UNFINISHED = {"python.jsonl", "bash_shell.jsonl", "git_commands.jsonl"}
_CONVERSATIONAL_TOKENS = ("conversation", "conversational", "chat", "dialogue")


def classify_local(staging_dir: str) -> tuple[dict[str, list[str]], list[tuple[str, str]], list[tuple[str, str]]]:
    """
    Walk the staging directory, returning:
        bucketed:  dict bucket-name -> list of file paths
        found:     list of (path, bucket) for files we will use
        skipped:   list of (path, reason)
    """
    bucketed: dict[str, list[str]] = defaultdict(list)
    found:   list[tuple[str, str]] = []
    skipped: list[tuple[str, str]] = []

    if not os.path.isdir(staging_dir):
        skipped.append((staging_dir, "staging directory does not exist"))
        return bucketed, found, skipped

    # Academic corpus -> _cleaned/corpus.jsonl
    academic_root = os.path.join(staging_dir, "Academic Corpus", "_cleaned")
    corpus_jsonl  = os.path.join(academic_root, "corpus.jsonl")
    if os.path.isfile(corpus_jsonl):
        bucketed["academic"].append(corpus_jsonl)
        found.append((corpus_jsonl, "academic"))
    else:
        skipped.append((corpus_jsonl, "Academic Corpus/_cleaned/corpus.jsonl not found"))

    # Prompt Completion Pairs
    pcp_dir = os.path.join(staging_dir, "Prompt Completion Pairs")
    if os.path.isdir(pcp_dir):
        for fname in sorted(os.listdir(pcp_dir)):
            full = os.path.join(pcp_dir, fname)
            low  = fname.lower()
            if not low.endswith(".jsonl"):
                continue
            if fname in _UNFINISHED:
                skipped.append((full, "explicitly excluded (unfinished)"))
                continue
            if not fname.startswith("D_"):
                skipped.append((full, "not a D_*.jsonl completion file"))
                continue
            if any(tok in low for tok in _CONVERSATIONAL_TOKENS):
                bucketed["conversational"].append(full)
                found.append((full, "conversational"))
            else:
                bucketed["stem"].append(full)
                found.append((full, "stem"))
    else:
        skipped.append((pcp_dir, "Prompt Completion Pairs directory not found"))

    return bucketed, found, skipped


# ── readers (yield plain text records) ───────────────────────────────────


def read_academic(path: str) -> Iterable[str]:
    with open(path, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                obj = json.loads(line)
            except json.JSONDecodeError:
                continue
            t = obj.get("text") or obj.get("content") or ""
            if isinstance(t, str) and t.strip():
                yield t


def read_d_pairs(path: str) -> Iterable[str]:
    """Yield plain chat text from a D_*.jsonl prompt/completion file."""
    with open(path, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                obj = json.loads(line)
            except json.JSONDecodeError:
                continue
            p = (obj.get("prompt") or obj.get("input")
                 or obj.get("instruction") or "").strip()
            c = (obj.get("completion") or obj.get("output")
                 or obj.get("response") or obj.get("text") or "").strip()
            if p and c:
                yield f"User: {p}\n\nAssistant: {c}"


def read_cosmopedia(max_samples: int | None = None) -> Iterable[str]:
    """Stream Cosmopedia-100k from HF, yielding the 'text' field per row."""
    from datasets import load_dataset
    ds = load_dataset("HuggingFaceTB/cosmopedia-100k", split="train")
    n = 0
    for row in ds:
        t = row.get("text") or row.get("completion") or ""
        if isinstance(t, str) and t.strip():
            yield t
            n += 1
            if max_samples and n >= max_samples:
                return


# ── tokenizer training (small subset of each bucket) ─────────────────────


def write_tokenizer_training_corpus(buckets: dict[str, list[str]],
                                     samples_per_bucket: int,
                                     out_path: str) -> None:
    os.makedirs(os.path.dirname(out_path) or ".", exist_ok=True)
    with open(out_path, "w", encoding="utf-8") as f:
        for name, records in buckets.items():
            for t in records[:samples_per_bucket]:
                f.write(t.strip())
                f.write("\n\n")


# ── main ──────────────────────────────────────────────────────────────────


def main() -> int:
    p = argparse.ArgumentParser(description="Assemble HAVOC v0 pretraining dataset.")
    p.add_argument("--staging_dir", default=r"C:\Users\Scott\OneDrive\Desktop\HAVOC train data")
    p.add_argument("--out_dir",     default="data/processed_v0")
    p.add_argument("--tokenizer_dir", default="models/tokenizers/havoc_v0")
    p.add_argument("--vocab_size",   type=int, default=16384)
    p.add_argument("--block_size",   type=int, default=2048)
    p.add_argument("--total_tokens", type=int, default=40_000_000,
                   help="Target total token count after upsampling/truncation.")
    p.add_argument("--mix_cosmopedia",     type=float, default=0.50)
    p.add_argument("--mix_academic",       type=float, default=0.25)
    p.add_argument("--mix_conversational", type=float, default=0.15)
    p.add_argument("--mix_stem",           type=float, default=0.10)
    p.add_argument("--val_split",          type=float, default=0.02)
    p.add_argument("--seed",               type=int,   default=1337)
    p.add_argument("--max_cosmopedia",     type=int,   default=None,
                   help="Cap cosmopedia rows (faster smoke runs).")
    p.add_argument("--no_train_tokenizer", action="store_true",
                   help="Reuse existing tokenizer at --tokenizer_dir.")
    p.add_argument("--tok_train_samples_per_bucket", type=int, default=4000,
                   help="Records per bucket fed to tokenizer training.")
    args = p.parse_args()

    rng = random.Random(args.seed)
    np.random.seed(args.seed)
    os.makedirs(args.out_dir, exist_ok=True)

    # ── 1. Classify local files ──────────────────────────────────────────
    print("\n=== 1. Scanning staging directory ===")
    print(f"  staging_dir = {args.staging_dir}")
    bucketed, found, skipped = classify_local(args.staging_dir)
    print(f"  academic files      : {len(bucketed['academic'])}")
    print(f"  conversational files: {len(bucketed['conversational'])}")
    print(f"  stem files          : {len(bucketed['stem'])}")
    print(f"  skipped             : {len(skipped)}")

    # ── 2. Load text records per bucket ──────────────────────────────────
    print("\n=== 2. Loading text records ===")
    records: dict[str, list[str]] = {b: [] for b in
                                     ("cosmopedia", "academic", "conversational", "stem")}

    for path in bucketed["academic"]:
        records["academic"].extend(read_academic(path))
    for path in bucketed["conversational"]:
        records["conversational"].extend(read_d_pairs(path))
    for path in bucketed["stem"]:
        records["stem"].extend(read_d_pairs(path))

    print(f"  academic       : {len(records['academic']):,} records")
    print(f"  conversational : {len(records['conversational']):,} records")
    print(f"  stem           : {len(records['stem']):,} records")

    print("\n  Streaming Cosmopedia-100k from HuggingFace ...")
    n_cosmo = 0
    for t in tqdm(read_cosmopedia(args.max_cosmopedia), desc="  cosmopedia", unit="rec"):
        records["cosmopedia"].append(t)
        n_cosmo += 1
    print(f"  cosmopedia     : {len(records['cosmopedia']):,} records")

    if not records["cosmopedia"]:
        print("ERROR: failed to load any cosmopedia records.", file=sys.stderr)
        return 1

    # ── 3. Tokenizer ─────────────────────────────────────────────────────
    print("\n=== 3. Tokenizer ===")
    tok_path = os.path.join(args.tokenizer_dir, "tokenizer.json")
    train_tokenizer = not (args.no_train_tokenizer and os.path.isfile(tok_path))
    if train_tokenizer:
        train_corpus = os.path.join(args.out_dir, "_tokenizer_corpus.txt")
        write_tokenizer_training_corpus(records, args.tok_train_samples_per_bucket, train_corpus)
        sz_mb = os.path.getsize(train_corpus) / 1024 ** 2
        print(f"  Training BPE on {sz_mb:.1f} MB sample (vocab_size={args.vocab_size})...")
        tok = HavocTokenizer.train(
            corpus_files = [train_corpus],
            vocab_size   = args.vocab_size,
            save_dir     = args.tokenizer_dir,
        )
    else:
        print(f"  Reusing tokenizer at {args.tokenizer_dir}")
        tok = HavocTokenizer.from_pretrained(args.tokenizer_dir)
    print(f"  vocab_size = {tok.vocab_size:,}")

    # ── 4. Tokenize each bucket ──────────────────────────────────────────
    print("\n=== 4. Tokenizing each bucket ===")
    bucket_ids: dict[str, np.ndarray] = {}
    eot = tok.eos_token_id
    for name, recs in records.items():
        ids: list[int] = []
        for t in tqdm(recs, desc=f"  {name}", unit="rec"):
            ids.extend(tok.encode(t, add_special=False))
            ids.append(eot)
        bucket_ids[name] = np.asarray(ids, dtype=np.int64)
        print(f"    {name}: {len(ids):,} tokens")

    # ── 5. Build mix ─────────────────────────────────────────────────────
    print("\n=== 5. Building sampling mix ===")
    mix = {
        "cosmopedia":     args.mix_cosmopedia,
        "academic":       args.mix_academic,
        "conversational": args.mix_conversational,
        "stem":           args.mix_stem,
    }
    s = sum(mix.values())
    if abs(s - 1.0) > 1e-6:
        print(f"  [WARN] proportions sum to {s:.4f}, not 1.0 - rescaling")
        for k in mix:
            mix[k] /= s

    targets = {k: int(args.total_tokens * v) for k, v in mix.items()}
    print(f"  Target total: {args.total_tokens:,} tokens")
    print(f"  Per-bucket targets:")
    for k, t in targets.items():
        have = len(bucket_ids[k])
        rep = t / max(have, 1)
        print(f"    {k:<14}: target {t:>14,}  have {have:>14,}  repeat {rep:>5.2f}x")

    # Sample with rotation: if target > have, repeat with shifting offsets
    pieces: list[np.ndarray] = []
    for k, t in targets.items():
        src = bucket_ids[k]
        if len(src) == 0 or t == 0:
            continue
        if len(src) >= t:
            start = rng.randint(0, len(src) - t)
            pieces.append(src[start:start + t].copy())
        else:
            n_full = t // len(src)
            n_rem  = t - n_full * len(src)
            chunks = [src] * n_full
            if n_rem > 0:
                # Take a random window so successive epochs don't all start at 0
                start = rng.randint(0, len(src) - n_rem) if len(src) > n_rem else 0
                chunks.append(src[start:start + n_rem])
            pieces.append(np.concatenate(chunks))

    final = np.concatenate(pieces)
    print(f"  Final corpus : {len(final):,} tokens")

    # ── 6. Train/val split ───────────────────────────────────────────────
    print("\n=== 6. Train / val split ===")
    val_n = int(len(final) * args.val_split)
    val_arr   = final[:val_n]
    train_arr = final[val_n:]
    print(f"  train tokens : {len(train_arr):,}")
    print(f"  val   tokens : {len(val_arr):,}")

    # ── 7. Write bin files ───────────────────────────────────────────────
    print("\n=== 7. Writing ===")
    if tok.vocab_size <= np.iinfo(np.uint16).max:
        dtype = np.dtype(np.uint16)
    else:
        dtype = np.dtype(np.uint32)

    train_path = os.path.join(args.out_dir, "train.bin")
    val_path   = os.path.join(args.out_dir, "val.bin")
    train_arr.astype(dtype).tofile(train_path)
    val_arr.astype(dtype).tofile(val_path)
    print(f"  {train_path}")
    print(f"  {val_path}")

    info = {
        "vocab_size":    tok.vocab_size,
        "block_size":    args.block_size,
        "train_tokens":  int(len(train_arr)),
        "val_tokens":    int(len(val_arr)),
        "train_file":    "train.bin",
        "val_file":      "val.bin",
        "token_dtype":   dtype.name,
        "tokenizer_dir": os.path.abspath(args.tokenizer_dir),
        "eot_token_id":      tok.eos_token_id,
        "sep_token_id":      tok.sep_token_id,
        "pad_token_id":      tok.pad_token_id,
        "think_token_id":    tok.think_token_id,
        "end_think_token_id": tok.end_think_token_id,
        "user_token_id":     tok.user_token_id,
        "end_user_token_id": tok.end_user_token_id,
        "asst_token_id":     tok.assistant_token_id,
        "end_asst_token_id": tok.end_assistant_token_id,
        "system_token_id":   tok.system_token_id,
        "end_system_token_id": tok.end_system_token_id,
        "v0_mix":              {k: float(v) for k, v in mix.items()},
        "v0_targets":          {k: int(v) for k, v in targets.items()},
        "v0_bucket_raw_tokens": {k: int(len(v)) for k, v in bucket_ids.items()},
    }
    with open(os.path.join(args.out_dir, "tokenizer_info.json"), "w") as f:
        json.dump(info, f, indent=2)

    # ── 8. Final report ──────────────────────────────────────────────────
    sep = "=" * 78
    print(f"\n{sep}\n  HAVOC v0 DATASET REPORT\n{sep}")
    print(f"  staging_dir   : {args.staging_dir}")
    print(f"  out_dir       : {os.path.abspath(args.out_dir)}")
    print(f"  tokenizer_dir : {os.path.abspath(args.tokenizer_dir)}")

    print(f"\n  Files found ({len(found)}):")
    for path, kind in found:
        rel = os.path.relpath(path, args.staging_dir)
        print(f"    [{kind:>14}] {rel}")
    print(f"\n  Files skipped ({len(skipped)}):")
    if not skipped:
        print("    (none)")
    for path, reason in skipped:
        rel = os.path.relpath(path, args.staging_dir)
        print(f"    - {rel} -> {reason}")

    print(f"\n  Cosmopedia: HuggingFaceTB/cosmopedia-100k  ({len(records['cosmopedia']):,} rows)")
    print(f"\n  Bucket classification (corpus vs D_ completions):")
    for path in bucketed["academic"]:
        print(f"    [academic corpus]  {os.path.relpath(path, args.staging_dir)}")
    for path in bucketed["conversational"]:
        print(f"    [D_ conversational]{os.path.relpath(path, args.staging_dir)}")
    for path in bucketed["stem"]:
        print(f"    [D_ STEM]          {os.path.relpath(path, args.staging_dir)}")

    print(f"\n  Token counts per group (raw, before sampling):")
    for k, ids in bucket_ids.items():
        recs = len(records[k])
        print(f"    {k:<14} : {len(ids):>14,} tokens  from {recs:>8,} records")

    print(f"\n  Final sampling mix (target_total = {args.total_tokens:,}):")
    for k, t in targets.items():
        have = len(bucket_ids[k])
        pct  = mix[k] * 100
        rep  = t / max(have, 1)
        print(f"    {k:<14} : {pct:>5.1f}%   target {t:>14,}   "
              f"raw {have:>14,}   repeat {rep:>5.2f}x")

    print(f"\n  Output:")
    print(f"    {os.path.abspath(train_path)}  ({len(train_arr):,} tokens)")
    print(f"    {os.path.abspath(val_path)}    ({len(val_arr):,} tokens)")
    print(f"    {os.path.abspath(os.path.join(args.out_dir, 'tokenizer_info.json'))}")

    print(f"\n  Source files were not modified.")
    print(f"{sep}\n")
    return 0


if __name__ == "__main__":
    sys.exit(main())
