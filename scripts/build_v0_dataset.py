"""
HAVOC v0 pretraining dataset assembler.

Walks a local staging directory, optionally streams external corpora from
HuggingFace, classifies every source into a bucket, trains a HavocTokenizer
on the combined corpus, then writes train.bin / val.bin sized to a
configurable target with the requested per-bucket sampling proportions.

Default buckets and mix:
    academic        15%   (Academic Corpus/_cleaned/corpus.jsonl)
    conversational  10%   (Prompt Completion Pairs/D_Conversations*.jsonl)
    stem             5%   (Prompt Completion Pairs/D_*.jsonl, excluding Conversations)
    fineweb_edu     35%   (chengjunyan1/smollm-12.5-corpus, fineweb-edu-dedup)
    cosmopedia_v2   20%   (chengjunyan1/smollm-12.5-corpus, cosmopedia-v2)
    tinystories     10%   (roneneldan/TinyStories)
    oasst2           5%   (local 2023-11-05_oasst2_ready.trees.jsonl.gz, en + quality>=0.5)

Skipped automatically:
    python.jsonl, bash_shell.jsonl, git_commands.jsonl  (unfinished)
    OpenWebMath                                         (excluded by request)

Each D_*.jsonl row is converted to plain chat text:
    User: {prompt}
    Assistant: {completion}

The original D_*.jsonl files are NEVER modified.

Usage on RunPod (default paths assume /workspace):
    python scripts/build_v0_dataset.py \\
        --staging_dir /workspace/havoc_train_data \\
        --out_dir /workspace/data/processed_v0 \\
        --tokenizer_dir /workspace/models/tokenizers/havoc_v0 \\
        --oasst2_path /workspace/data/raw/2023-11-05_oasst2_ready.trees.jsonl.gz \\
        --total_tokens 200_000_000

Smoke test (cap HF rows, smaller target):
    python scripts/build_v0_dataset.py \\
        --max_fineweb_edu 5000 --max_cosmopedia_v2 5000 --max_tinystories 5000 \\
        --total_tokens 5_000_000
"""

from __future__ import annotations

import argparse
import gzip
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


def read_hf_text(dataset_name: str,
                 config: str | None = None,
                 split: str = "train",
                 max_samples: int | None = None,
                 text_keys: tuple[str, ...] = ("text", "content", "completion")) -> Iterable[str]:
    """Stream a HuggingFace text dataset, yielding non-empty strings."""
    from datasets import load_dataset
    ds = load_dataset(dataset_name, config, split=split) if config else \
         load_dataset(dataset_name, split=split)
    n = 0
    for row in ds:
        t = ""
        for k in text_keys:
            v = row.get(k)
            if isinstance(v, str) and v.strip():
                t = v
                break
        if t:
            yield t
            n += 1
            if max_samples and n >= max_samples:
                return


def read_oasst2_trees(path: str,
                      min_quality: float = 0.5,
                      lang: str = "en") -> Iterable[str]:
    """
    Stream OASST2 conversation trees from a .jsonl.gz file.

    Each tree row contains a `prompt` node with nested `replies`. We walk
    every root->leaf path of (prompter, assistant, prompter, ...) turns,
    filtering on language and a labels.quality value when present, and emit
    a single chat-style record per path.
    """
    def _quality(node: dict) -> float:
        labels = (node.get("labels") or {})
        q = labels.get("quality")
        if isinstance(q, dict):
            return float(q.get("value", 0.0))
        if isinstance(q, (int, float)):
            return float(q)
        return 1.0  # no label = accept

    def _walk(node: dict, history: list[tuple[str, str]]) -> Iterable[list[tuple[str, str]]]:
        if node.get("lang", lang) != lang:
            return
        if _quality(node) < min_quality:
            return
        role = node.get("role", "")
        text = (node.get("text") or "").strip()
        if not text:
            return
        new_hist = history + [(role, text)]
        replies = node.get("replies") or []
        if not replies:
            yield new_hist
            return
        for r in replies:
            yield from _walk(r, new_hist)

    opener = gzip.open if path.endswith(".gz") else open
    with opener(path, "rt", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                tree = json.loads(line)
            except json.JSONDecodeError:
                continue
            root = tree.get("prompt") or tree
            for turns in _walk(root, []):
                parts = []
                for role, text in turns:
                    speaker = "User" if role == "prompter" else "Assistant"
                    parts.append(f"{speaker}: {text}")
                yield "\n\n".join(parts)


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
    # Local-source mix
    p.add_argument("--mix_academic",       type=float, default=0.15)
    p.add_argument("--mix_conversational", type=float, default=0.10)
    p.add_argument("--mix_stem",           type=float, default=0.05)
    # HuggingFace-source mix
    p.add_argument("--mix_fineweb_edu",    type=float, default=0.35,
                   help="Share for chengjunyan1/smollm-12.5-corpus (fineweb-edu-dedup).")
    p.add_argument("--mix_cosmopedia_v2",  type=float, default=0.20,
                   help="Share for chengjunyan1/smollm-12.5-corpus (cosmopedia-v2).")
    p.add_argument("--mix_tinystories",    type=float, default=0.10,
                   help="Share for roneneldan/TinyStories.")
    p.add_argument("--mix_oasst2",         type=float, default=0.05,
                   help="Share for OASST2 conversations (local .jsonl.gz).")
    # HuggingFace source caps (None = stream full split)
    p.add_argument("--max_fineweb_edu",    type=int, default=None)
    p.add_argument("--max_cosmopedia_v2",  type=int, default=None)
    p.add_argument("--max_tinystories",    type=int, default=None)
    # OASST2 inputs
    p.add_argument("--oasst2_path",        default=None,
                   help="Path to 2023-11-05_oasst2_ready.trees.jsonl.gz "
                        "(skips OASST2 bucket if omitted or missing).")
    p.add_argument("--oasst2_min_quality", type=float, default=0.5)
    p.add_argument("--oasst2_lang",        default="en")
    p.add_argument("--val_split",          type=float, default=0.02)
    p.add_argument("--seed",               type=int,   default=1337)
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
    bucket_names = ("academic", "conversational", "stem",
                    "fineweb_edu", "cosmopedia_v2", "tinystories", "oasst2")
    records: dict[str, list[str]] = {b: [] for b in bucket_names}

    for path in bucketed["academic"]:
        records["academic"].extend(read_academic(path))
    for path in bucketed["conversational"]:
        records["conversational"].extend(read_d_pairs(path))
    for path in bucketed["stem"]:
        records["stem"].extend(read_d_pairs(path))

    print(f"  academic       : {len(records['academic']):,} records")
    print(f"  conversational : {len(records['conversational']):,} records")
    print(f"  stem           : {len(records['stem']):,} records")

    # HuggingFace: chengjunyan1/smollm-12.5-corpus (fineweb-edu-dedup)
    if args.mix_fineweb_edu > 0:
        print("\n  Streaming smollm-12.5-corpus / fineweb-edu-dedup from HuggingFace ...")
        for t in tqdm(
            read_hf_text("chengjunyan1/smollm-12.5-corpus",
                         config="fineweb-edu-dedup",
                         split="train",
                         max_samples=args.max_fineweb_edu),
            desc="  fineweb-edu", unit="rec",
        ):
            records["fineweb_edu"].append(t)
        print(f"  fineweb_edu    : {len(records['fineweb_edu']):,} records")

    # HuggingFace: chengjunyan1/smollm-12.5-corpus (cosmopedia-v2)
    if args.mix_cosmopedia_v2 > 0:
        print("\n  Streaming smollm-12.5-corpus / cosmopedia-v2 from HuggingFace ...")
        for t in tqdm(
            read_hf_text("chengjunyan1/smollm-12.5-corpus",
                         config="cosmopedia-v2",
                         split="train",
                         max_samples=args.max_cosmopedia_v2),
            desc="  cosmopedia-v2", unit="rec",
        ):
            records["cosmopedia_v2"].append(t)
        print(f"  cosmopedia_v2  : {len(records['cosmopedia_v2']):,} records")

    # HuggingFace: roneneldan/TinyStories
    if args.mix_tinystories > 0:
        print("\n  Streaming TinyStories from HuggingFace ...")
        for t in tqdm(
            read_hf_text("roneneldan/TinyStories",
                         split="train",
                         max_samples=args.max_tinystories),
            desc="  tinystories", unit="rec",
        ):
            records["tinystories"].append(t)
        print(f"  tinystories    : {len(records['tinystories']):,} records")

    # Local OASST2 conversation trees (.jsonl.gz)
    if args.mix_oasst2 > 0 and args.oasst2_path and os.path.isfile(args.oasst2_path):
        print(f"\n  Loading OASST2 trees from {args.oasst2_path} ...")
        for t in tqdm(
            read_oasst2_trees(args.oasst2_path,
                              min_quality=args.oasst2_min_quality,
                              lang=args.oasst2_lang),
            desc="  oasst2", unit="conv",
        ):
            records["oasst2"].append(t)
        print(f"  oasst2         : {len(records['oasst2']):,} records")
    elif args.mix_oasst2 > 0:
        print(f"\n  [WARN] OASST2 requested but --oasst2_path not provided / not found - skipping")

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
        "academic":       args.mix_academic,
        "conversational": args.mix_conversational,
        "stem":           args.mix_stem,
        "fineweb_edu":    args.mix_fineweb_edu,
        "cosmopedia_v2":  args.mix_cosmopedia_v2,
        "tinystories":    args.mix_tinystories,
        "oasst2":         args.mix_oasst2,
    }
    # Drop any bucket that has no data — rescale the rest so we don't allocate
    # tokens to an empty source.
    for k in list(mix.keys()):
        if not records.get(k):
            if mix[k] > 0:
                print(f"  [INFO] no records for '{k}' - dropping from mix")
            mix[k] = 0.0
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

    print(f"\n  External sources:")
    print(f"    fineweb-edu-dedup  : {len(records['fineweb_edu']):,} rows  "
          f"(chengjunyan1/smollm-12.5-corpus)")
    print(f"    cosmopedia-v2      : {len(records['cosmopedia_v2']):,} rows  "
          f"(chengjunyan1/smollm-12.5-corpus)")
    print(f"    tinystories        : {len(records['tinystories']):,} rows  "
          f"(roneneldan/TinyStories)")
    print(f"    oasst2 (local .gz) : {len(records['oasst2']):,} conversations"
          + (f"  ({args.oasst2_path})" if args.oasst2_path else "  (path not set)"))
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
