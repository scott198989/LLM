"""
Data preprocessing pipeline for LLM training.

Sources supported:
  - JSONL files with {"prompt": "...", "completion": "..."} records
  - Word documents (.docx)
  - Plain text files (.txt)
  - Public-domain books auto-downloaded from Project Gutenberg (gap-filling)

Output:
  data/processed/train.bin         — token IDs for training records (95%)
  data/processed/val.bin           — token IDs for validation records (5%)
  data/processed/tokenizer_info.json

Usage:
  python scripts/preprocess.py --data_dir data/raw
  python scripts/preprocess.py --data_dir data/raw --gutenberg 1342 84 11  (Pride+Prejudice, Frankenstein, Alice)
  python scripts/preprocess.py --help
"""

import argparse
import json
import math
import os
import random
import re
import sys

import numpy as np
import requests
import torch
from transformers import GPT2TokenizerFast
from tqdm import tqdm


# ---------------------------------------------------------------------------
# Special tokens
# ---------------------------------------------------------------------------
SEP_TOKEN        = "<|sep|>"          # separates prompt from completion (legacy)
EOT_TOKEN        = "<|endoftext|>"    # GPT-2's built-in EOS / document boundary
THINK_TOKEN      = "<|think|>"        # opens  Chain-of-Thought reasoning block
END_THINK_TOKEN  = "<|/think|>"       # closes Chain-of-Thought reasoning block

# ChatML / system-prompt tokens
SYSTEM_TOKEN     = "<|system|>"       # opens  system-level instructions
END_SYSTEM_TOKEN = "<|/system|>"      # closes system-level instructions
USER_TOKEN       = "<|user|>"         # opens  user turn
END_USER_TOKEN   = "<|/user|>"        # closes user turn
ASST_TOKEN       = "<|assistant|>"    # opens  assistant turn (model generates from here)

DEFAULT_SPLIT_SEED = 1337


# ---------------------------------------------------------------------------
# Loaders
# ---------------------------------------------------------------------------

def load_jsonl(path: str) -> list[str]:
    """Load prompt/completion JSONL → formatted strings."""
    records = []
    with open(path, encoding="utf-8") as f:
        for line_no, line in enumerate(f, 1):
            line = line.strip()
            if not line:
                continue
            try:
                obj = json.loads(line)
            except json.JSONDecodeError as e:
                print(f"  [WARN] {path}:{line_no} JSON error: {e} — skipped")
                continue

            # Accept various key names people use
            prompt     = obj.get("prompt")     or obj.get("input")       or obj.get("instruction") or ""
            completion = obj.get("completion") or obj.get("output")      or obj.get("response")    or obj.get("text") or ""

            if not prompt and not completion:
                print(f"  [WARN] {path}:{line_no} no recognisable keys — skipped")
                continue

            if prompt and completion:
                # ChatML format — optionally include a per-record system field
                system = obj.get("system", "").strip()
                if system:
                    text = (
                        f"{EOT_TOKEN}"
                        f"{SYSTEM_TOKEN}{system}{END_SYSTEM_TOKEN}"
                        f"{USER_TOKEN}{prompt}{END_USER_TOKEN}"
                        f"{ASST_TOKEN}{completion}{EOT_TOKEN}"
                    )
                else:
                    text = (
                        f"{EOT_TOKEN}"
                        f"{USER_TOKEN}{prompt}{END_USER_TOKEN}"
                        f"{ASST_TOKEN}{completion}{EOT_TOKEN}"
                    )
            else:
                # completion-only record (pre-formatted text / plain document)
                text = f"{EOT_TOKEN}{prompt or completion}{EOT_TOKEN}"

            records.append(text)
    return records


def load_docx(path: str) -> list[str]:
    """Load a Word document → one string per non-empty paragraph."""
    from docx import Document
    doc = Document(path)
    chunks = []
    current = []
    for para in doc.paragraphs:
        text = para.text.strip()
        if not text:
            if current:
                chunks.append(" ".join(current))
                current = []
        else:
            current.append(text)
    if current:
        chunks.append(" ".join(current))
    # Wrap each paragraph group as a document
    return [f"{EOT_TOKEN}{c}{EOT_TOKEN}" for c in chunks if c]


def load_txt(path: str) -> list[str]:
    """Load a plain text file, splitting on blank lines."""
    with open(path, encoding="utf-8", errors="replace") as f:
        content = f.read()
    paragraphs = re.split(r"\n{2,}", content)
    return [f"{EOT_TOKEN}{p.strip()}{EOT_TOKEN}" for p in paragraphs if p.strip()]


def download_gutenberg(book_id: int, cache_dir: str = "data/raw/gutenberg") -> str | None:
    """Download a Project Gutenberg book by ID. Returns local path."""
    os.makedirs(cache_dir, exist_ok=True)
    path = os.path.join(cache_dir, f"{book_id}.txt")
    if os.path.exists(path):
        print(f"  [cache] Gutenberg #{book_id} already downloaded.")
        return path

    # Try common Gutenberg mirror URLs (most reliable first)
    urls = [
        f"https://www.gutenberg.org/cache/epub/{book_id}/pg{book_id}.txt",
        f"https://www.gutenberg.org/files/{book_id}/{book_id}-0.txt",
        f"https://www.gutenberg.org/files/{book_id}/{book_id}.txt",
    ]
    for url in urls:
        try:
            r = requests.get(url, timeout=30)
            if r.status_code == 200:
                with open(path, "wb") as f:
                    f.write(r.content)
                print(f"  [OK] Gutenberg #{book_id} → {path}")
                return path
        except Exception:
            continue
    print(f"  [FAIL] Could not download Gutenberg #{book_id}")
    return None


def strip_gutenberg_header_footer(text: str) -> str:
    """Remove standard Gutenberg boilerplate."""
    start_markers = ["*** START OF", "***START OF", "** START OF"]
    end_markers   = ["*** END OF",   "***END OF",   "** END OF"]
    start = 0
    for m in start_markers:
        idx = text.find(m)
        if idx != -1:
            start = text.find("\n", idx) + 1
            break
    end = len(text)
    for m in end_markers:
        idx = text.find(m)
        if idx != -1:
            end = idx
            break
    return text[start:end]


# ---------------------------------------------------------------------------
# Scan a directory for supported files
# ---------------------------------------------------------------------------

def load_directory(data_dir: str) -> list[str]:
    all_texts: list[str] = []
    supported = {".jsonl": load_jsonl, ".docx": load_docx, ".txt": load_txt}

    files = sorted(
        f for f in os.listdir(data_dir)
        if os.path.splitext(f)[1].lower() in supported
    )

    if not files:
        print(f"  [WARN] No supported files found in {data_dir}")
        return []

    for fname in files:
        path = os.path.join(data_dir, fname)
        ext  = os.path.splitext(fname)[1].lower()
        loader = supported[ext]
        print(f"  Loading {fname} ...", end=" ", flush=True)
        try:
            texts = loader(path)
            print(f"{len(texts)} segments")
            all_texts.extend(texts)
        except Exception as e:
            print(f"ERROR: {e}")

    return all_texts


# ---------------------------------------------------------------------------
# Tokenise + chunk
# ---------------------------------------------------------------------------

def tokenize_corpus(texts: list[str], tokenizer: GPT2TokenizerFast, label: str) -> torch.Tensor:
    """
    Tokenise all texts in a split and return a flat LongTensor.
    """
    all_ids: list[int] = []
    for text in tqdm(texts, desc=f"  {label}", unit="seg"):
        ids = tokenizer.encode(text, add_special_tokens=False)
        all_ids.extend(ids)
    return torch.tensor(all_ids, dtype=torch.long)


def split_records(texts: list[str], val_split: float, seed: int) -> tuple[list[str], list[str]]:
    """Split formatted records into train / validation lists using a fixed seed."""
    if not 0.0 <= val_split < 1.0:
        raise ValueError("--val_split must be in the range [0, 1).")
    if len(texts) < 2 or val_split == 0.0:
        return texts, []

    val_count = min(len(texts) - 1, max(1, math.floor(len(texts) * val_split + 0.5)))
    indices = list(range(len(texts)))
    random.Random(seed).shuffle(indices)
    val_idx = set(indices[:val_count])

    train_texts = [text for idx, text in enumerate(texts) if idx not in val_idx]
    val_texts = [text for idx, text in enumerate(texts) if idx in val_idx]
    return train_texts, val_texts


def token_storage_dtype(vocab_size: int) -> np.dtype:
    """Pick a compact binary dtype that still fits the tokenizer vocabulary."""
    if vocab_size <= np.iinfo(np.uint16).max:
        return np.dtype(np.uint16)
    if vocab_size <= np.iinfo(np.uint32).max:
        return np.dtype(np.uint32)
    return np.dtype(np.int64)


def save_bin(path: str, tokens: torch.Tensor, dtype: np.dtype) -> None:
    """Persist a flat token tensor to a binary .bin file."""
    np.asarray(tokens.cpu().numpy(), dtype=dtype).tofile(path)


# ---------------------------------------------------------------------------
# Diagnostics
# ---------------------------------------------------------------------------

def show_samples(tokens: torch.Tensor, tokenizer: GPT2TokenizerFast, block_size: int, n: int = 3):
    print(f"\n{'='*60}")
    print(f"TOKENISED SAMPLE OUTPUTS  (block_size={block_size})")
    print('='*60)
    step = max(1, len(tokens) // (n + 1))
    for i in range(n):
        start = step * (i + 1)
        chunk = tokens[start : start + block_size]
        decoded = tokenizer.decode(chunk.tolist())
        ids_preview = chunk[:20].tolist()
        print(f"\n--- Sample {i+1} (tokens {start}–{start+block_size}) ---")
        print(f"IDs (first 20):  {ids_preview}")
        print(f"Decoded text:\n{decoded[:300]}")
    print('='*60)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(description="Preprocess text data for LLM training")
    parser.add_argument("--data_dir",   default="data/raw",         help="Directory with .jsonl/.docx/.txt files")
    parser.add_argument("--out_dir",    default="data/processed",   help="Output directory")
    parser.add_argument("--block_size", type=int, default=2048,      help="Token sequence length per training example")
    parser.add_argument("--val_split",  type=float, default=0.05,    help="Fraction of records held out for validation")
    parser.add_argument("--gutenberg",  nargs="*", type=int,
                        metavar="BOOK_ID",
                        help="Project Gutenberg book IDs to download for gap-filling, e.g. --gutenberg 1342 84")
    args = parser.parse_args()

    os.makedirs(args.out_dir, exist_ok=True)

    print("\n=== 1. Loading data ===")
    texts = load_directory(args.data_dir)

    # Optional public-domain gap-filler books
    if args.gutenberg:
        print("\n  Fetching Project Gutenberg books ...")
        for book_id in args.gutenberg:
            path = download_gutenberg(book_id, cache_dir=os.path.join(args.data_dir, "gutenberg"))
            if path:
                with open(path, encoding="utf-8", errors="replace") as f:
                    raw = f.read()
                raw = strip_gutenberg_header_footer(raw)
                paragraphs = re.split(r"\n{2,}", raw)
                texts.extend(
                    f"{EOT_TOKEN}{p.strip()}{EOT_TOKEN}"
                    for p in paragraphs if p.strip()
                )

    if not texts:
        print("\nNo data loaded. Put .jsonl / .docx / .txt files in", args.data_dir)
        sys.exit(1)

    print(f"\n  Total segments: {len(texts):,}")

    print("\n=== 2. Train / validation split (record-level, reproducible) ===")
    train_texts, val_texts = split_records(texts, args.val_split, DEFAULT_SPLIT_SEED)
    print(f"  Split seed:    {DEFAULT_SPLIT_SEED}")
    print(f"  Train records: {len(train_texts):,}")
    print(f"  Val records:   {len(val_texts):,}")

    print("\n=== 3. Loading GPT-2 tokenizer ===")
    tokenizer = GPT2TokenizerFast.from_pretrained("gpt2")
    tokenizer.add_special_tokens({
        "additional_special_tokens": [
            SEP_TOKEN, THINK_TOKEN, END_THINK_TOKEN,
            SYSTEM_TOKEN, END_SYSTEM_TOKEN, USER_TOKEN, END_USER_TOKEN, ASST_TOKEN,
        ]
    })
    # GPT-2's built-in EOS is already <|endoftext|>; map pad → EOS
    tokenizer.pad_token = tokenizer.eos_token
    print(f"  Vocab size: {len(tokenizer):,}  "
          f"(GPT-2 base: 50257, +8 special tokens)")

    print("\n=== 4. Tokenising ===")
    train_t = tokenize_corpus(train_texts, tokenizer, label="Tokenising train")
    val_t   = tokenize_corpus(val_texts,   tokenizer, label="Tokenising val")
    print(f"  Train tokens: {len(train_t):,}  ({len(train_t)//args.block_size:,} full blocks)")
    print(f"  Val tokens:   {len(val_t):,}  ({len(val_t)//args.block_size:,} full blocks)")

    min_needed = args.block_size * 10
    if len(train_t) < min_needed:
        print(f"\n  [WARN] Only {len(train_t):,} training tokens — need at least {min_needed:,} for meaningful training.")
        print(  "         Add more data or use --gutenberg to pull in public-domain books.")
    if len(train_t) < 5_000_000:
        print(f"\n  [WARN] Training split has only {len(train_t):,} tokens.")
        print(  "         That's under the recommended 5,000,000 training-token floor.")

    print("\n=== 5. Saving ===")
    token_dtype = token_storage_dtype(len(tokenizer))
    train_path = os.path.join(args.out_dir, "train.bin")
    val_path   = os.path.join(args.out_dir, "val.bin")
    save_bin(train_path, train_t, token_dtype)
    save_bin(val_path,   val_t,   token_dtype)
    print(f"  {train_path}")
    print(f"  {val_path}")

    info = {
        "vocab_size":           len(tokenizer),
        "block_size":           args.block_size,
        "train_tokens":         len(train_t),
        "val_tokens":           len(val_t),
        "train_records":        len(train_texts),
        "val_records":          len(val_texts),
        "val_split":            args.val_split,
        "split_seed":           DEFAULT_SPLIT_SEED,
        "train_file":           os.path.basename(train_path),
        "val_file":             os.path.basename(val_path),
        "token_dtype":          token_dtype.name,
        # Token strings
        "sep_token":            SEP_TOKEN,
        "eot_token":            EOT_TOKEN,
        "think_token":          THINK_TOKEN,
        "end_think_token":      END_THINK_TOKEN,
        "system_token":         SYSTEM_TOKEN,
        "end_system_token":     END_SYSTEM_TOKEN,
        "user_token":           USER_TOKEN,
        "end_user_token":       END_USER_TOKEN,
        "asst_token":           ASST_TOKEN,
        # Token IDs
        "sep_token_id":         tokenizer.convert_tokens_to_ids(SEP_TOKEN),
        "eot_token_id":         tokenizer.eos_token_id,
        "think_token_id":       tokenizer.convert_tokens_to_ids(THINK_TOKEN),
        "end_think_token_id":   tokenizer.convert_tokens_to_ids(END_THINK_TOKEN),
        "system_token_id":      tokenizer.convert_tokens_to_ids(SYSTEM_TOKEN),
        "end_system_token_id":  tokenizer.convert_tokens_to_ids(END_SYSTEM_TOKEN),
        "user_token_id":        tokenizer.convert_tokens_to_ids(USER_TOKEN),
        "end_user_token_id":    tokenizer.convert_tokens_to_ids(END_USER_TOKEN),
        "asst_token_id":        tokenizer.convert_tokens_to_ids(ASST_TOKEN),
    }
    info_path = os.path.join(args.out_dir, "tokenizer_info.json")
    with open(info_path, "w") as f:
        json.dump(info, f, indent=2)
    print(f"  {info_path}")

    print("\n=== 6. Sample verification ===")
    sample_tokens = train_t if len(train_t) else val_t
    show_samples(sample_tokens, tokenizer, args.block_size)

    print("\n=== Done ===")
    print(f"  vocab_size             : {len(tokenizer)}")
    print(f"  total training tokens  : {len(train_t):,}")
    print(f"  total validation tokens: {len(val_t):,}")
    print(f"  think_token_id         : {info['think_token_id']}  (<|think|>)")
    print(f"  end_think_token_id     : {info['end_think_token_id']}  (<|/think|>)")
    print(f"  system_token_id        : {info['system_token_id']}  (<|system|>)")
    print(f"  user_token_id          : {info['user_token_id']}  (<|user|>)")
    print(f"  asst_token_id          : {info['asst_token_id']}  (<|assistant|>)")
    print(f"  Chain-of-Thought ready : yes — use model.generate_cot() at inference")
    print(f"  System-prompt ready    : yes — set via InferenceEngine.set_system_prompt()\n")


if __name__ == "__main__":
    main()
