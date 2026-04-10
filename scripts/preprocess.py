"""
Data preprocessing pipeline for LLM training.

Sources supported:
  - JSONL files with {"prompt": "...", "completion": "..."} records
  - Word documents (.docx)
  - Plain text files (.txt)
  - Public-domain books auto-downloaded from Project Gutenberg (gap-filling)

Output:
  data/processed/train_tokens.pt   — LongTensor of token IDs (80%)
  data/processed/val_tokens.pt     — LongTensor of token IDs (20%)
  data/processed/tokenizer_info.json

Usage:
  python scripts/preprocess.py --data_dir data/raw
  python scripts/preprocess.py --data_dir data/raw --gutenberg 1342 84 11  (Pride+Prejudice, Frankenstein, Alice)
  python scripts/preprocess.py --help
"""

import argparse
import json
import os
import re
import sys

import requests
import torch
from transformers import GPT2TokenizerFast
from tqdm import tqdm


# ---------------------------------------------------------------------------
# GPT-2 special tokens
# ---------------------------------------------------------------------------
SEP_TOKEN  = "<|sep|>"          # separates prompt from completion
EOT_TOKEN  = "<|endoftext|>"    # GPT-2's built-in EOS / document boundary


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
                text = f"{EOT_TOKEN}{prompt}{SEP_TOKEN}{completion}{EOT_TOKEN}"
            else:
                # completion-only record (pre-formatted text)
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

def tokenize_corpus(texts: list[str], tokenizer: GPT2TokenizerFast, block_size: int) -> torch.Tensor:
    """
    Tokenise all texts, concatenate, then return as a flat LongTensor.
    The DataLoader will slice this into (x, y) pairs of length block_size.
    """
    all_ids: list[int] = []
    for text in tqdm(texts, desc="  Tokenising", unit="seg"):
        ids = tokenizer.encode(text, add_special_tokens=False)
        all_ids.extend(ids)
    return torch.tensor(all_ids, dtype=torch.long)


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
    parser.add_argument("--block_size", type=int, default=512,       help="Token sequence length per training example")
    parser.add_argument("--val_split",  type=float, default=0.2,     help="Fraction held out for validation")
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

    print("\n=== 2. Loading GPT-2 tokenizer ===")
    tokenizer = GPT2TokenizerFast.from_pretrained("gpt2")
    tokenizer.add_special_tokens({"additional_special_tokens": [SEP_TOKEN]})
    # GPT-2's built-in EOS is already <|endoftext|>; map pad → EOS
    tokenizer.pad_token = tokenizer.eos_token
    print(f"  Vocab size: {len(tokenizer):,}  (GPT-2 base: 50257, +1 for <|sep|>)")

    print("\n=== 3. Tokenising ===")
    tokens = tokenize_corpus(texts, tokenizer, args.block_size)
    total  = len(tokens)
    print(f"  Total tokens: {total:,}")

    min_needed = args.block_size * 10
    if total < min_needed:
        print(f"\n  [WARN] Only {total} tokens — need at least {min_needed} for meaningful training.")
        print(  "         Add more data or use --gutenberg to pull in public-domain books.")

    print("\n=== 4. Train / validation split (80/20) ===")
    split   = int(total * (1 - args.val_split))
    # Align to block_size boundary
    split   = (split // args.block_size) * args.block_size
    train_t = tokens[:split]
    val_t   = tokens[split:]
    print(f"  Train tokens: {len(train_t):,}  ({len(train_t)//args.block_size:,} full blocks)")
    print(f"  Val tokens:   {len(val_t):,}  ({len(val_t)//args.block_size:,} full blocks)")

    print("\n=== 5. Saving ===")
    train_path = os.path.join(args.out_dir, "train_tokens.pt")
    val_path   = os.path.join(args.out_dir, "val_tokens.pt")
    torch.save(train_t, train_path)
    torch.save(val_t,   val_path)
    print(f"  {train_path}")
    print(f"  {val_path}")

    info = {
        "vocab_size":  len(tokenizer),
        "block_size":  args.block_size,
        "train_tokens": len(train_t),
        "val_tokens":   len(val_t),
        "sep_token":    SEP_TOKEN,
        "eot_token":    EOT_TOKEN,
        "sep_token_id": tokenizer.convert_tokens_to_ids(SEP_TOKEN),
        "eot_token_id": tokenizer.eos_token_id,
    }
    info_path = os.path.join(args.out_dir, "tokenizer_info.json")
    with open(info_path, "w") as f:
        json.dump(info, f, indent=2)
    print(f"  {info_path}")

    print("\n=== 6. Sample verification ===")
    show_samples(tokens, tokenizer, args.block_size)

    print("\n=== Done ===")
    print(f"  vocab_size to use in model config: {len(tokenizer)}")
    print(f"  Update Config.vocab_size = {len(tokenizer)} in scripts/train.py\n")


if __name__ == "__main__":
    main()
