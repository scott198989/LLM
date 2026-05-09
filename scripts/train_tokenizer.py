"""
CLI: train a HavocTokenizer from a corpus of plain-text files.

Examples:
    python scripts/train_tokenizer.py --corpus data/raw --vocab_size 16384 \\
        --out models/tokenizers/havoc_bpe

    python scripts/train_tokenizer.py --corpus data/raw data/extra/foo.txt \\
        --vocab_size 16384 --out models/tokenizers/havoc_bpe

The corpus argument accepts any mix of files and directories. Directories
are walked recursively and any *.txt files inside are picked up. JSONL
records and .docx files are NOT consumed here - run preprocess.py first
or convert manually. Tokenizer training is fast and only needs raw text.
"""

from __future__ import annotations

import argparse
import os
import sys

_HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, _HERE)

from tokenizer_havoc import HavocTokenizer        # noqa: E402


_SUPPORTED_EXTS = (".txt", ".jsonl")


def gather_text_files(paths: list[str]) -> list[str]:
    """Expand directories into their .txt/.jsonl contents; keep files as-is."""
    out: list[str] = []
    for p in paths:
        if os.path.isfile(p):
            out.append(p)
        elif os.path.isdir(p):
            for root, _dirs, files in os.walk(p):
                for f in files:
                    if f.lower().endswith(_SUPPORTED_EXTS):
                        out.append(os.path.join(root, f))
        else:
            print(f"  [WARN] skipping non-existent path: {p}")
    return sorted(set(out))


def main() -> int:
    parser = argparse.ArgumentParser(description="Train a HavocTokenizer.")
    parser.add_argument("--corpus", nargs="+", required=True,
                        help="Files and/or directories of .txt files.")
    parser.add_argument("--vocab_size", type=int, default=16384)
    parser.add_argument("--min_frequency", type=int, default=2)
    parser.add_argument("--out", required=True,
                        help="Output directory (will be created).")
    args = parser.parse_args()

    files = gather_text_files(args.corpus)
    if not files:
        print("ERROR: no .txt or .jsonl files found in the given corpus.\n"
              "Add data to data/raw/ or pass explicit file paths.",
              file=sys.stderr)
        return 1

    print(f"Training BPE on {len(files)} file(s):")
    for f in files[:6]:
        print(f"   - {f}")
    if len(files) > 6:
        print(f"   ... and {len(files) - 6} more")

    tok = HavocTokenizer.train(
        corpus_files  = files,
        vocab_size    = args.vocab_size,
        save_dir      = args.out,
        min_frequency = args.min_frequency,
    )

    print(f"\nTokenizer saved to: {args.out}")
    print(f"  vocab_size : {tok.vocab_size:,}")
    print(f"  eos id     : {tok.eos_token_id}   <|endoftext|>")
    print(f"  pad id     : {tok.pad_token_id}   <|pad|>")
    print(f"  think id   : {tok.think_token_id}   <|think|>")
    print(f"  /think id  : {tok.end_think_token_id}   <|/think|>")

    # quick round-trip sanity
    sample = "Hello, HAVOC. This is a tokenizer round-trip test."
    ids    = tok.encode(sample)
    back   = tok.decode(ids)
    print(f"\nRound-trip:")
    print(f"  in  : {sample!r}")
    print(f"  ids : {ids[:20]}{' ...' if len(ids) > 20 else ''}  ({len(ids)} tokens)")
    print(f"  out : {back!r}")
    print(("  PASS" if sample.strip() == back.strip() else "  WARN: round-trip mismatch") + "\n")
    return 0


if __name__ == "__main__":
    sys.exit(main())
