"""
Download and convert open-source datasets to prompt/completion JSONL format.

All output files land in data/raw/ and are immediately usable by preprocess.py.

Datasets included:
  --alpaca        tatsu-lab/alpaca              52k  instruction pairs  ~15M tokens
  --dolly         databricks/databricks-dolly-15k 15k instruction pairs   ~4M tokens
  --oasst         OpenAssistant/oasst1          88k  conversation turns   ~8M tokens
  --hermes        teknium/OpenHermes-2.5        1M+  instruction pairs  ~500M tokens (subset via --hermes_limit)
  --dailydialog   daily_dialog                  13k  multi-turn chats     ~1M tokens
  --wiki          wikipedia (en, 2022)          6.4GB articles         ~3B  tokens (subset via --wiki_limit)
  --gutenberg     (list of IDs)                 classic books           varies

Usage:
  # Grab everything modest (recommended starting point, ~530M tokens):
  python scripts/download_datasets.py --alpaca --dolly --oasst --dailydialog --hermes --wiki

  # Just the conversational sets (fast, ~28M tokens):
  python scripts/download_datasets.py --alpaca --dolly --oasst --dailydialog

  # Add specific Gutenberg books for domain flavour:
  python scripts/download_datasets.py --gutenberg 1342 84 11 74 1661 2701 64317

  # Full run with large corpora (do this on the 5090 box):
  python scripts/download_datasets.py --alpaca --dolly --oasst --dailydialog --hermes --hermes_limit 300000 --wiki --wiki_limit 50000
"""

import argparse
import json
import os
import re
import sys

import requests
from tqdm import tqdm

OUT_DIR = "data/raw"
EOT = "<|endoftext|>"
SEP = "<|sep|>"

os.makedirs(OUT_DIR, exist_ok=True)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def write_jsonl(path: str, records: list[dict], label: str):
    with open(path, "w", encoding="utf-8") as f:
        for r in records:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")
    print(f"  [{label}] Saved {len(records):,} records -> {path}")


def token_estimate(records: list[dict]) -> int:
    """Rough token count (chars / 4 is a good heuristic for English BPE)."""
    total_chars = sum(len(r.get("prompt","")) + len(r.get("completion","")) for r in records)
    return total_chars // 4


# ---------------------------------------------------------------------------
# Alpaca — 52k instruction / input / output triples
# ---------------------------------------------------------------------------

def download_alpaca():
    print("\n=== Alpaca (tatsu-lab/alpaca) ===")
    from datasets import load_dataset
    ds = load_dataset("tatsu-lab/alpaca", split="train")
    records = []
    for row in tqdm(ds, desc="  Converting"):
        prompt = row["instruction"]
        if row.get("input", "").strip():
            prompt = f"{prompt}\n{row['input']}"
        completion = row.get("output", "").strip()
        if prompt and completion:
            records.append({"prompt": prompt, "completion": completion})
    write_jsonl(os.path.join(OUT_DIR, "alpaca.jsonl"), records, "alpaca")
    print(f"  Estimated tokens: ~{token_estimate(records):,}")


# ---------------------------------------------------------------------------
# Dolly 15k — instruction / context / response
# ---------------------------------------------------------------------------

def download_dolly():
    print("\n=== Dolly 15k (databricks/databricks-dolly-15k) ===")
    from datasets import load_dataset
    ds = load_dataset("databricks/databricks-dolly-15k", split="train")
    records = []
    for row in tqdm(ds, desc="  Converting"):
        prompt = row["instruction"]
        if row.get("context", "").strip():
            prompt = f"{prompt}\nContext: {row['context']}"
        completion = row.get("response", "").strip()
        if prompt and completion:
            records.append({"prompt": prompt, "completion": completion})
    write_jsonl(os.path.join(OUT_DIR, "dolly.jsonl"), records, "dolly")
    print(f"  Estimated tokens: ~{token_estimate(records):,}")


# ---------------------------------------------------------------------------
# OpenAssistant OASST1 — conversation trees, extract human/assistant pairs
# ---------------------------------------------------------------------------

def download_oasst():
    print("\n=== OpenAssistant OASST1 ===")
    from datasets import load_dataset
    ds = load_dataset("OpenAssistant/oasst1", split="train")

    # Build a dict of message_id → row for parent lookups
    by_id = {row["message_id"]: row for row in ds}

    records = []
    for row in tqdm(ds, desc="  Pairing"):
        if row["role"] != "assistant":
            continue
        parent_id = row.get("parent_id")
        if not parent_id or parent_id not in by_id:
            continue
        parent = by_id[parent_id]
        if parent["role"] != "prompter":
            continue
        prompt     = parent["text"].strip()
        completion = row["text"].strip()
        if prompt and completion:
            records.append({"prompt": prompt, "completion": completion})

    write_jsonl(os.path.join(OUT_DIR, "oasst.jsonl"), records, "oasst")
    print(f"  Estimated tokens: ~{token_estimate(records):,}")


# ---------------------------------------------------------------------------
# DailyDialog — casual multi-turn; unroll into consecutive pairs
# ---------------------------------------------------------------------------

def download_dailydialog():
    """
    UltraChat 200k (HuggingFaceH4/ultrachat_200k) — high-quality multi-turn chat conversations.
    200k examples, maintained by HuggingFace, clean parquet format, no legacy scripts.
    """
    print("\n=== UltraChat 200k (HuggingFaceH4/ultrachat_200k) ===")
    from datasets import load_dataset
    ds = load_dataset("HuggingFaceH4/ultrachat_200k", split="train_sft")
    records = []
    for row in tqdm(ds, desc="  Converting"):
        messages = row.get("messages", [])
        for i in range(len(messages) - 1):
            if messages[i].get("role") == "user" and messages[i+1].get("role") == "assistant":
                prompt     = messages[i].get("content", "").strip()
                completion = messages[i+1].get("content", "").strip()
                if prompt and completion:
                    records.append({"prompt": prompt, "completion": completion})
    write_jsonl(os.path.join(OUT_DIR, "ultrachat.jsonl"), records, "ultrachat")
    print(f"  Estimated tokens: ~{token_estimate(records):,}")


# ---------------------------------------------------------------------------
# OpenHermes 2.5 — large instruction/chat dataset, sharegpt format
# ---------------------------------------------------------------------------

def download_hermes(limit: int = 0):
    print(f"\n=== OpenHermes 2.5 (teknium/OpenHermes-2.5){' limit=' + str(limit) if limit else ''} ===")
    from datasets import load_dataset
    ds = load_dataset("teknium/OpenHermes-2.5", split="train", streaming=True)
    records = []
    for row in tqdm(ds, desc="  Converting", total=limit or None):
        convs = row.get("conversations", [])
        # Extract consecutive human/gpt pairs
        for i in range(len(convs) - 1):
            if convs[i].get("from") == "human" and convs[i+1].get("from") == "gpt":
                prompt     = convs[i].get("value", "").strip()
                completion = convs[i+1].get("value", "").strip()
                if prompt and completion:
                    records.append({"prompt": prompt, "completion": completion})
        if limit and len(records) >= limit:
            break
    records = records[:limit] if limit else records
    write_jsonl(os.path.join(OUT_DIR, "hermes.jsonl"), records, "hermes")
    print(f"  Estimated tokens: ~{token_estimate(records):,}")


# ---------------------------------------------------------------------------
# Wikipedia — clean encyclopedic text, written as prompt=title / completion=article
# ---------------------------------------------------------------------------

def download_wiki(limit: int = 0):
    print(f"\n=== Wikipedia (en, 2022){' limit=' + str(limit) if limit else ' FULL ~6GB'} ===")
    if not limit:
        print("  WARNING: Full Wikipedia is ~6GB. Use --wiki_limit 50000 for a manageable subset.")
    from datasets import load_dataset
    ds = load_dataset("wikipedia", "20220301.en", split="train",
                      streaming=True, trust_remote_code=True)
    records = []
    for row in tqdm(ds, desc="  Converting", total=limit or None):
        title = row.get("title", "").strip()
        text  = row.get("text",  "").strip()
        if title and text:
            # Split long articles into paragraphs so context fits in block_size
            paragraphs = re.split(r"\n{2,}", text)
            first = paragraphs[0][:1200]   # lead paragraph as completion
            records.append({"prompt": title, "completion": first})
            # Also store remaining paragraphs as standalone text completions
            for para in paragraphs[1:]:
                para = para.strip()
                if len(para) > 100:
                    records.append({"prompt": "", "completion": para})
        if limit and len(records) >= limit:
            break
    records = records[:limit] if limit else records
    write_jsonl(os.path.join(OUT_DIR, "wikipedia.jsonl"), records, "wikipedia")
    print(f"  Estimated tokens: ~{token_estimate(records):,}")


# ---------------------------------------------------------------------------
# Project Gutenberg — public-domain books by ID
# ---------------------------------------------------------------------------

GUTENBERG_URLS = [
    "https://www.gutenberg.org/cache/epub/{id}/pg{id}.txt",
    "https://www.gutenberg.org/files/{id}/{id}-0.txt",
    "https://www.gutenberg.org/files/{id}/{id}.txt",
]

# Curated list of varied public-domain books (ID: title)
RECOMMENDED_BOOKS = {
    1342: "Pride and Prejudice — Austen",
    84:   "Frankenstein — Shelley",
    11:   "Alice in Wonderland — Carroll",
    74:   "The Adventures of Tom Sawyer — Twain",
    1661: "The Adventures of Sherlock Holmes — Doyle",
    2701: "Moby Dick — Melville",
    2600: "War and Peace — Tolstoy",
    98:   "A Tale of Two Cities — Dickens",
    1400: "Great Expectations — Dickens",
    345:  "Dracula — Stoker",
    1080: "A Modest Proposal — Swift",
    4300: "Ulysses — Joyce",
    5200: "Metamorphosis — Kafka",
    2554: "Crime and Punishment — Dostoevsky",
    1260: "Jane Eyre — Brontë",
    158:  "Emma — Austen",
    1232: "The Prince — Machiavelli",
    3207: "Leviathan — Hobbes",
    6593: "History of the Peloponnesian War — Thucydides",
    100:  "The Complete Works of Shakespeare",
}


def strip_gutenberg(text: str) -> str:
    for m in ["*** START OF", "***START OF"]:
        idx = text.find(m)
        if idx != -1:
            text = text[text.find("\n", idx) + 1:]
            break
    for m in ["*** END OF", "***END OF"]:
        idx = text.find(m)
        if idx != -1:
            text = text[:idx]
            break
    return text.strip()


def download_gutenberg(book_ids: list[int]):
    print(f"\n=== Project Gutenberg ({len(book_ids)} books) ===")
    cache_dir = os.path.join(OUT_DIR, "gutenberg")
    os.makedirs(cache_dir, exist_ok=True)
    records = []
    for book_id in book_ids:
        title = RECOMMENDED_BOOKS.get(book_id, f"Book #{book_id}")
        cached = os.path.join(cache_dir, f"{book_id}.txt")
        if not os.path.exists(cached):
            downloaded = False
            for url_tmpl in GUTENBERG_URLS:
                url = url_tmpl.format(id=book_id)
                try:
                    r = requests.get(url, timeout=30)
                    if r.status_code == 200:
                        with open(cached, "wb") as f:
                            f.write(r.content)
                        downloaded = True
                        break
                except Exception:
                    continue
            if not downloaded:
                print(f"  [FAIL] #{book_id} {title}")
                continue
        with open(cached, encoding="utf-8", errors="replace") as f:
            raw = f.read()
        text = strip_gutenberg(raw)
        paragraphs = [p.strip() for p in re.split(r"\n{2,}", text) if len(p.strip()) > 80]
        for i in range(0, len(paragraphs) - 1, 2):
            records.append({"prompt": paragraphs[i][:500], "completion": paragraphs[i+1][:800]})
        print(f"  [OK] #{book_id} {title} — {len(paragraphs)} paragraphs")
    out = os.path.join(OUT_DIR, "gutenberg.jsonl")
    write_jsonl(out, records, "gutenberg")
    print(f"  Estimated tokens: ~{token_estimate(records):,}")


# ---------------------------------------------------------------------------
# Summary
# ---------------------------------------------------------------------------

def print_summary():
    print("\n=== Dataset Summary ===")
    total_records = 0
    total_est_tokens = 0
    for fname in sorted(os.listdir(OUT_DIR)):
        if not fname.endswith(".jsonl"):
            continue
        path = os.path.join(OUT_DIR, fname)
        with open(path, encoding="utf-8") as f:
            lines = [l for l in f if l.strip()]
        records = []
        for l in lines:
            try: records.append(json.loads(l))
            except: pass
        est = token_estimate(records)
        total_records += len(records)
        total_est_tokens += est
        print(f"  {fname:<30} {len(records):>8,} records   ~{est:>12,} tokens")
    print(f"\n  {'TOTAL':<30} {total_records:>8,} records   ~{total_est_tokens:>12,} tokens")
    target = 300_000_000
    pct = total_est_tokens / target * 100
    print(f"\n  Target (min viable 30M model): {target:,} tokens")
    status = "READY" if pct >= 100 else f"need ~{(target-total_est_tokens)//1_000_000}M more tokens"
    print(f"  Coverage: {pct:.1f}%  ({status})")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--alpaca",       action="store_true")
    parser.add_argument("--dolly",        action="store_true")
    parser.add_argument("--oasst",        action="store_true")
    parser.add_argument("--dailydialog",  action="store_true")
    parser.add_argument("--hermes",       action="store_true")
    parser.add_argument("--hermes_limit", type=int, default=200000,
                        help="Max records from OpenHermes (default 200k ≈ 100M tokens)")
    parser.add_argument("--wiki",         action="store_true")
    parser.add_argument("--wiki_limit",   type=int, default=30000,
                        help="Max Wikipedia articles (default 30k ≈ 90M tokens)")
    parser.add_argument("--gutenberg",    nargs="*", type=int, metavar="BOOK_ID",
                        help="Gutenberg book IDs. Omit IDs to use the built-in curated list.")
    parser.add_argument("--all",          action="store_true",
                        help="Download all conversational sets (alpaca+dolly+oasst+dailydialog)")
    args = parser.parse_args()

    if args.all:
        args.alpaca = args.dolly = args.oasst = args.dailydialog = True

    ran_anything = False
    if args.alpaca:      download_alpaca();               ran_anything = True
    if args.dolly:       download_dolly();                ran_anything = True
    if args.oasst:       download_oasst();                ran_anything = True
    if args.dailydialog: download_dailydialog();           ran_anything = True
    if args.hermes:      download_hermes(args.hermes_limit); ran_anything = True
    if args.wiki:        download_wiki(args.wiki_limit);   ran_anything = True
    if args.gutenberg is not None:
        ids = args.gutenberg if args.gutenberg else list(RECOMMENDED_BOOKS.keys())
        download_gutenberg(ids)
        ran_anything = True

    if not ran_anything:
        parser.print_help()
        print("\nRecommended quick start (laptop-friendly, ~30M tokens):")
        print("  python scripts/download_datasets.py --all")
        print("\nFor 5090 at home (full run, ~600M+ tokens):")
        print("  python scripts/download_datasets.py --all --hermes --wiki --gutenberg")
        sys.exit(0)

    print_summary()
