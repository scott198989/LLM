"""
HavocTokenizer — a thin wrapper around `tokenizers.Tokenizer` that bakes
in our special-token contract (config.SPECIAL_TOKENS) and a stable on-disk
layout so any HAVOC script can load the same tokenizer the same way.

On disk:
    <save_dir>/
        tokenizer.json   - native `tokenizers` JSON
        meta.json        - vocab size, special-token ids, version

API:
    HavocTokenizer.train(corpus, vocab_size, save_dir)
    HavocTokenizer.from_pretrained(load_dir)
    tok.encode(text)            -> list[int]
    tok.decode(ids)             -> str
    tok.encode_chat(messages)   -> list[int]   (applies the HAVOC chat template)
    tok.vocab_size, tok.pad_token_id, tok.eos_token_id, ...
"""

from __future__ import annotations

import json
import os
from typing import Iterable

from tokenizers import Tokenizer, decoders, models, pre_tokenizers, trainers
from tokenizers.processors import TemplateProcessing

from config import SPECIAL_TOKENS


_META_FILE      = "meta.json"
_TOKENIZER_FILE = "tokenizer.json"
_VERSION        = 1

# Fields searched in JSONL records when training the tokenizer
_JSONL_TEXT_KEYS = (
    "text", "content",
    "prompt", "completion",
    "instruction", "input", "output", "response",
)


def _iter_corpus(paths: list[str]):
    """
    Stream raw text strings from a mix of .txt and .jsonl files.
    JSONL records contribute the concatenation of any present
    text fields from `_JSONL_TEXT_KEYS`. Non-string values are skipped.
    """
    for path in paths:
        ext = os.path.splitext(path)[1].lower()
        if ext == ".jsonl":
            with open(path, encoding="utf-8") as f:
                for line in f:
                    line = line.strip()
                    if not line:
                        continue
                    try:
                        obj = json.loads(line)
                    except json.JSONDecodeError:
                        continue
                    parts = []
                    for k in _JSONL_TEXT_KEYS:
                        v = obj.get(k)
                        if isinstance(v, str) and v:
                            parts.append(v)
                    if parts:
                        yield " ".join(parts)
        else:
            # treat as plain text - read whole file (small per-file memory cost)
            try:
                with open(path, encoding="utf-8", errors="replace") as f:
                    yield f.read()
            except OSError:
                continue


class HavocTokenizer:
    """Byte-level BPE tokenizer with HAVOC's fixed special-token slate."""

    # ── constructors ──────────────────────────────────────────────────────

    def __init__(self, tokenizer: Tokenizer, special_ids: dict[str, int]):
        self._tok          = tokenizer
        self._special_ids  = dict(special_ids)
        self._special_set  = set(special_ids)

    @classmethod
    def train(cls,
              corpus_files: Iterable[str],
              vocab_size:   int    = 16384,
              save_dir:     str    | None = None,
              min_frequency: int   = 2,
              ) -> "HavocTokenizer":
        """
        Train a byte-level BPE tokenizer on the given files.
        Accepts .txt files (raw text, one document per file) and .jsonl
        files (one JSON object per line; common text fields are extracted).
        All entries of SPECIAL_TOKENS are inserted with deterministic IDs.
        """
        files = [p for p in corpus_files if os.path.isfile(p)]
        if not files:
            raise FileNotFoundError(
                "HavocTokenizer.train received no readable files."
            )

        tokenizer = Tokenizer(models.BPE(unk_token=None))
        tokenizer.pre_tokenizer = pre_tokenizers.ByteLevel(add_prefix_space=False)
        tokenizer.decoder       = decoders.ByteLevel()

        trainer = trainers.BpeTrainer(
            vocab_size      = vocab_size,
            min_frequency   = min_frequency,
            special_tokens  = list(SPECIAL_TOKENS),
            initial_alphabet = pre_tokenizers.ByteLevel.alphabet(),
        )

        # Use train_from_iterator so we can mix .txt and .jsonl. Memory cost
        # is bounded by tokenizers' internal streaming - we just yield strings.
        tokenizer.train_from_iterator(_iter_corpus(files), trainer=trainer)

        # Resolve special-token IDs (BpeTrainer assigns them in the order given)
        special_ids = {tok: tokenizer.token_to_id(tok) for tok in SPECIAL_TOKENS}
        if any(v is None for v in special_ids.values()):
            missing = [k for k, v in special_ids.items() if v is None]
            raise RuntimeError(f"Special tokens missing after training: {missing}")

        # Post-processor: prepend <|endoftext|> for single-sequence encoding
        eos = "<|endoftext|>"
        tokenizer.post_processor = TemplateProcessing(
            single   = f"{eos}:0 $A:0 {eos}:0",
            pair     = f"{eos}:0 $A:0 {eos}:0 $B:1 {eos}:1",
            special_tokens = [(eos, special_ids[eos])],
        )

        ht = cls(tokenizer, special_ids)
        if save_dir:
            ht.save(save_dir)
        return ht

    @classmethod
    def from_pretrained(cls, load_dir: str) -> "HavocTokenizer":
        tok_path  = os.path.join(load_dir, _TOKENIZER_FILE)
        meta_path = os.path.join(load_dir, _META_FILE)
        if not os.path.isfile(tok_path):
            raise FileNotFoundError(f"{tok_path} not found.")
        tokenizer = Tokenizer.from_file(tok_path)
        special_ids: dict[str, int]
        if os.path.isfile(meta_path):
            with open(meta_path, encoding="utf-8") as f:
                meta = json.load(f)
            special_ids = meta.get("special_ids", {})
        else:
            special_ids = {tok: tokenizer.token_to_id(tok) for tok in SPECIAL_TOKENS}
        return cls(tokenizer, special_ids)

    # ── persistence ───────────────────────────────────────────────────────

    def save(self, save_dir: str) -> None:
        os.makedirs(save_dir, exist_ok=True)
        self._tok.save(os.path.join(save_dir, _TOKENIZER_FILE))
        meta = {
            "version":     _VERSION,
            "vocab_size":  self.vocab_size,
            "special_ids": self._special_ids,
        }
        with open(os.path.join(save_dir, _META_FILE), "w", encoding="utf-8") as f:
            json.dump(meta, f, indent=2)

    # ── core API ──────────────────────────────────────────────────────────

    def encode(self, text: str, add_special: bool = False) -> list[int]:
        """
        Encode `text` to token IDs.

        add_special=False  - raw text only (no auto-prepended BOS).
        add_special=True   - apply the post-processor that wraps text with EOS.
        """
        return self._tok.encode(text, add_special_tokens=add_special).ids

    def decode(self, ids: list[int], skip_special_tokens: bool = False) -> str:
        return self._tok.decode(ids, skip_special_tokens=skip_special_tokens)

    def token_to_id(self, token: str) -> int | None:
        return self._tok.token_to_id(token)

    def id_to_token(self, idx: int) -> str | None:
        return self._tok.id_to_token(idx)

    @property
    def vocab_size(self) -> int:
        return self._tok.get_vocab_size()

    # ── special token IDs (always present after train/load) ──────────────

    @property
    def eos_token_id(self) -> int:
        return self._special_ids["<|endoftext|>"]

    @property
    def pad_token_id(self) -> int:
        return self._special_ids["<|pad|>"]

    @property
    def sep_token_id(self) -> int:
        return self._special_ids["<|sep|>"]

    @property
    def think_token_id(self) -> int:
        return self._special_ids["<|think|>"]

    @property
    def end_think_token_id(self) -> int:
        return self._special_ids["<|/think|>"]

    @property
    def user_token_id(self) -> int:
        return self._special_ids["<|user|>"]

    @property
    def end_user_token_id(self) -> int:
        return self._special_ids["<|/user|>"]

    @property
    def assistant_token_id(self) -> int:
        return self._special_ids["<|assistant|>"]

    @property
    def end_assistant_token_id(self) -> int:
        return self._special_ids["<|/assistant|>"]

    @property
    def system_token_id(self) -> int:
        return self._special_ids["<|system|>"]

    @property
    def end_system_token_id(self) -> int:
        return self._special_ids["<|/system|>"]

    @property
    def special_ids(self) -> dict[str, int]:
        return dict(self._special_ids)

    # ── chat template ─────────────────────────────────────────────────────

    def encode_chat(self,
                    messages: list[dict[str, str]],
                    add_generation_prompt: bool = True,
                    ) -> list[int]:
        """
        Encode a list of {role: ..., content: ...} messages using the HAVOC
        chat template:

            <|endoftext|>
            <|system|> {system content} <|/system|>
            <|user|>   {user content}   <|/user|>
            <|assistant|> {assistant content} <|/assistant|>
            ...
            <|assistant|>     <-- if add_generation_prompt=True

        Returns a flat list of token IDs ready to feed to model.generate().
        """
        ids: list[int] = [self.eos_token_id]
        role_open = {
            "system":    self._special_ids["<|system|>"],
            "user":      self._special_ids["<|user|>"],
            "assistant": self._special_ids["<|assistant|>"],
        }
        role_close = {
            "system":    self._special_ids["<|/system|>"],
            "user":      self._special_ids["<|/user|>"],
            "assistant": self._special_ids["<|/assistant|>"],
        }
        for m in messages:
            role    = m.get("role", "user")
            content = m.get("content", "") or ""
            if role not in role_open:
                role = "user"
            ids.append(role_open[role])
            ids.extend(self.encode(content, add_special=False))
            ids.append(role_close[role])
        if add_generation_prompt:
            ids.append(role_open["assistant"])
        return ids
