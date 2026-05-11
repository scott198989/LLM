"""
RetrievalAgent: tiny RAG over a local knowledge directory.

Indexes every .txt and .md file under `knowledge_dir` into BM25 chunks
(non-overlapping windows of `chunk_words` words each). Queries return
the top-k highest-scoring chunks with their source path so the
orchestrator can splice them into the prompt.

Falls back to a TF-IDF style scoring if `rank_bm25` isn't installed.
"""

from __future__ import annotations

import math
import os
import re
from collections import Counter
from dataclasses import dataclass

try:
    from rank_bm25 import BM25Okapi
    _HAS_BM25 = True
except ImportError:
    _HAS_BM25 = False


_TOKEN_RX = re.compile(r"[a-zA-Z0-9]+")


def _tokenize(text: str) -> list[str]:
    return _TOKEN_RX.findall(text.lower())


@dataclass
class RetrievedChunk:
    text:   str
    source: str
    score:  float


class RetrievalAgent:
    def __init__(self, knowledge_dir: str = "data/knowledge",
                 chunk_words: int = 256):
        self.knowledge_dir = knowledge_dir
        self.chunk_words   = chunk_words
        self._chunks:    list[tuple[str, str]] = []   # (text, source)
        self._tok_chunks: list[list[str]]      = []
        self._bm25 = None
        self._idf: dict[str, float] = {}
        self._loaded = False

    def load(self) -> int:
        """Walk knowledge_dir, build the index. Returns chunk count."""
        self._chunks.clear()
        self._tok_chunks.clear()
        if not os.path.isdir(self.knowledge_dir):
            self._loaded = True
            return 0

        for root, _dirs, files in os.walk(self.knowledge_dir):
            for fname in files:
                if not fname.lower().endswith((".txt", ".md")):
                    continue
                path = os.path.join(root, fname)
                rel  = os.path.relpath(path, self.knowledge_dir)
                try:
                    with open(path, encoding="utf-8", errors="replace") as f:
                        text = f.read()
                except OSError:
                    continue
                words = text.split()
                for i in range(0, len(words), self.chunk_words):
                    chunk = " ".join(words[i:i + self.chunk_words])
                    if chunk.strip():
                        self._chunks.append((chunk, rel))
                        self._tok_chunks.append(_tokenize(chunk))

        if self._tok_chunks:
            if _HAS_BM25:
                self._bm25 = BM25Okapi(self._tok_chunks)
            else:
                # Cheap TF-IDF: idf for each term across chunks
                df: Counter = Counter()
                for toks in self._tok_chunks:
                    df.update(set(toks))
                n = len(self._tok_chunks)
                self._idf = {t: math.log(1 + n / (1 + c)) for t, c in df.items()}

        self._loaded = True
        return len(self._chunks)

    def query(self, q: str, top_k: int = 4) -> list[RetrievedChunk]:
        if not self._loaded:
            self.load()
        if not self._chunks:
            return []
        q_tokens = _tokenize(q)
        if not q_tokens:
            return []

        if self._bm25 is not None:
            scores = self._bm25.get_scores(q_tokens)
        else:
            # TF-IDF cosine-ish: sum of tf*idf for each query term in chunk
            scores = []
            for toks in self._tok_chunks:
                tf = Counter(toks)
                s  = sum(tf.get(t, 0) * self._idf.get(t, 0.0) for t in q_tokens)
                norm = math.sqrt(len(toks)) or 1.0
                scores.append(s / norm)

        ranked = sorted(range(len(self._chunks)), key=lambda i: scores[i], reverse=True)[:top_k]
        out = []
        for i in ranked:
            if scores[i] <= 0:
                continue
            text, src = self._chunks[i]
            out.append(RetrievedChunk(text=text, source=src, score=float(scores[i])))
        return out
