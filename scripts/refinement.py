"""
Iterative self-refinement engine (inference-time only).

This wraps any object that exposes a `generate_stream(prompt, ...)` method
(e.g. InferenceEngine) and runs up to N passes where each pass:

  1. Receives the user's question + a transcript of prior passes.
  2. Produces a brief reasoning trace, a tentative answer, and a confidence
     estimate (0-95% - the prompt explicitly forbids claiming 100%).
  3. Is parsed by deterministic regex to extract answer + confidence.

After every pass we check stability:
  - last two passes' confidence >= confidence_threshold AND
  - last two answers are textually similar (>= similarity_threshold)
If both are met, we stop early. Otherwise we continue up to max_passes.

A separate final pass asks the model to consolidate. Its output is
emitted as `final_complete` so the UI can render it distinctly from the
refinement trace.

The engine yields a stream of dict events:
    {"type": "pass_start",     "n": int}
    {"type": "pass_delta",     "n": int, "delta": str}
    {"type": "pass_complete",  "n": int, "answer": str, "confidence": float, "raw": str}
    {"type": "early_stop",     "reason": str, "passes": int}
    {"type": "final_start"}
    {"type": "final_delta",    "delta": str}
    {"type": "final_complete", "answer": str, "confidence": float, "raw": str, "passes_used": int}
    {"type": "done"}
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import Iterable, Iterator, Protocol


# ── Engine protocol ──────────────────────────────────────────────────────


class StreamingEngine(Protocol):
    """Anything that streams tokens from a prompt - InferenceEngine fits."""

    def generate_stream(self, prompt: str, **kwargs) -> Iterable[tuple[str, bool, object]]:
        ...

    def set_system_prompt(self, text: str) -> None:  # pragma: no cover - optional
        ...


# ── Parsers ───────────────────────────────────────────────────────────────


_CONF_RX = re.compile(
    r"(?i)(?:confidence|certainty|sure)\s*[:\-=]?\s*(\d{1,3})\s*%?"
)
_GENERIC_PCT_RX = re.compile(r"(\d{1,3})\s*%")
_ANSWER_RX = re.compile(
    r"(?ims)^\s*(?:answer|final\s+answer)\s*[:\-]\s*(.+?)(?=\n\s*(?:reasoning|confidence|pass\s+\d|$))"
)
_REASONING_RX = re.compile(
    r"(?ims)^\s*reasoning\s*[:\-]\s*(.+?)(?=\n\s*(?:answer|confidence|pass\s+\d|$))"
)


def parse_confidence(text: str) -> float:
    """
    Extract a confidence value from `text`. Returns a float in [0.0, 0.95].

    Looks first for explicit 'Confidence: NN%' style; falls back to the
    final '%' figure in the text. Returns 0.5 if nothing is found.
    Caps at 0.95 - we deliberately never let the model claim 100%.
    """
    m = _CONF_RX.search(text)
    if m is None:
        all_pct = list(_GENERIC_PCT_RX.finditer(text))
        if not all_pct:
            return 0.5
        m = all_pct[-1]
    try:
        v = int(m.group(1))
    except (ValueError, IndexError):
        return 0.5
    v = max(0, min(95, v))
    return v / 100.0


def parse_answer(text: str) -> str:
    """Pull the 'Answer: ...' block out of a pass; falls back to last paragraph."""
    m = _ANSWER_RX.search(text)
    if m:
        return m.group(1).strip()
    paragraphs = [p.strip() for p in text.strip().split("\n\n") if p.strip()]
    return paragraphs[-1] if paragraphs else text.strip()


def parse_reasoning(text: str) -> str:
    m = _REASONING_RX.search(text)
    return m.group(1).strip() if m else ""


# ── Stability check (cheap Jaccard over word sets) ────────────────────────


def _tokens(s: str) -> set[str]:
    return set(re.findall(r"[a-z0-9]+", s.lower()))


def jaccard(a: str, b: str) -> float:
    ta, tb = _tokens(a), _tokens(b)
    if not ta and not tb:
        return 1.0
    if not ta or not tb:
        return 0.0
    return len(ta & tb) / len(ta | tb)


# ── Engine ────────────────────────────────────────────────────────────────


@dataclass
class _PassRecord:
    n:          int
    raw:        str
    answer:     str
    confidence: float
    reasoning:  str = ""


@dataclass
class RefinementEngine:
    engine:               StreamingEngine
    max_passes:           int   = 10
    confidence_threshold: float = 0.85
    similarity_threshold: float = 0.9
    max_pass_tokens:      int   = 256
    max_final_tokens:     int   = 256
    # Default sampling for refinement passes — slightly cooler than chat
    temperature:          float = 0.6
    top_k:                int   = 40
    top_p:                float = 0.9
    stop_keywords:        tuple[str, ...] = field(default_factory=lambda: ("Pass ", "\nPass"))

    # ── prompt scaffolds ──────────────────────────────────────────────────

    def _system_block(self, user_system: str) -> str:
        base = (
            "You answer questions through iterative self-refinement. Each pass, "
            "produce three labelled lines:\n"
            "  Reasoning: a short rationale.\n"
            "  Answer: your current best answer.\n"
            "  Confidence: a number 0-95 (never claim 100). Be honest about uncertainty.\n"
            "When asked for a Final answer, give a concise, separated final response."
        )
        if user_system.strip():
            return user_system.strip() + "\n\n" + base
        return base

    def _build_pass_prompt(self,
                           user_question: str,
                           passes: list[_PassRecord],
                           n: int,
                           system_prompt: str) -> str:
        sys_block = self._system_block(system_prompt)
        parts = [sys_block, "", f"User: {user_question}", ""]
        if passes:
            parts.append("Previous passes:")
            for p in passes:
                parts.append(
                    f"  Pass {p.n}: answer={p.answer!r}  confidence={int(p.confidence*100)}%"
                )
            parts.append("")
        if n == 1:
            parts.append(f"Pass {n}:")
        else:
            parts.append(
                f"Pass {n}: Review the previous answer for missed wording, "
                "hidden assumptions, or ambiguity. Then produce updated lines."
            )
        parts.append("Reasoning:")
        return "\n".join(parts)

    def _build_final_prompt(self,
                            user_question: str,
                            passes: list[_PassRecord],
                            system_prompt: str) -> str:
        sys_block = self._system_block(system_prompt)
        parts = [sys_block, "", f"User: {user_question}", ""]
        parts.append("Refinement transcript:")
        for p in passes:
            parts.append(
                f"  Pass {p.n}: answer={p.answer!r}  confidence={int(p.confidence*100)}%"
            )
        parts.append("")
        parts.append(
            "Final answer: write a short consolidated answer, on a single line "
            "labelled 'Answer:'. Then on the next line write 'Confidence:' "
            "with an honest estimate (cap at 95)."
        )
        parts.append("Answer:")
        return "\n".join(parts)

    # ── streaming helpers ────────────────────────────────────────────────

    def _stream_pass(self, prompt: str, max_tokens: int) -> Iterator[tuple[str, str]]:
        """
        Yield ('delta', token_text) tuples and finally ('done', full_text).

        Stops early if the model starts a new 'Pass N:' header.
        """
        full = ""
        for tok, done, _ in self.engine.generate_stream(
            prompt          = prompt,
            max_new_tokens  = max_tokens,
            temperature     = self.temperature,
            top_k           = self.top_k,
            top_p           = self.top_p,
        ):
            if tok:
                full += tok
                # Soft stop: if a new pass header appears, cut the tail
                cut = self._find_stop(full)
                if cut is not None:
                    full = full[:cut]
                    yield ("delta", "")  # keep stream alive
                    yield ("done", full)
                    return
                yield ("delta", tok)
            if done:
                break
        yield ("done", full)

    def _find_stop(self, text: str) -> int | None:
        for kw in self.stop_keywords:
            idx = text.find(kw)
            if idx > 0:        # require some content before the marker
                return idx
        return None

    # ── main entry point ─────────────────────────────────────────────────

    def stream(self, user_question: str, system_prompt: str = ""):
        passes: list[_PassRecord] = []
        passes_used = 0

        for n in range(1, self.max_passes + 1):
            yield {"type": "pass_start", "n": n}
            prompt = self._build_pass_prompt(user_question, passes, n, system_prompt)
            full   = ""

            for kind, payload in self._stream_pass(prompt, self.max_pass_tokens):
                if kind == "delta":
                    if payload:
                        yield {"type": "pass_delta", "n": n, "delta": payload}
                else:  # 'done'
                    full = payload

            ans  = parse_answer(full)
            conf = parse_confidence(full)
            rea  = parse_reasoning(full)
            rec  = _PassRecord(n=n, raw=full, answer=ans, confidence=conf, reasoning=rea)
            passes.append(rec)
            passes_used = n

            yield {
                "type":       "pass_complete",
                "n":          n,
                "answer":     rec.answer,
                "confidence": rec.confidence,
                "reasoning":  rec.reasoning,
                "raw":        rec.raw,
            }

            if self._should_stop(passes):
                yield {"type": "early_stop", "reason":
                       "two consecutive high-confidence and stable answers",
                       "passes": passes_used}
                break

        # Final synthesis
        final_prompt = self._build_final_prompt(user_question, passes, system_prompt)
        yield {"type": "final_start"}
        final_full = ""
        for kind, payload in self._stream_pass(final_prompt, self.max_final_tokens):
            if kind == "delta":
                if payload:
                    yield {"type": "final_delta", "delta": payload}
            else:
                final_full = payload

        final_ans  = parse_answer(final_full) or (passes[-1].answer if passes else "")
        final_conf = parse_confidence(final_full)
        if final_conf == 0.5 and passes:        # nothing parsed -> reuse last pass
            final_conf = passes[-1].confidence

        yield {
            "type":         "final_complete",
            "answer":       final_ans,
            "confidence":   final_conf,
            "raw":          final_full,
            "passes_used":  passes_used,
        }
        yield {"type": "done"}

    # ── stability check ─────────────────────────────────────────────────

    def _should_stop(self, passes: list[_PassRecord]) -> bool:
        if len(passes) < 2:
            return False
        a, b = passes[-2], passes[-1]
        if a.confidence < self.confidence_threshold:
            return False
        if b.confidence < self.confidence_threshold:
            return False
        if jaccard(a.answer, b.answer) < self.similarity_threshold:
            return False
        return True
