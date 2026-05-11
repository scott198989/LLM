"""Multi-pass reasoning + scoring for the nanoGPT-style HAVOC pretrain model.

Runs N passes. Each pass conditions on the best-scoring prior answer via
a simple "first attempt / more accurate answer" scaffold (pretrain models
can follow that kind of continuation pattern even without instruction
tuning). Every candidate answer is scored intrinsically by its average
per-token log-probability under the model — this is the model's own
assessment of how plausible the text is.

Two probabilities are reported:
  - per-pass `confidence` = exp(avg log-prob per token), the geometric
    mean per-token probability. Bounded in [0,1]; interpretable as
    "how natural this answer feels to the model, per token".
  - final `confidence` = softmax over all candidates' avg log-probs,
    interpretable as "this candidate's share of model preference
    across the N answers it generated".

NOTE: This measures the model's *self-likelihood*, not factual correctness.
A confidently-generated wrong answer can still score high. With a 64M
pretrain-only checkpoint the absolute numbers are noisy — useful for
ranking candidates, not for absolute reliability.

Emits the same event shape as scripts/refinement.py so the existing
"Refine" UI panel renders it without changes.
"""

from __future__ import annotations

import math
import re
from dataclasses import dataclass

import torch
import torch.nn.functional as F


# ── helpers ─────────────────────────────────────────────────────────────────


def _softmax(xs: list[float], temperature: float = 1.0) -> list[float]:
    if not xs:
        return []
    m = max(xs)
    es = [math.exp((x - m) / max(temperature, 1e-6)) for x in xs]
    s = sum(es) or 1.0
    return [e / s for e in es]


def _strip_answer(text: str) -> str:
    """Trim the streamed candidate at the first scaffold marker the model echoed."""
    for marker in ("\nFirst attempt:", "\nA more accurate answer:",
                   "\nAnswer:", "\nQ:", "\n\n\n"):
        i = text.find(marker)
        if i > 0:
            text = text[:i]
            break
    return text.strip()


@dataclass
class _Pass:
    n:        int
    answer:   str
    score:    float    # avg log-prob per token (intrinsic)
    per_tok:  float    # exp(score), in [0,1]


# ── engine ──────────────────────────────────────────────────────────────────


class ReasoningEngine:
    """N-pass best-of-N with intrinsic log-prob scoring, bound to a NanoGPTEngine."""

    def __init__(self,
                 engine,                              # NanoGPTEngine
                 n_passes:        int   = 10,
                 max_pass_tokens: int   = 96,
                 temperature:     float = 0.85,
                 top_k:           int   = 40,
                 top_p:           float = 0.9,
                 softmax_temp:    float = 0.5):
        self.engine          = engine
        self.n_passes        = n_passes
        self.max_pass_tokens = max_pass_tokens
        self.temperature     = temperature
        self.top_k           = top_k
        self.top_p           = top_p
        self.softmax_temp    = softmax_temp

    # ── prompt scaffolding ──────────────────────────────────────────────────

    def _initial_prompt(self, question: str, system_prompt: str) -> str:
        sys_part = (system_prompt.strip() + "\n\n") if system_prompt.strip() else ""
        return f"{sys_part}{question.strip()}\n\nAnswer:"

    def _refine_prompt(self, question: str, best_answer: str, system_prompt: str) -> str:
        sys_part = (system_prompt.strip() + "\n\n") if system_prompt.strip() else ""
        return (
            f"{sys_part}{question.strip()}\n\n"
            f"First attempt: {best_answer.strip()}\n\n"
            f"A more accurate answer:"
        )

    # ── intrinsic scoring (avg log-prob per token) ──────────────────────────

    def _score_logprob(self, prompt_ids: list[int], answer_ids: list[int]) -> float:
        """Return mean log p(answer_t | prompt, answer_<t>) under the model."""
        if not answer_ids:
            return float("-inf")

        model      = self.engine.model
        device     = self.engine.device
        block_size = self.engine.cfg.block_size

        # Truncate prompt from the left if combined length exceeds block_size
        max_prompt = max(1, block_size - len(answer_ids))
        if len(prompt_ids) > max_prompt:
            prompt_ids = prompt_ids[-max_prompt:]
        full = prompt_ids + answer_ids
        if len(full) > block_size:
            full = full[-block_size:]
            # recompute prompt boundary after truncation
            P = block_size - len(answer_ids)
        else:
            P = len(prompt_ids)

        idx = torch.tensor([full], dtype=torch.long, device=device)

        amp_dtype = torch.bfloat16 if (device == "cuda" and torch.cuda.is_bf16_supported()) else torch.float16
        use_amp = (device == "cuda")

        with torch.no_grad():
            with torch.amp.autocast(device, dtype=amp_dtype, enabled=use_amp):
                # Reproduce the model body but keep ALL positions (the model's
                # forward returns only the last position for inference speed).
                pos = torch.arange(idx.size(1), dtype=torch.long, device=device)
                x = model.drop(model.tok_emb(idx) + model.pos_emb(pos))
                for blk in model.blocks:
                    x = blk(x)
                x = model.ln_f(x)
                logits = model.lm_head(x)            # [1, T, V]

        # Token at position p+1 is predicted by logits at position p.
        # Answer span tokens are at full[P : P+|A|]. Their predicting
        # logits are at positions P-1 .. P+|A|-2.
        pred = logits[0, P - 1: P - 1 + len(answer_ids), :].float()
        tgt  = torch.tensor(answer_ids, dtype=torch.long, device=device)
        log_probs = F.log_softmax(pred, dim=-1)
        sel = log_probs.gather(1, tgt.unsqueeze(1)).squeeze(1)
        return float(sel.mean().item())

    # ── streaming candidate generation ──────────────────────────────────────

    def _stream_candidate(self, scaffold_prompt: str):
        """Yield ('delta', tok) and finally ('done', full_text)."""
        full = ""
        # Use the engine's generate_stream but bypass its system-prompt prefix —
        # we built the prompt ourselves. Setting an empty system prompt is enough
        # since NanoGPTEngine prepends only when it's non-empty.
        prior_system = self.engine.get_system_prompt()
        self.engine.set_system_prompt("")
        try:
            for tok, done, _ in self.engine.generate_stream(
                prompt          = scaffold_prompt,
                max_new_tokens  = self.max_pass_tokens,
                temperature     = self.temperature,
                top_k           = self.top_k,
                top_p           = self.top_p,
            ):
                if tok:
                    full += tok
                    yield ("delta", tok)
                if done:
                    break
        finally:
            self.engine.set_system_prompt(prior_system)
        yield ("done", full)

    # ── main entry point ────────────────────────────────────────────────────

    def stream(self, question: str, system_prompt: str = ""):
        tokenizer = self.engine.tokenizer
        if tokenizer is None or self.engine.model is None:
            yield {"type": "error", "content": "engine not loaded"}
            yield {"type": "done"}
            return

        passes: list[_Pass] = []

        # Use the question alone (no system prompt) for scoring, so the score
        # reflects p(answer | question) — not how well it followed the scaffold.
        score_prompt_ids = tokenizer.encode(question.strip(), disallowed_special=())

        for n in range(1, self.n_passes + 1):
            yield {"type": "pass_start", "n": n}

            # Pass 1 uses a plain answer-completion scaffold; later passes
            # condition on the highest-scoring prior answer.
            if not passes:
                scaffold = self._initial_prompt(question, system_prompt)
            else:
                best = max(passes, key=lambda p: p.score)
                scaffold = self._refine_prompt(question, best.answer, system_prompt)

            full = ""
            for kind, payload in self._stream_candidate(scaffold):
                if kind == "delta" and payload:
                    yield {"type": "pass_delta", "n": n, "delta": payload}
                elif kind == "done":
                    full = payload

            answer_text = _strip_answer(full)
            answer_ids  = tokenizer.encode(answer_text, disallowed_special=()) if answer_text else []
            score       = self._score_logprob(score_prompt_ids, answer_ids)
            per_tok     = math.exp(score) if math.isfinite(score) else 0.0
            rec = _Pass(n=n, answer=answer_text, score=score, per_tok=per_tok)
            passes.append(rec)

            yield {
                "type":       "pass_complete",
                "n":          n,
                "answer":     answer_text,
                "confidence": per_tok,            # exp(avg logprob), bounded [0,1]
                "reasoning":  f"score={score:.3f} nats/tok",
                "raw":        full,
            }

        # Final ranking: softmax over all candidates' raw log-prob scores
        valid = [p for p in passes if math.isfinite(p.score)]
        if not valid:
            yield {"type": "final_complete", "answer": "", "confidence": 0.0,
                   "passes_used": len(passes), "raw": ""}
            yield {"type": "done"}
            return

        scores = [p.score for p in valid]
        probs  = _softmax(scores, temperature=self.softmax_temp)
        best_i = max(range(len(valid)), key=lambda i: scores[i])
        winner = valid[best_i]

        yield {
            "type":        "final_start",
        }
        # Emit the winner's text as a final_delta block so the UI gets a body
        yield {"type": "final_delta", "delta": winner.answer}

        yield {
            "type":        "final_complete",
            "answer":      winner.answer,
            "confidence":  float(probs[best_i]),   # softmax share over candidates
            "passes_used": len(passes),
            "raw":         f"best pass={winner.n}  raw_score={winner.score:.3f} nats/tok  "
                           f"softmax_share={probs[best_i]*100:.1f}%",
        }
        yield {"type": "done"}
