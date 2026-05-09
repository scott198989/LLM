"""
HAVOC inference engine.

Loads a checkpoint produced by pretrain.py / sft.py and exposes a
streaming generation interface. Three modes:

  - generate_stream()              token-by-token streaming (default)
  - generate_with_refinement()     iterative self-refinement (refinement.py)
  - generate_with_orchestration()  full agent + tool pipeline (orchestrator.py)

The streaming interface is deliberately stable - gui_app.py and chat_ui
both consume `(token_text, is_done, GenStats)` tuples.
"""

from __future__ import annotations

import os
import sys
import threading
import time
from dataclasses import dataclass

import torch
import torch.nn.functional as F

_HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, _HERE)

from config           import HavocConfig                  # noqa: E402
from model            import HavocModel                   # noqa: E402
from tokenizer_havoc  import HavocTokenizer               # noqa: E402


# ── sampling helpers ──────────────────────────────────────────────────────


def _top_k_filter(logits: torch.Tensor, k: int) -> torch.Tensor:
    if k <= 0:
        return logits
    k = min(k, logits.size(-1))
    values, _ = torch.topk(logits, k)
    threshold = values[:, -1].unsqueeze(-1)
    return logits.masked_fill(logits < threshold, -float("inf"))


def _top_p_filter(logits: torch.Tensor, p: float) -> torch.Tensor:
    if p >= 1.0:
        return logits
    probs                    = F.softmax(logits, dim=-1)
    sorted_probs, sorted_idx = torch.sort(probs, dim=-1, descending=True)
    cumulative               = torch.cumsum(sorted_probs, dim=-1)
    remove                   = (cumulative - sorted_probs) > p
    sorted_probs[remove]     = 0.0
    probs = torch.zeros_like(logits).scatter_(-1, sorted_idx, sorted_probs)
    return logits.masked_fill(probs == 0.0, -float("inf"))


def _repetition_penalty(logits: torch.Tensor,
                        generated: torch.Tensor,
                        penalty: float) -> torch.Tensor:
    if abs(penalty - 1.0) < 1e-6:
        return logits
    for tok_id in generated.unique():
        if logits[0, tok_id] > 0:
            logits[0, tok_id] /= penalty
        else:
            logits[0, tok_id] *= penalty
    return logits


# ── Stats ─────────────────────────────────────────────────────────────────


@dataclass
class GenStats:
    n_tokens:    int   = 0
    elapsed_s:   float = 0.0
    tok_per_sec: float = 0.0
    status:      str   = "idle"

    def update(self, n_tokens: int, t_start: float, status: str = "running") -> None:
        self.n_tokens    = n_tokens
        self.elapsed_s   = max(time.perf_counter() - t_start, 1e-9)
        self.tok_per_sec = n_tokens / self.elapsed_s
        self.status      = status


# ── Engine ────────────────────────────────────────────────────────────────


class InferenceEngine:
    """Loads a HAVOC checkpoint and streams generated text."""

    def __init__(self) -> None:
        self.model:      HavocModel | None      = None
        self.tokenizer:  HavocTokenizer | None  = None
        self.cfg:        HavocConfig | None     = None
        self.device:     str                    = "cpu"
        self.ckpt_meta:  dict                   = {}
        self.loaded:     bool                   = False
        self._system_prompt: str                = ""

        # Lazy: refinement / orchestration engines (built on demand)
        self._refiner    = None
        self._orchestrator = None

    # ── system prompt ────────────────────────────────────────────────────

    def set_system_prompt(self, text: str) -> None:
        self._system_prompt = text.strip()

    def get_system_prompt(self) -> str:
        return self._system_prompt

    def load_system_prompt(self, path: str) -> str:
        try:
            with open(path, encoding="utf-8") as f:
                self._system_prompt = f.read().strip()
        except OSError as exc:
            print(f"  [inference] Could not load system prompt from {path}: {exc}")
        return self._system_prompt

    def save_system_prompt(self, path: str) -> None:
        with open(path, "w", encoding="utf-8") as f:
            f.write(self._system_prompt)

    # ── model loading ────────────────────────────────────────────────────

    def load_model(self,
                   ckpt_path:     str,
                   tokenizer_dir: str | None = None,
                   ) -> dict:
        device = "cuda" if torch.cuda.is_available() else "cpu"

        ck = torch.load(ckpt_path, map_location=device, weights_only=False)
        # Reconstruct config from checkpoint, dropping fields HavocConfig doesn't know
        valid = {f.name for f in HavocConfig.__dataclass_fields__.values()}
        cfg_kwargs = {k: v for k, v in ck.get("cfg", {}).items() if k in valid}
        cfg = HavocConfig(**cfg_kwargs)
        cfg.gradient_checkpointing = False
        cfg.dropout                = 0.0

        # Build model and load weights
        model = HavocModel(cfg).to(device)
        model.eval()
        sd = {k.replace("_orig_mod.", ""): v for k, v in ck["model"].items()}
        missing, unexpected = model.load_state_dict(sd, strict=False)
        if missing:
            print(f"  [inference] Missing keys  : {missing[:5]}")
        if unexpected:
            print(f"  [inference] Unexpected keys: {unexpected[:5]}")

        # Load tokenizer
        # 1) explicit arg, 2) checkpoint metadata, 3) default location
        tok_dir = tokenizer_dir or ck.get("tokenizer_dir") or "models/tokenizers/havoc_bpe"
        tokenizer = None
        if os.path.isfile(os.path.join(tok_dir, "tokenizer.json")):
            tokenizer = HavocTokenizer.from_pretrained(tok_dir)
        else:
            print(f"  [inference] WARN: no tokenizer at {tok_dir} - decoding will show ids")

        self.model     = model
        self.tokenizer = tokenizer
        self.cfg       = cfg
        self.device    = device
        self.loaded    = True

        n_unique = sum(p.numel() for p in {id(p): p for p in model.parameters()}.values())
        self.ckpt_meta = {
            "path":       ckpt_path,
            "device":     device,
            "n_params":   n_unique,
            "epoch":      ck.get("epoch", "?"),
            "step":       ck.get("step", "?"),
            "val_loss":   ck.get("val_loss", float("nan")),
            "vocab_size": cfg.vocab_size,
            "max_seq_len": cfg.max_seq_len,
            "num_layers":  cfg.num_layers,
            "num_heads":   cfg.num_heads,
            "hidden_size": cfg.hidden_size,
        }
        # Reset wrapped engines so they pick up the new model
        self._refiner = None
        self._orchestrator = None
        return self.ckpt_meta

    # ── core streaming generation ────────────────────────────────────────

    def generate_stream(
        self,
        prompt:             str,
        max_new_tokens:     int           = 300,
        temperature:        float         = 0.8,
        top_k:              int           = 40,
        top_p:              float         = 0.9,
        repetition_penalty: float         = 1.1,
        sampling_mode:      str           = "top_kp",   # top_kp|top_k|top_p|greedy
        stop_event:         threading.Event | None = None,
        wrap_chat:          bool          = True,
        cot:                bool          = False,
    ):
        """
        Yields (token_text: str, is_done: bool, stats: GenStats).

        wrap_chat=True (default) wraps the prompt in the HAVOC chat template
        using the configured system prompt. Pass wrap_chat=False when you've
        already crafted the full prompt (the refinement scaffold does this).
        """
        if not self.loaded or self.model is None:
            yield ("", True, GenStats(status="error"))
            return

        model     = self.model
        tokenizer = self.tokenizer
        device    = self.device
        cfg       = self.cfg
        stats     = GenStats()
        t_start   = time.perf_counter()

        # Encode
        if tokenizer is not None:
            if wrap_chat:
                messages = []
                if self._system_prompt:
                    messages.append({"role": "system", "content": self._system_prompt})
                messages.append({"role": "user", "content": prompt})
                enc = tokenizer.encode_chat(messages, add_generation_prompt=True)
            else:
                enc = tokenizer.encode(prompt, add_special=False)
            if not enc:
                enc = [tokenizer.eos_token_id]
            # CoT mode: nudge the model into reasoning by appending <|think|>
            # so the next generated tokens are inside the thinking block until
            # <|/think|> is emitted.
            if cot:
                enc = list(enc) + [tokenizer.think_token_id]
            eos_id = tokenizer.eos_token_id
            asst_close_id = tokenizer.end_assistant_token_id
        else:
            enc           = [0]
            eos_id        = 0
            asst_close_id = None

        idx = torch.tensor([enc], dtype=torch.long, device=device)
        generated_ids: list[int] = []
        amp_dtype = torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16
        use_amp   = (device == "cuda")

        with torch.no_grad():
            for _ in range(max_new_tokens):
                if stop_event is not None and stop_event.is_set():
                    stats.update(len(generated_ids), t_start, "stopped")
                    yield ("", True, stats)
                    return

                ctx = idx[:, -cfg.max_seq_len:]
                with torch.amp.autocast(device, dtype=amp_dtype, enabled=use_amp):
                    logits, _ = model(ctx)
                logits = logits[:, -1, :].float()

                if sampling_mode == "greedy":
                    next_id = logits.argmax(dim=-1, keepdim=True)
                else:
                    if temperature > 0:
                        logits = logits / max(temperature, 1.0e-8)
                    if generated_ids:
                        past = torch.tensor([generated_ids], device=device)
                        logits = _repetition_penalty(logits, past, repetition_penalty)
                    if sampling_mode in ("top_k", "top_kp") and top_k > 0:
                        logits = _top_k_filter(logits, top_k)
                    if sampling_mode in ("top_p", "top_kp") and top_p < 1.0:
                        logits = _top_p_filter(logits, top_p)
                    probs = F.softmax(logits, dim=-1)
                    if probs.sum() < 1e-8:
                        probs = torch.ones_like(probs) / probs.size(-1)
                    next_id = torch.multinomial(probs, num_samples=1)

                tok_id = next_id.item()
                if tok_id == eos_id:
                    stats.update(len(generated_ids), t_start, "eot")
                    yield ("", True, stats)
                    return
                if asst_close_id is not None and tok_id == asst_close_id:
                    stats.update(len(generated_ids), t_start, "asst_close")
                    yield ("", True, stats)
                    return

                generated_ids.append(tok_id)
                idx = torch.cat([idx, next_id], dim=1)

                tok_text = (tokenizer.decode([tok_id], skip_special_tokens=False)
                            if tokenizer is not None else f"[{tok_id}]")
                stats.update(len(generated_ids), t_start, "running")
                yield (tok_text, False, stats)

        stats.update(len(generated_ids), t_start, "max_tokens")
        yield ("", True, stats)

    # ── convenience: full text in one shot ────────────────────────────────

    def generate(self, prompt: str, **kwargs) -> str:
        parts: list[str] = []
        for tok, done, _ in self.generate_stream(prompt, **kwargs):
            if tok:
                parts.append(tok)
            if done:
                break
        return "".join(parts)

    # ── refinement mode ──────────────────────────────────────────────────

    def _refiner_engine(self):
        if self._refiner is None:
            from refinement import RefinementEngine
            self._refiner = RefinementEngine(
                engine                = _BareStreamingAdapter(self),
                max_passes            = self.cfg.refinement_max_passes if self.cfg else 10,
                confidence_threshold  = (self.cfg.refinement_confidence_threshold
                                         if self.cfg else 0.85),
                similarity_threshold  = (self.cfg.refinement_similarity_threshold
                                         if self.cfg else 0.9),
            )
        return self._refiner

    def generate_with_refinement(self, prompt: str):
        """Yield refinement events. See refinement.py for event schema."""
        if not self.loaded:
            yield {"type": "error", "content": "model not loaded"}
            return
        ref = self._refiner_engine()
        yield from ref.stream(prompt, system_prompt=self._system_prompt)

    # ── orchestration mode ──────────────────────────────────────────────

    def _orchestrator(self):
        if self._orchestrator is None:
            from orchestrator import Orchestrator
            self._orchestrator = Orchestrator.from_engine(self)
        return self._orchestrator

    def generate_with_orchestration(self, prompt: str, **kwargs):
        """Yield orchestrator events. See orchestrator.py for schema."""
        if not self.loaded:
            yield {"type": "error", "content": "model not loaded"}
            return
        orch = self._orchestrator()
        yield from orch.stream(prompt, system_prompt=self._system_prompt, **kwargs)


# ── Adapter so refinement always sees a 'wrap_chat=False' channel ──────


class _BareStreamingAdapter:
    """
    Refinement scaffolds construct entire prompts by hand (system + chat markers
    are baked into the scaffold text). We must therefore tell the underlying
    engine NOT to re-wrap the prompt in another chat template.
    """

    def __init__(self, engine: InferenceEngine):
        self.engine = engine

    def generate_stream(self, prompt: str, **kwargs):
        kwargs.setdefault("wrap_chat", False)
        yield from self.engine.generate_stream(prompt, **kwargs)

    def set_system_prompt(self, text: str) -> None:
        self.engine.set_system_prompt(text)


# ── CLI ────────────────────────────────────────────────────────────────────


if __name__ == "__main__":
    import argparse
    p = argparse.ArgumentParser(description="HAVOC inference (streaming).")
    p.add_argument("--ckpt",          required=True)
    p.add_argument("--tokenizer_dir", default=None)
    p.add_argument("--prompt",        required=True)
    p.add_argument("--system_prompt", default="")
    p.add_argument("--max_new_tokens", type=int, default=200)
    p.add_argument("--temperature",   type=float, default=0.8)
    p.add_argument("--top_k",         type=int, default=40)
    p.add_argument("--top_p",         type=float, default=0.9)
    p.add_argument("--refine",        action="store_true",
                   help="Use iterative self-refinement (refinement.py).")
    p.add_argument("--orchestrate",   action="store_true",
                   help="Use the full agent + tool orchestrator.")
    args = p.parse_args()

    eng = InferenceEngine()
    meta = eng.load_model(args.ckpt, tokenizer_dir=args.tokenizer_dir)
    if args.system_prompt:
        eng.set_system_prompt(args.system_prompt)

    print(f"\nLoaded {meta['n_params']:,} params from {meta['path']}")
    print(f"Layers/Heads/Hidden = {meta['num_layers']}/{meta['num_heads']}/{meta['hidden_size']}")

    if args.refine:
        print("\n=== REFINEMENT MODE ===\n")
        for ev in eng.generate_with_refinement(args.prompt):
            t = ev["type"]
            if t == "pass_start":
                print(f"\n--- Pass {ev['n']} ---")
            elif t == "pass_delta":
                print(ev["delta"], end="", flush=True)
            elif t == "pass_complete":
                print(f"\n[answer: {ev['answer']!r}  confidence: {int(ev['confidence']*100)}%]")
            elif t == "early_stop":
                print(f"\n[early stop after {ev['passes']} passes - {ev['reason']}]")
            elif t == "final_start":
                print("\n\n=== FINAL ANSWER ===")
            elif t == "final_delta":
                print(ev["delta"], end="", flush=True)
            elif t == "final_complete":
                print(f"\n\nFinal: {ev['answer']!r}  confidence: {int(ev['confidence']*100)}%  "
                      f"({ev['passes_used']} passes used)")
    elif args.orchestrate:
        print("\n=== ORCHESTRATE MODE ===\n")
        for ev in eng.generate_with_orchestration(args.prompt):
            print(ev)
    else:
        for tok, done, stats in eng.generate_stream(
            prompt          = args.prompt,
            max_new_tokens  = args.max_new_tokens,
            temperature     = args.temperature,
            top_k           = args.top_k,
            top_p           = args.top_p,
        ):
            if tok:
                print(tok, end="", flush=True)
            if done:
                print(f"\n\n[{stats.n_tokens} tokens in {stats.elapsed_s:.2f}s "
                      f"= {stats.tok_per_sec:.1f} tok/s, status={stats.status}]")
                break
