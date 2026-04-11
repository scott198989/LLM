"""
Inference engine for the trained GPT model.

Handles model loading, tokenisation, and token-by-token streaming generation
with full sampling control: temperature, top-k, top-p (nucleus), repetition
penalty, and greedy decoding.

This module is imported by gui_app.py but can also be used standalone:

    from inference import InferenceEngine
    engine = InferenceEngine()
    engine.load_model("models/checkpoints/best.pt")

    for token, done, stats in engine.generate_stream("Once upon a time", max_new_tokens=200):
        print(token, end="", flush=True)
        if done:
            break
"""

from __future__ import annotations

import os
import sys
import threading
import time
from dataclasses import dataclass, field

import torch
import torch.nn.functional as F

# ── Import model architecture from train.py ───────────────────────────────────
# train.py has no side-effects at import time (training code is under
# `if __name__ == "__main__"`) so this is safe.
_SCRIPTS_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, _SCRIPTS_DIR)

from train import Config, GPT  # noqa: E402

# ── Optional: HuggingFace tokenizer ───────────────────────────────────────────
try:
    from transformers import GPT2TokenizerFast
    _HAS_TOKENIZER = True
except ImportError:
    _HAS_TOKENIZER = False


# ---------------------------------------------------------------------------
# Sampling helpers  (static, pure-function style for testability)
# ---------------------------------------------------------------------------

def _top_k_filter(logits: torch.Tensor, k: int) -> torch.Tensor:
    """Zero out all logits outside the top-k."""
    if k <= 0:
        return logits
    k = min(k, logits.size(-1))
    values, _ = torch.topk(logits, k)
    threshold  = values[:, -1].unsqueeze(-1)
    logits     = logits.masked_fill(logits < threshold, -float("inf"))
    return logits


def _top_p_filter(logits: torch.Tensor, p: float) -> torch.Tensor:
    """
    Nucleus (top-p) sampling filter.

    Keeps the smallest set of tokens whose cumulative probability ≥ p and
    zeroes out the rest.  Always keeps at least 1 token.
    """
    if p >= 1.0:
        return logits
    probs                       = F.softmax(logits, dim=-1)
    sorted_probs, sorted_idx    = torch.sort(probs, dim=-1, descending=True)
    cumulative                  = torch.cumsum(sorted_probs, dim=-1)
    # Shift right by 1: we want to *include* the token that pushes us over p
    remove                      = (cumulative - sorted_probs) > p
    sorted_probs[remove]        = 0.0
    # Scatter back to original vocabulary order
    probs = torch.zeros_like(logits).scatter_(-1, sorted_idx, sorted_probs)
    # Replace original logits so downstream code can use softmax or multinomial
    logits = logits.masked_fill(probs == 0.0, -float("inf"))
    return logits


def _repetition_penalty(logits: torch.Tensor,
                         generated: torch.Tensor,
                         penalty: float) -> torch.Tensor:
    """
    Penalise tokens that have already appeared in the context.

    Positive logits are divided by `penalty`; negative logits are multiplied.
    penalty=1.0 → no effect.
    """
    if abs(penalty - 1.0) < 1e-6:
        return logits
    for tok_id in generated.unique():
        if logits[0, tok_id] > 0:
            logits[0, tok_id] /= penalty
        else:
            logits[0, tok_id] *= penalty
    return logits


# ---------------------------------------------------------------------------
# Generation statistics
# ---------------------------------------------------------------------------

@dataclass
class GenStats:
    n_tokens:    int   = 0
    elapsed_s:   float = 0.0
    tok_per_sec: float = 0.0
    status:      str   = "idle"   # "running" | "done" | "stopped" | "error"

    def update(self, n_tokens: int, t_start: float, status: str = "running") -> None:
        self.n_tokens    = n_tokens
        self.elapsed_s   = max(time.perf_counter() - t_start, 1e-9)
        self.tok_per_sec = n_tokens / self.elapsed_s
        self.status      = status


# ---------------------------------------------------------------------------
# Inference engine
# ---------------------------------------------------------------------------

class InferenceEngine:
    """
    Loads a checkpoint produced by train.py and exposes a streaming
    generation interface.

    Usage
    -----
        engine = InferenceEngine()
        info   = engine.load_model("models/checkpoints/best.pt")

        stop = threading.Event()
        for tok, done, stats in engine.generate_stream(
            prompt="What is gravity?",
            max_new_tokens=200,
            temperature=0.8,
            top_k=40,
            top_p=0.9,
            stop_event=stop,
        ):
            print(tok, end="", flush=True)
            if done:
                break
    """

    def __init__(self) -> None:
        self.model:      GPT | None   = None
        self.tokenizer                 = None
        self.device:     str          = "cpu"
        self.model_cfg:  Config | None = None
        self.ckpt_meta:  dict         = {}
        self.loaded:     bool         = False

        # Special token IDs populated after load
        self._eot_id:        int | None = None
        self._think_id:      int | None = None
        self._end_think_id:  int | None = None
        self._sep_id:        int | None = None

        # System prompt — applied to every generate_stream() call
        self._system_prompt: str = ""

    # ── System prompt ─────────────────────────────────────────────────────────

    def set_system_prompt(self, text: str) -> None:
        """Set the system prompt string directly."""
        self._system_prompt = text.strip()

    def get_system_prompt(self) -> str:
        """Return the current system prompt."""
        return self._system_prompt

    def load_system_prompt(self, path: str) -> str:
        """
        Load the system prompt from a text file.
        Returns the loaded text.  Silently keeps the previous prompt on error.
        """
        try:
            with open(path, encoding="utf-8") as f:
                self._system_prompt = f.read().strip()
        except OSError as exc:
            print(f"  [inference] Could not load system prompt from {path}: {exc}")
        return self._system_prompt

    def save_system_prompt(self, path: str) -> None:
        """Persist the current system prompt to a text file."""
        with open(path, "w", encoding="utf-8") as f:
            f.write(self._system_prompt)

    # ── Model loading ──────────────────────────────────────────────────────────

    def load_model(self, ckpt_path: str) -> dict:
        """
        Load a checkpoint file.  Returns a metadata dict with model info.

        Works with both raw checkpoints and checkpoints from torch.compile()
        (which prefix state-dict keys with "_orig_mod.").
        """
        self.loaded = False
        device = "cuda" if torch.cuda.is_available() else "cpu"

        ckpt = torch.load(ckpt_path, map_location=device, weights_only=False)

        # ── Reconstruct Config from saved cfg ─────────────────────────────────
        saved_cfg   = ckpt.get("cfg", {})
        model_cfg   = Config()

        # Copy only recognised model fields; skip device/path fields
        _skip = {"processed_dir", "ckpt_dir", "log_dir", "device"}
        for k, v in saved_cfg.items():
            if k not in _skip and hasattr(model_cfg, k):
                setattr(model_cfg, k, v)

        # Inference overrides
        model_cfg.gradient_checkpointing = False
        model_cfg.dropout                = 0.0
        model_cfg.device                 = device

        # ── Build model and load weights ───────────────────────────────────────
        model = GPT(model_cfg).to(device)
        model.eval()

        state_dict = ckpt["model"]
        # Strip torch.compile prefix if present
        state_dict = {
            k.replace("_orig_mod.", ""): v
            for k, v in state_dict.items()
        }
        missing, unexpected = model.load_state_dict(state_dict, strict=False)
        if missing:
            print(f"  [inference] Missing keys  : {missing[:5]}")
        if unexpected:
            print(f"  [inference] Unexpected keys: {unexpected[:5]}")

        # ── Tokenizer ──────────────────────────────────────────────────────────
        tokenizer = None
        if _HAS_TOKENIZER:
            try:
                tokenizer = GPT2TokenizerFast.from_pretrained("gpt2")
                tokenizer.add_special_tokens({
                    "additional_special_tokens": [
                        "<|sep|>", "<|think|>", "<|/think|>",
                        "<|system|>", "<|/system|>",
                        "<|user|>", "<|/user|>", "<|assistant|>",
                    ]
                })
            except Exception as exc:
                print(f"  [inference] Tokenizer load failed: {exc}")

        self.model     = model
        self.tokenizer = tokenizer
        self.device    = device
        self.model_cfg = model_cfg
        self.loaded    = True

        # ── Resolve special token IDs ──────────────────────────────────────────
        if tokenizer is not None:
            self._eot_id       = tokenizer.eos_token_id
            self._sep_id       = tokenizer.convert_tokens_to_ids("<|sep|>")
            self._think_id     = tokenizer.convert_tokens_to_ids("<|think|>")
            self._end_think_id = tokenizer.convert_tokens_to_ids("<|/think|>")
        else:
            self._eot_id = 50256

        # ── Metadata ───────────────────────────────────────────────────────────
        n_unique = sum(p.numel() for p in set(model.parameters()))
        self.ckpt_meta = {
            "path":       ckpt_path,
            "device":     device,
            "n_params":   n_unique,
            "epoch":      ckpt.get("epoch", "?"),
            "step":       ckpt.get("step", "?"),
            "val_loss":   ckpt.get("val_loss", float("nan")),
            "vocab_size": model_cfg.vocab_size,
            "block_size": model_cfg.block_size,
            "n_layer":    model_cfg.n_layer,
            "n_head":     model_cfg.n_head,
            "n_embd":     model_cfg.n_embd,
        }
        return self.ckpt_meta

    # ── Generation ─────────────────────────────────────────────────────────────

    def generate_stream(
        self,
        prompt:             str,
        max_new_tokens:     int            = 300,
        temperature:        float          = 0.8,
        top_k:              int            = 40,
        top_p:              float          = 0.9,
        repetition_penalty: float          = 1.1,
        sampling_mode:      str            = "top_kp",   # top_kp|top_k|top_p|greedy
        use_cot:            bool           = False,
        stop_event:         threading.Event | None = None,
    ):
        """
        Streaming token generator.

        Yields
        ------
        (token_text : str,  is_done : bool,  stats : GenStats)

        token_text  — decoded text of the new token (empty string on final yield)
        is_done     — True on the final yield only
        stats       — live GenStats object (same object mutated each step)

        Sampling modes
        --------------
        "top_kp"  — top-k filter THEN top-p filter (recommended)
        "top_k"   — top-k filter only
        "top_p"   — top-p filter only
        "greedy"  — argmax, ignores temperature / top-k / top-p
        """
        if not self.loaded or self.model is None:
            yield ("", True, GenStats(status="error"))
            return

        model     = self.model
        tokenizer = self.tokenizer
        device    = self.device
        cfg       = self.model_cfg
        stats     = GenStats()
        t_start   = time.perf_counter()

        # ── Encode prompt (with optional system prompt prefix) ─────────────────
        if tokenizer is not None:
            if self._system_prompt:
                full_prompt = (
                    f"<|system|>{self._system_prompt}<|/system|>"
                    f"<|user|>{prompt}<|/user|><|assistant|>"
                )
            else:
                full_prompt = prompt
            enc = tokenizer.encode(full_prompt, add_special_tokens=False)
            if not enc:
                enc = [self._eot_id or 50256]
        else:
            enc = [self._eot_id or 50256]

        # Inject CoT opening token if requested
        if use_cot and self._think_id is not None:
            enc = enc + [self._think_id]

        idx = torch.tensor([enc], dtype=torch.long, device=device)

        generated_ids: list[int] = []
        use_amp = (device == "cuda")
        amp_dtype = cfg.dtype if hasattr(cfg, "dtype") else torch.bfloat16

        # ── Token-by-token generation ──────────────────────────────────────────
        with torch.no_grad():
            for _ in range(max_new_tokens):

                # ── Stop signal ────────────────────────────────────────────────
                if stop_event is not None and stop_event.is_set():
                    stats.update(len(generated_ids), t_start, "stopped")
                    yield ("", True, stats)
                    return

                # ── Forward pass ────────────────────────────────────────────────
                idx_cond = idx[:, -cfg.block_size:]
                with torch.amp.autocast(device, dtype=amp_dtype, enabled=use_amp):
                    logits, _ = model(idx_cond)

                logits = logits[:, -1, :].float()    # (1, vocab) in fp32

                # ── Greedy shortcut ────────────────────────────────────────────
                if sampling_mode == "greedy":
                    next_id = logits.argmax(dim=-1, keepdim=True)

                else:
                    # Temperature scaling
                    if temperature > 0:
                        logits = logits / max(temperature, 1e-8)

                    # Repetition penalty
                    if generated_ids:
                        past = torch.tensor([generated_ids], device=device)
                        logits = _repetition_penalty(logits, past, repetition_penalty)

                    # Sampling filters
                    if sampling_mode in ("top_k", "top_kp") and top_k > 0:
                        logits = _top_k_filter(logits, top_k)

                    if sampling_mode in ("top_p", "top_kp") and top_p < 1.0:
                        logits = _top_p_filter(logits, top_p)

                    # Sample from resulting distribution
                    probs   = F.softmax(logits, dim=-1)
                    # Guard against all-zero rows (extreme filtering)
                    if probs.sum() < 1e-8:
                        probs = torch.ones_like(probs) / probs.size(-1)
                    next_id = torch.multinomial(probs, num_samples=1)

                tok_id = next_id.item()

                # ── Stop at end-of-text ────────────────────────────────────────
                if self._eot_id is not None and tok_id == self._eot_id:
                    stats.update(len(generated_ids), t_start, "eot")
                    yield ("", True, stats)
                    return

                generated_ids.append(tok_id)
                idx = torch.cat([idx, next_id], dim=1)

                # ── Decode token to text ───────────────────────────────────────
                if tokenizer is not None:
                    tok_text = tokenizer.decode(
                        [tok_id], skip_special_tokens=False
                    )
                else:
                    tok_text = f"[{tok_id}]"

                stats.update(len(generated_ids), t_start, "running")
                yield (tok_text, False, stats)

        stats.update(len(generated_ids), t_start, "max_tokens")
        yield ("", True, stats)

    # ── Convenience: non-streaming full generation ─────────────────────────────

    def generate(self, prompt: str, **kwargs) -> str:
        """
        Blocking full-string generation.  Returns the complete generated text.
        Useful for testing outside the GUI.
        """
        parts: list[str] = []
        for tok, done, _ in self.generate_stream(prompt, **kwargs):
            if tok:
                parts.append(tok)
            if done:
                break
        return "".join(parts)
