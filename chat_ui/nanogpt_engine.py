"""Inference engine for the nanoGPT-style HAVOC checkpoint
(model.havoc.HavocGPT + GPT-2 BPE via tiktoken).

Mirrors enough of InferenceEngine's surface that chat_ui/app.py can
route to it transparently: `loaded`, `ckpt_meta`, `set_system_prompt`,
`generate_stream`. Pretraining checkpoints have no chat tokens, so the
prompt is fed as raw text completion (system prompt prepended if set).
"""

from __future__ import annotations

import importlib.util
import math
import os
import sys
import threading
import time
from dataclasses import dataclass

import torch
import torch.nn.functional as F

_HERE = os.path.dirname(os.path.abspath(__file__))
_PROJ = os.path.dirname(_HERE)

# Load model/havoc.py by absolute path. We can't `from model.havoc import ...`
# because chat_ui/app.py puts scripts/ at the front of sys.path, where
# scripts/model.py shadows the model/ package.
_HAVOC_PATH = os.path.join(_PROJ, "model", "havoc.py")
_MOD_NAME   = "havoc_nanogpt_module"
_spec = importlib.util.spec_from_file_location(_MOD_NAME, _HAVOC_PATH)
_havoc_mod = importlib.util.module_from_spec(_spec)
sys.modules[_MOD_NAME] = _havoc_mod   # required for @dataclass to resolve the class's module
_spec.loader.exec_module(_havoc_mod)
HavocConfig = _havoc_mod.HavocConfig
HavocGPT    = _havoc_mod.HavocGPT


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


def _top_k_filter(logits: torch.Tensor, k: int) -> torch.Tensor:
    if k <= 0:
        return logits
    k = min(k, logits.size(-1))
    v, _ = torch.topk(logits, k)
    return logits.masked_fill(logits < v[:, -1:], -float("inf"))


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


def looks_like_nanogpt_ckpt(ck: dict) -> bool:
    cfg = ck.get("cfg") or {}
    return "block_size" in cfg and "n_embd" in cfg


class NanoGPTEngine:
    """Loads a nanoGPT-style HAVOC checkpoint and streams generated text."""

    def __init__(self) -> None:
        self.model:     HavocGPT | None     = None
        self.tokenizer                       = None
        self.cfg:       HavocConfig | None   = None
        self.device:    str                  = "cpu"
        self.ckpt_meta: dict                 = {}
        self.loaded:    bool                 = False
        self._system_prompt: str             = ""
        self.phase:     str                  = "pretrain"   # "pretrain" | "sft"

    def set_system_prompt(self, text: str) -> None:
        self._system_prompt = (text or "").strip()

    def get_system_prompt(self) -> str:
        return self._system_prompt

    def load_model(self, ckpt_path: str, tokenizer_dir: str | None = None) -> dict:
        import tiktoken

        device = "cuda" if torch.cuda.is_available() else "cpu"
        ck = torch.load(ckpt_path, map_location=device, weights_only=False)

        valid = {f.name for f in HavocConfig.__dataclass_fields__.values()}
        cfg_kwargs = {k: v for k, v in ck.get("cfg", {}).items() if k in valid}
        cfg = HavocConfig(**cfg_kwargs)
        cfg.dropout = 0.0

        model = HavocGPT(cfg).to(device)
        model.eval()
        sd = {k.replace("_orig_mod.", ""): v for k, v in ck["model"].items()}
        missing, unexpected = model.load_state_dict(sd, strict=False)
        if missing:
            print(f"  [nanogpt] Missing keys  : {missing[:5]}")
        if unexpected:
            print(f"  [nanogpt] Unexpected keys: {unexpected[:5]}")

        self.tokenizer = tiktoken.get_encoding("gpt2")
        self.model     = model
        self.cfg       = cfg
        self.device    = device
        self.loaded    = True
        self.phase     = ck.get("phase") or "pretrain"

        n_params = sum(p.numel() for p in model.parameters())
        val_loss = ck.get("val_loss")
        if isinstance(val_loss, float) and math.isnan(val_loss):
            val_loss = None
        self.ckpt_meta = {
            "path":        ckpt_path,
            "device":      device,
            "n_params":    n_params,
            "step":        ck.get("step"),
            "tokens_seen": ck.get("tokens_seen"),
            "val_loss":    val_loss,
            "vocab_size":  cfg.vocab_size,
            "max_seq_len": cfg.block_size,
            "num_layers":  cfg.n_layer,
            "num_heads":   cfg.n_head,
            "hidden_size": cfg.n_embd,
            "arch":        "nanogpt",
            "phase":       self.phase,
        }
        return self.ckpt_meta

    def generate_stream(
        self,
        prompt:             str,
        max_new_tokens:     int   = 256,
        temperature:        float = 0.8,
        top_k:              int   = 40,
        top_p:              float = 0.9,
        repetition_penalty: float = 1.0,
        sampling_mode:      str   = "top_kp",
        stop_event:         threading.Event | None = None,
        wrap_chat:          bool  = True,   # ignored — pretrain ckpt has no chat tokens
        cot:                bool  = False,  # ignored
    ):
        """Yields `(token_text, is_done, GenStats)` triples, matching InferenceEngine."""
        if not self.loaded or self.model is None or self.tokenizer is None:
            yield ("", True, GenStats(status="error"))
            return

        device = self.device
        cfg    = self.cfg
        stats  = GenStats()
        t_start = time.perf_counter()

        # SFT checkpoints were trained on "User: {p}\n\nAssistant: {c}<|endoftext|>"
        # — wrap user input the same way so the model recognizes the format.
        # Pretrain checkpoints keep raw-completion behavior.
        if self.phase == "sft" and wrap_chat:
            text = ""
            if self._system_prompt:
                text += self._system_prompt.rstrip() + "\n\n"
            text += f"User: {prompt}\n\nAssistant: "
        else:
            text = ""
            if self._system_prompt:
                text += self._system_prompt.rstrip() + "\n\n"
            text += prompt

        eot_id = self.tokenizer.eot_token  # GPT-2: 50256
        ids = self.tokenizer.encode(text, disallowed_special=())
        if not ids:
            ids = [eot_id]
        idx = torch.tensor([ids], dtype=torch.long, device=device)

        amp_dtype = torch.bfloat16 if (device == "cuda" and torch.cuda.is_bf16_supported()) else torch.float16
        use_amp = (device == "cuda")

        emitted_ids:  list[int] = []
        emitted_text: str       = ""

        with torch.no_grad():
            for _ in range(max_new_tokens):
                if stop_event is not None and stop_event.is_set():
                    stats.update(len(emitted_ids), t_start, "stopped")
                    yield ("", True, stats)
                    return

                ctx = idx[:, -cfg.block_size:]
                with torch.amp.autocast(device, dtype=amp_dtype, enabled=use_amp):
                    logits, _ = self.model(ctx)
                logits = logits[:, -1, :].float()

                if sampling_mode == "greedy" or temperature <= 0:
                    next_id = logits.argmax(dim=-1, keepdim=True)
                else:
                    logits = logits / max(temperature, 1e-8)
                    if sampling_mode in ("top_k", "top_kp") and top_k > 0:
                        logits = _top_k_filter(logits, top_k)
                    if sampling_mode in ("top_p", "top_kp") and top_p < 1.0:
                        logits = _top_p_filter(logits, top_p)
                    probs = F.softmax(logits, dim=-1)
                    if not torch.isfinite(probs).all() or probs.sum() < 1e-8:
                        probs = torch.ones_like(probs) / probs.size(-1)
                    next_id = torch.multinomial(probs, num_samples=1)

                tok_id = int(next_id.item())
                if tok_id == eot_id:
                    stats.update(len(emitted_ids), t_start, "eot")
                    yield ("", True, stats)
                    return

                emitted_ids.append(tok_id)
                idx = torch.cat([idx, next_id], dim=1)

                # tiktoken handles multi-byte tokens correctly only on the full
                # id list — decode the prefix and emit the new suffix delta.
                full_text = self.tokenizer.decode(emitted_ids)
                delta = full_text[len(emitted_text):]
                emitted_text = full_text

                stats.update(len(emitted_ids), t_start, "running")
                if delta:
                    yield (delta, False, stats)

        stats.update(len(emitted_ids), t_start, "max_tokens")
        yield ("", True, stats)
