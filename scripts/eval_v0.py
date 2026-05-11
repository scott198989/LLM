"""HAVOC v0 desktop eval — runs the 10 spec prompts."""

from __future__ import annotations

import argparse
import os
import sys

_HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, _HERE)

from inference import InferenceEngine  # noqa: E402

PROMPTS = [
    "hello",
    "good morning",
    "What is a book?",
    "What is a cloud?",
    "What color are clouds?",
    "What is motion?",
    "Define density.",
    "What is Ohm's law?",
    "Describe voltage.",
    "Why is the sky blue?",
]

_SOFT_STOPS = ("\nUser:", "\n\nUser:", "User:", "\nuser:", "<|user|>")


def _clip(text: str) -> str:
    cuts = [text.find(s) for s in _SOFT_STOPS]
    cuts = [c for c in cuts if c > 0]
    return text[: min(cuts)].rstrip() if cuts else text.rstrip()


def main() -> int:
    p = argparse.ArgumentParser()
    p.add_argument("--ckpt",          required=True)
    p.add_argument("--tokenizer_dir", default="models/tokenizers/havoc_v0")
    p.add_argument("--system_prompt",
                   default="You are HAVOC, a helpful assistant. Reply briefly and clearly.")
    p.add_argument("--max_new_tokens", type=int,   default=80)
    p.add_argument("--temperature",    type=float, default=0.7)
    p.add_argument("--top_p",          type=float, default=0.9)
    p.add_argument("--top_k",          type=int,   default=40)
    p.add_argument("--rep_penalty",    type=float, default=1.1)
    args = p.parse_args()

    eng = InferenceEngine()
    meta = eng.load_model(args.ckpt, tokenizer_dir=args.tokenizer_dir)
    eng.set_system_prompt(args.system_prompt)
    print(f"\nLoaded {meta['n_params']:,} params from {meta['path']}")
    print(f"  layers={meta['num_layers']}  heads={meta['num_heads']}  "
          f"hidden={meta['hidden_size']}  ctx={meta['max_seq_len']}")
    print(f"  step={meta['step']}  val_loss={meta['val_loss']}")
    print(f"\nSampling: temp={args.temperature}  top_p={args.top_p}  "
          f"top_k={args.top_k}  rep_penalty={args.rep_penalty}  "
          f"max_new_tokens={args.max_new_tokens}")

    def _safe_print(s: str) -> None:
        try:
            sys.stdout.buffer.write((s + "\n").encode("utf-8", "replace"))
            sys.stdout.flush()
        except Exception:
            print(s.encode("ascii", "replace").decode("ascii"))

    _safe_print("\n" + "=" * 78)
    for i, prompt in enumerate(PROMPTS, 1):
        text = eng.generate(
            prompt              = prompt,
            max_new_tokens      = args.max_new_tokens,
            temperature         = args.temperature,
            top_p               = args.top_p,
            top_k               = args.top_k,
            repetition_penalty  = args.rep_penalty,
        )
        text = _clip(text)
        _safe_print(f"\n[{i}] User: {prompt}")
        _safe_print(f"    HAVOC: {text}")
    _safe_print("\n" + "=" * 78 + "\n")
    return 0


if __name__ == "__main__":
    sys.exit(main())
