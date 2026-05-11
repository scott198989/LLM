"""
text_parser: small text-extraction helpers.

Args format: 'mode:<text>'
    sentences:<text>   - split into sentences
    words:<text>       - whitespace tokens
    numbers:<text>     - extract numeric literals (ints, floats, percents)
    lines:<text>       - non-empty lines
"""

from __future__ import annotations

import re

from .router import ToolResult, ToolError


_SENT_RX = re.compile(r"(?<=[.!?])\s+(?=[A-Z])")
_NUM_RX  = re.compile(r"[-+]?\d*\.?\d+(?:[eE][-+]?\d+)?%?")


def text_parser(args: str) -> ToolResult:
    raw = args or ""
    if ":" not in raw:
        return ToolError(name="parse", args=args,
                         output="expected 'mode:text' (modes: sentences, words, numbers, lines)")
    mode, text = raw.split(":", 1)
    mode = mode.strip().lower()
    text = text.strip()

    if mode == "sentences":
        items = [s.strip() for s in _SENT_RX.split(text) if s.strip()]
    elif mode == "words":
        items = text.split()
    elif mode == "numbers":
        items = _NUM_RX.findall(text)
    elif mode == "lines":
        items = [l for l in text.splitlines() if l.strip()]
    else:
        return ToolError(name="parse", args=args, output=f"unknown mode: {mode}")

    body = "\n".join(f"  {i+1}. {it}" for i, it in enumerate(items[:50]))
    extra = f"\n  ... and {len(items)-50} more" if len(items) > 50 else ""
    return ToolResult(name="parse", args=args,
                      output=f"{len(items)} {mode}\n{body}{extra}",
                      meta={"items": items, "mode": mode})
