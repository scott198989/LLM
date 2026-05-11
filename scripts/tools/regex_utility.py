"""
regex_utility: '<pattern> || <text>' returns up to 50 matches.
Optional flags via '<pattern> ||i<text>' (i=ignore-case, m=multiline, s=dotall).
"""

from __future__ import annotations

import re

from .router import ToolResult, ToolError


def regex_utility(args: str) -> ToolResult:
    if "||" not in args:
        return ToolError(name="regex", args=args,
                         output="expected '<pattern> || <text>'")
    pattern_part, text = (s.strip() for s in args.split("||", 1))
    flags = 0
    pattern = pattern_part
    # Optional flag suffix: `^pattern$|im`
    if "|" in pattern_part:
        head, tail = pattern_part.rsplit("|", 1)
        if all(c in "imsx" for c in tail):
            pattern = head
            for c in tail:
                flags |= {"i": re.I, "m": re.M, "s": re.S, "x": re.X}[c]
    try:
        rx = re.compile(pattern, flags)
    except re.error as exc:
        return ToolError(name="regex", args=args, output=f"bad regex: {exc}")
    matches = rx.findall(text)
    rendered = "\n".join(f"  {i+1}. {m!r}" for i, m in enumerate(matches[:50]))
    extra = f"\n  ... and {len(matches)-50} more" if len(matches) > 50 else ""
    return ToolResult(name="regex", args=args,
                      output=f"{len(matches)} matches\n{rendered}{extra}",
                      meta={"matches": matches})
