"""
Deterministic verifier for orchestrator outputs.

NOT an LLM-based check. Each verification routine is a pure function
returning (ok, message). Routines cover the cases where the model
might produce a plausible but factually wrong output:

  - JSON validity (does it parse? does it match a schema?)
  - Numeric consistency (re-evaluates arithmetic claimed in the answer)
  - Tool output type checks
  - Regex / format matching
  - Length / range bounds

The orchestrator calls these after a tool runs (to confirm the tool's
own output is well-formed) and after a generation (to flag claims that
don't match retrieved evidence or tool results).
"""

from __future__ import annotations

import json
import re

from tools.calculator import calculator


# ── single-call verification primitives ──────────────────────────────────


def is_valid_json(text: str) -> tuple[bool, str]:
    try:
        json.loads(text)
        return True, "valid JSON"
    except json.JSONDecodeError as exc:
        return False, f"invalid JSON: {exc}"


def matches_schema(text: str, schema: dict) -> tuple[bool, str]:
    try:
        from jsonschema import validate, ValidationError
    except ImportError:
        return False, "jsonschema not installed"
    try:
        data = json.loads(text)
        validate(instance=data, schema=schema)
        return True, "matches schema"
    except (json.JSONDecodeError, ValidationError) as exc:
        return False, str(exc)


def matches_regex(text: str, pattern: str) -> tuple[bool, str]:
    try:
        rx = re.compile(pattern)
    except re.error as exc:
        return False, f"bad pattern: {exc}"
    return (bool(rx.search(text)), "regex match" if rx.search(text) else "no regex match")


def in_range(value: float, lo: float, hi: float) -> tuple[bool, str]:
    ok = lo <= value <= hi
    return ok, f"{value} {'in' if ok else 'NOT in'} [{lo}, {hi}]"


def length_within(text: str, max_chars: int) -> tuple[bool, str]:
    n = len(text)
    return (n <= max_chars, f"{n} <= {max_chars}" if n <= max_chars
            else f"{n} > {max_chars}")


# ── numeric consistency ───────────────────────────────────────────────────


_EQ_RX = re.compile(r"([\-\+\*\/\d\.\s\(\)\^]+?)\s*=\s*([\-\+]?\d+(?:\.\d+)?)")


def numeric_consistency(answer_text: str, tolerance: float = 1.0e-3) -> tuple[bool, str]:
    """
    Walk the answer text for "<expr> = <value>" claims and re-evaluate each
    expression with the safe calculator. Returns (False, message) on the
    FIRST mismatch found, (True, "...") if every check passes (or none found).
    """
    found = 0
    for m in _EQ_RX.finditer(answer_text):
        expr = m.group(1).strip().replace("^", "**")
        try:
            claimed = float(m.group(2))
        except ValueError:
            continue
        result = calculator(expr)
        if not result.ok:
            continue
        try:
            actual = float(result.output)
        except ValueError:
            continue
        found += 1
        if abs(actual - claimed) > tolerance:
            return False, f"claimed {expr} = {claimed} but evaluates to {actual}"
    return True, (f"{found} numeric claim(s) verified" if found
                  else "no numeric claims to verify")


# ── single-batch runner ──────────────────────────────────────────────────


def run_checks(answer_text: str,
               *,
               require_json:   bool          = False,
               json_schema:    dict | None   = None,
               regex_required: str  | None   = None,
               max_chars:      int  | None   = None,
               check_numeric:  bool          = True,
               ) -> dict:
    """Run the requested verifications. Returns a dict with `passed`/`failures`."""
    failures: list[tuple[str, str]] = []

    if require_json:
        ok, msg = is_valid_json(answer_text)
        if not ok:
            failures.append(("json", msg))

    if json_schema is not None:
        ok, msg = matches_schema(answer_text, json_schema)
        if not ok:
            failures.append(("schema", msg))

    if regex_required:
        ok, msg = matches_regex(answer_text, regex_required)
        if not ok:
            failures.append(("regex", msg))

    if max_chars is not None:
        ok, msg = length_within(answer_text, max_chars)
        if not ok:
            failures.append(("length", msg))

    if check_numeric:
        ok, msg = numeric_consistency(answer_text)
        if not ok:
            failures.append(("numeric", msg))

    return {
        "passed":   not failures,
        "failures": failures,
    }
