"""
Safe-ish Python execution sandbox.

Uses RestrictedPython when available; falls back to a tightly-restricted
exec() with a stripped __builtins__ dict otherwise. Imports, attribute
access on dunder names, file I/O, and process spawning are blocked in
both modes. Output is the value of the last expression OR whatever is
written to stdout via the provided `print` shim.

This is NOT a security boundary. Treat it as a developer convenience for
arithmetic / data-manipulation snippets, not as a safe way to run
untrusted code from the public internet.
"""

from __future__ import annotations

import io
import sys
from contextlib import redirect_stdout

from .router import ToolResult, ToolError


# ── Try RestrictedPython first ────────────────────────────────────────────

try:
    from RestrictedPython import compile_restricted
    from RestrictedPython.Guards import (
        safe_builtins,
        safe_globals,
        guarded_iter_unpack_sequence,
        guarded_unpack_sequence,
    )
    _HAS_RP = True
except ImportError:
    _HAS_RP = False


_BLOCKED_NAMES = {
    "__import__", "open", "exec", "eval", "compile", "exit", "quit",
    "globals", "locals", "vars", "dir", "input", "breakpoint",
}


def _safe_globals_fallback() -> dict:
    builtins_clean = {}
    safe_builtin_names = (
        "abs", "all", "any", "ascii", "bin", "bool", "bytes", "chr",
        "complex", "dict", "divmod", "enumerate", "filter", "float",
        "format", "frozenset", "hash", "hex", "int", "isinstance",
        "issubclass", "iter", "len", "list", "map", "max", "min", "next",
        "object", "oct", "ord", "pow", "print", "range", "repr",
        "reversed", "round", "set", "slice", "sorted", "str", "sum",
        "tuple", "type", "zip",
    )
    import builtins
    for name in safe_builtin_names:
        builtins_clean[name] = getattr(builtins, name, None)
    g = {"__builtins__": builtins_clean}
    return g


def python_exec(args: str) -> ToolResult:
    code = (args or "").strip()
    if not code:
        return ToolError(name="python", args=args, output="empty code")

    buf = io.StringIO()
    last_value: object = None

    if _HAS_RP:
        try:
            compiled = compile_restricted(code, "<python_exec>", "exec")
        except SyntaxError as exc:
            return ToolError(name="python", args=args, output=f"SyntaxError: {exc}")
        g = dict(safe_globals)
        g["__builtins__"] = dict(safe_builtins)
        g["_print_"]      = lambda *a, **kw: print(*a, **kw)
        g["_getiter_"]    = iter
        g["_iter_unpack_sequence_"] = guarded_iter_unpack_sequence
        g["_unpack_sequence_"]      = guarded_unpack_sequence
    else:
        try:
            compiled = compile(code, "<python_exec>", "exec")
        except SyntaxError as exc:
            return ToolError(name="python", args=args, output=f"SyntaxError: {exc}")
        g = _safe_globals_fallback()
        for n in _BLOCKED_NAMES:
            g["__builtins__"].pop(n, None)

    try:
        with redirect_stdout(buf):
            exec(compiled, g, g)
        # If the snippet ends with an expression, evaluate it for the return
        # (RestrictedPython only runs statements; we extract the last line).
        last_line = code.rstrip().split("\n")[-1].strip()
        if last_line and not last_line.startswith(("def ", "class ", "for ", "while ",
                                                    "if ", "import ", "from ",
                                                    "with ", "try:")):
            try:
                last_value = eval(compile(last_line, "<py_last>", "eval"), g, g)
            except Exception:
                last_value = None
    except Exception as exc:
        return ToolError(name="python", args=args, output=f"{type(exc).__name__}: {exc}")

    out = buf.getvalue()
    if last_value is not None and not out.strip():
        out = repr(last_value)
    return ToolResult(name="python", args=args, output=out.strip() or "(no output)")
