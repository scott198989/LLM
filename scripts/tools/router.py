"""
ToolRouter — turns a tool name + argument string into a ToolResult.

Tool functions all match the same simple signature:
    fn(args: str) -> ToolResult

`args` is whatever string the model produced inside the tool tag /
JSON call. Each tool parses its own arguments. The router catches any
exception and packages it as a ToolError (so a buggy tool can never
take down the orchestrator).
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Callable


@dataclass
class ToolResult:
    name:    str
    args:    str
    output:  str
    ok:      bool = True
    meta:    dict = field(default_factory=dict)

    def render(self) -> str:
        head = f"[{self.name}({self.args!r}) -> {'OK' if self.ok else 'ERR'}]"
        return f"{head}\n{self.output}"


@dataclass
class ToolError(ToolResult):
    ok: bool = False


class ToolRouter:
    """
    Simple keyed dispatch with descriptions for prompt embedding.

    Build via `ToolRouter.default()` for the standard HAVOC tool set.
    """

    def __init__(self):
        self._tools: dict[str, Callable[[str], ToolResult]] = {}
        self._descs: dict[str, str] = {}

    # ── registration ─────────────────────────────────────────────────────

    def register(self, name: str, fn: Callable[[str], ToolResult], description: str) -> None:
        self._tools[name] = fn
        self._descs[name] = description

    def names(self) -> list[str]:
        return list(self._tools.keys())

    def describe(self) -> str:
        """Human-readable list of tools, suitable for prompt context."""
        lines = []
        for n in self.names():
            lines.append(f"  - {n}: {self._descs[n]}")
        return "\n".join(lines)

    # ── dispatch ─────────────────────────────────────────────────────────

    def call(self, name: str, args: str) -> ToolResult:
        if name not in self._tools:
            return ToolError(name=name, args=args,
                             output=f"unknown tool {name!r}; available: {', '.join(self.names())}")
        try:
            return self._tools[name](args)
        except Exception as exc:
            return ToolError(name=name, args=args, output=f"{type(exc).__name__}: {exc}")

    # ── default tool set ─────────────────────────────────────────────────

    @classmethod
    def default(cls) -> "ToolRouter":
        from .calculator     import calculator
        from .python_exec    import python_exec
        from .json_validator import json_validator
        from .file_reader    import file_reader
        from .text_parser    import text_parser
        from .unit_converter import unit_converter
        from .regex_utility  import regex_utility

        r = cls()
        r.register("calc",           calculator,
                   "evaluate an arithmetic expression (e.g. '12.7 * 84').")
        r.register("python",         python_exec,
                   "run a short Python snippet in a sandbox; the last expression is returned.")
        r.register("json",           json_validator,
                   "validate JSON; if a schema is given as 'data || schema', validate against it.")
        r.register("read_file",      file_reader,
                   "read a UTF-8 text file (paths confined to the project root).")
        r.register("parse",          text_parser,
                   "parse text: 'sentences:<text>', 'words:<text>', or 'numbers:<text>'.")
        r.register("convert",        unit_converter,
                   "convert units: '5 km in mi', '212 F in C', '3 hours in seconds'.")
        r.register("regex",          regex_utility,
                   "regex match: '<pattern> || <text>' returns all matches.")
        return r
