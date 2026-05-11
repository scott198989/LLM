"""
file_reader: read a UTF-8 text file. Paths are confined to the project
root (the directory containing the `scripts/` folder) to prevent the
model from poking at /etc/passwd.

Args: a single path string. Optionally a 'path:max_chars' form to cap
the read size (default 8192).
"""

from __future__ import annotations

import os

from .router import ToolResult, ToolError


_PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


def _is_within(path: str, root: str) -> bool:
    try:
        rp = os.path.realpath(path)
        rr = os.path.realpath(root)
        return os.path.commonpath([rp, rr]) == rr
    except ValueError:
        return False


def file_reader(args: str) -> ToolResult:
    text = (args or "").strip()
    if not text:
        return ToolError(name="read_file", args=args, output="empty path")

    max_chars = 8192
    if ":" in text and text.rsplit(":", 1)[-1].isdigit():
        path, n = text.rsplit(":", 1)
        path = path.strip()
        max_chars = int(n)
    else:
        path = text

    full = path if os.path.isabs(path) else os.path.join(_PROJECT_ROOT, path)
    if not _is_within(full, _PROJECT_ROOT):
        return ToolError(name="read_file", args=args,
                         output=f"path outside project root: {path}")
    if not os.path.isfile(full):
        return ToolError(name="read_file", args=args, output=f"not a file: {path}")

    try:
        with open(full, encoding="utf-8", errors="replace") as f:
            content = f.read(max_chars + 1)
    except OSError as exc:
        return ToolError(name="read_file", args=args, output=f"OSError: {exc}")

    truncated = len(content) > max_chars
    if truncated:
        content = content[:max_chars]
    rel = os.path.relpath(full, _PROJECT_ROOT)
    suffix = f"\n... [truncated at {max_chars} chars]" if truncated else ""
    return ToolResult(name="read_file", args=args,
                      output=f"# {rel}\n{content}{suffix}",
                      meta={"path": rel, "truncated": truncated})
