"""
JSON validator. Args formats:
    {"key": "value"}                         - parse as JSON, report ok / err
    {"a": 1} || {"type": "object", ...}      - parse data and validate against schema
"""

from __future__ import annotations

import json

from .router import ToolResult, ToolError

try:
    from jsonschema import validate, ValidationError
    _HAS_JSONSCHEMA = True
except ImportError:
    _HAS_JSONSCHEMA = False


def json_validator(args: str) -> ToolResult:
    text = (args or "").strip()
    if not text:
        return ToolError(name="json", args=args, output="empty input")

    if "||" in text:
        data_str, schema_str = (s.strip() for s in text.split("||", 1))
    else:
        data_str, schema_str = text, ""

    try:
        data = json.loads(data_str)
    except json.JSONDecodeError as exc:
        return ToolError(name="json", args=args, output=f"invalid JSON: {exc}")

    if schema_str:
        if not _HAS_JSONSCHEMA:
            return ToolError(name="json", args=args,
                             output="jsonschema package not installed; cannot validate")
        try:
            schema = json.loads(schema_str)
        except json.JSONDecodeError as exc:
            return ToolError(name="json", args=args, output=f"invalid schema JSON: {exc}")
        try:
            validate(instance=data, schema=schema)
        except ValidationError as exc:
            return ToolError(name="json", args=args, output=f"schema mismatch: {exc.message}")

    return ToolResult(name="json", args=args,
                      output=f"valid JSON, type={type(data).__name__}",
                      meta={"data": data})
