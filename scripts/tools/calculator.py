"""
Safe arithmetic evaluator. Parses with `ast` and walks the tree allowing
only numeric literals and a fixed whitelist of operators / functions.
Never calls eval() / exec().
"""

from __future__ import annotations

import ast
import math
import operator as op

from .router import ToolResult, ToolError


_BIN_OPS = {
    ast.Add:      op.add,
    ast.Sub:      op.sub,
    ast.Mult:     op.mul,
    ast.Div:      op.truediv,
    ast.FloorDiv: op.floordiv,
    ast.Mod:      op.mod,
    ast.Pow:      op.pow,
}
_UNARY_OPS = {
    ast.UAdd: op.pos,
    ast.USub: op.neg,
}
_FUNCS = {
    "sqrt":  math.sqrt,
    "log":   math.log,
    "log10": math.log10,
    "exp":   math.exp,
    "sin":   math.sin,
    "cos":   math.cos,
    "tan":   math.tan,
    "asin":  math.asin,
    "acos":  math.acos,
    "atan":  math.atan,
    "abs":   abs,
    "min":   min,
    "max":   max,
    "round": round,
    "floor": math.floor,
    "ceil":  math.ceil,
}
_CONSTS = {
    "pi": math.pi,
    "e":  math.e,
}


def _eval(node):
    if isinstance(node, ast.Expression):
        return _eval(node.body)
    if isinstance(node, ast.Constant) and isinstance(node.value, (int, float)):
        return node.value
    if isinstance(node, ast.Name):
        if node.id in _CONSTS:
            return _CONSTS[node.id]
        raise ValueError(f"name {node.id!r} not allowed")
    if isinstance(node, ast.BinOp):
        if type(node.op) not in _BIN_OPS:
            raise ValueError(f"operator {type(node.op).__name__} not allowed")
        return _BIN_OPS[type(node.op)](_eval(node.left), _eval(node.right))
    if isinstance(node, ast.UnaryOp):
        if type(node.op) not in _UNARY_OPS:
            raise ValueError(f"unary op {type(node.op).__name__} not allowed")
        return _UNARY_OPS[type(node.op)](_eval(node.operand))
    if isinstance(node, ast.Call):
        if not isinstance(node.func, ast.Name) or node.func.id not in _FUNCS:
            raise ValueError("only whitelisted functions can be called")
        return _FUNCS[node.func.id](*(_eval(a) for a in node.args))
    raise ValueError(f"node {type(node).__name__} not allowed")


def calculator(args: str) -> ToolResult:
    expr = (args or "").strip()
    if not expr:
        return ToolError(name="calc", args=args, output="empty expression")
    try:
        tree = ast.parse(expr, mode="eval")
        value = _eval(tree)
    except Exception as exc:
        return ToolError(name="calc", args=args, output=f"{type(exc).__name__}: {exc}")
    return ToolResult(name="calc", args=args, output=str(value), meta={"value": value})
