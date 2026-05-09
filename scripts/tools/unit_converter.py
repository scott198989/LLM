"""
unit_converter: '5 km in mi', '212 F in C', '90 minutes in seconds'.

Uses `pint` if installed. Falls back to a small built-in lookup table
covering common length / mass / temperature / time / speed / volume
conversions so simple cases work without any extra dependency.
"""

from __future__ import annotations

import re

from .router import ToolResult, ToolError


_RX = re.compile(r"^\s*([-+]?\d+(?:\.\d+)?(?:[eE][-+]?\d+)?)\s*([a-zA-Z]+)\s+(?:in|to)\s+([a-zA-Z]+)\s*$")

# Linear conversions (factor + offset relative to canonical SI unit)
# canonical: m, kg, s, K, m/s, L
_TABLES = {
    # length -> meters
    "length": {
        "m":  (1.0,        0.0), "km": (1000.0,        0.0),
        "cm": (0.01,       0.0), "mm": (0.001,         0.0),
        "in": (0.0254,     0.0), "ft": (0.3048,        0.0),
        "yd": (0.9144,     0.0), "mi": (1609.344,      0.0),
    },
    # mass -> kilograms
    "mass": {
        "kg": (1.0,         0.0), "g":  (0.001,        0.0),
        "lb": (0.45359237,  0.0), "oz": (0.0283495,    0.0),
        "t":  (1000.0,      0.0),
    },
    # time -> seconds
    "time": {
        "s":   (1.0,    0.0), "ms":   (0.001, 0.0),
        "min": (60.0,   0.0), "hour": (3600.0, 0.0), "h": (3600.0, 0.0),
        "day": (86400.0,0.0), "week": (604800.0, 0.0),
    },
    # temperature -> Kelvin (factor*x + offset)
    "temp": {
        "K":  (1.0,   0.0),
        "C":  (1.0,   273.15),
        "F":  (5/9,   (273.15 - 32 * 5/9)),
        "R":  (5/9,   0.0),
    },
    # speed -> m/s
    "speed": {
        "m/s": (1.0, 0.0), "kph": (1000/3600, 0.0), "mph": (1609.344/3600, 0.0),
    },
    # volume -> liters
    "volume": {
        "l":   (1.0, 0.0), "ml":  (0.001, 0.0),
        "gal": (3.78541, 0.0), "qt": (0.946353, 0.0),
        "pt":  (0.473176, 0.0), "cup": (0.24, 0.0),
    },
}


def _lookup(unit: str) -> tuple[str, tuple[float, float]] | None:
    u = unit.lower()
    aliases = {
        "meter": "m", "meters": "m", "metre": "m", "metres": "m",
        "kilometer": "km", "kilometers": "km",
        "millimeter": "mm", "millimeters": "mm",
        "centimeter": "cm", "centimeters": "cm",
        "inch": "in", "inches": "in", "feet": "ft", "foot": "ft",
        "yard": "yd", "yards": "yd", "mile": "mi", "miles": "mi",
        "kilogram": "kg", "kilograms": "kg", "gram": "g", "grams": "g",
        "pound": "lb", "pounds": "lb", "ounce": "oz", "ounces": "oz",
        "ton": "t", "tons": "t", "tonne": "t", "tonnes": "t",
        "second": "s", "seconds": "s", "millisecond": "ms",
        "minute": "min", "minutes": "min", "mins": "min",
        "hour": "hour", "hours": "hour", "hr": "hour", "hrs": "hour",
        "days": "day", "weeks": "week",
        "kelvin": "K", "celsius": "C", "fahrenheit": "F", "rankine": "R",
        "k": "K", "c": "C", "f": "F", "r": "R",
        "kilometers per hour": "kph", "miles per hour": "mph",
        "liter": "l", "liters": "l", "litre": "l", "litres": "l",
        "milliliter": "ml", "milliliters": "ml",
        "gallon": "gal", "gallons": "gal", "quart": "qt", "quarts": "qt",
        "pint": "pt", "pints": "pt", "cups": "cup",
    }
    u = aliases.get(u, u)
    for cat, table in _TABLES.items():
        # match preserving original case for temperature K/C/F/R
        for k, v in table.items():
            if k.lower() == u:
                return cat, v
    return None


def _try_pint(amount: float, src: str, dst: str) -> tuple[bool, str]:
    try:
        import pint
        ureg = pint.UnitRegistry()
        v = (amount * ureg(src)).to(dst)
        return True, f"{amount} {src} = {v.magnitude:.6g} {dst}"
    except Exception as exc:
        return False, str(exc)


def unit_converter(args: str) -> ToolResult:
    text = (args or "").strip()
    m = _RX.match(text)
    if not m:
        return ToolError(name="convert", args=args,
                         output="expected '<number> <from-unit> in <to-unit>'")

    amount   = float(m.group(1))
    src_unit = m.group(2)
    dst_unit = m.group(3)

    # Try pint first (better)
    ok, out = _try_pint(amount, src_unit, dst_unit)
    if ok:
        return ToolResult(name="convert", args=args, output=out)

    # Fallback: built-in tables
    src = _lookup(src_unit)
    dst = _lookup(dst_unit)
    if not src or not dst:
        return ToolError(name="convert", args=args,
                         output=f"unknown unit(s): {src_unit}, {dst_unit}")
    if src[0] != dst[0]:
        return ToolError(name="convert", args=args,
                         output=f"incompatible categories: {src[0]} vs {dst[0]}")
    cat = src[0]
    sf, so = src[1]
    df, do = dst[1]
    canonical = amount * sf + so
    converted = (canonical - do) / df
    return ToolResult(name="convert", args=args,
                      output=f"{amount} {src_unit} = {converted:.6g} {dst_unit}",
                      meta={"category": cat, "value": converted})
