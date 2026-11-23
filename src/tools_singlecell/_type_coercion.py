from __future__ import annotations

import json
from typing import Any, Iterable, List, Optional


def _parse_list_string(s: str):
    """Parse a JSON / Python list literal like '["CD3E","CD4"]' or "['a', 'b']".
    Returns a list, or None if `s` is not a bracketed list literal."""
    s = s.strip()
    if not (s.startswith("[") and s.endswith("]")):
        return None
    try:
        val = json.loads(s)
    except ValueError:
        import ast
        try:
            val = ast.literal_eval(s)
        except (ValueError, SyntaxError):
            return None
    return list(val) if isinstance(val, (list, tuple)) else None


def _is_none_like(v: Any) -> bool:
    return v is None or (isinstance(v, str) and v.strip().lower() in {"none", "null", ""})


def coerce_int(v: Any, *, default: Optional[int] = None) -> Optional[int]:
    if _is_none_like(v):
        return default
    if isinstance(v, bool):
        return int(v)
    if isinstance(v, int):
        return v
    f = float(v)  # handles "3", "3.0"
    if not f.is_integer():
        raise ValueError(f"Expected an integer, got {v!r} (non-integral values are not rounded)")
    return int(f)


def coerce_float(v: Any, *, default: Optional[float] = None) -> Optional[float]:
    if _is_none_like(v):
        return default
    if isinstance(v, bool):
        return float(int(v))
    if isinstance(v, (int, float)):
        return float(v)
    return float(v)


def coerce_bool(v: Any, *, default: Optional[bool] = None) -> Optional[bool]:
    if _is_none_like(v):
        return default
    if isinstance(v, bool):
        return v
    if isinstance(v, (int, float)):
        return bool(v)
    if isinstance(v, str):
        s = v.strip().lower()
        if s in {"true", "t", "yes", "y", "1"}:
            return True
        if s in {"false", "f", "no", "n", "0"}:
            return False
    raise ValueError(f"Cannot coerce to bool: {v!r}")


def coerce_str(v: Any, *, default: Optional[str] = None) -> Optional[str]:
    if _is_none_like(v):
        return default
    if isinstance(v, str):
        return v
    return str(v)


def coerce_int_list(v: Any, *, default: Optional[List[int]] = None) -> Optional[List[int]]:
    if _is_none_like(v):
        return default
    if isinstance(v, (list, tuple)):
        return [coerce_int(x) for x in v]
    if isinstance(v, str):
        s = v.strip()
        if not s:
            return default
        parsed = _parse_list_string(s)
        if parsed is not None:
            return [coerce_int(x) for x in parsed]
        # allow "1,2,3"
        parts = [p.strip() for p in s.split(",") if p.strip()]
        return [coerce_int(p) for p in parts]
    if isinstance(v, Iterable):
        return [coerce_int(x) for x in v]
    return [coerce_int(v)]


def coerce_str_list(v: Any, *, default: Optional[List[str]] = None) -> Optional[List[str]]:
    if _is_none_like(v):
        return default
    if isinstance(v, list):
        return [str(x) for x in v]
    if isinstance(v, tuple):
        return [str(x) for x in list(v)]
    if isinstance(v, str):
        s = v.strip()
        if not s:
            return default
        parsed = _parse_list_string(s)
        if parsed is not None:
            return [str(x) for x in parsed]
        parts = [p.strip() for p in s.split(",") if p.strip()]
        return parts
    if isinstance(v, Iterable):
        return [str(x) for x in v]
    return [str(v)]

