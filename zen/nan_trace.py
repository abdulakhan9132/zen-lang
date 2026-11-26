"""Runtime NaN / Inf tracer for `--nan-trace` scientific debugging."""

from __future__ import annotations

import math
from typing import Any

from zen.errors import ZenRuntimeError

ENABLED: bool = False
_CURRENT_LINE: int = 0
_CURRENT_COL: int = 1


def set_enabled(flag: bool) -> None:
    global ENABLED
    ENABLED = bool(flag)


def set_location(line: int, col: int = 1) -> None:
    global _CURRENT_LINE, _CURRENT_COL
    if line:
        _CURRENT_LINE = line
        _CURRENT_COL = col or 1


def _describe_shapes(left: Any, right: Any) -> str:
    def _s(v: Any) -> str:
        if hasattr(v, "shape"):
            return str(list(v.shape))
        if isinstance(v, list):
            return f"list[{len(v)}]"
        return type(v).__name__

    if right is None:
        return _s(left)
    return f"{_s(left)}, {_s(right)}"


def check_value(val: Any, op_name: str, left: Any = None, right: Any = None, line: int = 0, col: int = 0) -> None:
    """Raise E014 ZenRuntimeError if `val` contains NaN or Inf when `--nan-trace` is active."""
    if not ENABLED:
        return
    use_line = line or _CURRENT_LINE
    use_col = col or _CURRENT_COL
    if isinstance(val, float):
        if math.isnan(val) or math.isinf(val):
            kind = "NaN" if math.isnan(val) else "Inf"
            shapes = _describe_shapes(left if left is not None else val, right)
            raise ZenRuntimeError(
                f"{kind} produced by '{op_name}' (shapes: {shapes})",
                use_line,
                use_col,
                code="E014",
            )
        return
    if hasattr(val, "data") and hasattr(val, "shape"):
        data = val.data
        for x in data:
            if isinstance(x, float) and (math.isnan(x) or math.isinf(x)):
                kind = "NaN" if math.isnan(x) else "Inf"
                shapes = _describe_shapes(left if left is not None else val, right)
                raise ZenRuntimeError(
                    f"{kind} produced by '{op_name}' (out shape: {list(val.shape)}, inputs: {shapes})",
                    use_line,
                    use_col,
                    code="E014",
                )
