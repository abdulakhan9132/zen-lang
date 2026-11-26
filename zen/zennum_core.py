"""NumPy-like N-dimensional Array engine for ZEN (pure Python + optional native kernels)."""

from __future__ import annotations

from array import array
from functools import lru_cache
from itertools import repeat
import math
import operator
import random
from typing import Any, Callable

from zen.errors import ZenRuntimeError
from zen import native_kernels

VALID_DTYPES = ("float64", "float32", "int64", "bool")


@lru_cache(maxsize=512)
def _strides(shape: tuple[int, ...]) -> tuple[int, ...]:
    s = [1] * len(shape)
    acc = 1
    for i in range(len(shape) - 1, -1, -1):
        s[i] = acc
        acc *= shape[i]
    return tuple(s)


@lru_cache(maxsize=512)
def _numel(shape: tuple[int, ...]) -> int:
    return math.prod(shape)


@lru_cache(maxsize=512)
def _broadcast_shapes(s1: tuple[int, ...], s2: tuple[int, ...]) -> tuple[int, ...]:
    ndim = max(len(s1), len(s2))
    p1 = (1,) * (ndim - len(s1)) + s1
    p2 = (1,) * (ndim - len(s2)) + s2
    res: list[int] = []
    for d1, d2 in zip(p1, p2):
        if d1 == d2:
            res.append(d1)
        elif d1 == 1:
            res.append(d2)
        elif d2 == 1:
            res.append(d1)
        else:
            raise ZenRuntimeError(f"Shape mismatch: cannot broadcast shapes {list(s1)} and {list(s2)}")
    return tuple(res)


def _normalize_shape(shape: Any) -> tuple[int, ...]:
    if isinstance(shape, int) and not isinstance(shape, bool):
        raw = [shape]
    elif isinstance(shape, (list, tuple)):
        raw = list(shape)
    else:
        raise ZenRuntimeError("Shape must be an integer or list of integers")
    out: list[int] = []
    for d in raw:
        if isinstance(d, bool) or not isinstance(d, int) or d < 0:
            raise ZenRuntimeError(f"Invalid dimension size: {d!r}")
        out.append(d)
    return tuple(out)


def _make_buffer(values: Any, dtype: str) -> array | list[Any]:
    if dtype not in VALID_DTYPES:
        raise ZenRuntimeError(f"Unsupported dtype '{dtype}'; expected one of {VALID_DTYPES}")
    if dtype == "float64":
        return array("d", (float(v) for v in values))
    if dtype == "float32":
        return array("f", (float(v) for v in values))
    if dtype == "int64":
        return [int(v) for v in values]
    return [bool(v) for v in values]


def _zeros_buffer(size: int, dtype: str) -> array | list[Any]:
    if dtype == "float64":
        return array("d", [0.0]) * size
    if dtype == "float32":
        return array("f", [0.0]) * size
    if dtype == "int64":
        return [0] * size
    if dtype == "bool":
        return [False] * size
    raise ZenRuntimeError(f"Unsupported dtype '{dtype}'")


def _promote_dtype(d1: str, d2: str) -> str:
    if d1 == "float64" or d2 == "float64":
        return "float64"
    if d1 == "float32" or d2 == "float32":
        return "float32"
    if d1 == "int64" or d2 == "int64":
        return "int64"
    return "bool"


def _infer_nested(data: Any) -> tuple[list[Any], tuple[int, ...], str]:
    if isinstance(data, bool):
        return [data], (), "bool"
    if isinstance(data, int):
        return [data], (), "int64"
    if isinstance(data, float):
        return [data], (), "float64"
    if not isinstance(data, list):
        raise ZenRuntimeError("Array data must be a number, bool, or nested list")
    if len(data) == 0:
        return [], (0,), "float64"
    if isinstance(data[0], list):
        sub_shape: tuple[int, ...] | None = None
        flat: list[Any] = []
        dtype = "bool"
        for item in data:
            sf, ss, sd = _infer_nested(item)
            if sub_shape is None:
                sub_shape = ss
            elif ss != sub_shape:
                raise ZenRuntimeError(f"Shape mismatch: ragged nested lists ({list(sub_shape)} vs {list(ss)})")
            flat.extend(sf)
            dtype = _promote_dtype(dtype, sd)
        assert sub_shape is not None
        return flat, (len(data),) + sub_shape, dtype
    flat = []
    dtype = "bool"
    for item in data:
        if isinstance(item, bool):
            flat.append(item)
        elif isinstance(item, int):
            flat.append(item)
            dtype = _promote_dtype(dtype, "int64")
        elif isinstance(item, float):
            flat.append(item)
            dtype = _promote_dtype(dtype, "float64")
        else:
            raise ZenRuntimeError("Array elements must be numbers or booleans")
    return flat, (len(data),), dtype


class ZenArray:
    """NumPy-like N-dimensional array backed by flat typed buffers."""

    def __init__(self, data: array | list[Any], shape: tuple[int, ...], dtype: str = "float64") -> None:
        self.data = data
        self.shape = shape
        self.dtype = dtype

    @property
    def ndim(self) -> int:
        return len(self.shape)

    @property
    def size(self) -> int:
        return len(self.data)

    @classmethod
    def from_list(cls, raw: Any, dtype: str | None = None) -> "ZenArray":
        if isinstance(raw, ZenArray):
            dt = dtype or raw.dtype
            return cls(_make_buffer(raw.data, dt), raw.shape, dt)
        # Support ZenTensor duck-typed without circular import
        if hasattr(raw, "data") and hasattr(raw, "shape") and not isinstance(raw, (list, tuple)):
            dt = dtype or "float64"
            return cls(_make_buffer(raw.data, dt), tuple(raw.shape), dt)
        flat, shape, inferred = _infer_nested(raw)
        if dtype is None and inferred == "int64":
            dt = "int64"
        else:
            dt = dtype or (inferred if inferred != "bool" else "bool")
        return cls(_make_buffer(flat, dt), shape, dt)

    @classmethod
    def zeros(cls, shape: Any, dtype: str = "float64") -> "ZenArray":
        s = _normalize_shape(shape)
        return cls(_zeros_buffer(_numel(s), dtype), s, dtype)

    @classmethod
    def ones(cls, shape: Any, dtype: str = "float64") -> "ZenArray":
        s = _normalize_shape(shape)
        val = 1.0 if dtype in ("float64", "float32") else (1 if dtype == "int64" else True)
        return cls(_make_buffer(repeat(val, _numel(s)), dtype), s, dtype)

    @classmethod
    def full(cls, shape: Any, fill_value: Any, dtype: str | None = None) -> "ZenArray":
        s = _normalize_shape(shape)
        if dtype is None:
            dtype = "bool" if isinstance(fill_value, bool) else ("int64" if isinstance(fill_value, int) else "float64")
        return cls(_make_buffer(repeat(fill_value, _numel(s)), dtype), s, dtype)

    @classmethod
    def arange(cls, start: Any, stop: Any = None, step: Any = 1, dtype: str | None = None) -> "ZenArray":
        if stop is None:
            start_v, stop_v = 0, start
        else:
            start_v, stop_v = start, stop
        if step == 0:
            raise ZenRuntimeError("arange step cannot be 0")
        if dtype is None:
            is_f = any(isinstance(x, float) for x in (start_v, stop_v, step))
            dtype = "float64" if is_f else "int64"
        vals: list[Any] = []
        cur = start_v
        if step > 0:
            while cur < stop_v:
                vals.append(cur)
                cur += step
        else:
            while cur > stop_v:
                vals.append(cur)
                cur += step
        return cls(_make_buffer(vals, dtype), (len(vals),), dtype)

    @classmethod
    def linspace(cls, start: float, stop: float, num: int = 50, dtype: str = "float64") -> "ZenArray":
        if num < 0:
            raise ZenRuntimeError("linspace num must be >= 0")
        if num == 0:
            return cls(_zeros_buffer(0, dtype), (0,), dtype)
        if num == 1:
            return cls(_make_buffer([float(start)], dtype), (1,), dtype)
        step = (float(stop) - float(start)) / (num - 1)
        vals = [float(start) + i * step for i in range(num)]
        vals[-1] = float(stop)
        return cls(_make_buffer(vals, dtype), (num,), dtype)

    @classmethod
    def eye(cls, n: int, m: int | None = None, dtype: str = "float64") -> "ZenArray":
        cols = n if m is None else m
        buf = _zeros_buffer(n * cols, dtype)
        one = 1.0 if dtype in ("float64", "float32") else (1 if dtype == "int64" else True)
        for i in range(min(n, cols)):
            buf[i * cols + i] = one
        return cls(buf, (n, cols), dtype)

    @classmethod
    def rand(cls, shape: Any, rng: random.Random, dtype: str = "float64") -> "ZenArray":
        s = _normalize_shape(shape)
        return cls(_make_buffer((rng.random() for _ in range(_numel(s))), dtype), s, dtype)

    @classmethod
    def randn(cls, shape: Any, rng: random.Random, dtype: str = "float64") -> "ZenArray":
        s = _normalize_shape(shape)
        return cls(_make_buffer((rng.gauss(0.0, 1.0) for _ in range(_numel(s))), dtype), s, dtype)

    def copy(self) -> "ZenArray":
        return ZenArray(_make_buffer(self.data, self.dtype), self.shape, self.dtype)

    def astype(self, dtype: str) -> "ZenArray":
        return ZenArray(_make_buffer(self.data, dtype), self.shape, dtype)

    def tolist(self) -> Any:
        if len(self.shape) == 0:
            v = self.data[0]
            return bool(v) if self.dtype == "bool" else (int(v) if self.dtype == "int64" else float(v))
        if len(self.shape) == 1:
            return list(self.data)

        def _build(off: int, dims: tuple[int, ...]) -> list[Any]:
            if len(dims) == 1:
                return list(self.data[off : off + dims[0]])
            step = _numel(dims[1:])
            return [_build(off + i * step, dims[1:]) for i in range(dims[0])]

        return _build(0, self.shape)

    def reshape(self, new_shape: Any) -> "ZenArray":
        if isinstance(new_shape, int) and not isinstance(new_shape, bool):
            raw = [new_shape]
        elif isinstance(new_shape, (list, tuple)):
            raw = list(new_shape)
        else:
            raise ZenRuntimeError("reshape expects an integer or shape list")
        total = len(self.data)
        neg_idx = -1
        prod = 1
        dims: list[int] = []
        for idx, d in enumerate(raw):
            if isinstance(d, bool) or not isinstance(d, int):
                raise ZenRuntimeError(f"Invalid reshape dimension: {d!r}")
            if d == -1:
                if neg_idx != -1:
                    raise ZenRuntimeError("Only one dimension can be -1 in reshape")
                neg_idx = idx
                dims.append(-1)
            elif d < 0:
                raise ZenRuntimeError(f"Invalid reshape dimension: {d}")
            else:
                prod *= d
                dims.append(d)
        if neg_idx != -1:
            if prod == 0 or total % prod != 0:
                raise ZenRuntimeError(f"Shape mismatch: cannot reshape size {total} into {raw}")
            dims[neg_idx] = total // prod
        final_shape = tuple(dims)
        if _numel(final_shape) != total:
            raise ZenRuntimeError(f"Shape mismatch: cannot reshape {list(self.shape)} (size {total}) to {list(final_shape)}")
        return ZenArray(_make_buffer(self.data, self.dtype), final_shape, self.dtype)

    def flatten(self) -> "ZenArray":
        return ZenArray(_make_buffer(self.data, self.dtype), (len(self.data),), self.dtype)

    def transpose(self, axes: list[int] | None = None) -> "ZenArray":
        ndim = len(self.shape)
        if ndim <= 1:
            return self.copy()
        if ndim == 2 and (axes is None or axes == [1, 0]):
            rows, cols = self.shape
            out = _zeros_buffer(rows * cols, self.dtype)
            d = self.data
            idx = 0
            for j in range(cols):
                col_vals = d[j::cols]
                out[idx : idx + rows] = col_vals
                idx += rows
            return ZenArray(out, (cols, rows), self.dtype)
        perm = list(range(ndim - 1, -1, -1)) if axes is None else list(axes)
        if len(perm) != ndim or set(perm) != set(range(ndim)):
            raise ZenRuntimeError(f"Invalid transpose axes {perm} for shape {list(self.shape)}")
        old_st = _strides(self.shape)
        new_shape = tuple(self.shape[p] for p in perm)
        new_st = _strides(new_shape)
        total = len(self.data)
        out = _zeros_buffer(total, self.dtype)
        for flat_idx in range(total):
            rem = flat_idx
            coords = [0] * ndim
            for i in range(ndim):
                coords[i], rem = divmod(rem, old_st[i])
            nf = 0
            for i, p in enumerate(perm):
                nf += coords[p] * new_st[i]
            out[nf] = self.data[flat_idx]
        return ZenArray(out, new_shape, self.dtype)

    def _resolve_key_indices(self, key: Any) -> tuple[list[int], tuple[int, ...]]:
        if isinstance(key, ZenArray) and key.dtype == "bool":
            if key.shape != self.shape:
                raise ZenRuntimeError(f"Shape mismatch: boolean mask {list(key.shape)} vs array {list(self.shape)}")
            idxs = [i for i, flag in enumerate(key.data) if flag]
            return idxs, (len(idxs),)

        keys = list(key) if isinstance(key, tuple) else [key]
        if len(keys) > len(self.shape):
            raise ZenRuntimeError(f"Too many indices ({len(keys)}) for {len(self.shape)}D array of shape {list(self.shape)}")
        while len(keys) < len(self.shape):
            keys.append(slice(None))

        axis_lists: list[list[int]] = []
        out_shape: list[int] = []
        for ax, k in enumerate(keys):
            dim = self.shape[ax]
            if isinstance(k, int) and not isinstance(k, bool):
                idx = k + dim if k < 0 else k
                if idx < 0 or idx >= dim:
                    raise ZenRuntimeError(f"Index {k} out of bounds for axis {ax} with size {dim}")
                axis_lists.append([idx])
            elif isinstance(k, slice):
                rng = range(*k.indices(dim))
                axis_lists.append(list(rng))
                out_shape.append(len(rng))
            else:
                raise ZenRuntimeError(f"Unsupported index type '{type(k).__name__}'")

        st = _strides(self.shape)
        flat_indices: list[int] = [0]
        for ax, ax_idxs in enumerate(axis_lists):
            step = st[ax]
            flat_indices = [base + pos * step for base in flat_indices for pos in ax_idxs]
        return flat_indices, tuple(out_shape)

    def get_item(self, key: Any) -> Any:
        if len(self.shape) == 0:
            raise ZenRuntimeError("Cannot index into 0D array")
        flat_indices, out_shape = self._resolve_key_indices(key)
        if len(out_shape) == 0:
            v = self.data[flat_indices[0]]
            return bool(v) if self.dtype == "bool" else (int(v) if self.dtype == "int64" else float(v))
        vals = [self.data[i] for i in flat_indices]
        return ZenArray(_make_buffer(vals, self.dtype), out_shape, self.dtype)

    def set_item(self, key: Any, value: Any) -> None:
        if len(self.shape) == 0:
            raise ZenRuntimeError("Cannot index-assign into 0D array")
        flat_indices, out_shape = self._resolve_key_indices(key)
        if isinstance(value, ZenArray):
            if len(out_shape) == 0:
                if value.size != 1:
                    raise ZenRuntimeError(f"Shape mismatch: cannot assign {list(value.shape)} to scalar slot")
                self.data[flat_indices[0]] = value.data[0]
                return
            bcast = value if value.shape == out_shape else value._broadcast_to(out_shape)
            for dst, v in zip(flat_indices, bcast.data):
                self.data[dst] = v
            return
        if isinstance(value, list):
            arr = ZenArray.from_list(value, self.dtype)
            self.set_item(key, arr)
            return
        cast_v = bool(value) if self.dtype == "bool" else (int(value) if self.dtype == "int64" else float(value))
        for dst in flat_indices:
            self.data[dst] = cast_v

    def _broadcast_to(self, target_shape: tuple[int, ...]) -> "ZenArray":
        if self.shape == target_shape:
            return self
        _broadcast_shapes(self.shape, target_shape)
        ndim = len(target_shape)
        s = (1,) * (ndim - len(self.shape)) + self.shape
        st = _strides(s)
        b_st = tuple(0 if d == 1 else stride for d, stride in zip(s, st))
        out_st = _strides(target_shape)
        total = _numel(target_shape)
        out = _zeros_buffer(total, self.dtype)
        d = self.data
        for idx in range(total):
            rem = idx
            src_i = 0
            for ax in range(ndim):
                coord, rem = divmod(rem, out_st[ax])
                src_i += coord * b_st[ax]
            out[idx] = d[src_i]
        return ZenArray(out, target_shape, self.dtype)

    def _binary(self, other: Any, fn: Callable[[Any, Any], Any], out_dtype: str | None = None, reverse: bool = False) -> "ZenArray":
        if isinstance(other, (int, float, bool)):
            od = out_dtype or _promote_dtype(
                self.dtype, "bool" if isinstance(other, bool) else ("int64" if isinstance(other, int) else "float64")
            )
            val = other
            if reverse:
                vals = map(fn, repeat(val), self.data)
            else:
                vals = map(fn, self.data, repeat(val))
            return ZenArray(_make_buffer(vals, od), self.shape, od)
        if not isinstance(other, ZenArray):
            raise ZenRuntimeError(f"Array operation requires another Array or number, got '{type(other).__name__}'")

        od = out_dtype or _promote_dtype(self.dtype, other.dtype)
        if self.shape == other.shape:
            vals = map(fn, self.data, other.data)
            return ZenArray(_make_buffer(vals, od), self.shape, od)

        out_shape = _broadcast_shapes(self.shape, other.shape)
        ndim = len(out_shape)
        total = _numel(out_shape)
        s1 = (1,) * (ndim - len(self.shape)) + self.shape
        s2 = (1,) * (ndim - len(other.shape)) + other.shape
        b_st1 = tuple(0 if d == 1 else st for d, st in zip(s1, _strides(s1)))
        b_st2 = tuple(0 if d == 1 else st for d, st in zip(s2, _strides(s2)))
        out_st = _strides(out_shape)
        out = _zeros_buffer(total, od)
        d1, d2 = self.data, other.data
        for idx in range(total):
            rem = idx
            i1 = 0
            i2 = 0
            for ax in range(ndim):
                c, rem = divmod(rem, out_st[ax])
                i1 += c * b_st1[ax]
                i2 += c * b_st2[ax]
            out[idx] = fn(d1[i1], d2[i2])
        return ZenArray(out, out_shape, od)

    def add(self, other: Any) -> "ZenArray":
        return self._binary(other, operator.add)

    def sub(self, other: Any) -> "ZenArray":
        return self._binary(other, operator.sub)

    def rsub(self, scalar: Any) -> "ZenArray":
        return self._binary(scalar, operator.sub, reverse=True)

    def mul(self, other: Any) -> "ZenArray":
        return self._binary(other, operator.mul)

    def div(self, other: Any) -> "ZenArray":
        def _safe_div(a: float, b: float) -> float:
            if b == 0:
                raise ZenRuntimeError("Division by zero in array")
            return a / b

        od = "float64" if self.dtype == "int64" or (isinstance(other, ZenArray) and other.dtype == "int64") else None
        return self._binary(other, _safe_div, out_dtype=od or _promote_dtype(self.dtype, "float64"))

    def rdiv(self, scalar: Any) -> "ZenArray":
        def _safe_div(a: float, b: float) -> float:
            if b == 0:
                raise ZenRuntimeError("Division by zero in array")
            return a / b

        return self._binary(scalar, _safe_div, out_dtype="float64", reverse=True)

    def pow(self, other: Any) -> "ZenArray":
        return self._binary(other, operator.pow)

    def neg(self) -> "ZenArray":
        return ZenArray(_make_buffer((-x for x in self.data), self.dtype), self.shape, self.dtype)

    def cmp(self, op: str, other: Any) -> "ZenArray":
        ops = {
            "<": operator.lt,
            "<=": operator.le,
            ">": operator.gt,
            ">=": operator.ge,
            "==": operator.eq,
            "!=": operator.ne,
        }
        return self._binary(other, ops[op], out_dtype="bool")

    def unary_math(self, name: str) -> "ZenArray":
        if name == "abs":
            return ZenArray(_make_buffer(map(abs, self.data), self.dtype), self.shape, self.dtype)
        dt = "float64" if self.dtype in ("int64", "bool") else self.dtype
        if name == "sqrt":
            vals = []
            for x in self.data:
                if x < 0:
                    raise ZenRuntimeError("Array sqrt domain error (value < 0)")
                vals.append(math.sqrt(x))
            return ZenArray(_make_buffer(vals, dt), self.shape, dt)
        if name == "exp":
            return ZenArray(_make_buffer(map(math.exp, self.data), dt), self.shape, dt)
        if name == "log":
            vals = []
            for x in self.data:
                if x <= 0:
                    raise ZenRuntimeError("Array log domain error (value <= 0)")
                vals.append(math.log(x))
            return ZenArray(_make_buffer(vals, dt), self.shape, dt)
        if name == "sin":
            return ZenArray(_make_buffer(map(math.sin, self.data), dt), self.shape, dt)
        if name == "cos":
            return ZenArray(_make_buffer(map(math.cos, self.data), dt), self.shape, dt)
        if name == "tanh":
            return ZenArray(_make_buffer(map(math.tanh, self.data), dt), self.shape, dt)
        raise ZenRuntimeError(f"Unknown math op '{name}'")

    def clip(self, low: float | None, high: float | None) -> "ZenArray":
        lo = -math.inf if low is None else float(low)
        hi = math.inf if high is None else float(high)
        vals = [lo if x < lo else (hi if x > hi else x) for x in self.data]
        return ZenArray(_make_buffer(vals, self.dtype), self.shape, self.dtype)

    def reduce(self, mode: str, axis: int | None = None, keepdims: bool = False) -> Any:
        if not self.data:
            raise ZenRuntimeError(f"Cannot compute {mode} of empty array")

        def _calc(vals: list[Any]) -> Any:
            if mode == "sum":
                return sum(vals)
            if mode == "mean":
                return sum(vals) / len(vals)
            if mode == "var":
                m = sum(vals) / len(vals)
                return sum((x - m) * (x - m) for x in vals) / len(vals)
            if mode == "std":
                m = sum(vals) / len(vals)
                return math.sqrt(sum((x - m) * (x - m) for x in vals) / len(vals))
            if mode == "min":
                return min(vals)
            if mode == "max":
                return max(vals)
            if mode == "argmax":
                best_i, best_v = 0, vals[0]
                for idx in range(1, len(vals)):
                    if vals[idx] > best_v:
                        best_v = vals[idx]
                        best_i = idx
                return best_i
            if mode == "argmin":
                best_i, best_v = 0, vals[0]
                for idx in range(1, len(vals)):
                    if vals[idx] < best_v:
                        best_v = vals[idx]
                        best_i = idx
                return best_i
            if mode == "norm":
                return math.sqrt(sum(x * x for x in vals))
            raise ZenRuntimeError(f"Unknown reduction '{mode}'")

        out_dt = "int64" if mode in ("argmax", "argmin") else ("float64" if mode in ("mean", "var", "std", "norm") else self.dtype)

        if axis is None:
            res = _calc(list(self.data))
            if keepdims:
                return ZenArray(_make_buffer([res], out_dt), (1,) * len(self.shape), out_dt)
            return int(res) if out_dt == "int64" else (bool(res) if out_dt == "bool" else float(res))

        ndim = len(self.shape)
        ax = axis + ndim if axis < 0 else axis
        if ax < 0 or ax >= ndim:
            raise ZenRuntimeError(f"Axis {axis} out of bounds for {ndim}D array of shape {list(self.shape)}")

        out_shape = self.shape[:ax] + ((1,) if keepdims else ()) + self.shape[ax + 1 :]
        if len(out_shape) == 0:
            return self.reduce(mode, None, False)

        outer = _numel(self.shape[:ax])
        ax_len = self.shape[ax]
        inner = _numel(self.shape[ax + 1 :])
        out_vals: list[Any] = []
        for o in range(outer):
            base = o * ax_len * inner
            for inn in range(inner):
                vals = [self.data[base + k * inner + inn] for k in range(ax_len)]
                out_vals.append(_calc(vals))
        return ZenArray(_make_buffer(out_vals, out_dt), out_shape, out_dt)

    def sort(self, axis: int = -1) -> "ZenArray":
        if len(self.shape) == 0:
            return self.copy()
        ndim = len(self.shape)
        ax = axis + ndim if axis < 0 else axis
        if ax < 0 or ax >= ndim:
            raise ZenRuntimeError(f"Axis {axis} out of bounds for shape {list(self.shape)}")
        outer = _numel(self.shape[:ax])
        ax_len = self.shape[ax]
        inner = _numel(self.shape[ax + 1 :])
        out = _zeros_buffer(len(self.data), self.dtype)
        for o in range(outer):
            base = o * ax_len * inner
            for inn in range(inner):
                vals = sorted(self.data[base + k * inner + inn] for k in range(ax_len))
                for k in range(ax_len):
                    out[base + k * inner + inn] = vals[k]
        return ZenArray(out, self.shape, self.dtype)

    def cumsum(self, axis: int | None = None) -> "ZenArray":
        if axis is None:
            out: list[Any] = []
            acc = 0.0 if self.dtype in ("float64", "float32") else 0
            for v in self.data:
                acc += v
                out.append(acc)
            dt = "int64" if self.dtype in ("int64", "bool") else self.dtype
            return ZenArray(_make_buffer(out, dt), (len(out),), dt)
        ndim = len(self.shape)
        ax = axis + ndim if axis < 0 else axis
        if ax < 0 or ax >= ndim:
            raise ZenRuntimeError(f"Axis {axis} out of bounds for shape {list(self.shape)}")
        outer = _numel(self.shape[:ax])
        ax_len = self.shape[ax]
        inner = _numel(self.shape[ax + 1 :])
        dt = "int64" if self.dtype in ("int64", "bool") else self.dtype
        out_buf = _zeros_buffer(len(self.data), dt)
        for o in range(outer):
            base = o * ax_len * inner
            for inn in range(inner):
                acc = 0.0 if dt in ("float64", "float32") else 0
                for k in range(ax_len):
                    acc += self.data[base + k * inner + inn]
                    out_buf[base + k * inner + inn] = acc
        return ZenArray(out_buf, self.shape, dt)

    def matmul(self, other: "ZenArray") -> Any:
        if not isinstance(other, ZenArray):
            raise ZenRuntimeError("Matrix multiply '@' requires two arrays")
        if len(self.shape) == 1 and len(other.shape) == 1:
            if self.shape[0] != other.shape[0]:
                raise ZenRuntimeError(f"Shape mismatch in dot/matmul: {list(self.shape)} and {list(other.shape)}")
            res = sum(a * b for a, b in zip(self.data, other.data))
            return int(res) if self.dtype == "int64" and other.dtype == "int64" else float(res)
        if len(self.shape) == 2 and len(other.shape) == 1:
            m, k = self.shape
            if k != other.shape[0]:
                raise ZenRuntimeError(f"Shape mismatch in matmul: {list(self.shape)} @ {list(other.shape)}")
            out = [sum(self.data[i * k + p] * other.data[p] for p in range(k)) for i in range(m)]
            dt = _promote_dtype(self.dtype, other.dtype)
            return ZenArray(_make_buffer(out, dt), (m,), dt)
        if len(self.shape) == 1 and len(other.shape) == 2:
            k = self.shape[0]
            k2, n = other.shape
            if k != k2:
                raise ZenRuntimeError(f"Shape mismatch in matmul: {list(self.shape)} @ {list(other.shape)}")
            out = [sum(self.data[p] * other.data[p * n + j] for p in range(k)) for j in range(n)]
            dt = _promote_dtype(self.dtype, other.dtype)
            return ZenArray(_make_buffer(out, dt), (n,), dt)
        if len(self.shape) == 2 and len(other.shape) == 2:
            m, k = self.shape
            k2, n = other.shape
            if k != k2:
                raise ZenRuntimeError(f"Shape mismatch in matmul: {list(self.shape)} @ {list(other.shape)}")
            dt = _promote_dtype(self.dtype, other.dtype)
            a_buf = array("d", (float(x) for x in self.data))
            b_buf = array("d", (float(x) for x in other.data))
            c_buf = native_kernels.matmul_2d_buf(a_buf, 0, m, k, b_buf, 0, n)
            return ZenArray(_make_buffer(c_buf, dt), (m, n), dt)
        if len(self.shape) >= 3 and len(other.shape) == 2:
            k, n = other.shape
            if self.shape[-1] != k:
                raise ZenRuntimeError(f"Shape mismatch in matmul: {list(self.shape)} @ {list(other.shape)}")
            m_tot = _numel(self.shape[:-1])
            dt = _promote_dtype(self.dtype, other.dtype)
            a_buf = array("d", (float(x) for x in self.data))
            b_buf = array("d", (float(x) for x in other.data))
            c_buf = native_kernels.matmul_2d_buf(a_buf, 0, m_tot, k, b_buf, 0, n)
            return ZenArray(_make_buffer(c_buf, dt), self.shape[:-1] + (n,), dt)
        if len(self.shape) >= 2 and len(other.shape) >= 2:
            m, k = self.shape[-2], self.shape[-1]
            k2, n = other.shape[-2], other.shape[-1]
            if k != k2:
                raise ZenRuntimeError(f"Shape mismatch in matmul: {list(self.shape)} @ {list(other.shape)}")
            batch_out = _broadcast_shapes(self.shape[:-2], other.shape[:-2])
            b_ndim = len(batch_out)
            ba = (1,) * (b_ndim - len(self.shape[:-2])) + self.shape[:-2]
            bb = (1,) * (b_ndim - len(other.shape[:-2])) + other.shape[:-2]
            b_sta = tuple(0 if d == 1 else s for d, s in zip(ba, _strides(ba)))
            b_stb = tuple(0 if d == 1 else s for d, s in zip(bb, _strides(bb)))
            out_st = _strides(batch_out)
            a_buf = array("d", (float(x) for x in self.data))
            b_buf = array("d", (float(x) for x in other.data))
            out_vals: list[float] = []
            for idx in range(_numel(batch_out)):
                rem = idx
                ia = 0
                ib = 0
                for ax in range(b_ndim):
                    c, rem = divmod(rem, out_st[ax])
                    ia += c * b_sta[ax]
                    ib += c * b_stb[ax]
                c_buf = native_kernels.matmul_2d_buf(a_buf, ia * m * k, m, k, b_buf, ib * k * n, n)
                out_vals.extend(c_buf)
            dt = _promote_dtype(self.dtype, other.dtype)
            return ZenArray(_make_buffer(out_vals, dt), batch_out + (m, n), dt)
        raise ZenRuntimeError(f"Shape mismatch in matmul: {list(self.shape)} @ {list(other.shape)}")


def arr_concat(arrays: list[ZenArray], axis: int = 0) -> ZenArray:
    if not arrays:
        raise ZenRuntimeError("concat requires at least one array")
    first = arrays[0]
    ndim = len(first.shape)
    ax = axis + ndim if axis < 0 else axis
    if ax < 0 or ax >= ndim:
        raise ZenRuntimeError(f"Axis {axis} out of bounds for {ndim}D array")
    dt = first.dtype
    for arr in arrays[1:]:
        if len(arr.shape) != ndim:
            raise ZenRuntimeError(f"Shape mismatch in concat: {list(first.shape)} vs {list(arr.shape)}")
        for i in range(ndim):
            if i != ax and arr.shape[i] != first.shape[i]:
                raise ZenRuntimeError(f"Shape mismatch in concat: {list(first.shape)} vs {list(arr.shape)}")
        dt = _promote_dtype(dt, arr.dtype)
    new_dim = sum(a.shape[ax] for a in arrays)
    out_shape = first.shape[:ax] + (new_dim,) + first.shape[ax + 1 :]
    outer = _numel(first.shape[:ax])
    inner = _numel(first.shape[ax + 1 :])
    out_vals: list[Any] = []
    for o in range(outer):
        for a in arrays:
            chunk = a.shape[ax] * inner
            start = o * chunk
            out_vals.extend(a.data[start : start + chunk])
    return ZenArray(_make_buffer(out_vals, dt), out_shape, dt)


def arr_stack(arrays: list[ZenArray], axis: int = 0) -> ZenArray:
    if not arrays:
        raise ZenRuntimeError("stack requires at least one array")
    base_shape = arrays[0].shape
    for a in arrays[1:]:
        if a.shape != base_shape:
            raise ZenRuntimeError(f"Shape mismatch in stack: {list(base_shape)} vs {list(a.shape)}")
    ndim = len(base_shape) + 1
    ax = axis + ndim if axis < 0 else axis
    if ax < 0 or ax >= ndim:
        raise ZenRuntimeError(f"Axis {axis} out of bounds for stack")
    expanded_shape = base_shape[:ax] + (1,) + base_shape[ax:]
    expanded = [ZenArray(a.data, expanded_shape, a.dtype) for a in arrays]
    return arr_concat(expanded, axis=ax)


def arr_split(arr: ZenArray, sections_or_indices: int | list[int], axis: int = 0) -> list[ZenArray]:
    ndim = len(arr.shape)
    ax = axis + ndim if axis < 0 else axis
    if ax < 0 or ax >= ndim:
        raise ZenRuntimeError(f"Axis {axis} out of bounds for shape {list(arr.shape)}")
    dim = arr.shape[ax]
    if isinstance(sections_or_indices, int) and not isinstance(sections_or_indices, bool):
        n = sections_or_indices
        if n <= 0 or dim % n != 0:
            raise ZenRuntimeError(f"Array of axis length {dim} cannot be evenly split into {n} sections")
        step = dim // n
        bounds = [(i * step, (i + 1) * step) for i in range(n)]
    elif isinstance(sections_or_indices, list):
        pts = [0] + list(sections_or_indices) + [dim]
        bounds = [(pts[i], pts[i + 1]) for i in range(len(pts) - 1)]
    else:
        raise ZenRuntimeError("split expects an integer or list of split indices")
    results: list[ZenArray] = []
    for lo, hi in bounds:
        key = tuple(slice(lo, hi) if i == ax else slice(None) for i in range(ndim))
        sub = arr.get_item(key)
        assert isinstance(sub, ZenArray)
        results.append(sub)
    return results


def arr_where(cond: ZenArray, x: Any, y: Any) -> ZenArray:
    x_arr = x if isinstance(x, ZenArray) else ZenArray.full(cond.shape, x)
    y_arr = y if isinstance(y, ZenArray) else ZenArray.full(cond.shape, y)
    out_shape = _broadcast_shapes(cond.shape, _broadcast_shapes(x_arr.shape, y_arr.shape))
    c_b = cond._broadcast_to(out_shape)
    x_b = x_arr._broadcast_to(out_shape)
    y_b = y_arr._broadcast_to(out_shape)
    dt = _promote_dtype(x_b.dtype, y_b.dtype)
    vals = [xv if cv else yv for cv, xv, yv in zip(c_b.data, x_b.data, y_b.data)]
    return ZenArray(_make_buffer(vals, dt), out_shape, dt)


def arr_outer(a: ZenArray, b: ZenArray) -> ZenArray:
    af = a.flatten()
    bf = b.flatten()
    dt = _promote_dtype(af.dtype, bf.dtype)
    vals = [x * y for x in af.data for y in bf.data]
    return ZenArray(_make_buffer(vals, dt), (af.size, bf.size), dt)


def _tri_mask(arr: ZenArray, k: int, *, upper: bool) -> ZenArray:
    op_name = "triu" if upper else "tril"
    if len(arr.shape) < 2:
        raise ZenRuntimeError(f"{op_name} requires at least a 2D array, got shape {list(arr.shape)}")
    if isinstance(k, bool) or not isinstance(k, int):
        raise ZenRuntimeError(f"{op_name} diagonal offset 'k' must be an integer")
    rows, cols = arr.shape[-2], arr.shape[-1]
    batch = _numel(arr.shape[:-2])
    mat_size = rows * cols
    zero_val: Any = 0.0 if arr.dtype in ("float64", "float32") else (0 if arr.dtype == "int64" else False)
    out = _zeros_buffer(len(arr.data), arr.dtype)
    for b in range(batch):
        base = b * mat_size
        for r in range(rows):
            row_off = base + r * cols
            for c in range(cols):
                keep = (c - r >= k) if upper else (c - r <= k)
                out[row_off + c] = arr.data[row_off + c] if keep else zero_val
    return ZenArray(out, arr.shape, arr.dtype)


def arr_tril(arr: ZenArray, k: int = 0) -> ZenArray:
    return _tri_mask(arr, k, upper=False)


def arr_triu(arr: ZenArray, k: int = 0) -> ZenArray:
    return _tri_mask(arr, k, upper=True)
