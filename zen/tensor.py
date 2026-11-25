"""Pure-Python Tensor engine for ZEN (no external libraries)."""

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


@lru_cache(maxsize=512)
def _compute_strides(shape: tuple[int, ...]) -> tuple[int, ...]:
    strides = [1] * len(shape)
    acc = 1
    for i in range(len(shape) - 1, -1, -1):
        strides[i] = acc
        acc *= shape[i]
    return tuple(strides)


@lru_cache(maxsize=512)
def _numel(shape: tuple[int, ...]) -> int:
    return math.prod(shape)


@lru_cache(maxsize=512)
def _broadcast_shapes_cached(s1: tuple[int, ...], s2: tuple[int, ...]) -> tuple[int, ...]:
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
            raise ZenRuntimeError(f"Cannot broadcast shapes {list(s1)} and {list(s2)}")
    return tuple(res)


def _infer_nested(data: Any) -> tuple[list[float], tuple[int, ...]]:
    if isinstance(data, (int, float)) and not isinstance(data, bool):
        return [float(data)], ()
    if not isinstance(data, list):
        raise ZenRuntimeError("Tensor data must be a number or nested list of numbers")
    if len(data) == 0:
        return [], (0,)
    first = data[0]
    if isinstance(first, list):
        sub_shape: tuple[int, ...] | None = None
        flat: list[float] = []
        for item in data:
            sub_flat, item_shape = _infer_nested(item)
            if sub_shape is None:
                sub_shape = item_shape
            elif item_shape != sub_shape:
                raise ZenRuntimeError("Ragged nested lists cannot form a Tensor")
            flat.extend(sub_flat)
        assert sub_shape is not None
        return flat, (len(data),) + sub_shape
    flat = []
    for item in data:
        if isinstance(item, bool) or not isinstance(item, (int, float)):
            raise ZenRuntimeError("Tensor elements must be numbers")
        flat.append(float(item))
    return flat, (len(data),)


def _normalize_shape(shape: Any) -> tuple[int, ...]:
    if isinstance(shape, int) and not isinstance(shape, bool):
        shape_list = [shape]
    elif isinstance(shape, list):
        shape_list = shape
    elif isinstance(shape, tuple):
        shape_list = list(shape)
    else:
        raise ZenRuntimeError("Shape must be an integer or list of integers")
    out: list[int] = []
    for d in shape_list:
        if isinstance(d, bool) or not isinstance(d, int) or d < 0:
            raise ZenRuntimeError(f"Invalid tensor dimension: {d!r}")
        out.append(d)
    return tuple(out)


def _broadcast_shapes(s1: tuple[int, ...], s2: tuple[int, ...]) -> tuple[int, ...]:
    return _broadcast_shapes_cached(s1, s2)


class ZenTensor:
    """N-dimensional dense float tensor backed by a flat Python list."""

    def __init__(self, data: list[float], shape: tuple[int, ...]) -> None:
        self.data = data
        self.shape = shape

    @classmethod
    def from_list(cls, raw: Any) -> "ZenTensor":
        if isinstance(raw, ZenTensor):
            return cls(list(raw.data), raw.shape)
        flat, shape = _infer_nested(raw)
        return cls(flat, shape)

    @classmethod
    def zeros(cls, shape: Any) -> "ZenTensor":
        s = _normalize_shape(shape)
        return cls([0.0] * _numel(s), s)

    @classmethod
    def ones(cls, shape: Any) -> "ZenTensor":
        s = _normalize_shape(shape)
        return cls([1.0] * _numel(s), s)

    @classmethod
    def rand(cls, shape: Any, rng: random.Random) -> "ZenTensor":
        s = _normalize_shape(shape)
        return cls([rng.random() for _ in range(_numel(s))], s)

    @classmethod
    def randn(cls, shape: Any, rng: random.Random) -> "ZenTensor":
        s = _normalize_shape(shape)
        return cls([rng.gauss(0.0, 1.0) for _ in range(_numel(s))], s)

    def tolist(self) -> Any:
        if len(self.shape) == 0:
            return self.data[0]
        if len(self.shape) == 1:
            return list(self.data)

        def _build(offset: int, dims: tuple[int, ...]) -> list[Any]:
            if len(dims) == 1:
                return self.data[offset : offset + dims[0]]
            step = _numel(dims[1:])
            return [_build(offset + i * step, dims[1:]) for i in range(dims[0])]

        return _build(0, self.shape)

    def reshape(self, new_shape: Any) -> "ZenTensor":
        if isinstance(new_shape, int) and not isinstance(new_shape, bool):
            raw_dims = [new_shape]
        elif isinstance(new_shape, (list, tuple)):
            raw_dims = list(new_shape)
        else:
            raise ZenRuntimeError("reshape expects a shape list or integer")

        total = len(self.data)
        neg_idx = -1
        known_prod = 1
        dims: list[int] = []
        for idx, d in enumerate(raw_dims):
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
                known_prod *= d
                dims.append(d)

        if neg_idx != -1:
            if known_prod == 0 or total % known_prod != 0:
                raise ZenRuntimeError(f"Cannot reshape size {total} into {raw_dims}")
            dims[neg_idx] = total // known_prod

        final_shape = tuple(dims)
        if _numel(final_shape) != total:
            raise ZenRuntimeError(
                f"Cannot reshape tensor of size {total} into shape {list(final_shape)}"
            )
        return ZenTensor(list(self.data), final_shape)

    def transpose(self, axes: list[int] | None = None) -> "ZenTensor":
        ndim = len(self.shape)
        if ndim <= 1:
            return ZenTensor(list(self.data), self.shape)
        if ndim == 2 and (axes is None or axes == [1, 0]):
            rows, cols = self.shape
            d = self.data
            out: list[float] = []
            for j in range(cols):
                out.extend(d[j::cols])
            return ZenTensor(out, (cols, rows))
        if axes is None:
            perm = list(range(ndim - 1, -1, -1))
        else:
            if len(axes) != ndim or set(axes) != set(range(ndim)):
                raise ZenRuntimeError(f"Invalid transpose axes {axes} for {ndim}D tensor")
            perm = list(axes)

        old_strides = _compute_strides(self.shape)
        new_shape = tuple(self.shape[p] for p in perm)
        new_strides = _compute_strides(new_shape)
        total = len(self.data)
        out = [0.0] * total

        for flat_idx in range(total):
            rem = flat_idx
            old_coords = [0] * ndim
            for i in range(ndim):
                old_coords[i], rem = divmod(rem, old_strides[i])
            new_flat = 0
            for i, p in enumerate(perm):
                new_flat += old_coords[p] * new_strides[i]
            out[new_flat] = self.data[flat_idx]

        return ZenTensor(out, new_shape)

    def _elementwise(self, other: "ZenTensor | int | float", fn: Callable[[float, float], float]) -> "ZenTensor":
        if isinstance(other, (int, float)) and not isinstance(other, bool):
            return ZenTensor(list(map(fn, self.data, repeat(float(other)))), self.shape)
        if not isinstance(other, ZenTensor):
            raise ZenRuntimeError("Tensor operation requires another Tensor or a number")

        if self.shape == other.shape:
            return ZenTensor(list(map(fn, self.data, other.data)), self.shape)

        out_shape = _broadcast_shapes(self.shape, other.shape)
        ndim = len(out_shape)
        out_total = _numel(out_shape)
        if out_total == 0:
            return ZenTensor([], out_shape)

        s1 = (1,) * (ndim - len(self.shape)) + self.shape
        s2 = (1,) * (ndim - len(other.shape)) + other.shape
        st1 = _compute_strides(s1)
        st2 = _compute_strides(s2)
        b_st1 = tuple(0 if d == 1 else st for d, st in zip(s1, st1))
        b_st2 = tuple(0 if d == 1 else st for d, st in zip(s2, st2))
        out_st = _compute_strides(out_shape)

        out = [0.0] * out_total
        d1 = self.data
        d2 = other.data
        for idx in range(out_total):
            rem = idx
            i1 = 0
            i2 = 0
            for axis in range(ndim):
                coord, rem = divmod(rem, out_st[axis])
                i1 += coord * b_st1[axis]
                i2 += coord * b_st2[axis]
            out[idx] = fn(d1[i1], d2[i2])
        return ZenTensor(out, out_shape)

    def _rev_elementwise(self, scalar: int | float, fn: Callable[[float, float], float]) -> "ZenTensor":
        s_val = float(scalar)
        return ZenTensor([fn(s_val, x) for x in self.data], self.shape)

    def add(self, other: Any) -> "ZenTensor":
        return self._elementwise(other, operator.add)

    def sub(self, other: Any) -> "ZenTensor":
        return self._elementwise(other, operator.sub)

    def rsub(self, scalar: int | float) -> "ZenTensor":
        return self._rev_elementwise(scalar, operator.sub)

    def mul(self, other: Any) -> "ZenTensor":
        return self._elementwise(other, operator.mul)

    def div(self, other: Any) -> "ZenTensor":
        def _safe_div(a: float, b: float) -> float:
            if b == 0.0:
                raise ZenRuntimeError("Division by zero in Tensor")
            return a / b

        return self._elementwise(other, _safe_div)

    def rdiv(self, scalar: int | float) -> "ZenTensor":
        def _safe_div(a: float, b: float) -> float:
            if b == 0.0:
                raise ZenRuntimeError("Division by zero in Tensor")
            return a / b

        return self._rev_elementwise(scalar, _safe_div)

    def exp(self) -> "ZenTensor":
        return ZenTensor(list(map(math.exp, self.data)), self.shape)

    def log(self) -> "ZenTensor":
        out: list[float] = []
        for x in self.data:
            if x <= 0.0:
                raise ZenRuntimeError("Tensor log domain error (value <= 0)")
            out.append(math.log(x))
        return ZenTensor(out, self.shape)

    def tanh(self) -> "ZenTensor":
        return ZenTensor(list(map(math.tanh, self.data)), self.shape)

    def sqrt(self) -> "ZenTensor":
        out: list[float] = []
        for x in self.data:
            if x < 0.0:
                raise ZenRuntimeError("Tensor sqrt domain error (value < 0)")
            out.append(math.sqrt(x))
        return ZenTensor(out, self.shape)

    def clamp_min(self, min_val: float = 0.0) -> "ZenTensor":
        mv = float(min_val)
        return ZenTensor([x if x > mv else mv for x in self.data], self.shape)

    def gt(self, val: float = 0.0) -> "ZenTensor":
        v = float(val)
        return ZenTensor([1.0 if x > v else 0.0 for x in self.data], self.shape)

    @staticmethod
    def _matmul_2d(a_data: list[float], a_off: int, m: int, k: int, b_data: list[float], b_off: int, n: int) -> list[float]:
        if native_kernels.NATIVE_AVAILABLE and a_off == 0 and b_off == 0:
            res_buf = native_kernels.matmul_2d_buf(array("d", a_data), 0, m, k, array("d", b_data), 0, n)
            return list(res_buf)
        b_end = b_off + k * n
        b_cols = [b_data[b_off + j : b_end : n] for j in range(n)]
        out = [0.0] * (m * n)
        mul = operator.mul
        for i in range(m):
            a_row = a_data[a_off + i * k : a_off + (i + 1) * k]
            row_off = i * n
            for j in range(n):
                out[row_off + j] = sum(map(mul, a_row, b_cols[j]))
        return out

    def matmul(self, other: "ZenTensor") -> "ZenTensor | float":
        if not isinstance(other, ZenTensor):
            raise ZenRuntimeError("Matrix multiply '@' requires two Tensors")

        if len(self.shape) == 1 and len(other.shape) == 1:
            if self.shape[0] != other.shape[0]:
                raise ZenRuntimeError(f"Dot product shape mismatch: {list(self.shape)} @ {list(other.shape)}")
            return sum(a * b for a, b in zip(self.data, other.data))

        if len(self.shape) == 2 and len(other.shape) == 1:
            m, k = self.shape
            if k != other.shape[0]:
                raise ZenRuntimeError(f"Matmul shape mismatch: {list(self.shape)} @ {list(other.shape)}")
            out = [0.0] * m
            for i in range(m):
                row_off = i * k
                out[i] = sum(self.data[row_off + p] * other.data[p] for p in range(k))
            return ZenTensor(out, (m,))

        if len(self.shape) == 1 and len(other.shape) == 2:
            k = self.shape[0]
            k2, n = other.shape
            if k != k2:
                raise ZenRuntimeError(f"Matmul shape mismatch: {list(self.shape)} @ {list(other.shape)}")
            out = [0.0] * n
            for j in range(n):
                out[j] = sum(self.data[p] * other.data[p * n + j] for p in range(k))
            return ZenTensor(out, (n,))

        if len(self.shape) == 2 and len(other.shape) == 2:
            m, k = self.shape
            k2, n = other.shape
            if k != k2:
                raise ZenRuntimeError(f"Matmul shape mismatch: {list(self.shape)} @ {list(other.shape)}")
            return ZenTensor(self._matmul_2d(self.data, 0, m, k, other.data, 0, n), (m, n))

        if len(self.shape) >= 3 and len(other.shape) == 2:
            k, n = other.shape
            if self.shape[-1] != k:
                raise ZenRuntimeError(f"Matmul shape mismatch: {list(self.shape)} @ {list(other.shape)}")
            m_total = _numel(self.shape[:-1])
            out = self._matmul_2d(self.data, 0, m_total, k, other.data, 0, n)
            return ZenTensor(out, self.shape[:-1] + (n,))

        if len(self.shape) >= 2 and len(other.shape) >= 2:
            m, k = self.shape[-2], self.shape[-1]
            k2, n = other.shape[-2], other.shape[-1]
            if k != k2:
                raise ZenRuntimeError(f"Matmul shape mismatch: {list(self.shape)} @ {list(other.shape)}")
            batch_out = _broadcast_shapes(self.shape[:-2], other.shape[:-2])
            b_ndim = len(batch_out)
            ba = (1,) * (b_ndim - len(self.shape[:-2])) + self.shape[:-2]
            bb = (1,) * (b_ndim - len(other.shape[:-2])) + other.shape[:-2]
            sta = _compute_strides(ba)
            stb = _compute_strides(bb)
            b_sta = tuple(0 if d == 1 else s for d, s in zip(ba, sta))
            b_stb = tuple(0 if d == 1 else s for d, s in zip(bb, stb))
            out_st = _compute_strides(batch_out)
            n_batches = _numel(batch_out)
            a_step = m * k
            b_step = k * n
            out: list[float] = []
            for idx in range(n_batches):
                rem = idx
                ia = 0
                ib = 0
                for ax in range(b_ndim):
                    coord, rem = divmod(rem, out_st[ax])
                    ia += coord * b_sta[ax]
                    ib += coord * b_stb[ax]
                out.extend(self._matmul_2d(self.data, ia * a_step, m, k, other.data, ib * b_step, n))
            return ZenTensor(out, batch_out + (m, n))

        raise ZenRuntimeError(f"Matmul unsupported shapes {list(self.shape)} and {list(other.shape)}")

    def _reduce_axis(self, axis: int | None, mode: str, keepdims: bool = False) -> "ZenTensor | float":
        if not self.data:
            raise ZenRuntimeError(f"Cannot compute {mode} of empty Tensor")
        if axis is None:
            if mode == "sum":
                val = sum(self.data)
            elif mode == "mean":
                val = sum(self.data) / len(self.data)
            else:
                val = max(self.data)
            if keepdims:
                return ZenTensor([val], (1,) * len(self.shape))
            return val

        ndim = len(self.shape)
        if isinstance(axis, bool) or not isinstance(axis, int):
            raise ZenRuntimeError("Axis must be an integer or none")
        if axis < 0:
            axis += ndim
        if axis < 0 or axis >= ndim:
            raise ZenRuntimeError(f"Axis {axis} out of bounds for {ndim}D tensor")

        if keepdims:
            out_shape = self.shape[:axis] + (1,) + self.shape[axis + 1 :]
        else:
            out_shape = self.shape[:axis] + self.shape[axis + 1 :]
            if len(out_shape) == 0:
                return self._reduce_axis(None, mode, keepdims=False)

        outer = _numel(self.shape[:axis])
        axis_len = self.shape[axis]
        inner = _numel(self.shape[axis + 1 :])
        out: list[float] = []

        for o in range(outer):
            base = o * axis_len * inner
            for inn in range(inner):
                vals = [self.data[base + k * inner + inn] for k in range(axis_len)]
                if mode == "sum":
                    out.append(sum(vals))
                elif mode == "mean":
                    out.append(sum(vals) / axis_len)
                else:
                    out.append(max(vals))
        return ZenTensor(out, out_shape)

    def sum(self, axis: int | None = None, keepdims: bool = False) -> "ZenTensor | float":
        return self._reduce_axis(axis, "sum", keepdims)

    def mean(self, axis: int | None = None, keepdims: bool = False) -> "ZenTensor | float":
        return self._reduce_axis(axis, "mean", keepdims)

    def max(self, axis: int | None = None, keepdims: bool = False) -> "ZenTensor | float":
        return self._reduce_axis(axis, "max", keepdims)

    def gather(self, indices: Any) -> "ZenTensor":
        if len(self.shape) < 1:
            raise ZenRuntimeError("Cannot gather from a 0D Tensor")
        row_size = _numel(self.shape[1:])
        n_rows = self.shape[0]
        flat_idx: list[int] = []

        def _collect(node: Any) -> tuple[int, ...]:
            if isinstance(node, int) and not isinstance(node, bool):
                idx = node + n_rows if node < 0 else node
                if idx < 0 or idx >= n_rows:
                    raise ZenRuntimeError(f"Gather index {node} out of range [0, {n_rows})")
                flat_idx.append(idx)
                return ()
            if isinstance(node, list):
                if not node:
                    return (0,)
                sub_s: tuple[int, ...] | None = None
                for item in node:
                    s = _collect(item)
                    if sub_s is None:
                        sub_s = s
                    elif s != sub_s:
                        raise ZenRuntimeError("Ragged index list in gather")
                assert sub_s is not None
                return (len(node),) + sub_s
            raise ZenRuntimeError("Gather indices must be ints or nested lists of ints")

        idx_shape = _collect(indices)
        out: list[float] = []
        for idx in flat_idx:
            start = idx * row_size
            out.extend(self.data[start : start + row_size])
        return ZenTensor(out, idx_shape + self.shape[1:])

    @classmethod
    def scatter_add_rows(cls, target_shape: Any, indices: Any, values: "ZenTensor") -> "ZenTensor":
        s = _normalize_shape(target_shape)
        if len(s) < 1:
            raise ZenRuntimeError("scatter_add target_shape must be at least 1D")
        n_rows = s[0]
        row_size = _numel(s[1:])
        out = [0.0] * _numel(s)
        flat_idx: list[int] = []

        def _collect(node: Any) -> None:
            if isinstance(node, int) and not isinstance(node, bool):
                idx = node + n_rows if node < 0 else node
                if idx < 0 or idx >= n_rows:
                    raise ZenRuntimeError(f"Scatter index {node} out of range")
                flat_idx.append(idx)
            elif isinstance(node, list):
                for item in node:
                    _collect(item)
            else:
                raise ZenRuntimeError("Scatter indices must be ints or nested lists of ints")

        _collect(indices)
        if len(flat_idx) * row_size != len(values.data):
            raise ZenRuntimeError("Scatter values size mismatch")
        for pos, idx in enumerate(flat_idx):
            dst = idx * row_size
            src = pos * row_size
            for k in range(row_size):
                out[dst + k] += values.data[src + k]
        return cls(out, s)

    def get_index(self, idx: int) -> "ZenTensor | float":
        if len(self.shape) == 0:
            raise ZenRuntimeError("Cannot index into a 0D Tensor")
        dim0 = self.shape[0]
        if idx < 0:
            idx += dim0
        if idx < 0 or idx >= dim0:
            raise ZenRuntimeError(f"Tensor index {idx} out of bounds for axis 0 of size {dim0}")
        if len(self.shape) == 1:
            return self.data[idx]
        stride = _numel(self.shape[1:])
        start = idx * stride
        return ZenTensor(self.data[start : start + stride], self.shape[1:])

    def __eq__(self, other: object) -> bool:
        if not isinstance(other, ZenTensor):
            return False
        return self.shape == other.shape and self.data == other.data

    def __repr__(self) -> str:
        return f"Tensor({self.tolist()})"
