"""Standard built-in functions for the ZEN runtime (pure Python stdlib only)."""

from __future__ import annotations

import json
import math
import random
import time
from pathlib import Path
from typing import TYPE_CHECKING, Any

from zen.errors import ZenRuntimeError
from zen import native_kernels
from zen.tensor import ZenTensor
from zen.zennum_core import (
    ZenArray,
    arr_concat,
    arr_outer,
    arr_split,
    arr_stack,
    arr_tril,
    arr_triu,
    arr_where,
)

if TYPE_CHECKING:
    from zen.interpreter import Interpreter, ZenBuiltin

_GLOBAL_RNG = random.Random()
_RECORDED_SEED: list[Any] = []


def get_recorded_seed() -> Any:
    """Return the most recent seed passed to seed() / zn.seed(), or None."""
    return _RECORDED_SEED[-1] if _RECORDED_SEED else None


def reset_recorded_seed() -> None:
    """Clear the recorded seed tracker (used before --record runs)."""
    _RECORDED_SEED.clear()


def register_phase4_builtins(interp: "Interpreter", make_builtin: type["ZenBuiltin"]) -> None:
    """Register Phase 4 built-in functions into the interpreter's global scope."""
    rng = _GLOBAL_RNG

    def _check_num(x: Any, fn_name: str) -> float | int:
        if isinstance(x, bool) or not isinstance(x, (int, float)):
            raise ZenRuntimeError(f"'{fn_name}' expects a number")
        return x

    def _range(*args: Any) -> list[int]:
        if len(args) == 1:
            return list(range(int(_check_num(args[0], "range"))))
        if len(args) == 2:
            return list(range(int(_check_num(args[0], "range")), int(_check_num(args[1], "range"))))
        if len(args) == 3:
            step = int(_check_num(args[2], "range"))
            if step == 0:
                raise ZenRuntimeError("'range' step cannot be 0")
            return list(range(int(_check_num(args[0], "range")), int(_check_num(args[1], "range")), step))
        raise ZenRuntimeError(f"'range' expected 1 to 3 arguments, got {len(args)}")

    def _push(lst: Any, item: Any) -> list[Any]:
        if not isinstance(lst, list):
            raise ZenRuntimeError("'push' expects a list as first argument")
        lst.append(item)
        return lst

    def _pop(lst: Any, idx: Any = None) -> Any:
        if not isinstance(lst, list):
            raise ZenRuntimeError("'pop' expects a list")
        if not lst:
            raise ZenRuntimeError("Cannot pop from an empty list")
        if idx is None:
            return lst.pop()
        if isinstance(idx, bool) or not isinstance(idx, int):
            raise ZenRuntimeError("'pop' index must be an integer")
        try:
            return lst.pop(idx)
        except IndexError as exc:
            raise ZenRuntimeError(f"'pop' index {idx} out of range") from exc

    def _map(lst: Any, fn: Any) -> list[Any]:
        if not isinstance(lst, list):
            raise ZenRuntimeError("'map' expects a list as first argument")
        return [interp._call_value(fn, [item], 0, 0) for item in lst]

    def _filter(lst: Any, fn: Any) -> list[Any]:
        from zen.interpreter import is_truthy

        if not isinstance(lst, list):
            raise ZenRuntimeError("'filter' expects a list as first argument")
        return [item for item in lst if is_truthy(interp._call_value(fn, [item], 0, 0))]

    def _reduce(lst: Any, fn: Any, init: Any) -> Any:
        if not isinstance(lst, list):
            raise ZenRuntimeError("'reduce' expects a list as first argument")
        acc = init
        for item in lst:
            acc = interp._call_value(fn, [acc, item], 0, 0)
        return acc

    def _floor_div(a: Any, b: Any) -> Any:
        na = _check_num(a, "floor_div")
        nb = _check_num(b, "floor_div")
        if nb == 0:
            raise ZenRuntimeError("Division by zero", code="E007")
        return na // nb

    def _input(prompt: Any = "") -> str:
        from zen.interpreter import zen_format

        p_str = "" if prompt is None else zen_format(prompt)
        if p_str:
            interp.stdout.write(p_str)
            interp.stdout.flush()
        in_stream = getattr(interp, "stdin", None)
        if in_stream is None:
            import sys as _sys
            in_stream = _sys.stdin
        line = in_stream.readline()
        return line.rstrip("\r\n")

    def _seed(val: Any) -> None:
        _RECORDED_SEED.append(val)
        rng.seed(val)

    def _random() -> float:
        return rng.random()

    def _randint(a: Any, b: Any) -> int:
        return rng.randint(int(_check_num(a, "randint")), int(_check_num(b, "randint")))

    def _uniform(a: Any, b: Any) -> float:
        return rng.uniform(float(_check_num(a, "uniform")), float(_check_num(b, "uniform")))

    def _min(*args: Any) -> Any:
        if len(args) == 1 and isinstance(args[0], list):
            if not args[0]:
                raise ZenRuntimeError("'min' on empty list")
            return min(args[0])
        if not args:
            raise ZenRuntimeError("'min' expects at least 1 argument")
        return min(args)

    def _max(*args: Any) -> Any:
        if len(args) >= 1 and isinstance(args[0], ZenTensor):
            axis = args[1] if len(args) >= 2 else None
            keepdims = bool(args[2]) if len(args) >= 3 else False
            return args[0].max(axis, keepdims)
        if len(args) == 1 and isinstance(args[0], list):
            if not args[0]:
                raise ZenRuntimeError("'max' on empty list")
            return max(args[0])
        if not args:
            raise ZenRuntimeError("'max' expects at least 1 argument")
        return max(args)

    def _sum(xs: Any, axis: Any = None, keepdims: Any = False) -> Any:
        if isinstance(xs, ZenTensor):
            return xs.sum(axis, bool(keepdims))
        if not isinstance(xs, list):
            raise ZenRuntimeError("'sum' expects a list or Tensor")
        return sum(xs)

    def _mean(xs: Any, axis: Any = None, keepdims: Any = False) -> Any:
        if isinstance(xs, ZenTensor):
            return xs.mean(axis, bool(keepdims))
        if not isinstance(xs, list) or not xs:
            raise ZenRuntimeError("'mean' expects a non-empty list or Tensor")
        return sum(xs) / len(xs)

    def _split(s: Any, sep: Any = None) -> list[str]:
        if not isinstance(s, str):
            raise ZenRuntimeError("'split' expects a string")
        return s.split(sep)

    def _join(lst: Any, sep: Any = "") -> str:
        from zen.interpreter import zen_format

        if not isinstance(lst, list) or not isinstance(sep, str):
            raise ZenRuntimeError("'join' expects (list, string)")
        return sep.join(zen_format(x) for x in lst)

    def _trim(s: Any) -> str:
        if not isinstance(s, str):
            raise ZenRuntimeError("'trim' expects a string")
        return s.strip()

    def _upper(s: Any) -> str:
        if not isinstance(s, str):
            raise ZenRuntimeError("'upper' expects a string")
        return s.upper()

    def _lower(s: Any) -> str:
        if not isinstance(s, str):
            raise ZenRuntimeError("'lower' expects a string")
        return s.lower()

    def _contains(container: Any, item: Any) -> bool:
        if isinstance(container, str):
            if not isinstance(item, str):
                raise ZenRuntimeError("'contains' on string expects a string needle")
            return item in container
        if isinstance(container, (list, dict)):
            return item in container
        raise ZenRuntimeError("'contains' expects a string, list, or map")

    def _replace(s: Any, old: Any, new: Any) -> str:
        if not (isinstance(s, str) and isinstance(old, str) and isinstance(new, str)):
            raise ZenRuntimeError("'replace' expects three strings")
        return s.replace(old, new)

    def _chars(s: Any) -> list[str]:
        if not isinstance(s, str):
            raise ZenRuntimeError("'chars' expects a string")
        return list(s)

    def _ord(ch: Any) -> int:
        if not isinstance(ch, str) or len(ch) != 1:
            raise ZenRuntimeError("'ord' expects a 1-character string")
        return ord(ch)

    def _chr(code: Any) -> str:
        return chr(int(_check_num(code, "chr")))

    def _keys(m: Any) -> list[Any]:
        if not isinstance(m, dict):
            raise ZenRuntimeError("'keys' expects a map")
        return list(m.keys())

    def _values(m: Any) -> list[Any]:
        if not isinstance(m, dict):
            raise ZenRuntimeError("'values' expects a map")
        return list(m.values())

    def _slice(seq: Any, start: Any = 0, end: Any = None) -> Any:
        if not isinstance(seq, (list, str)):
            raise ZenRuntimeError("'slice' expects a list or string")
        s = int(_check_num(start, "slice"))
        e = None if end is None else int(_check_num(end, "slice"))
        return seq[s:e]

    def _resolve_zen_path(path: str) -> Path:
        return Path(path.replace("\\", "/"))

    def _read_file(path: Any) -> str:
        if not isinstance(path, str):
            raise ZenRuntimeError("'read_file' expects a path string")
        p = _resolve_zen_path(path)
        if not p.is_file() and not p.is_absolute():
            alt = interp.base_dir / p
            if alt.is_file():
                p = alt
        if not p.is_file():
            raise ZenRuntimeError(f"File not found: '{path}'")
        return p.read_text(encoding="utf-8")

    def _write_file(path: Any, content: Any) -> None:
        from zen.interpreter import zen_format

        if not isinstance(path, str):
            raise ZenRuntimeError("'write_file' expects a path string")
        p = _resolve_zen_path(path)
        p.parent.mkdir(parents=True, exist_ok=True)
        p.write_text(
            content if isinstance(content, str) else zen_format(content),
            encoding="utf-8",
            newline="\n",
        )

    def _file_exists(path: Any) -> bool:
        if not isinstance(path, str):
            raise ZenRuntimeError("'file_exists' expects a path string")
        p = _resolve_zen_path(path)
        if p.is_file():
            return True
        if not p.is_absolute():
            return (interp.base_dir / p).is_file()
        return False

    def _serialize_weights(obj: Any) -> Any:
        from zen.interpreter import ZenInstance

        if isinstance(obj, ZenTensor):
            return {"__zen_tensor__": True, "shape": list(obj.shape), "data": obj.data}
        if isinstance(obj, ZenInstance) and "data" in obj.fields and isinstance(obj.fields["data"], ZenTensor):
            t = obj.fields["data"]
            return {"__zen_tensor__": True, "shape": list(t.shape), "data": t.data}
        if isinstance(obj, list):
            return [_serialize_weights(x) for x in obj]
        if isinstance(obj, dict):
            return {str(k): _serialize_weights(v) for k, v in obj.items()}
        if obj is None or isinstance(obj, (bool, int, float, str)):
            return obj
        raise ZenRuntimeError(f"Cannot serialize object of type '{type(obj).__name__}' to weights file")

    def _deserialize_weights(obj: Any) -> Any:
        if isinstance(obj, list):
            return [_deserialize_weights(x) for x in obj]
        if isinstance(obj, dict):
            if obj.get("__zen_tensor__") is True:
                return ZenTensor([float(x) for x in obj["data"]], tuple(int(d) for d in obj["shape"]))
            return {k: _deserialize_weights(v) for k, v in obj.items()}
        return obj

    def _save_weights(path: Any, payload: Any) -> None:
        if not isinstance(path, str):
            raise ZenRuntimeError("'save_weights' expects a file path string")
        p = _resolve_zen_path(path)
        p.parent.mkdir(parents=True, exist_ok=True)
        serialized = _serialize_weights(payload)
        p.write_text(json.dumps(serialized), encoding="utf-8", newline="\n")

    def _load_weights(path: Any) -> Any:
        if not isinstance(path, str):
            raise ZenRuntimeError("'load_weights' expects a file path string")
        p = _resolve_zen_path(path)
        if not p.is_file() and not p.is_absolute():
            alt = interp.base_dir / p
            if alt.is_file():
                p = alt
        if not p.is_file():
            raise ZenRuntimeError(f"Weights file not found: '{path}'")
        return _deserialize_weights(json.loads(p.read_text(encoding="utf-8")))

    def _assert(cond: Any, message: Any = "Assertion failed") -> None:
        from zen.interpreter import is_truthy

        if not is_truthy(cond):
            raise ZenRuntimeError(str(message))

    def _fields(obj: Any) -> dict[str, Any]:
        from zen.interpreter import ZenInstance

        if not isinstance(obj, ZenInstance):
            raise ZenRuntimeError("'fields' expects a class instance")
        return dict(obj.fields)

    def _make_zennum_rt():
        from zen.interpreter import ZenModule

        def _wrap(name: str, fn: Any) -> Any:
            return make_builtin(name, fn)

        rt_exports = {
            "array": _wrap("array", lambda data, dtype=None: ZenArray.from_list(data, dtype)),
            "zeros": _wrap("zeros", lambda shape, dtype="float64": ZenArray.zeros(shape, dtype)),
            "ones": _wrap("ones", lambda shape, dtype="float64": ZenArray.ones(shape, dtype)),
            "full": _wrap("full", lambda shape, val, dtype=None: ZenArray.full(shape, val, dtype)),
            "arange": _wrap("arange", lambda start, stop=None, step=1, dtype=None: ZenArray.arange(start, stop, step, dtype)),
            "linspace": _wrap("linspace", lambda start, stop, num=50, dtype="float64": ZenArray.linspace(start, stop, num, dtype)),
            "eye": _wrap("eye", lambda n, m=None, dtype="float64": ZenArray.eye(n, m, dtype)),
            "rand": _wrap("rand", lambda shape, dtype="float64": ZenArray.rand(shape, rng, dtype)),
            "randn": _wrap("randn", lambda shape, dtype="float64": ZenArray.randn(shape, rng, dtype)),
            "seed": _wrap("seed", _seed),
            "reshape": _wrap("reshape", lambda a, s: ZenArray.from_list(a).reshape(s)),
            "transpose": _wrap("transpose", lambda a, axes=None: ZenArray.from_list(a).transpose(axes)),
            "concat": _wrap("concat", lambda arrs, axis=0: arr_concat([ZenArray.from_list(x) for x in arrs], axis)),
            "stack": _wrap("stack", lambda arrs, axis=0: arr_stack([ZenArray.from_list(x) for x in arrs], axis)),
            "split": _wrap("split", lambda a, sec, axis=0: arr_split(ZenArray.from_list(a), sec, axis)),
            "tril": _wrap("tril", lambda a, k=0: arr_tril(ZenArray.from_list(a), k)),
            "triu": _wrap("triu", lambda a, k=0: arr_triu(ZenArray.from_list(a), k)),
            "where": _wrap("where", lambda c, x, y: arr_where(ZenArray.from_list(c, "bool"), x, y)),
            "clip": _wrap("clip", lambda a, lo, hi: ZenArray.from_list(a).clip(lo, hi)),
            "abs": _wrap("abs", lambda a: ZenArray.from_list(a).unary_math("abs")),
            "sqrt": _wrap("sqrt", lambda a: ZenArray.from_list(a).unary_math("sqrt")),
            "exp": _wrap("exp", lambda a: ZenArray.from_list(a).unary_math("exp")),
            "log": _wrap("log", lambda a: ZenArray.from_list(a).unary_math("log")),
            "sin": _wrap("sin", lambda a: ZenArray.from_list(a).unary_math("sin")),
            "cos": _wrap("cos", lambda a: ZenArray.from_list(a).unary_math("cos")),
            "tanh": _wrap("tanh", lambda a: ZenArray.from_list(a).unary_math("tanh")),
            "pow": _wrap("pow", lambda a, p: ZenArray.from_list(a).pow(p)),
            "sum": _wrap("sum", lambda a, axis=None, keepdims=False: ZenArray.from_list(a).reduce("sum", axis, bool(keepdims))),
            "mean": _wrap("mean", lambda a, axis=None, keepdims=False: ZenArray.from_list(a).reduce("mean", axis, bool(keepdims))),
            "std": _wrap("std", lambda a, axis=None, keepdims=False: ZenArray.from_list(a).reduce("std", axis, bool(keepdims))),
            "var": _wrap("var", lambda a, axis=None, keepdims=False: ZenArray.from_list(a).reduce("var", axis, bool(keepdims))),
            "min": _wrap("min", lambda a, axis=None, keepdims=False: ZenArray.from_list(a).reduce("min", axis, bool(keepdims))),
            "max": _wrap("max", lambda a, axis=None, keepdims=False: ZenArray.from_list(a).reduce("max", axis, bool(keepdims))),
            "argmax": _wrap("argmax", lambda a, axis=None, keepdims=False: ZenArray.from_list(a).reduce("argmax", axis, bool(keepdims))),
            "argmin": _wrap("argmin", lambda a, axis=None, keepdims=False: ZenArray.from_list(a).reduce("argmin", axis, bool(keepdims))),
            "sort": _wrap("sort", lambda a, axis=-1: ZenArray.from_list(a).sort(axis)),
            "cumsum": _wrap("cumsum", lambda a, axis=None: ZenArray.from_list(a).cumsum(axis)),
            "dot": _wrap("dot", lambda a, b: ZenArray.from_list(a).matmul(ZenArray.from_list(b))),
            "matmul": _wrap("matmul", lambda a, b: ZenArray.from_list(a).matmul(ZenArray.from_list(b))),
            "outer": _wrap("outer", lambda a, b: arr_outer(ZenArray.from_list(a), ZenArray.from_list(b))),
            "norm": _wrap("norm", lambda a, axis=None, keepdims=False: ZenArray.from_list(a).reduce("norm", axis, bool(keepdims))),
            "gather": _wrap("gather", lambda table, idx: _arr_gather(ZenArray.from_list(table), idx)),
            "scatter_add": _wrap("scatter_add", lambda shape, idx, vals: _arr_scatter_add(shape, idx, ZenArray.from_list(vals))),
            "backend": native_kernels.NATIVE_BACKEND,
        }
        return ZenModule("_zennum_rt", rt_exports)

    def _arr_gather(table: ZenArray, indices: Any) -> ZenArray:
        if len(table.shape) < 1:
            raise ZenRuntimeError("Cannot gather from 0D array")
        n_rows = table.shape[0]
        row_size = math.prod(table.shape[1:])
        flat_idx: list[int] = []

        def _collect(node: Any) -> tuple[int, ...]:
            if isinstance(node, int) and not isinstance(node, bool):
                idx = node + n_rows if node < 0 else node
                if idx < 0 or idx >= n_rows:
                    raise ZenRuntimeError(f"Gather index {node} out of bounds for size {n_rows}")
                flat_idx.append(idx)
                return ()
            if isinstance(node, ZenArray):
                return _collect(node.tolist())
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
            raise ZenRuntimeError("Gather indices must be ints or nested lists")

        idx_shape = _collect(indices)
        out_vals: list[Any] = []
        for idx in flat_idx:
            start = idx * row_size
            out_vals.extend(table.data[start : start + row_size])
        from zen.zennum_core import _make_buffer
        return ZenArray(_make_buffer(out_vals, table.dtype), idx_shape + table.shape[1:], table.dtype)

    def _arr_scatter_add(target_shape: Any, indices: Any, values: ZenArray) -> ZenArray:
        from zen.zennum_core import _normalize_shape, _zeros_buffer
        s = _normalize_shape(target_shape)
        n_rows = s[0]
        row_size = math.prod(s[1:])
        out = _zeros_buffer(math.prod(s), values.dtype)
        flat_idx: list[int] = []

        def _collect(node: Any) -> None:
            if isinstance(node, int) and not isinstance(node, bool):
                idx = node + n_rows if node < 0 else node
                if idx < 0 or idx >= n_rows:
                    raise ZenRuntimeError(f"Scatter index {node} out of bounds")
                flat_idx.append(idx)
            elif isinstance(node, ZenArray):
                _collect(node.tolist())
            elif isinstance(node, list):
                for item in node:
                    _collect(item)
            else:
                raise ZenRuntimeError("Scatter indices must be ints or nested lists")

        _collect(indices)
        if len(flat_idx) * row_size != values.size:
            raise ZenRuntimeError("Scatter values size mismatch")
        for pos, idx in enumerate(flat_idx):
            dst = idx * row_size
            src = pos * row_size
            for k in range(row_size):
                out[dst + k] += values.data[src + k]
        return ZenArray(out, s, values.dtype)

    builtins: dict[str, Any] = {
        "range": _range,
        "push": _push,
        "append": _push,
        "pop": _pop,
        "map": _map,
        "filter": _filter,
        "reduce": _reduce,
        "sqrt": lambda x: x.sqrt() if isinstance(x, ZenTensor) else math.sqrt(_check_num(x, "sqrt")),
        "exp": lambda x: x.exp() if isinstance(x, ZenTensor) else math.exp(_check_num(x, "exp")),
        "log": lambda x, base=None: (
            x.log()
            if isinstance(x, ZenTensor) and base is None
            else (
                math.log(_check_num(x, "log"))
                if base is None
                else math.log(_check_num(x, "log"), _check_num(base, "log"))
            )
        ),
        "sin": lambda x: math.sin(_check_num(x, "sin")),
        "cos": lambda x: math.cos(_check_num(x, "cos")),
        "tanh": lambda x: x.tanh() if isinstance(x, ZenTensor) else math.tanh(_check_num(x, "tanh")),
        "pow": lambda x, y: math.pow(_check_num(x, "pow"), _check_num(y, "pow")),
        "abs": lambda x: abs(_check_num(x, "abs")),
        "floor": lambda x: math.floor(_check_num(x, "floor")),
        "floor_div": _floor_div,
        "ceil": lambda x: math.ceil(_check_num(x, "ceil")),
        "round": lambda x, n=0: round(_check_num(x, "round"), int(_check_num(n, "round"))) if n else round(_check_num(x, "round")),
        "min": _min,
        "max": _max,
        "sum": _sum,
        "mean": _mean,
        "tensor": lambda data: ZenTensor.from_list(data),
        "zeros": lambda shape: ZenTensor.zeros(shape),
        "ones": lambda shape: ZenTensor.ones(shape),
        "tensor_rand": lambda shape: ZenTensor.rand(shape, rng),
        "randn": lambda shape: ZenTensor.randn(shape, rng),
        "shape": lambda t: list(t.shape) if isinstance(t, ZenTensor) else (_ for _ in ()).throw(ZenRuntimeError("'shape' expects a Tensor")),
        "reshape": lambda t, s: t.reshape(s) if isinstance(t, ZenTensor) else (_ for _ in ()).throw(ZenRuntimeError("'reshape' expects a Tensor")),
        "transpose": lambda t, axes=None: t.transpose(axes) if isinstance(t, ZenTensor) else (_ for _ in ()).throw(ZenRuntimeError("'transpose' expects a Tensor")),
        "matmul": lambda a, b: a.matmul(b) if isinstance(a, ZenTensor) else (_ for _ in ()).throw(ZenRuntimeError("'matmul' expects Tensors")),
        "tolist": lambda t: t.tolist() if isinstance(t, ZenTensor) else (_ for _ in ()).throw(ZenRuntimeError("'tolist' expects a Tensor")),
        "tensor_sum": lambda t, axis=None, keepdims=False: t.sum(axis, bool(keepdims)) if isinstance(t, ZenTensor) else (_ for _ in ()).throw(ZenRuntimeError("'tensor_sum' expects a Tensor")),
        "tensor_mean": lambda t, axis=None, keepdims=False: t.mean(axis, bool(keepdims)) if isinstance(t, ZenTensor) else (_ for _ in ()).throw(ZenRuntimeError("'tensor_mean' expects a Tensor")),
        "tensor_max": lambda t, axis=None, keepdims=False: t.max(axis, bool(keepdims)) if isinstance(t, ZenTensor) else (_ for _ in ()).throw(ZenRuntimeError("'tensor_max' expects a Tensor")),
        "tensor_exp": lambda t: t.exp() if isinstance(t, ZenTensor) else (_ for _ in ()).throw(ZenRuntimeError("'tensor_exp' expects a Tensor")),
        "tensor_log": lambda t: t.log() if isinstance(t, ZenTensor) else (_ for _ in ()).throw(ZenRuntimeError("'tensor_log' expects a Tensor")),
        "tensor_tanh": lambda t: t.tanh() if isinstance(t, ZenTensor) else (_ for _ in ()).throw(ZenRuntimeError("'tensor_tanh' expects a Tensor")),
        "tensor_gt": lambda t, val=0.0: t.gt(val) if isinstance(t, ZenTensor) else (_ for _ in ()).throw(ZenRuntimeError("'tensor_gt' expects a Tensor")),
        "tensor_clamp_min": lambda t, val=0.0: t.clamp_min(val) if isinstance(t, ZenTensor) else (_ for _ in ()).throw(ZenRuntimeError("'tensor_clamp_min' expects a Tensor")),
        "tensor_gather": lambda t, indices: t.gather(indices) if isinstance(t, ZenTensor) else (_ for _ in ()).throw(ZenRuntimeError("'tensor_gather' expects a Tensor")),
        "tensor_scatter_add": lambda shape, indices, vals: ZenTensor.scatter_add_rows(shape, indices, vals),
        "seed": _seed,
        "random": _random,
        "randint": _randint,
        "uniform": _uniform,
        "int": lambda x: int(x) if not isinstance(x, bool) else int(x),
        "float": lambda x: float(x),
        "split": _split,
        "join": _join,
        "trim": _trim,
        "upper": _upper,
        "lower": _lower,
        "contains": _contains,
        "replace": _replace,
        "chars": _chars,
        "ord": _ord,
        "chr": _chr,
        "input": _input,
        "keys": _keys,
        "values": _values,
        "slice": _slice,
        "read_file": _read_file,
        "write_file": _write_file,
        "file_exists": _file_exists,
        "save_weights": _save_weights,
        "load_weights": _load_weights,
        "fields": _fields,
        "clock": time.perf_counter,
    }

    for name, fn in builtins.items():
        interp.globals.define(name, make_builtin(name, fn))
    interp.globals.define("_zennum_rt", _make_zennum_rt())
