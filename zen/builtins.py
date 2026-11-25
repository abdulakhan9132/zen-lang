"""Standard built-in functions for the ZEN runtime (pure Python stdlib only)."""

from __future__ import annotations

import json
import math
import random
import time
from pathlib import Path
from typing import TYPE_CHECKING, Any

from zen.errors import ZenRuntimeError
from zen.tensor import ZenTensor

if TYPE_CHECKING:
    from zen.interpreter import Interpreter, ZenBuiltin


def register_phase4_builtins(interp: "Interpreter", make_builtin: type["ZenBuiltin"]) -> None:
    """Register Phase 4 built-in functions into the interpreter's global scope."""
    rng = random.Random()

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

    def _seed(val: Any) -> None:
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

    def _read_file(path: Any) -> str:
        if not isinstance(path, str):
            raise ZenRuntimeError("'read_file' expects a path string")
        p = Path(path)
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
        p = Path(path)
        p.parent.mkdir(parents=True, exist_ok=True)
        p.write_text(content if isinstance(content, str) else zen_format(content), encoding="utf-8")

    def _file_exists(path: Any) -> bool:
        if not isinstance(path, str):
            raise ZenRuntimeError("'file_exists' expects a path string")
        p = Path(path)
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
        p = Path(path)
        p.parent.mkdir(parents=True, exist_ok=True)
        serialized = _serialize_weights(payload)
        p.write_text(json.dumps(serialized), encoding="utf-8")

    def _load_weights(path: Any) -> Any:
        if not isinstance(path, str):
            raise ZenRuntimeError("'load_weights' expects a file path string")
        p = Path(path)
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
        "keys": _keys,
        "values": _values,
        "slice": _slice,
        "read_file": _read_file,
        "write_file": _write_file,
        "file_exists": _file_exists,
        "save_weights": _save_weights,
        "load_weights": _load_weights,
        "clock": time.perf_counter,
    }

    for name, fn in builtins.items():
        interp.globals.define(name, make_builtin(name, fn))
