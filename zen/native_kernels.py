"""Optional native C kernel loader with automatic pure-Python fallback.

Uses flat `array('d')` / `array('f')` buffers. Set `ZEN_NO_NATIVE=1` to force pure-Python mode.
"""

from __future__ import annotations

from array import array
import ctypes
import math
import operator
import os
from pathlib import Path
import shutil
import subprocess
import sys

_NATIVE_DIR = Path(__file__).resolve().parent / "native"


def _shared_lib_candidates() -> list[Path]:
    names = (
        ["zen_kernels.dll"]
        if sys.platform.startswith("win")
        else (["zen_kernels.dylib", "zen_kernels.so"] if sys.platform == "darwin" else ["zen_kernels.so"])
    )
    candidates: list[Path] = []
    for base in (_NATIVE_DIR, Path(__file__).resolve().parent):
        for name in names:
            candidates.append(base / name)
        candidates.extend(base.glob("*zen_kernels*.pyd"))
        candidates.extend(base.glob("*zen_kernels*.so"))
    return candidates


def detect_c_compiler() -> str | None:
    """Return path to available C compiler (gcc, clang, or MSVC cl), or None."""
    for tool in ("gcc", "clang", "cl"):
        found = shutil.which(tool)
        if found:
            return found
    return None


def compile_native_if_available() -> Path | None:
    """Attempt to compile zen_kernels.c into a shared library if a C compiler exists."""
    c_src = _NATIVE_DIR / "zen_kernels.c"
    if not c_src.is_file():
        return None
    compiler = detect_c_compiler()
    if compiler is None:
        return None
    out_name = "zen_kernels.dll" if sys.platform.startswith("win") else ("zen_kernels.dylib" if sys.platform == "darwin" else "zen_kernels.so")
    out_path = _NATIVE_DIR / out_name
    base_name = Path(compiler).stem.lower()
    try:
        if "cl" == base_name:
            cmd = [compiler, "/O2", "/LD", str(c_src), f"/Fe:{out_path}"]
        else:
            cmd = [compiler, "-O3", "-shared", "-fPIC", str(c_src), "-o", str(out_path), "-lm"]
        res = subprocess.run(cmd, cwd=_NATIVE_DIR, capture_output=True, text=True)
        if res.returncode == 0 and out_path.is_file():
            return out_path
    except OSError:
        return None
    return None


def _load_native_lib():
    if os.environ.get("ZEN_NO_NATIVE") == "1":
        return None, "pure-python (ZEN_NO_NATIVE=1)"
    for cand in _shared_lib_candidates():
        if cand.is_file():
            try:
                lib = ctypes.CDLL(str(cand))
                c_dp = ctypes.POINTER(ctypes.c_double)
                c_ip = ctypes.POINTER(ctypes.c_int)
                lib.zen_matmul_f64.argtypes = [c_dp, c_dp, c_dp, ctypes.c_int, ctypes.c_int, ctypes.c_int]
                lib.zen_add_f64.argtypes = [c_dp, c_dp, c_dp, ctypes.c_int]
                lib.zen_sub_f64.argtypes = [c_dp, c_dp, c_dp, ctypes.c_int]
                lib.zen_mul_f64.argtypes = [c_dp, c_dp, c_dp, ctypes.c_int]
                lib.zen_div_f64.argtypes = [c_dp, c_dp, c_dp, ctypes.c_int]
                lib.zen_exp_f64.argtypes = [c_dp, c_dp, ctypes.c_int]
                lib.zen_log_f64.argtypes = [c_dp, c_dp, ctypes.c_int]
                lib.zen_tanh_f64.argtypes = [c_dp, c_dp, ctypes.c_int]
                lib.zen_relu_f64.argtypes = [c_dp, c_dp, ctypes.c_int]
                lib.zen_sum_f64.argtypes = [c_dp, ctypes.c_int]
                lib.zen_sum_f64.restype = ctypes.c_double
                lib.zen_max_f64.argtypes = [c_dp, ctypes.c_int]
                lib.zen_max_f64.restype = ctypes.c_double
                lib.zen_min_f64.argtypes = [c_dp, ctypes.c_int]
                lib.zen_min_f64.restype = ctypes.c_double
                lib.zen_softmax_2d_f64.argtypes = [c_dp, c_dp, ctypes.c_int, ctypes.c_int]
                lib.zen_layernorm_2d_f64.argtypes = [c_dp, c_dp, c_dp, c_dp, ctypes.c_int, ctypes.c_int, ctypes.c_double]
                lib.zen_embedding_f64.argtypes = [c_dp, c_ip, c_dp, ctypes.c_int, ctypes.c_int]
                return lib, f"native ({cand.name})"
            except OSError:
                continue
    return None, "pure-python (no native library loaded)"


_LIB, NATIVE_BACKEND = _load_native_lib()
NATIVE_AVAILABLE = _LIB is not None


def _ptr_d(buf: array) -> ctypes.POINTER(ctypes.c_double):
    return (ctypes.c_double * len(buf)).from_buffer(buf)


def matmul_2d_buf(a: array, a_off: int, m: int, k: int, b: array, b_off: int, n: int) -> array:
    out = array("d", [0.0]) * (m * n)
    if _LIB is not None and os.environ.get("ZEN_NO_NATIVE") != "1" and a_off == 0 and b_off == 0:
        _LIB.zen_matmul_f64(_ptr_d(a), _ptr_d(b), _ptr_d(out), m, k, n)
        return out
    mul = operator.mul
    b_end = b_off + k * n
    b_cols = [list(b[b_off + j : b_end : n]) for j in range(n)]
    for i in range(m):
        a_row = list(a[a_off + i * k : a_off + (i + 1) * k])
        row_off = i * n
        for j in range(n):
            out[row_off + j] = sum(map(mul, a_row, b_cols[j]))
    return out


def softmax_2d_buf(x: array, rows: int, cols: int) -> array:
    out = array("d", [0.0]) * (rows * cols)
    if _LIB is not None and os.environ.get("ZEN_NO_NATIVE") != "1":
        _LIB.zen_softmax_2d_f64(_ptr_d(x), _ptr_d(out), rows, cols)
        return out
    m_exp = math.exp
    for r in range(rows):
        off = r * cols
        row = x[off : off + cols]
        mv = max(row)
        exps = [m_exp(v - mv) for v in row]
        inv = 1.0 / sum(exps)
        for c in range(cols):
            out[off + c] = exps[c] * inv
    return out


def layernorm_2d_buf(
    x: array,
    gamma: array | None,
    beta: array | None,
    rows: int,
    cols: int,
    eps: float = 1e-5,
) -> array:
    out = array("d", [0.0]) * (rows * cols)
    if _LIB is not None and os.environ.get("ZEN_NO_NATIVE") != "1":
        gp = _ptr_d(gamma) if gamma is not None else ctypes.POINTER(ctypes.c_double)()
        bp = _ptr_d(beta) if beta is not None else ctypes.POINTER(ctypes.c_double)()
        _LIB.zen_layernorm_2d_f64(_ptr_d(x), gp, bp, _ptr_d(out), rows, cols, eps)
        return out
    m_sqrt = math.sqrt
    for r in range(rows):
        off = r * cols
        row = x[off : off + cols]
        mean = sum(row) / cols
        var = sum((v - mean) * (v - mean) for v in row) / cols
        inv_std = 1.0 / m_sqrt(var + eps)
        for c in range(cols):
            norm = (row[c] - mean) * inv_std
            g = gamma[c] if gamma is not None else 1.0
            b = beta[c] if beta is not None else 0.0
            out[off + c] = norm * g + b
    return out


def embedding_lookup_buf(table: array, indices: list[int], dim: int) -> array:
    n_idx = len(indices)
    out = array("d", [0.0]) * (n_idx * dim)
    if _LIB is not None and os.environ.get("ZEN_NO_NATIVE") != "1":
        idx_arr = array("i", indices)
        idx_ptr = (ctypes.c_int * n_idx).from_buffer(idx_arr)
        _LIB.zen_embedding_f64(_ptr_d(table), idx_ptr, _ptr_d(out), n_idx, dim)
        return out
    for i, idx in enumerate(indices):
        src = idx * dim
        dst = i * dim
        out[dst : dst + dim] = table[src : src + dim]
    return out
