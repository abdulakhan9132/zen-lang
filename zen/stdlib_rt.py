"""Pure-Python-stdlib backed runtime helpers for ZEN Stage 3 standard library modules."""

from __future__ import annotations

import csv as py_csv
from datetime import datetime as py_datetime
import io
import json as py_json
import os
from pathlib import Path
import platform
import re as py_re
import shutil
import statistics as py_stats
import subprocess
import sys
import urllib.error
import urllib.request
from typing import Any

from zen.errors import ZenRuntimeError
from zen.recorder import record_data_access

_SCRIPT_ARGV: list[str] = []


def set_script_argv(argv: list[str]) -> None:
    global _SCRIPT_ARGV
    _SCRIPT_ARGV = [str(a) for a in argv]


def get_script_argv() -> list[str]:
    return list(_SCRIPT_ARGV)


def make_stdlib_rt(interp: Any, make_builtin: Any) -> Any:
    from zen.interpreter import ZenModule, zen_format

    def _w(name: str, fn: Any) -> Any:
        return make_builtin(name, fn)

    def _resolve(path_str: str) -> Path:
        if not isinstance(path_str, str):
            raise ZenRuntimeError("Expected path string", code="E004")
        p = Path(path_str.replace("\\", "/"))
        if not p.is_file() and not p.is_absolute():
            alt = interp.base_dir / p
            if alt.is_file():
                return alt
        return p

    # --- JSON ---
    def _json_parse(text: str) -> Any:
        if not isinstance(text, str):
            raise ZenRuntimeError("json.parse expects a string", code="E004")
        try:
            return py_json.loads(text)
        except py_json.JSONDecodeError as exc:
            raise ZenRuntimeError(f"Invalid JSON: {exc}", code="E004") from exc

    def _json_stringify(val: Any, indent: int | None = None) -> str:
        try:
            return py_json.dumps(val, indent=indent)
        except TypeError as exc:
            raise ZenRuntimeError(f"Cannot serialize to JSON: {exc}", code="E004") from exc

    # --- Files & Paths ---
    def _file_read(path: str) -> str:
        p = _resolve(path)
        if not p.is_file():
            raise ZenRuntimeError(f"File not found: '{path}'", code="E009")
        record_data_access(p)
        return p.read_text(encoding="utf-8")

    def _file_write(path: str, content: Any) -> None:
        p = Path(path.replace("\\", "/"))
        if not p.is_absolute():
            p = interp.base_dir / p if not p.parent.exists() and interp.base_dir != Path.cwd() else p
        p.parent.mkdir(parents=True, exist_ok=True)
        txt = content if isinstance(content, str) else zen_format(content)
        p.write_text(txt, encoding="utf-8", newline="\n")

    def _file_append(path: str, content: Any) -> None:
        p = Path(path.replace("\\", "/"))
        p.parent.mkdir(parents=True, exist_ok=True)
        txt = content if isinstance(content, str) else zen_format(content)
        with p.open("a", encoding="utf-8", newline="\n") as f:
            f.write(txt)

    def _file_exists(path: str) -> bool:
        p = _resolve(path)
        return p.exists()

    def _file_is_file(path: str) -> bool:
        return _resolve(path).is_file()

    def _file_is_dir(path: str) -> bool:
        return _resolve(path).is_dir()

    def _file_mkdir(path: str) -> None:
        Path(path.replace("\\", "/")).mkdir(parents=True, exist_ok=True)

    def _file_remove(path: str) -> None:
        p = _resolve(path)
        if p.is_dir():
            shutil.rmtree(p)
        elif p.is_file():
            p.unlink()

    def _file_list_dir(path: str = ".") -> list[str]:
        p = _resolve(path)
        if not p.is_dir():
            raise ZenRuntimeError(f"Directory not found: '{path}'", code="E009")
        return sorted(entry.name for entry in p.iterdir())

    def _file_copy(src: str, dst: str) -> None:
        s = _resolve(src)
        d = Path(dst.replace("\\", "/"))
        d.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(s, d)

    def _path_join(parts: Any) -> str:
        if not isinstance(parts, list) or not parts:
            raise ZenRuntimeError("path.join expects a non-empty list of path segments", code="E004")
        return Path(*(str(x).replace("\\", "/") for x in parts)).as_posix()

    # --- Regex ---
    def _re_match(pattern: str, text: str) -> bool:
        return py_re.match(pattern, text) is not None

    def _re_search(pattern: str, text: str) -> str | None:
        m = py_re.search(pattern, text)
        return m.group(0) if m else None

    def _re_find_all(pattern: str, text: str) -> list[Any]:
        return [list(x) if isinstance(x, tuple) else x for x in py_re.findall(pattern, text)]

    def _re_replace(pattern: str, repl: str, text: str) -> str:
        return py_re.sub(pattern, repl, text)

    def _re_split(pattern: str, text: str) -> list[str]:
        return py_re.split(pattern, text)

    # --- Datetime ---
    def _dt_now() -> str:
        return py_datetime.now().strftime("%Y-%m-%d %H:%M:%S")

    def _dt_today() -> str:
        return py_datetime.now().strftime("%Y-%m-%d")

    def _dt_timestamp() -> float:
        return py_datetime.now().timestamp()

    def _dt_format(ts: float, fmt: str = "%Y-%m-%d %H:%M:%S") -> str:
        return py_datetime.fromtimestamp(float(ts)).strftime(fmt)

    def _dt_parse(text: str, fmt: str = "%Y-%m-%d %H:%M:%S") -> float:
        return py_datetime.strptime(text, fmt).timestamp()

    def _dt_iso(ts: float | None = None) -> str:
        dt = py_datetime.now() if ts is None else py_datetime.fromtimestamp(float(ts))
        return dt.isoformat(timespec="seconds")

    # --- CSV ---
    def _csv_parse(text: str) -> list[list[str]]:
        reader = py_csv.reader(io.StringIO(text))
        return [row for row in reader]

    def _csv_stringify(rows: list[list[Any]]) -> str:
        buf = io.StringIO()
        writer = py_csv.writer(buf, lineterminator="\n")
        for row in rows:
            writer.writerow(row)
        return buf.getvalue()

    def _csv_read(path: str) -> list[list[str]]:
        return _csv_parse(_file_read(path))

    def _csv_write(path: str, rows: list[list[Any]]) -> None:
        _file_write(path, _csv_stringify(rows))

    # --- Process ---
    def _proc_run(cmd: Any, cwd: str | None = None) -> dict[str, Any]:
        if isinstance(cmd, list):
            args = [str(x) for x in cmd]
            shell = False
        elif isinstance(cmd, str):
            args = cmd
            shell = True
        else:
            raise ZenRuntimeError("process.run expects a command list or string", code="E004")
        res = subprocess.run(
            args,
            cwd=cwd or str(interp.base_dir),
            shell=shell,
            capture_output=True,
            text=True,
        )
        return {"code": res.returncode, "stdout": res.stdout, "stderr": res.stderr}

    # --- HTTP ---
    def _http_req(method: str, url: str, body: Any = None, headers: dict[str, str] | None = None, timeout: float = 10.0) -> dict[str, Any]:
        hdrs = {str(k): str(v) for k, v in (headers or {}).items()}
        data_bytes: bytes | None = None
        if body is not None:
            if isinstance(body, (dict, list)):
                data_bytes = py_json.dumps(body).encode("utf-8")
                hdrs.setdefault("Content-Type", "application/json")
            elif isinstance(body, str):
                data_bytes = body.encode("utf-8")
            else:
                data_bytes = zen_format(body).encode("utf-8")
        req = urllib.request.Request(url, data=data_bytes, headers=hdrs, method=method)
        try:
            with urllib.request.urlopen(req, timeout=float(timeout)) as resp:
                raw = resp.read().decode("utf-8", errors="replace")
                return {"status": int(resp.status), "body": raw, "headers": dict(resp.headers.items())}
        except urllib.error.HTTPError as exc:
            raw = exc.read().decode("utf-8", errors="replace") if exc.fp else ""
            return {"status": int(exc.code), "body": raw, "headers": dict(exc.headers.items()) if exc.headers else {}}
        except urllib.error.URLError as exc:
            raise ZenRuntimeError(f"HTTP {method} failed for '{url}': {exc.reason}", code="E009") from exc

    # --- Statistics ---
    def _num_seq(xs: Any, fn_name: str) -> list[float]:
        if hasattr(xs, "tolist"):
            xs = xs.tolist()
        if not isinstance(xs, list) or not xs:
            raise ZenRuntimeError(f"stats.{fn_name} expects a non-empty list", code="E004")
        return [float(x) for x in xs]

    def _stats_var(xs: Any, ddof: int = 0) -> float:
        vals = _num_seq(xs, "var")
        return py_stats.variance(vals) if ddof == 1 else py_stats.pvariance(vals)

    def _stats_std(xs: Any, ddof: int = 0) -> float:
        vals = _num_seq(xs, "std")
        return py_stats.stdev(vals) if ddof == 1 else py_stats.pstdev(vals)

    exports = {
        "json_parse": _w("json_parse", _json_parse),
        "json_stringify": _w("json_stringify", _json_stringify),
        "file_read": _w("file_read", _file_read),
        "file_write": _w("file_write", _file_write),
        "file_append": _w("file_append", _file_append),
        "file_exists": _w("file_exists", _file_exists),
        "file_is_file": _w("file_is_file", _file_is_file),
        "file_is_dir": _w("file_is_dir", _file_is_dir),
        "file_mkdir": _w("file_mkdir", _file_mkdir),
        "file_remove": _w("file_remove", _file_remove),
        "file_list_dir": _w("file_list_dir", _file_list_dir),
        "file_copy": _w("file_copy", _file_copy),
        "path_join": _w("path_join", _path_join),
        "path_basename": _w("path_basename", lambda p: Path(str(p).replace("\\", "/")).name),
        "path_dirname": _w("path_dirname", lambda p: Path(str(p).replace("\\", "/")).parent.as_posix()),
        "path_stem": _w("path_stem", lambda p: Path(str(p).replace("\\", "/")).stem),
        "path_ext": _w("path_ext", lambda p: Path(str(p).replace("\\", "/")).suffix),
        "path_resolve": _w("path_resolve", lambda p: Path(str(p).replace("\\", "/")).resolve().as_posix()),
        "path_cwd": _w("path_cwd", lambda: Path.cwd().as_posix()),
        "path_sep": _w("path_sep", lambda: os.sep),
        "re_match": _w("re_match", _re_match),
        "re_search": _w("re_search", _re_search),
        "re_find_all": _w("re_find_all", _re_find_all),
        "re_replace": _w("re_replace", _re_replace),
        "re_split": _w("re_split", _re_split),
        "dt_now": _w("dt_now", _dt_now),
        "dt_today": _w("dt_today", _dt_today),
        "dt_timestamp": _w("dt_timestamp", _dt_timestamp),
        "dt_format": _w("dt_format", _dt_format),
        "dt_parse": _w("dt_parse", _dt_parse),
        "dt_iso": _w("dt_iso", _dt_iso),
        "env_argv": _w("env_argv", get_script_argv),
        "env_get": _w("env_get", lambda k, d=None: os.environ.get(str(k), d)),
        "env_set": _w("env_set", lambda k, v: os.environ.__setitem__(str(k), str(v))),
        "env_has": _w("env_has", lambda k: str(k) in os.environ),
        "env_all": _w("env_all", lambda: dict(os.environ)),
        "env_platform": _w("env_platform", lambda: sys.platform),
        "env_os_name": _w("env_os_name", platform.system),
        "csv_parse": _w("csv_parse", _csv_parse),
        "csv_stringify": _w("csv_stringify", _csv_stringify),
        "csv_read": _w("csv_read", _csv_read),
        "csv_write": _w("csv_write", _csv_write),
        "proc_run": _w("proc_run", _proc_run),
        "http_get": _w("http_get", lambda url, headers=None, timeout=10.0: _http_req("GET", url, None, headers, timeout)),
        "http_post": _w("http_post", lambda url, body=None, headers=None, timeout=10.0: _http_req("POST", url, body, headers, timeout)),
        "stats_mean": _w("stats_mean", lambda xs: py_stats.mean(_num_seq(xs, "mean"))),
        "stats_median": _w("stats_median", lambda xs: py_stats.median(_num_seq(xs, "median"))),
        "stats_std": _w("stats_std", _stats_std),
        "stats_var": _w("stats_var", _stats_var),
        "stats_min": _w("stats_min", lambda xs: min(_num_seq(xs, "min"))),
        "stats_max": _w("stats_max", lambda xs: max(_num_seq(xs, "max"))),
        "stats_sum": _w("stats_sum", lambda xs: sum(_num_seq(xs, "sum"))),
    }
    return ZenModule("_zenstd_rt", exports)
