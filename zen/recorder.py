"""Reproducible run recording (`--record`) and replay verification (`zen replay`)."""

from __future__ import annotations

from dataclasses import dataclass, field
import hashlib
import io
import json
from pathlib import Path
import platform
import sys
from typing import Any

from zen.spec import VERSION


def _sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        while True:
            chunk = f.read(65536)
            if not chunk:
                break
            h.update(chunk)
    return h.hexdigest()


def _rel_posix(path: Path, base: Path) -> str:
    try:
        return path.resolve().relative_to(base.resolve()).as_posix()
    except ValueError:
        return path.resolve().as_posix()


@dataclass
class RunRecorder:
    root_dir: Path
    zen_files: dict[str, str] = field(default_factory=dict)
    data_files: dict[str, str] = field(default_factory=dict)

    def add_zen_file(self, path: Path) -> None:
        resolved = path.resolve()
        if resolved.is_file():
            self.zen_files[_rel_posix(resolved, self.root_dir)] = _sha256_file(resolved)

    def add_data_file(self, path: Path) -> None:
        resolved = path.resolve()
        if resolved.is_file():
            self.data_files[_rel_posix(resolved, self.root_dir)] = _sha256_file(resolved)


_ACTIVE_RECORDER: RunRecorder | None = None


def get_active_recorder() -> RunRecorder | None:
    return _ACTIVE_RECORDER


def set_active_recorder(rec: RunRecorder | None) -> None:
    global _ACTIVE_RECORDER
    _ACTIVE_RECORDER = rec


def record_zen_access(path: Path) -> None:
    if _ACTIVE_RECORDER is not None:
        _ACTIVE_RECORDER.add_zen_file(path)


def record_data_access(path: Path) -> None:
    if _ACTIVE_RECORDER is not None:
        _ACTIVE_RECORDER.add_data_file(path)


def write_manifest(
    manifest_path: Path,
    *,
    recorder: RunRecorder,
    entry_file: Path,
    script_args: list[str],
    tree_walk: bool,
    no_check: bool,
    nan_trace_flag: bool,
    max_depth: int,
    stdout_text: str,
) -> dict[str, Any]:
    from zen.builtins import get_recorded_seed

    out_hash = hashlib.sha256(stdout_text.encode("utf-8")).hexdigest()
    manifest: dict[str, Any] = {
        "zen_version": VERSION,
        "os": f"{platform.system()} {platform.release()} ({sys.platform})",
        "python_version": platform.python_version(),
        "seed": get_recorded_seed(),
        "command": {
            "file": _rel_posix(entry_file, recorder.root_dir),
            "args": list(script_args),
            "tree_walk": bool(tree_walk),
            "no_check": bool(no_check),
            "nan_trace": bool(nan_trace_flag),
            "max_depth": int(max_depth),
        },
        "zen_files": dict(sorted(recorder.zen_files.items())),
        "data_files": dict(sorted(recorder.data_files.items())),
        "output_sha256": out_hash,
        "output_lines": len(stdout_text.splitlines()),
    }
    manifest_path.parent.mkdir(parents=True, exist_ok=True)
    manifest_path.write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8", newline="\n")
    return manifest


def replay_manifest(manifest_path: Path, *, cwd: Path | None = None) -> tuple[bool, str]:
    """Replay a recorded manifest and return (matched, human_report)."""
    from zen import nan_trace
    from zen.builtins import reset_recorded_seed
    from zen.interpreter import Interpreter
    from zen.stdlib_rt import set_script_argv
    from zen.vm import VM

    base = (cwd or manifest_path.resolve().parent).resolve()
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    cmd = manifest["command"]
    entry = Path(cmd["file"])
    if not entry.is_absolute():
        cand = (base / entry).resolve()
        if not cand.is_file():
            cand = (Path.cwd() / entry).resolve()
        entry = cand

    input_diffs: list[str] = []
    for rel, expected_hash in {**manifest.get("zen_files", {}), **manifest.get("data_files", {})}.items():
        p = Path(rel)
        if not p.is_absolute():
            p_cand = (base / p).resolve()
            if not p_cand.is_file():
                p_cand = (Path.cwd() / p).resolve()
            p = p_cand
        if not p.is_file():
            input_diffs.append(f"missing input: {rel}")
        else:
            actual_h = _sha256_file(p)
            if actual_h != expected_hash:
                input_diffs.append(f"modified input: {rel}")

    if not entry.is_file():
        return False, f"MISMATCH: entry file not found: {entry}"

    buf = io.StringIO()
    tree_walk = bool(cmd.get("tree_walk", False))
    no_check = bool(cmd.get("no_check", False))
    nan_flag = bool(cmd.get("nan_trace", False))
    max_depth = int(cmd.get("max_depth", 10000))
    script_args = [str(entry)] + [str(a) for a in cmd.get("args", [])]

    reset_recorded_seed()
    set_script_argv(script_args)
    nan_trace.set_enabled(nan_flag)
    try:
        engine: VM | Interpreter = (
            Interpreter(stdout=buf, max_depth=max_depth) if tree_walk else VM(stdout=buf, max_depth=max_depth)
        )
        engine.run_file(entry, type_check=not no_check)
    finally:
        nan_trace.set_enabled(False)

    actual_out = buf.getvalue()
    actual_hash = hashlib.sha256(actual_out.encode("utf-8")).hexdigest()
    expected_out_hash = manifest["output_sha256"]
    matched = actual_hash == expected_out_hash and not input_diffs

    lines = [
        f"{'MATCH' if matched else 'MISMATCH'}: {entry.name}",
        f"  expected output sha256: {expected_out_hash}",
        f"  actual   output sha256: {actual_hash}",
    ]
    if input_diffs:
        lines.append("  input differences: " + ", ".join(input_diffs))
    return matched, "\n".join(lines)
