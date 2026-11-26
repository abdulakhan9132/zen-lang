"""Command-line interface and interactive REPL for the ZEN language."""

from __future__ import annotations

import argparse
import io
import os
import platform
import subprocess
import sys
from pathlib import Path
from typing import TextIO

from zen import nan_trace
from zen import native_kernels
from zen.builtins import reset_recorded_seed
from zen.doctest import run_doctests
from zen.errors import ZenError, format_error_explanation
from zen.formatter import format_file
from zen.interpreter import Interpreter, zen_format
from zen.paths import resolve_docs_paths, resolve_guide_pdf_path, resolve_stdlib_path
from zen.recorder import RunRecorder, replay_manifest, set_active_recorder, write_manifest
from zen.spec import VERSION
from zen.stdlib_rt import set_script_argv
from zen.typechecker import type_check_source
from zen.vm import VM


def _brace_balance(text: str) -> int:
    balance = 0
    in_str = False
    escape = False
    i = 0
    n = len(text)
    while i < n:
        ch = text[i]
        if in_str:
            if escape:
                escape = False
            elif ch == "\\":
                escape = True
            elif ch == '"':
                in_str = False
        else:
            if ch == "/" and i + 1 < n and text[i + 1] == "/":
                while i < n and text[i] != "\n":
                    i += 1
                continue
            if ch == '"':
                in_str = True
            elif ch == "{":
                balance += 1
            elif ch == "}":
                balance -= 1
        i += 1
    return balance


def run_repl(
    stdin: TextIO | None = None,
    stdout: TextIO | None = None,
    stderr: TextIO | None = None,
    *,
    tree_walk: bool = False,
) -> int:
    """Interactive ZEN REPL."""
    in_stream = stdin if stdin is not None else sys.stdin
    out_stream = stdout if stdout is not None else sys.stdout
    err_stream = stderr if stderr is not None else sys.stderr
    interactive = in_stream is sys.stdin and in_stream.isatty()

    engine: VM | Interpreter = Interpreter(stdout=out_stream) if tree_walk else VM(stdout=out_stream)
    out_stream.write(f"ZEN {VERSION} REPL ({'tree-walk' if tree_walk else 'bytecode VM'}). Type 'exit' to quit.\n")
    out_stream.flush()

    buffer: list[str] = []
    while True:
        if interactive:
            out_stream.write("zen> " if not buffer else "...  ")
            out_stream.flush()
        line = in_stream.readline()
        if line == "":
            break
        stripped = line.strip()
        if not buffer and stripped in ("exit", "quit", "exit()", "quit()"):
            break
        if not buffer and not stripped:
            continue

        buffer.append(line)
        src = "".join(buffer)
        if _brace_balance(src) > 0:
            continue

        buffer.clear()
        try:
            if isinstance(engine, VM):
                res = engine.run(src, return_last_expr=True)
                if res is not None:
                    out_stream.write(zen_format(res) + "\n")
            else:
                engine.run(src)
        except ZenError as exc:
            err_stream.write(f"ZEN Error: {exc}\n")
        out_stream.flush()
    return 0


def main(argv: list[str] | None = None) -> int:
    raw_argv = list(sys.argv[1:] if argv is None else argv)
    if not raw_argv:
        return run_repl()

    parser = argparse.ArgumentParser(prog="zen", description="ZEN programming language CLI")
    parser.add_argument("--version", action="version", version=f"zen {VERSION}")
    parser.add_argument("--debug", action="store_true", help="Show raw Python tracebacks on internal errors")
    subparsers = parser.add_subparsers(dest="command", required=True)

    run_parser = subparsers.add_parser("run", help="Run a .zen source file")
    run_parser.add_argument("file", help="Path to the .zen file")
    run_parser.add_argument(
        "--tree-walk",
        action="store_true",
        help="Use the fallback tree-walking interpreter instead of the bytecode VM",
    )
    run_parser.add_argument(
        "--no-check",
        action="store_true",
        help="Skip static type checking before running",
    )
    run_parser.add_argument(
        "--max-depth",
        type=int,
        default=10000,
        help="Maximum function call recursion depth (default: 10000)",
    )
    run_parser.add_argument(
        "--nan-trace",
        action="store_true",
        help="Stop at the first operation producing NaN or Inf and report op, shapes, and line",
    )
    run_parser.add_argument(
        "--record",
        default=None,
        help="Record run manifest (seed, file SHA-256s, output hash) to JSON file",
    )
    run_parser.add_argument(
        "--debug",
        action="store_true",
        help="Show raw Python tracebacks on internal errors",
    )
    run_parser.add_argument(
        "script_args",
        nargs=argparse.REMAINDER,
        help="Arguments passed to env.argv()",
    )

    repl_parser = subparsers.add_parser("repl", help="Start interactive ZEN REPL")
    repl_parser.add_argument(
        "--tree-walk",
        action="store_true",
        help="Use the fallback tree-walking interpreter in the REPL",
    )

    fmt_parser = subparsers.add_parser("fmt", help="Format a .zen source file")
    fmt_parser.add_argument("file", help="Path to the .zen file")
    fmt_parser.add_argument("--check", action="store_true", help="Exit non-zero if formatting is needed")

    check_parser = subparsers.add_parser("check", help="Type-check a .zen source file without running it")
    check_parser.add_argument("file", help="Path to the .zen file")

    test_parser = subparsers.add_parser("test", help="Run all .zen test files in tests_zen/")
    test_parser.add_argument("dir", nargs="?", default=None, help="Test directory or file (default: tests_zen)")
    test_parser.add_argument("--doc", action="store_true", help="Run //> doctests in .zen files")

    replay_parser = subparsers.add_parser("replay", help="Replay a recorded run.json manifest and verify MATCH/MISMATCH")
    replay_parser.add_argument("manifest", help="Path to recorded run.json manifest")

    guide_parser = subparsers.add_parser("guide", help="Open the Learn-ZEN.pdf beginner guide")
    guide_parser.add_argument("--path", action="store_true", help="Print the path to Learn-ZEN.pdf without opening it")

    subparsers.add_parser("docs", help="Print paths to ZENNUM.md and ZENNET.md reference docs")

    explain_parser = subparsers.add_parser("explain", help="Explain a ZEN error code (e.g. E001)")
    explain_parser.add_argument("code", nargs="?", default=None, help="Error code to explain")

    subparsers.add_parser("doctor", help="Print ZEN environment and backend diagnostics")

    args = parser.parse_args(raw_argv)
    debug_mode = bool(getattr(args, "debug", False))
    if args.command == "repl":
        return run_repl(tree_walk=args.tree_walk)

    if args.command == "explain":
        print(format_error_explanation(args.code))
        return 0

    if args.command == "doctor":
        stdlib_path = resolve_stdlib_path()
        pdf_path = resolve_guide_pdf_path()
        backend_mode = "native" if native_kernels.NATIVE_AVAILABLE else "pure-python fallback"
        print(f"ZEN version: {VERSION}")
        print(f"OS: {platform.platform()} ({sys.platform})")
        print(f"Python version: {platform.python_version()} ({sys.executable})")
        print(f"Math backend: {backend_mode} [{native_kernels.NATIVE_BACKEND}]")
        print(f"Stdlib path: {stdlib_path if stdlib_path is not None else 'not found'}")
        print(f"Guide PDF path: {pdf_path if pdf_path is not None else 'not found'}")
        return 0

    if args.command == "docs":
        doc_paths = resolve_docs_paths()
        if not doc_paths:
            print("Error: ZENNUM.md and ZENNET.md not found.", file=sys.stderr)
            return 1
        for name, p in doc_paths.items():
            print(f"{name}: {p}")
        return 0

    if args.command == "guide":
        pdf_path = resolve_guide_pdf_path()
        if pdf_path is None:
            print("Error: Learn-ZEN.pdf not found.", file=sys.stderr)
            return 1
        if args.path:
            print(str(pdf_path))
            return 0
        try:
            if sys.platform.startswith("win"):
                os.startfile(str(pdf_path))  # type: ignore[attr-defined]
            elif sys.platform == "darwin":
                subprocess.run(["open", str(pdf_path)], check=True)
            else:
                subprocess.run(["xdg-open", str(pdf_path)], check=True)
        except Exception:
            print(f"Open the ZEN guide at: {pdf_path}")
        return 0

    if args.command == "fmt":
        file_path = Path(args.file)
        if not file_path.is_file():
            print(f"Error: file not found: {file_path}", file=sys.stderr)
            return 1
        try:
            changed = format_file(str(file_path), check_only=args.check)
        except ZenError as exc:
            print(f"ZEN Error: {exc}", file=sys.stderr)
            return 1
        except Exception:
            if debug_mode:
                raise
            print("internal error, please report", file=sys.stderr)
            return 1
        if args.check and changed:
            print(f"Needs formatting: {file_path}", file=sys.stderr)
            return 1
        return 0

    if args.command == "check":
        file_path = Path(args.file)
        if not file_path.is_file():
            print(f"Error: file not found: {file_path}", file=sys.stderr)
            return 1
        try:
            src = file_path.read_text(encoding="utf-8")
            type_check_source(src, file_path=file_path)
        except ZenError as exc:
            print(f"ZEN Error: {exc}", file=sys.stderr)
            return 1
        except Exception:
            if debug_mode:
                raise
            print("internal error, please report", file=sys.stderr)
            return 1
        print(f"OK: {file_path}")
        return 0

    if args.command == "test":
        if args.doc:
            if args.dir is None:
                doc_paths: list[Path] = []
                for root_d in (Path("docs/guide_examples"), Path("tests_zen")):
                    if root_d.is_dir():
                        doc_paths.extend(sorted(root_d.glob("*.zen")))
            else:
                target = Path(args.dir)
                if target.is_file():
                    doc_paths = [target]
                elif target.is_dir():
                    doc_paths = sorted(target.glob("*.zen"))
                else:
                    print(f"Error: path not found: {target}", file=sys.stderr)
                    return 1
            if not doc_paths:
                print("No .zen files found for doctests", file=sys.stderr)
                return 1
            passed, failed, msgs = run_doctests(doc_paths)
            for m in msgs:
                print(m, file=sys.stderr if m.startswith("FAIL") else sys.stdout)
            print(f"ZEN doctests: {passed} passed, {failed} failed")
            return 0 if failed == 0 else 1

        test_dir = Path(args.dir or "tests_zen")
        if not test_dir.is_dir():
            print(f"Error: test directory not found: {test_dir}", file=sys.stderr)
            return 1
        files = sorted(test_dir.glob("*.zen"))
        if not files:
            print(f"No .zen test files found in {test_dir}", file=sys.stderr)
            return 1
        passed = 0
        failed = 0
        for tf in files:
            try:
                VM(stdout=io.StringIO()).run_file(tf, type_check=True)
                print(f"PASS {tf}")
                passed += 1
            except ZenError as exc:
                print(f"FAIL {tf}: {exc}", file=sys.stderr)
                failed += 1
            except Exception:
                if debug_mode:
                    raise
                print(f"FAIL {tf}: internal error, please report", file=sys.stderr)
                failed += 1
        print(f"ZEN tests: {passed} passed, {failed} failed")
        return 0 if failed == 0 else 1

    if args.command == "replay":
        manifest_path = Path(args.manifest)
        if not manifest_path.is_file():
            print(f"Error: manifest not found: {manifest_path}", file=sys.stderr)
            return 1
        try:
            matched, report = replay_manifest(manifest_path)
            print(report)
            return 0 if matched else 1
        except ZenError as exc:
            print(f"ZEN Error: {exc}", file=sys.stderr)
            return 1
        except Exception:
            if debug_mode:
                raise
            print("internal error, please report", file=sys.stderr)
            return 1

    if args.command == "run":
        file_path = Path(args.file)
        if not file_path.is_file():
            print(f"Error: file not found: {file_path}", file=sys.stderr)
            return 1
        script_args = list(args.script_args)
        if script_args and script_args[0] == "--":
            script_args = script_args[1:]
        set_script_argv([str(file_path)] + script_args)

        recorder: RunRecorder | None = None
        capture_buf: io.StringIO | None = None
        if args.record:
            reset_recorded_seed()
            recorder = RunRecorder(root_dir=Path.cwd().resolve())
            set_active_recorder(recorder)
            capture_buf = io.StringIO()

        nan_trace.set_enabled(bool(args.nan_trace))
        engine: VM | Interpreter = (
            Interpreter(stdout=capture_buf, max_depth=args.max_depth)
            if args.tree_walk
            else VM(stdout=capture_buf, max_depth=args.max_depth)
        )
        try:
            engine.run_file(file_path, type_check=not args.no_check)
            if recorder is not None and capture_buf is not None:
                out_text = capture_buf.getvalue()
                if out_text:
                    sys.stdout.write(out_text)
                write_manifest(
                    Path(args.record),
                    recorder=recorder,
                    entry_file=file_path,
                    script_args=script_args,
                    tree_walk=bool(args.tree_walk),
                    no_check=bool(args.no_check),
                    nan_trace_flag=bool(args.nan_trace),
                    max_depth=int(args.max_depth),
                    stdout_text=out_text,
                )
        except ZenError as exc:
            print(f"ZEN Error: {exc}", file=sys.stderr)
            return 1
        except Exception:
            if debug_mode:
                raise
            line_info = f" at ZEN line {engine.current_line}" if getattr(engine, "current_line", 0) else ""
            print(f"internal error, please report{line_info}", file=sys.stderr)
            return 1
        finally:
            nan_trace.set_enabled(False)
            set_active_recorder(None)
        return 0
    return 1


if __name__ == "__main__":
    raise SystemExit(main())
