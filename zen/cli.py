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

from zen import native_kernels
from zen.errors import ZenError
from zen.formatter import format_file
from zen.interpreter import Interpreter, zen_format
from zen.paths import resolve_docs_paths, resolve_guide_pdf_path, resolve_stdlib_path
from zen.spec import VERSION
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
    test_parser.add_argument("dir", nargs="?", default="tests_zen", help="Test directory (default: tests_zen)")

    guide_parser = subparsers.add_parser("guide", help="Open the Learn-ZEN.pdf beginner guide")
    guide_parser.add_argument("--path", action="store_true", help="Print the path to Learn-ZEN.pdf without opening it")

    subparsers.add_parser("docs", help="Print paths to ZENNUM.md and ZENNET.md reference docs")

    subparsers.add_parser("doctor", help="Print ZEN environment and backend diagnostics")

    args = parser.parse_args(raw_argv)
    if args.command == "repl":
        return run_repl(tree_walk=args.tree_walk)

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
        print(f"OK: {file_path}")
        return 0

    if args.command == "test":
        test_dir = Path(args.dir)
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
        print(f"ZEN tests: {passed} passed, {failed} failed")
        return 0 if failed == 0 else 1

    if args.command == "run":
        file_path = Path(args.file)
        if not file_path.is_file():
            print(f"Error: file not found: {file_path}", file=sys.stderr)
            return 1
        engine: VM | Interpreter = Interpreter() if args.tree_walk else VM()
        try:
            engine.run_file(file_path, type_check=not args.no_check)
        except ZenError as exc:
            print(f"ZEN Error: {exc}", file=sys.stderr)
            return 1
        return 0
    return 1


if __name__ == "__main__":
    raise SystemExit(main())
