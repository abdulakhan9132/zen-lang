"""Tests for Phase 12: Bytecode VM, fallback flag, benchmark, REPL, and source-line errors."""

import io
import time
from pathlib import Path

import pytest
from zen.cli import main as cli_main
from zen.cli import run_repl
from zen.errors import ZenRuntimeError, ZenSyntaxError
from zen.interpreter import Interpreter
from zen.vm import VM

ROOT = Path(__file__).resolve().parent.parent


def test_vm_matches_interpreter_across_core_and_ai_examples() -> None:
    for rel in (
        "examples/hello.zen",
        "examples/phase3_closures.zen",
        "examples/phase4_builtins.zen",
        "examples/phase5_tensors.zen",
        "examples/phase6_autograd.zen",
        "examples/phase7_xor.zen",
        "examples/phase9_tensor_autograd.zen",
    ):
        p = ROOT / rel
        buf_interp = io.StringIO()
        buf_vm = io.StringIO()
        Interpreter(stdout=buf_interp).run_file(p)
        VM(stdout=buf_vm).run_file(p)
        assert buf_vm.getvalue() == buf_interp.getvalue(), f"Mismatch on {rel}"


def test_vm_handles_for_break_continue_and_const() -> None:
    buf = io.StringIO()
    VM(stdout=buf).run("""
    let total = 0
    for i in range(10) {
        if i == 2 { continue }
        if i == 5 { break }
        total = total + i
    }
    say total
    """)
    assert buf.getvalue().splitlines() == ["8"]

    with pytest.raises(ZenRuntimeError, match="constant"):
        VM().run("fn f() {\n  const x = 1\n  x = 2\n}\nf()")


def test_vm_is_faster_than_tree_walker_benchmark() -> None:
    p = ROOT / "examples" / "phase12_benchmark.zen"

    t0 = time.perf_counter()
    buf_tw = io.StringIO()
    Interpreter(stdout=buf_tw).run_file(p)
    tw_time = time.perf_counter() - t0

    t1 = time.perf_counter()
    buf_vm = io.StringIO()
    VM(stdout=buf_vm).run_file(p)
    vm_time = time.perf_counter() - t1

    vm_lines = buf_vm.getvalue().splitlines()
    tw_lines = buf_tw.getvalue().splitlines()
    assert vm_lines[0].split(" | ")[0] == tw_lines[0].split(" | ")[0] == "fib total = 225075000"
    assert vm_lines[1].split(" | ")[0] == tw_lines[1].split(" | ")[0]
    assert vm_time < tw_time, f"Expected VM ({vm_time:.3f}s) < tree-walker ({tw_time:.3f}s)"


def test_cli_vm_default_and_tree_walk_fallback(capsys: pytest.CaptureFixture[str]) -> None:
    assert cli_main(["run", str(ROOT / "examples" / "hello.zen")]) == 0
    out_vm = capsys.readouterr().out
    assert cli_main(["run", "--tree-walk", str(ROOT / "examples" / "hello.zen")]) == 0
    out_tw = capsys.readouterr().out
    assert out_vm == out_tw == "I am ZEN\n0.1\n"


def test_repl_multiline_and_error_recovery() -> None:
    stdin = io.StringIO(
        "let x = 10\n"
        "fn add(a, b) {\n"
        "  return a + b\n"
        "}\n"
        "add(x, 5)\n"
        "say unknown_var\n"
        "x * 2\n"
        "exit\n"
    )
    stdout = io.StringIO()
    stderr = io.StringIO()
    code = run_repl(stdin=stdin, stdout=stdout, stderr=stderr)
    assert code == 0
    out_lines = stdout.getvalue().splitlines()
    assert "15" in out_lines
    assert "20" in out_lines
    assert "Undefined variable 'unknown_var'" in stderr.getvalue()


def test_clear_error_messages_include_source_line_and_caret() -> None:
    with pytest.raises(ZenRuntimeError) as exc:
        VM().run("let a = 1\nsay missing_name")
    msg = str(exc.value)
    assert "[line 2, col 5]" in msg
    assert "2 | say missing_name" in msg
    assert "^" in msg

    with pytest.raises(ZenSyntaxError) as exc_syn:
        VM().run("let a = 1\nlet b = ")
    msg_syn = str(exc_syn.value)
    assert "[line 2, col 9]" in msg_syn
    assert "2 | let b =" in msg_syn
    assert "^" in msg_syn
