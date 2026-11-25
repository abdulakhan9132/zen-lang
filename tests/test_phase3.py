"""Tests for Phase 3: Interpreter, scopes, closures, classes, and CLI."""

import io
import subprocess
import sys
from pathlib import Path

import pytest
from zen.cli import main as cli_main
from zen.errors import ZenRuntimeError
from zen.interpreter import Interpreter

ROOT = Path(__file__).resolve().parent.parent


def run_zen(source: str) -> str:
    buf = io.StringIO()
    interp = Interpreter(stdout=buf)
    interp.run(source)
    return buf.getvalue()


def test_milestone_hello_zen() -> None:
    buf = io.StringIO()
    interp = Interpreter(stdout=buf)
    interp.run_file(ROOT / "examples" / "hello.zen")
    assert buf.getvalue().splitlines() == ["I am ZEN", "0.1"]


def test_let_const_and_scopes() -> None:
    out = run_zen("""
    let x = 10
    const y = 5
    if true {
        let x = 99
        say x
    }
    say x + y
    print(x, y)
    """)
    assert out.splitlines() == ["99", "15", "10 5"]

    with pytest.raises(ZenRuntimeError, match="constant"):
        run_zen("const a = 1\na = 2")


def test_closures_and_recursion() -> None:
    out = run_zen("""
    fn make_adder(base) {
        fn add(x) {
            return base + x
        }
        return add
    }
    let add5 = make_adder(5)
    say add5(7)

    fn fib(n) {
        if n <= 1 { return n }
        return fib(n - 1) + fib(n - 2)
    }
    say fib(7)
    """)
    assert out.splitlines() == ["12", "13"]


def test_loops_lists_maps_and_classes() -> None:
    out = run_zen("""
    let xs = [1, 2, 3, 4]
    xs[1] = 20
    let total = 0
    for x in xs {
        if x == 3 { continue }
        total = total + x
    }
    say total

    let m = {"k": 7}
    m["z"] = 3
    say m["k"] + m["z"]

    let i = 0
    while i < 5 {
        if i == 2 { break }
        i = i + 1
    }
    say i

    class Pair {
        fn init(self, a, b) {
            self.a = a
            self.b = b
        }
        fn sum(self) {
            return self.a + self.b
        }
    }
    let p = Pair(4, 6)
    say p.sum()
    """)
    assert out.splitlines() == ["25", "10", "2", "10"]


def test_runtime_errors_report_line() -> None:
    with pytest.raises(ZenRuntimeError) as exc:
        run_zen("let x = 1\nsay y")
    assert exc.value.line == 2

    with pytest.raises(ZenRuntimeError) as exc2:
        run_zen("let a = 1 / 0")
    assert exc2.value.line == 1


def test_cli_run_hello_zen(capsys: pytest.CaptureFixture[str]) -> None:
    code = cli_main(["run", str(ROOT / "examples" / "hello.zen")])
    assert code == 0
    captured = capsys.readouterr()
    assert captured.out.splitlines() == ["I am ZEN", "0.1"]

    res = subprocess.run(
        [sys.executable, "-m", "zen", "run", str(ROOT / "examples" / "hello.zen")],
        cwd=ROOT,
        capture_output=True,
        text=True,
        check=True,
    )
    assert res.stdout.splitlines() == ["I am ZEN", "0.1"]
