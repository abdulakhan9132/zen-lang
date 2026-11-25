"""Tests for Phase 4: Built-ins, math, strings, file I/O, and import."""

import io
from pathlib import Path

import pytest
from zen.errors import ZenRuntimeError
from zen.interpreter import Interpreter

ROOT = Path(__file__).resolve().parent.parent


def run_zen(source: str, base_dir: Path | None = None) -> str:
    buf = io.StringIO()
    interp = Interpreter(stdout=buf, base_dir=base_dir)
    interp.run(source)
    return buf.getvalue()


def test_range_and_list_helpers() -> None:
    out = run_zen("""
    let a = range(4)
    let b = range(1, 6, 2)
    push(a, 99)
    let last = pop(a)
    let doubled = map(b, fn(x) { return x * 2 })
    let big = filter(doubled, fn(x) { return x > 4 })
    say a
    say last
    say doubled
    say big
    say sum(big)
    """)
    assert out.splitlines() == [
        "[0, 1, 2, 3]",
        "99",
        "[2, 6, 10]",
        "[6, 10]",
        "16",
    ]


def test_math_and_seeded_random() -> None:
    out1 = run_zen("""
    seed(123)
    say round(sqrt(25), 1)
    say round(exp(0), 1)
    say round(log(1), 1)
    say round(sin(0), 1)
    say round(cos(0), 1)
    say pow(2, 5)
    say abs(-7)
    say random()
    """)
    out2 = run_zen("""
    seed(123)
    say round(sqrt(25), 1)
    say round(exp(0), 1)
    say round(log(1), 1)
    say round(sin(0), 1)
    say round(cos(0), 1)
    say pow(2, 5)
    say abs(-7)
    say random()
    """)
    assert out1 == out2
    lines = out1.splitlines()
    assert lines[:7] == ["5.0", "1.0", "0.0", "0.0", "1.0", "32.0", "7"]


def test_string_functions() -> None:
    out = run_zen("""
    let s = trim("  Alpha Beta  ")
    say upper(s)
    say lower(s)
    say contains(s, "Beta")
    say replace(s, "Beta", "Gamma")
    say join(split(s, " "), "_")
    say chars("zen")
    say ord("A")
    say chr(65)
    """)
    assert out.splitlines() == [
        "ALPHA BETA",
        "alpha beta",
        "true",
        "Alpha Gamma",
        "Alpha_Beta",
        "['z', 'e', 'n']",
        "65",
        "A",
    ]


def test_file_io_and_import(tmp_path: Path) -> None:
    mod_file = tmp_path / "helper.zen"
    mod_file.write_text("fn triple(x) { return x * 3 }\n", encoding="utf-8")

    data_file = tmp_path / "note.txt"
    main_file = tmp_path / "main.zen"
    main_file.write_text(
        f"""
        import "helper.zen" as h
        write_file("{data_file.as_posix()}", "val=" + str(h.triple(4)))
        say read_file("{data_file.as_posix()}")
        """,
        encoding="utf-8",
    )

    buf = io.StringIO()
    interp = Interpreter(stdout=buf)
    interp.run_file(main_file)
    assert buf.getvalue().splitlines() == ["val=12"]

    with pytest.raises(ZenRuntimeError, match="not found"):
        run_zen('import "does_not_exist.zen"')


def test_phase4_example_runs() -> None:
    buf = io.StringIO()
    interp = Interpreter(stdout=buf)
    interp.run_file(ROOT / "examples" / "phase4_builtins.zen")
    lines = buf.getvalue().splitlines()
    assert lines[0] == "nums: [1, 2, 3, 4, 5]"
    assert lines[1] == "squares: [1, 4, 9, 16, 25]"
    assert lines[2] == "even squares: [4, 16]"
    assert lines[3] == "sqrt(16) = 4.0"
    assert lines[5] == "HELLO-ZEN-WORLD"
