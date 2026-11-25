"""Tests for Phase 15: assert statement, zen fmt, zen check, zen test, and tests_zen/."""

from pathlib import Path

import pytest
from zen.cli import main as cli_main
from zen.errors import ZenRuntimeError
from zen.formatter import format_source
from zen.interpreter import Interpreter
from zen.vm import VM

ROOT = Path(__file__).resolve().parent.parent


def test_assert_statement_passes_and_fails_with_clear_message() -> None:
    VM().run('assert 2 + 2 == 4, "math broke"')
    Interpreter().run('assert 2 + 2 == 4, "math broke"')

    for engine in (VM(), Interpreter()):
        with pytest.raises(ZenRuntimeError) as exc:
            engine.run('let x = 1\nassert x == 2, "x must be 2"')
        msg = str(exc.value)
        assert "Assertion failed: x must be 2" in msg
        assert "[line 2, col 1]" in msg
        assert '2 | assert x == 2, "x must be 2"' in msg
        assert "^" in msg


def test_zen_fmt_formats_consistently_and_is_idempotent(tmp_path: Path) -> None:
    messy = (
        "// header comment\n"
        "fn add(a:int,b:int)->int{\n"
        "let x=a+b// inline note\n"
        "if x>0{\n"
        "return x\n"
        "}else{\n"
        "return -x\n"
        "}\n"
        "}\n"
    )
    formatted = format_source(messy)
    assert "fn add(a: int, b: int) -> int {" in formatted
    assert "    let x = a + b  // inline note" in formatted
    assert "    if x > 0 {" in formatted
    assert "        return -x" in formatted
    assert format_source(formatted) == formatted

    f = tmp_path / "messy.zen"
    f.write_text(messy, encoding="utf-8")
    assert cli_main(["fmt", "--check", str(f)]) == 1
    assert cli_main(["fmt", str(f)]) == 0
    assert cli_main(["fmt", "--check", str(f)]) == 0


def test_zen_check_cli(tmp_path: Path, capsys: pytest.CaptureFixture[str]) -> None:
    good = ROOT / "examples" / "phase15_tools.zen"
    assert cli_main(["check", str(good)]) == 0
    assert "OK:" in capsys.readouterr().out

    bad = tmp_path / "bad_types.zen"
    bad.write_text('let s: int = "hello"\n', encoding="utf-8")
    assert cli_main(["check", str(bad)]) == 1
    err = capsys.readouterr().err
    assert "cannot assign 'string' to 'int'" in err


def test_zen_test_runs_tests_zen_directory(tmp_path: Path, capsys: pytest.CaptureFixture[str]) -> None:
    zen_tests = sorted((ROOT / "tests_zen").glob("*.zen"))
    assert len(zen_tests) >= 5

    assert cli_main(["test", str(ROOT / "tests_zen")]) == 0
    out = capsys.readouterr().out
    assert f"ZEN tests: {len(zen_tests)} passed, 0 failed" in out

    # Failing test file returns non-zero
    fail_file = tmp_path / "test_fail.zen"
    fail_file.write_text('assert 1 == 2, "expected failure"\n', encoding="utf-8")
    assert cli_main(["test", str(tmp_path)]) == 1
    err = capsys.readouterr().err
    assert "Assertion failed: expected failure" in err


def test_phase15_example_runs(capsys: pytest.CaptureFixture[str]) -> None:
    assert cli_main(["run", str(ROOT / "examples" / "phase15_tools.zen")]) == 0
    assert capsys.readouterr().out.strip() == "All Phase 15 checks passed: res = 42"


def test_crlf_source_and_forward_slash_import(tmp_path: Path, capsys: pytest.CaptureFixture[str]) -> None:
    sub = tmp_path / "pkg"
    sub.mkdir()
    mod = sub / "helper.zen"
    mod.write_bytes(b'fn greet() -> string {\r\n    return "ok"\r\n}\r\n')

    main_file = tmp_path / "main.zen"
    main_file.write_bytes(b'import "pkg/helper.zen" as h\r\nsay h.greet()\r\n')

    assert cli_main(["run", str(main_file)]) == 0
    assert capsys.readouterr().out.strip() == "ok"

    assert cli_main(["fmt", str(mod)]) == 0
    assert b"\r\n" not in mod.read_bytes()
