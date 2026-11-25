"""Tests for Phase 13: Static type checker, annotations, inference, and --no-check."""

import io
from pathlib import Path

import pytest
from zen.cli import main as cli_main
from zen.errors import ZenTypeError
from zen.typechecker import type_check_source
from zen.vm import VM

ROOT = Path(__file__).resolve().parent.parent


def test_type_annotations_and_inference_run_cleanly() -> None:
    buf = io.StringIO()
    VM(stdout=buf).run_file(ROOT / "examples" / "phase13_types.zen", type_check=True)
    assert buf.getvalue().splitlines() == ["ZEN Type Checker", "13.5", "Hello, AI"]


def test_all_existing_zen_files_pass_type_checker() -> None:
    zen_files = list((ROOT / "examples").glob("*.zen")) + list((ROOT / "stdlib" / "ai").glob("*.zen"))
    assert len(zen_files) >= 10
    for path in zen_files:
        src = path.read_text(encoding="utf-8")
        type_check_source(src, file_path=path)


def test_type_checker_catches_undefined_const_and_arity() -> None:
    with pytest.raises(ZenTypeError) as exc1:
        type_check_source("let a = 1\nsay missing")
    msg1 = str(exc1.value)
    assert "[line 2, col 5]" in msg1
    assert "2 | say missing" in msg1
    assert "^" in msg1

    with pytest.raises(ZenTypeError, match="constant"):
        type_check_source("const X: int = 5\nX = 6")

    with pytest.raises(ZenTypeError, match="expected 2 argument"):
        type_check_source("fn add(a: int, b: int) -> int { return a + b }\nadd(1)")


def test_type_checker_catches_return_and_operator_mismatches() -> None:
    with pytest.raises(ZenTypeError, match="Return type mismatch"):
        type_check_source('fn f() -> int {\n  return "nope"\n}')

    with pytest.raises(ZenTypeError, match="may not return"):
        type_check_source("fn g() -> int {\n  say 1\n}")

    with pytest.raises(ZenTypeError) as exc_op:
        type_check_source('let bad = "a" - 1')
    assert "[line 1, col 15]" in str(exc_op.value)
    assert '1 | let bad = "a" - 1' in str(exc_op.value)
    assert "^" in str(exc_op.value)

    with pytest.raises(ZenTypeError, match="cannot assign 'float' to 'int'"):
        type_check_source("let n: int = 2.5")

    with pytest.raises(ZenTypeError, match="inferred|Argument|expected"):
        type_check_source("fn sq(x: int) -> int { return x * x }\nsq(2.5)")


def test_cli_no_check_skips_type_checker(tmp_path: Path, capsys: pytest.CaptureFixture[str]) -> None:
    f = tmp_path / "skip_check.zen"
    f.write_text('if false {\n  let x = "a" - 1\n}\nsay "ok"\n', encoding="utf-8")

    # Fails when type checker runs (default)
    assert cli_main(["run", str(f)]) == 1
    err = capsys.readouterr().err
    assert "not supported between 'string' and 'int'" in err

    # Passes when --no-check is passed
    assert cli_main(["run", "--no-check", str(f)]) == 0
    out = capsys.readouterr().out
    assert out.strip() == "ok"
