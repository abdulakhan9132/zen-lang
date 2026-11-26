"""Stage 1 tests: Trust features (div, RNG across processes, f-strings, recursion, hints/codes, narrowing)."""

from __future__ import annotations

from io import StringIO
from pathlib import Path
import subprocess
import sys

import pytest

from zen.cli import main as cli_main
from zen.errors import ZenRuntimeError, ZenTypeError
from zen.interpreter import Interpreter
from zen.typechecker import type_check_source
from zen.vm import VM

REPO = Path(__file__).resolve().parent.parent


def _run_both(code: str) -> tuple[str, str, str, str]:
    o1, e1 = StringIO(), StringIO()
    Interpreter(stdout=o1, stderr=e1, base_dir=REPO).run(code, type_check=True)
    o2, e2 = StringIO(), StringIO()
    VM(stdout=o2, stderr=e2, base_dir=REPO).run(code, type_check=True)
    return o1.getvalue().strip(), o2.getvalue().strip(), e1.getvalue(), e2.getvalue()


def test_div_operator_floor_div_and_comment_warning(tmp_path: Path) -> None:
    o1, o2, e1, e2 = _run_both("say 7 div 2\nsay floor_div(7, 2)\nsay -7 div 2")
    assert o1 == o2 == "3\n3\n-4"
    assert e1 == "" and e2 == ""

    # Suspicious trailing comments: // 2 and // b (variable in scope)
    warn_code = "let a = 7\nlet b = 2\nsay a // 2\nsay a // b\nsay a // normal note\n"
    err_buf = StringIO()
    type_check_source(warn_code, stderr=err_buf)
    w = err_buf.getvalue()
    assert "// 2" in w and "not integer division" in w
    assert "// b" in w and "variable 'b'" in w
    assert "normal note" not in w

    # `zen check` and `zen run` both warn
    f = tmp_path / "div_warn.zen"
    f.write_text("let x = 7 // 2\nsay x\n", encoding="utf-8")
    p_chk = subprocess.run([sys.executable, "-m", "zen", "check", str(f)], cwd=REPO, capture_output=True, text=True)
    assert p_chk.returncode == 0
    assert "// 2" in p_chk.stderr
    p_run = subprocess.run([sys.executable, "-m", "zen", "run", str(f)], cwd=REPO, capture_output=True, text=True)
    assert p_run.returncode == 0
    assert p_run.stdout.strip() == "7"
    assert "// 2" in p_run.stderr


def test_shared_rng_matches_across_two_separate_processes(tmp_path: Path) -> None:
    script = tmp_path / "rng_proc.zen"
    script.write_text(
        """import zennum as zn
import zennet as nn
zn.seed(1234)
let x = zn.randn([2, 3])
let r = random()
let m = nn.Linear(3, 2)
say str(x.tolist()) + "|" + str(r) + "|" + str(m.weight.data.tolist())
""",
        encoding="utf-8",
    )
    p1 = subprocess.run([sys.executable, "-m", "zen", "run", str(script)], cwd=REPO, capture_output=True, text=True, check=True)
    p2 = subprocess.run([sys.executable, "-m", "zen", "run", str(script)], cwd=REPO, capture_output=True, text=True, check=True)
    assert p1.stdout == p2.stdout and len(p1.stdout.strip()) > 10


def test_fstrings_and_plain_string_brace_warning() -> None:
    code = '''let name = "ZEN"
let x = 3
say f"lang={name}, next={x + 1}, braces={{ok}}"
say "plain={x}"
'''
    o1, o2, e1, e2 = _run_both(code)
    assert o1 == o2 == "lang=ZEN, next=4, braces={ok}\nplain={x}"
    assert "plain string contains '{x}'" in e1
    assert "plain string contains '{x}'" in e2


def test_deep_recursion_10000_and_max_depth_error(tmp_path: Path) -> None:
    deep_code = """
fn sum_down(n: int, acc: int) -> int {
    if n == 0 { return acc }
    return sum_down(n - 1, acc + 1)
}
say sum_down(10000, 0)
"""
    out = StringIO()
    VM(stdout=out, max_depth=10050).run(deep_code, type_check=True)
    assert out.getvalue().strip() == "10000"

    inf_file = tmp_path / "inf.zen"
    inf_file.write_text("fn boom(n) {\n    return boom(n + 1)\n}\nboom(1)\n", encoding="utf-8")
    p = subprocess.run(
        [sys.executable, "-m", "zen", "run", "--max-depth", "50", str(inf_file)],
        cwd=REPO,
        capture_output=True,
        text=True,
    )
    assert p.returncode == 1
    assert "E010" in p.stderr
    assert "boom" in p.stderr
    assert "50" in p.stderr
    assert "line 2" in p.stderr
    assert "Traceback" not in p.stderr


def test_did_you_mean_error_codes_explain_and_internal_error(tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]) -> None:
    with pytest.raises(ZenTypeError, match=r"E002.*Did you mean 'count'"):
        type_check_source("let count = 1\nsay couny\n")

    with pytest.raises(ZenTypeError, match=r"E003.*Did you mean 'sum'"):
        type_check_source("import zennum as zn\nlet a = zn.ones([2])\nsay zn.summ(a)\n")

    assert cli_main(["explain", "E001"]) == 0
    assert "E001: Syntax error" in capsys.readouterr().out

    # Internal error should never dump raw traceback unless --debug
    f = tmp_path / "ok.zen"
    f.write_text("say 1\n", encoding="utf-8")
    monkeypatch.setattr(VM, "run_file", lambda *a, **kw: (_ for _ in ()).throw(RuntimeError("boom")))
    assert cli_main(["run", str(f)]) == 1
    err = capsys.readouterr().err
    assert "internal error, please report" in err
    assert "Traceback" not in err


def test_optional_type_any_and_flow_narrowing() -> None:
    code = """
fn handle(x: int?, raw: any) -> int {
    let base = 0
    if x != none {
        base = x + 5
    }
    if type(raw) == "list" {
        return base + len(raw)
    }
    return base
}
say handle(10, [1, 2, 3])
say handle(none, 99)
"""
    o1, o2, _, _ = _run_both(code)
    assert o1 == o2 == "18\n0"

    with pytest.raises(ZenTypeError, match="E004"):
        type_check_source("let x: int? = none\nsay x + 1\n")
