"""Phase G tests: verify all 5 ZenNum/ZenNet recipes run cleanly."""

from __future__ import annotations

from io import StringIO
from pathlib import Path
from contextlib import redirect_stdout

from zen.cli import main


REPO = Path(__file__).resolve().parent.parent


def _run_cli(path: Path) -> str:
    buf = StringIO()
    with redirect_stdout(buf):
        rc = main(["run", str(path)])
    assert rc == 0, f"Failed running {path}"
    return buf.getvalue().strip()


def test_recipe_zennum_basics():
    out = _run_cli(REPO / "examples" / "zennum_basics.zen")
    assert "Shape: [3, 4]" in out


def test_recipe_linear_regression():
    out = _run_cli(REPO / "examples" / "linear_regression.zen")
    assert "2.5" in out and "1.0" in out


def test_recipe_mlp_xor():
    p = REPO / "examples" / "mlp_xor.zen"
    lines = [ln for ln in p.read_text(encoding="utf-8").splitlines() if ln.strip()]
    assert len(lines) <= 18
    out = _run_cli(p)
    assert "XOR predictions:" in out


def test_recipe_tiny_classifier():
    out = _run_cli(REPO / "examples" / "tiny_classifier.zen")
    assert "Validation accuracy: 100.0%" in out


def test_recipe_gpt_in_40_lines():
    p = REPO / "examples" / "gpt_in_40_lines.zen"
    lines = p.read_text(encoding="utf-8").splitlines()
    assert len(lines) <= 45
    out = _run_cli(p)
    assert "Start loss:" in out and "End loss:" in out and "Sample:" in out
