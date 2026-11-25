"""Tests for Phase 6: Autograd written in .zen verified against hand-calculated gradients."""

import io
from pathlib import Path

from zen.interpreter import Interpreter

ROOT = Path(__file__).resolve().parent.parent


def run_zen(source: str, base_dir: Path | None = None) -> str:
    buf = io.StringIO()
    interp = Interpreter(stdout=buf, base_dir=base_dir or ROOT)
    interp.run(source)
    return buf.getvalue()


def test_hand_calculated_polynomial_gradients() -> None:
    # f(x, y) = (x * y + x**2) / y
    # At x=3, y=2: f = (6 + 9) / 2 = 7.5
    # df/dx = 1 + 2*x/y = 1 + 3 = 4.0
    # df/dy = -x**2 / y**2 = -9 / 4 = -2.25
    out = run_zen("""
    import "stdlib/ai/autograd.zen" as ag
    let x = ag.value(3.0)
    let y = ag.value(2.0)
    let f = (x * y + x ** 2) / y
    f.backward()
    say f.data
    say x.grad
    say y.grad
    """)
    f_val, dx, dy = [float(line) for line in out.splitlines()]
    assert abs(f_val - 7.5) < 1e-9
    assert abs(dx - 4.0) < 1e-9
    assert abs(dy - (-2.25)) < 1e-9


def test_activations_exp_log_and_shared_node_accumulation() -> None:
    # g(x) = tanh(2*x - 1) + sigmoid(x) + relu(-x)
    # At x = 1.0:
    # u = 1.0, tanh(1) = 0.7615941559557649, dtanh/dx = 2 * (1 - tanh(1)^2)
    # sig(1) = 1 / (1 + e^-1), dsig/dx = sig(1) * (1 - sig(1))
    # relu(-1) = 0, drelu/dx = 0
    out = run_zen("""
    import "stdlib/ai/autograd.zen" as ag
    let x = ag.value(1.0)
    let g = (2.0 * x - 1.0).tanh() + x.sigmoid() + (-x).relu()
    g.backward()
    say g.data
    say x.grad

    let z = ag.value(2.0)
    let h = z.exp().log()
    h.backward()
    say h.data
    say z.grad
    """)
    lines = [float(l) for l in out.splitlines()]
    import math

    t1 = math.tanh(1.0)
    s1 = 1.0 / (1.0 + math.exp(-1.0))
    expected_g = t1 + s1
    expected_dg = 2.0 * (1.0 - t1 * t1) + s1 * (1.0 - s1)
    assert abs(lines[0] - expected_g) < 1e-9
    assert abs(lines[1] - expected_dg) < 1e-9
    assert abs(lines[2] - 2.0) < 1e-9
    assert abs(lines[3] - 1.0) < 1e-9


def test_phase6_example_runs() -> None:
    buf = io.StringIO()
    interp = Interpreter(stdout=buf)
    interp.run_file(ROOT / "examples" / "phase6_autograd.zen")
    assert buf.getvalue().splitlines() == [
        "f = 4.0",
        "df/da = -3.0",
        "df/db = 2.0",
        "df/dc = 1.0",
    ]
