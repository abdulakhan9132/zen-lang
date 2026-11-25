"""Tests for Phase 9: Tensor autograd in .zen verified against numerical gradients."""

import io
from pathlib import Path

from zen.interpreter import Interpreter

ROOT = Path(__file__).resolve().parent.parent


def run_zen(source: str) -> str:
    buf = io.StringIO()
    interp = Interpreter(stdout=buf, base_dir=ROOT)
    interp.run(source)
    return buf.getvalue()


def test_elementwise_exp_log_builtins_on_tensor() -> None:
    out = run_zen("""
    let t = tensor([[0.0, 1.0], [2.0, 3.0]])
    let e = exp(t)
    let l = log(e)
    say round(sum(l), 6)
    say round(sum(tensor_log(tensor_exp(t))), 6)
    """)
    assert out.splitlines() == ["6.0", "6.0"]


def test_numerical_gradient_check_for_all_ops() -> None:
    # Finite-difference check inside ZEN comparing autograd .grad vs (f(x+eps)-f(x-eps))/(2*eps)
    out = run_zen("""
    import "stdlib/ai/tensor_autograd.zen" as ta

    fn check_grad(name, make_loss, param_data, eps) {
        let p = ta.from(param_data)
        let loss = make_loss(p)
        loss.backward()
        let auto_g = p.grad.tolist()

        let flat_in = reshape(tensor(param_data), [-1]).tolist()
        let flat_auto = reshape(p.grad, [-1]).tolist()
        let s = shape(tensor(param_data))
        let max_err = 0.0

        for idx in range(len(flat_in)) {
            let plus = []
            let minus = []
            for j in range(len(flat_in)) {
                if j == idx {
                    push(plus, flat_in[j] + eps)
                    push(minus, flat_in[j] - eps)
                } else {
                    push(plus, flat_in[j])
                    push(minus, flat_in[j])
                }
            }
            let tp = ta.from(reshape(tensor(plus), s))
            let tm = ta.from(reshape(tensor(minus), s))
            let lp = make_loss(tp).data.tolist()
            let lm = make_loss(tm).data.tolist()
            let num_g = (lp - lm) / (2.0 * eps)
            let err = abs(flat_auto[idx] - num_g)
            if err > max_err {
                max_err = err
            }
        }
        say name + ": " + str(max_err < 0.0001)
    }

    let c = ta.from([[1.2, -0.7], [0.4, 1.5]])
    let bias = ta.from([0.3, -0.4])

    check_grad("add_sub_mul_div", fn(x) {
        return (((x + bias) * c - 0.5) / (c + 2.0)).sum()
    }, [[0.5, 1.1], [-0.3, 0.8]], 0.0001)

    check_grad("matmul_transpose_reshape", fn(x) {
        let y = (x @ c.transpose()).reshape([4, 1])
        return (y * y).mean()
    }, [[0.4, -0.6], [1.2, 0.3]], 0.0001)

    check_grad("exp_log_tanh_relu", fn(x) {
        let a = (x.tanh() + 2.0).log()
        let b = (x * 0.5).exp() + x.relu()
        return (a * b).sum()
    }, [[0.6, -0.5], [1.1, 0.2]], 0.0001)

    check_grad("softmax", fn(x) {
        let p = x.softmax()
        let w = ta.from([[1.0, -0.5, 0.2], [0.3, 0.8, -1.1]])
        return (p * w).sum()
    }, [[1.0, 2.0, -0.5], [0.2, -0.3, 0.9]], 0.0001)

    check_grad("gather", fn(emb) {
        let g = emb.gather([0, 2, 0])
        return (g * ta.from([[1.0, 2.0], [-1.0, 0.5], [0.3, -0.7]])).sum()
    }, [[0.2, -0.1], [0.4, 0.5], [-0.3, 0.8]], 0.0001)
    """)
    assert out.splitlines() == [
        "add_sub_mul_div: true",
        "matmul_transpose_reshape: true",
        "exp_log_tanh_relu: true",
        "softmax: true",
        "gather: true",
    ]


def test_phase9_example_runs() -> None:
    buf = io.StringIO()
    interp = Interpreter(stdout=buf)
    interp.run_file(ROOT / "examples" / "phase9_tensor_autograd.zen")
    lines = buf.getvalue().splitlines()
    assert len(lines) == 4
    assert lines[0].startswith("h = Tensor(")
    assert lines[2].startswith("dL/dw = Tensor(")
