"""Phase C tests: ZenNet autograd engine verified against numerical gradients."""

from __future__ import annotations

from io import StringIO
import json
from pathlib import Path
import pytest

from zen.cli import main
from zen.interpreter import Interpreter
from zen.vm import VM


REPO = Path(__file__).resolve().parent.parent


def _run_json(code: str) -> tuple[dict, dict]:
    buf1 = StringIO()
    Interpreter(stdout=buf1, base_dir=REPO).run(code, type_check=True)
    buf2 = StringIO()
    VM(stdout=buf2, base_dir=REPO).run(code, type_check=True)
    d1 = json.loads(buf1.getvalue().strip().splitlines()[-1])
    d2 = json.loads(buf2.getvalue().strip().splitlines()[-1])
    return d1, d2


def _assert_close(a, b, tol=1e-4):
    if isinstance(a, list):
        assert len(a) == len(b)
        for x, y in zip(a, b):
            _assert_close(x, y, tol)
    else:
        assert abs(a - b) <= tol, f"Mismatch: analytical={a} vs numerical={b}"


def _check_op_grad(op_expr: str, x_vals, shape, extra_setup: str = "", tol: float = 1e-4):
    code = f"""
import zennum as zn
import zennet.autograd as ag

{extra_setup}

fn forward_fn(x_arr) {{
    let x = ag.tensor(x_arr, true)
    let w = ag.tensor(zn.array([1.3, -0.7, 2.1, 0.4, -1.2, 0.8])[:zn.array(x_arr).size].reshape({shape}), false)
    let y = {op_expr}
    let loss = (y * w).sum()
    return [loss, x]
}}

let x0 = zn.array({x_vals}, "float64")
let res0 = forward_fn(x0)
let loss = res0[0]
let x_t = res0[1]
loss.backward()
let ana = x_t.grad.flatten().tolist()

let eps = 1e-5
let flat = x0.flatten().tolist()
let num = []
let n = len(flat)
for i in range(n) {{
    let xp = zn.array(flat, "float64")
    let xm = zn.array(flat, "float64")
    xp[i] = flat[i] + eps
    xm[i] = flat[i] - eps
    let lp = ag.no_grad(fn() {{ return forward_fn(xp.reshape({shape}))[0].item() }})
    let lm = ag.no_grad(fn() {{ return forward_fn(xm.reshape({shape}))[0].item() }})
    push(num, (lp - lm) / (2.0 * eps))
}}
say "{{\\"ana\\": " + str(ana) + ", \\"num\\": " + str(num) + "}}"
"""
    d1, d2 = _run_json(code)
    _assert_close(d1["ana"], d1["num"], tol)
    _assert_close(d2["ana"], d2["num"], tol)
    _assert_close(d1["ana"], d2["ana"], 1e-7)


@pytest.mark.parametrize(
    "expr,vals,shape",
    [
        ("(x + 1.5) * 2.0 - x / 3.0", [[1.2, -0.5, 2.0], [0.7, 1.4, -1.1]], [2, 3]),
        ("x ** 2.5 + (-x)", [[1.2, 0.5, 2.0], [0.7, 1.4, 3.1]], [2, 3]),
        ("x.reshape([3, 2]).transpose()", [[1.2, -0.5, 2.0], [0.7, 1.4, -1.1]], [2, 3]),
        ("x.sum(1, false).mean()", [[1.2, -0.5, 2.0], [0.7, 1.4, -1.1]], [2, 3]),
        ("x.abs() + x.sqrt()", [[1.2, 0.5, 2.0], [0.7, 1.4, 3.1]], [2, 3]),
        ("x.exp() + x.log()", [[0.4, 0.8, 1.2], [0.7, 1.4, 0.9]], [2, 3]),
        ("x.sin() + x.cos() + x.clip(-0.8, 1.0)", [[0.4, -0.5, 1.3], [0.7, -1.4, 0.9]], [2, 3]),
        ("x.relu() + x.gelu() + x.sigmoid() + x.tanh()", [[0.4, -0.5, 1.3], [0.7, -1.4, 0.9]], [2, 3]),
        ("x.softmax(1) + x.log_softmax(1)", [[0.4, -0.5, 1.3], [0.7, -1.4, 0.9]], [2, 3]),
        ("x.masked_fill([[true, false, false], [false, true, false]], -9.0)", [[0.4, -0.5, 1.3], [0.7, -1.4, 0.9]], [2, 3]),
        ("x.layernorm()", [[0.4, -0.5, 1.3], [0.7, -1.4, 0.9]], [2, 3]),
    ],
)
def test_autograd_ops_numerical(expr, vals, shape):
    _check_op_grad(expr, vals, shape)


def test_autograd_matmul_layernorm_params_gather_dropout_and_slice():
    code = """
import zennum as zn
import zennet.autograd as ag

zn.seed(123)

// 1. Batched matmul gradient + broadcast bias
let A = ag.tensor([[[1.0, 2.0], [3.0, 4.0]], [[-1.0, 0.5], [2.0, 1.5]]], true)
let B = ag.tensor([[0.5, -1.0], [1.5, 0.25]], true)
let b = ag.tensor([0.1, -0.2], true)
let out = (A @ B) + b
let loss = (out * out).sum()
loss.backward()

let eps = 1e-5
let A_flat = A.data.flatten().tolist()
let num_A = []
for i in range(len(A_flat)) {
    let ap = zn.array(A_flat, "float64")
    let am = zn.array(A_flat, "float64")
    ap[i] = A_flat[i] + eps
    am[i] = A_flat[i] - eps
    let lp = ag.no_grad(fn() { let z = (ag.tensor(ap.reshape([2, 2, 2])) @ ag.tensor(B.data)) + b.data; return (z * z).sum().item() })
    let lm = ag.no_grad(fn() { let z = (ag.tensor(am.reshape([2, 2, 2])) @ ag.tensor(B.data)) + b.data; return (z * z).sum().item() })
    push(num_A, (lp - lm) / (2.0 * eps))
}

// 2. Gather / Embedding gradient
let emb = ag.tensor([[1.0, 2.0], [3.0, 4.0], [5.0, 6.0]], true)
let g = ag.embedding(emb, [0, 2, 0])
let gloss = (g * ag.tensor([[1.0, 1.0], [2.0, 2.0], [3.0, 3.0]])).sum()
gloss.backward()

// 3. Slice backward
let s = ag.tensor([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]], true)
let sub = s[0:2, 1:3]
(sub * 2.0).sum().backward()

// 4. Dropout & detach & no_grad
let d_in = ag.tensor([[1.0, 2.0], [3.0, 4.0]], true)
let d_out = d_in.dropout(0.5, true)
d_out.sum().backward()
let d_det = d_in.detach()
let ng = ag.no_grad(fn() { return d_in * 3.0 })

say "{\\"ana_A\\": " + str(A.grad.flatten().tolist()) + ", \\"num_A\\": " + str(num_A) + ", \\"emb_grad\\": " + str(emb.grad.tolist()) + ", \\"slice_grad\\": " + str(s.grad.tolist()) + ", \\"dropout_grad_sum\\": " + str(zn.sum(d_in.grad)) + ", \\"det_req\\": " + str(d_det.requires_grad) + ", \\"ng_req\\": " + str(ng.requires_grad) + "}"
"""
    d1, d2 = _run_json(code)
    _assert_close(d1["ana_A"], d1["num_A"], 1e-4)
    assert d1["emb_grad"] == [[4.0, 4.0], [0.0, 0.0], [2.0, 2.0]]
    assert d1["slice_grad"] == [[0.0, 2.0, 2.0], [0.0, 2.0, 2.0]]
    assert d1["det_req"] is False
    assert d1["ng_req"] is False
    assert d1["ana_A"] == d2["ana_A"]


def test_phaseC_example_runs():
    assert main(["run", str(REPO / "examples" / "phaseC_zennet_autograd.zen")]) == 0
