"""Tests for Phase 5: Pure-Python Tensor engine exposed to .zen."""

import io
from pathlib import Path

import pytest
from zen.errors import ZenRuntimeError
from zen.interpreter import Interpreter

ROOT = Path(__file__).resolve().parent.parent


def run_zen(source: str) -> str:
    buf = io.StringIO()
    interp = Interpreter(stdout=buf)
    interp.run(source)
    return buf.getvalue()


def test_tensor_creation_and_elementwise_broadcasting() -> None:
    out = run_zen("""
    let a = tensor([[1, 2], [3, 4]])
    let b = ones([2, 2]) * 2
    let z = zeros([2, 2])
    let row = tensor([10, 20])
    say shape(a)
    say (a + b).tolist()
    say (a - b).tolist()
    say (a * 3).tolist()
    say (b / 2).tolist()
    say (a + row).tolist()
    say (10 - z).tolist()
    """)
    assert out.splitlines() == [
        "[2, 2]",
        "[[3.0, 4.0], [5.0, 6.0]]",
        "[[-1.0, 0.0], [1.0, 2.0]]",
        "[[3.0, 6.0], [9.0, 12.0]]",
        "[[1.0, 1.0], [1.0, 1.0]]",
        "[[11.0, 22.0], [13.0, 24.0]]",
        "[[10.0, 10.0], [10.0, 10.0]]",
    ]


def test_tensor_matmul_transpose_reshape() -> None:
    out = run_zen("""
    let a = tensor([[1, 2, 3], [4, 5, 6]])
    let b = reshape(tensor([1, 2, 3, 4, 5, 6]), [3, -1])
    let c = a @ b
    say shape(c)
    say c.tolist()
    say transpose(a).tolist()
    say a.T.tolist()
    say matmul(tensor([1, 2, 3]), tensor([4, 5, 6]))
    """)
    assert out.splitlines() == [
        "[2, 2]",
        "[[22.0, 28.0], [49.0, 64.0]]",
        "[[1.0, 4.0], [2.0, 5.0], [3.0, 6.0]]",
        "[[1.0, 4.0], [2.0, 5.0], [3.0, 6.0]]",
        "32.0",
    ]


def test_tensor_reductions_indexing_and_random() -> None:
    out = run_zen("""
    seed(7)
    let r = tensor_rand([2, 3])
    say shape(r)
    let t = tensor([[1, 5, 2], [4, 0, 6]])
    say sum(t)
    say mean(t)
    say max(t)
    say tensor_sum(t, 0).tolist()
    say tensor_mean(t, 1).tolist()
    say tensor_max(t, 1).tolist()
    say t[0].tolist()
    say t[1][2]
    """)
    assert out.splitlines() == [
        "[2, 3]",
        "18.0",
        "3.0",
        "6.0",
        "[5.0, 5.0, 8.0]",
        "[2.6666666666666665, 3.3333333333333335]",
        "[5.0, 6.0]",
        "[1.0, 5.0, 2.0]",
        "6.0",
    ]


def test_tensor_errors_and_phase5_example() -> None:
    with pytest.raises(ZenRuntimeError, match="broadcast"):
        run_zen("let x = zeros([2, 3]) + zeros([2, 2])")
    with pytest.raises(ZenRuntimeError, match="mismatch"):
        run_zen("let x = zeros([2, 3]) @ zeros([2, 3])")

    buf = io.StringIO()
    interp = Interpreter(stdout=buf)
    interp.run_file(ROOT / "examples" / "phase5_tensors.zen")
    lines = buf.getvalue().splitlines()
    assert "shifted: Tensor([[11.0, 22.0, 33.0], [14.0, 25.0, 36.0]])" == lines[0]
    assert "a @ b = Tensor([[22.0, 28.0], [49.0, 64.0]])" == lines[1]
