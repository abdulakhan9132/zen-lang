"""Phase B tests: ZenNum library, slicing, broadcasting, reductions, linalg."""

from __future__ import annotations

from io import StringIO
from pathlib import Path
import pytest

from zen.cli import main
from zen.errors import ZenRuntimeError
from zen.interpreter import Interpreter
from zen.parser import parse
from zen.typechecker import type_check_program
from zen.vm import VM


REPO = Path(__file__).resolve().parent.parent


def _run_both(code: str) -> tuple[str, str]:
    buf1 = StringIO()
    Interpreter(stdout=buf1, base_dir=REPO).run(code, type_check=True)

    buf2 = StringIO()
    VM(stdout=buf2, base_dir=REPO).run(code, type_check=True)

    return buf1.getvalue().strip(), buf2.getvalue().strip()


def test_zennum_imports_and_creation():
    code = """
import zennum as zn
let z = zn.zeros([2, 3])
let o = zn.ones([2, 3])
let f = zn.full([2, 2], 7)
let ar = zn.arange(0, 10, 2)
let ls = zn.linspace(0, 1, 5)
let id = zn.eye(3)
say z.shape
say o.size
say f.tolist()
say ar.tolist()
say ls.tolist()
say id.tolist()
"""
    out1, out2 = _run_both(code)
    assert out1 == out2
    lines = out1.splitlines()
    assert lines[0] == "[2, 3]"
    assert lines[1] == "6"
    assert lines[2] == "[[7, 7], [7, 7]]"
    assert lines[3] == "[0, 2, 4, 6, 8]"
    assert lines[4] == "[0.0, 0.25, 0.5, 0.75, 1.0]"
    assert lines[5] == "[[1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]]"


def test_zennum_import_without_alias_and_default_args():
    code = """
import zennum
zn_check = 0
say zennum.arange(5).tolist()
say zennum.eye(2).tolist()
"""
    # note: use let!
    code = """
import zennum
say zennum.arange(5).tolist()
say zennum.eye(2).tolist()
"""
    out1, out2 = _run_both(code)
    assert out1 == out2
    assert "[0, 1, 2, 3, 4]" in out1
    assert "[[1.0, 0.0], [0.0, 1.0]]" in out1


def test_zennum_slicing_indexing_and_bool_mask():
    code = """
import zennum as zn
let a = zn.arange(1, 13).reshape([3, 4])
say a[1:3, :].tolist()
say a[:, 1:3].tolist()
say a[-1, -1]
let mask = a > 8
say a[mask].tolist()
a[0, 0] = 99
say a[0, 0]
"""
    out1, out2 = _run_both(code)
    assert out1 == out2
    lines = out1.splitlines()
    assert lines[0] == "[[5, 6, 7, 8], [9, 10, 11, 12]]"
    assert lines[1] == "[[2, 3], [6, 7], [10, 11]]"
    assert lines[2] == "12"
    assert lines[3] == "[9, 10, 11, 12]"
    assert lines[4] == "99"


def test_zennum_broadcasting_and_shape_error():
    code = """
import zennum as zn
let a = zn.array([[1, 2, 3], [4, 5, 6]])
let b = zn.array([10, 20, 30])
say (a + b).tolist()
say (a * 2).tolist()
"""
    out1, out2 = _run_both(code)
    assert out1 == out2
    assert "[[11, 22, 33], [14, 25, 36]]" in out1
    assert "[[2, 4, 6], [8, 10, 12]]" in out1

    bad = """
import zennum as zn
let a = zn.zeros([2, 3])
let b = zn.zeros([4, 5])
let c = a + b
"""
    with pytest.raises(ZenRuntimeError) as exc:
        Interpreter(base_dir=REPO).run(bad)
    msg = str(exc.value)
    assert "[2, 3]" in msg and "[4, 5]" in msg


def test_zennum_math_reductions_shape_and_linalg():
    code = """
import zennum as zn
zn.seed(7)
let x = zn.array([[3.0, 1.0, 2.0], [6.0, 4.0, 5.0]])
say zn.sum(x, 1).tolist()
say zn.mean(x, 0).tolist()
say zn.max(x)
say zn.min(x)
say zn.argmax(x, 1).tolist()
say zn.argmin(x, 1).tolist()
say zn.sort(x, 1).tolist()
say zn.cumsum(zn.array([1, 2, 3, 4])).tolist()
say zn.concat([zn.ones([2, 2]), zn.zeros([2, 1])], 1).shape
say zn.stack([zn.array([1, 2]), zn.array([3, 4])], 0).tolist()
say len(zn.split(zn.arange(6), 3))
say zn.where(zn.array([true, false, true]), zn.array([1, 2, 3]), zn.array([10, 20, 30])).tolist()
say zn.clip(zn.array([-2, 0, 5, 10]), 0, 6).tolist()
say zn.dot(zn.array([1.0, 2.0, 3.0]), zn.array([4.0, 5.0, 6.0]))
say zn.outer(zn.array([1, 2]), zn.array([3, 4, 5])).tolist()
let n = zn.norm(zn.array([3.0, 4.0]))
say n
let d = zn.sum(zn.abs(zn.sqrt(zn.array([4.0, 9.0])) - zn.array([2.0, 3.0])))
say d
"""
    out1, out2 = _run_both(code)
    assert out1 == out2
    lines = out1.splitlines()
    assert lines[0] == "[6.0, 15.0]"
    assert lines[1] == "[4.5, 2.5, 3.5]"
    assert lines[2] == "6.0"
    assert lines[3] == "1.0"
    assert lines[4] == "[0, 0]"
    assert lines[5] == "[1, 1]"
    assert lines[6] == "[[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]]"
    assert lines[7] == "[1, 3, 6, 10]"
    assert lines[8] == "[2, 3]"
    assert lines[9] == "[[1, 2], [3, 4]]"
    assert lines[10] == "3"
    assert lines[11] == "[1, 20, 3]"
    assert lines[12] == "[0, 0, 5, 6]"
    assert lines[13] == "32.0"
    assert lines[14] == "[[3, 4, 5], [6, 8, 10]]"
    assert lines[15] == "5.0"
    assert lines[16] == "0.0"


def test_zennum_example_runs():
    assert main(["run", str(REPO / "examples" / "zennum_basics.zen")]) == 0
