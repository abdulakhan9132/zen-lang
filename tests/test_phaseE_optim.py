"""Phase E tests: ZenNet optimizers, gradient clipping, and schedulers."""

from __future__ import annotations

from io import StringIO
import json
from pathlib import Path

from zen.cli import main
from zen.interpreter import Interpreter
from zen.vm import VM


REPO = Path(__file__).resolve().parent.parent


def _run_both(code: str) -> tuple[str, str]:
    buf1 = StringIO()
    Interpreter(stdout=buf1, base_dir=REPO).run(code, type_check=True)
    buf2 = StringIO()
    VM(stdout=buf2, base_dir=REPO).run(code, type_check=True)
    return buf1.getvalue().strip(), buf2.getvalue().strip()


def test_sgd_adam_adamw_and_clip_grad_norm():
    code = """
import zennum as zn
import zennet.autograd as ag
import zennet.nn as nn
import zennet.optim as optim

zn.seed(1)

// SGD with momentum minimizes (w - 3)^2
let w1 = ag.tensor([0.0], true)
let sgd = optim.SGD([w1], 0.2, 0.5)
for i in range(25) {
    sgd.zero_grad()
    let l = ((w1 - 3.0) ** 2.0).sum()
    l.backward()
    sgd.step()
}

// Adam minimizes (w + 2)^2
let w2 = ag.tensor([5.0], true)
let adam = optim.Adam([w2], 0.25)
for i in range(120) {
    adam.zero_grad()
    let l = ((w2 + 2.0) ** 2.0).sum()
    l.backward()
    adam.step()
}

// AdamW decoupled weight decay shrinks unused/zero-grad weight
let w3 = ag.tensor([10.0], true)
w3.grad = zn.array([0.0])
let adamw = optim.AdamW([w3], 0.1, [0.9, 0.999], 1e-8, 0.5)
adamw.step()

// clip_grad_norm caps gradient L2 norm
let wg = ag.tensor([3.0, 4.0], true)
wg.grad = zn.array([3.0, 4.0])
let pre_norm = optim.clip_grad_norm([wg], 1.0)
let post_norm = zn.norm(wg.grad)

say "{\\"w1\\": " + str(w1.item()) + ", \\"w2\\": " + str(w2.item()) + ", \\"w3\\": " + str(w3.item()) + ", \\"pre\\": " + str(pre_norm) + ", \\"post\\": " + str(post_norm) + "}"
"""
    o1, o2 = _run_both(code)
    assert o1 == o2
    d = json.loads(o1)
    assert abs(d["w1"] - 3.0) < 0.05
    assert abs(d["w2"] + 2.0) < 0.2
    assert abs(d["w3"] - 9.5) < 1e-6
    assert abs(d["pre"] - 5.0) < 1e-6
    assert abs(d["post"] - 1.0) < 1e-6


def test_lr_schedulers():
    code = """
import zennet.autograd as ag
import zennet.optim as optim

let w = ag.tensor([1.0], true)
let o1 = optim.SGD([w], 1.0)
let s1 = optim.StepLR(o1, 2, 0.5)
let lrs1 = [s1.step(), s1.step(), s1.step(), s1.step()]

let o2 = optim.SGD([w], 1.0)
let s2 = optim.CosineAnnealingLR(o2, 4, 0.0)
let lrs2 = [s2.step(), s2.step(), s2.step(), s2.step()]

let o3 = optim.SGD([w], 1.0)
let s3 = optim.WarmupLR(o3, 2)
let lrs3 = [s3.step(), s3.step(), s3.step()]

say "{\\"step\\": " + str(lrs1) + ", \\"cos\\": " + str(lrs2) + ", \\"warm\\": " + str(lrs3) + "}"
"""
    o1, o2 = _run_both(code)
    assert o1 == o2
    d = json.loads(o1)
    assert d["step"] == [1.0, 0.5, 0.5, 0.25]
    assert abs(d["cos"][1] - 0.5) < 1e-6 and abs(d["cos"][3] - 0.0) < 1e-6
    assert d["warm"] == [0.5, 1.0, 1.0]


def test_phaseE_example_runs():
    assert main(["run", str(REPO / "examples" / "phaseE_optim.zen")]) == 0
