"""Phase D tests: ZenNet layers, losses, Module save/load, and GPT."""

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


def test_layers_sequential_and_losses():
    code = """
import zennum as zn
import zennet.autograd as ag
import zennet.nn as nn

zn.seed(7)
let mlp = nn.Sequential([
    nn.Linear(3, 6),
    nn.ReLU(),
    nn.Dropout(0.25),
    nn.Linear(6, 2),
])
let x = [[1.0, 2.0, -1.0], [0.5, -0.5, 2.0]]
let out = mlp(x)
let ce = nn.cross_entropy(out, [0, 1])
ce.backward()

let probs = out.softmax(-1)
let bce = nn.binary_cross_entropy(probs, [[1.0, 0.0], [0.0, 1.0]])

mlp.eval()
let d_eval = mlp.layers[2].training
mlp.train()
let d_train = mlp.layers[2].training

let p0 = mlp.parameters()[0]
let has_grad = p0.grad != none

say "{\\"n_params\\": " + str(len(mlp.parameters())) + ", \\"out_shape\\": " + str(out.shape) + ", \\"d_eval\\": " + str(d_eval) + ", \\"d_train\\": " + str(d_train) + ", \\"has_grad\\": " + str(has_grad) + ", \\"ce\\": " + str(ce.item()) + ", \\"bce\\": " + str(bce.item()) + "}"
"""
    o1, o2 = _run_both(code)
    assert o1 == o2
    d = json.loads(o1)
    assert d["n_params"] == 4
    assert d["out_shape"] == [2, 2]
    assert d["d_eval"] is False
    assert d["d_train"] is True
    assert d["has_grad"] is True
    assert d["ce"] > 0.0 and d["bce"] > 0.0


def test_gpt_forward_backward_causal_and_save_load(tmp_path):
    ckpt = (tmp_path / "gpt_weights.json").as_posix()
    code = f"""
import zennum as zn
import zennet.nn as nn

zn.seed(11)
let gpt = nn.GPT({{"vocab_size": 12, "n_layer": 2, "n_head": 2, "d_model": 8, "block_size": 6, "dropout": 0.0}})
gpt.eval()
let x1 = [[1, 2, 3, 4]]
let x2 = [[1, 2, 3, 9]]
let l1 = gpt(x1)
let l2 = gpt(x2)
// Causal check: positions 0, 1, 2 logits must be identical when token 3 changes!
let diff = zn.sum(zn.abs(l1.data[0, 0:3, :] - l2.data[0, 0:3, :]))

let loss = nn.cross_entropy(l1, [[2, 3, 4, 5]])
loss.backward()
let gnorm = zn.norm(gpt.tok_emb.weight.grad)

gpt.save("{ckpt}")
let gpt2 = nn.GPT({{"vocab_size": 12, "n_layer": 2, "n_head": 2, "d_model": 8, "block_size": 6, "dropout": 0.0}})
gpt2.load("{ckpt}")
let l_loaded = gpt2(x1)
let load_diff = zn.sum(zn.abs(l1.data - l_loaded.data))

say "{{\\"shape\\": " + str(l1.shape) + ", \\"causal_diff\\": " + str(diff) + ", \\"gnorm\\": " + str(gnorm) + ", \\"load_diff\\": " + str(load_diff) + "}}"
"""
    o1, o2 = _run_both(code)
    assert o1 == o2
    d = json.loads(o1)
    assert d["shape"] == [1, 4, 12]
    assert d["causal_diff"] < 1e-10
    assert d["gnorm"] > 0.0
    assert d["load_diff"] < 1e-10


def test_phaseD_example_runs():
    assert main(["run", str(REPO / "examples" / "phaseD_layers_gpt.zen")]) == 0
