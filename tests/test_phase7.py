"""Tests for Phase 7: Neural network in .zen (Dense, activations, losses, SGD, XOR)."""

import io
from pathlib import Path

from zen.interpreter import Interpreter

ROOT = Path(__file__).resolve().parent.parent


def run_zen(source: str) -> str:
    buf = io.StringIO()
    interp = Interpreter(stdout=buf, base_dir=ROOT)
    interp.run(source)
    return buf.getvalue()


def test_activations_softmax_and_losses() -> None:
    out = run_zen("""
    import "stdlib/ai/autograd.zen" as ag
    import "stdlib/ai/nn.zen" as nn

    let probs = nn.softmax([ag.value(1.0), ag.value(2.0), ag.value(3.0)])
    let sum_p = probs[0].data + probs[1].data + probs[2].data
    say round(sum_p, 6)

    let ce = nn.cross_entropy_loss(probs, 2)
    ce.backward()
    say ce.data > 0.0
    say probs[2].grad < 0.0
    say nn.relu(-2.0).data
    say round(nn.sigmoid(0.0).data, 4)
    """)
    assert out.splitlines() == ["1.0", "true", "true", "0.0", "0.5"]


def test_xor_training_learns_and_loss_drops() -> None:
    buf = io.StringIO()
    interp = Interpreter(stdout=buf)
    interp.run_file(ROOT / "examples" / "phase7_xor.zen")
    lines = buf.getvalue().splitlines()

    loss_lines = [l for l in lines if l.startswith("epoch ")]
    losses = [float(l.split("loss = ")[1]) for l in loss_lines]
    assert len(losses) >= 2
    assert losses[-1] < losses[0] * 0.25, f"Loss did not drop enough: {losses}"
    assert losses[-1] < 0.05, f"Final XOR loss too high: {losses[-1]}"

    # Check predictions match XOR targets
    pred_lines = [l for l in lines if " -> " in l]
    assert len(pred_lines) == 4
    preds = [float(l.split(" -> ")[1].split(" ")[0]) for l in pred_lines]
    assert preds[0] < 0.25
    assert preds[1] > 0.75
    assert preds[2] > 0.75
    assert preds[3] < 0.25
