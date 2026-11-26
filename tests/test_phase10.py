"""Tests for Phase 10: Tensor-autograd MLP + GPT + Adam optimizer."""

import io
from pathlib import Path

import pytest

from zen.interpreter import Interpreter

ROOT = Path(__file__).resolve().parent.parent


@pytest.mark.slow
def test_phase10_xor_and_gpt_learn_with_adam() -> None:
    buf = io.StringIO()
    interp = Interpreter(stdout=buf)
    interp.run_file(ROOT / "examples" / "phase10_compare.zen")
    lines = buf.getvalue().splitlines()

    xor_new = [l for l in lines if l.startswith("XOR (new tensor Adam")][0]
    loss_part = xor_new.split("loss=")[1].split(" ")[0]
    start_x, end_x = [float(v) for v in loss_part.split("->")]
    assert end_x < start_x * 0.25
    assert end_x < 0.05

    gpt_new = [l for l in lines if l.startswith("GPT (new autograd Adam")][0]
    gpt_loss_part = gpt_new.split("loss=")[1].split(" ")[0]
    start_g, end_g = [float(v) for v in gpt_loss_part.split("->")]
    assert end_g < start_g * 0.6

    sample_line = [l for l in lines if l.startswith("sample: ")][0]
    assert sample_line.startswith("sample: zen ")
