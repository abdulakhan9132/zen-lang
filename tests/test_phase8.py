"""Tests for Phase 8: Tiny character-level language model written in .zen."""

import io
from pathlib import Path

from zen.interpreter import Interpreter

ROOT = Path(__file__).resolve().parent.parent


def test_tiny_char_lm_trains_and_generates() -> None:
    buf = io.StringIO()
    interp = Interpreter(stdout=buf)
    interp.run_file(ROOT / "examples" / "phase8_char_lm.zen")
    output = buf.getvalue()
    lines = output.splitlines()

    step_lines = [l for l in lines if l.startswith("step ")]
    losses = [float(l.split("loss = ")[1]) for l in step_lines]
    assert len(losses) >= 3
    assert losses[-1] < losses[0] * 0.6, f"Loss did not decrease enough: {losses}"

    gen_idx = lines.index("generated:")
    generated = "\n".join(lines[gen_idx + 1 :])
    assert generated.startswith("zen ")
    assert len(generated) >= 40

    # All generated characters must belong to tiny.txt's vocabulary
    tiny_chars = set((ROOT / "examples" / "tiny.txt").read_text(encoding="utf-8"))
    assert set(generated).issubset(tiny_chars)
