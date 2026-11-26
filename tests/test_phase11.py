"""Tests for Phase 11: Bigger text test (200-500 KB, mini-batches, train/val split)."""

import io
from pathlib import Path

import pytest

from zen.interpreter import Interpreter

ROOT = Path(__file__).resolve().parent.parent


def test_corpus_size_between_200kb_and_500kb() -> None:
    corpus = ROOT / "examples" / "data" / "corpus.txt"
    assert corpus.is_file()
    size = corpus.stat().st_size
    assert 200_000 <= size <= 500_000, f"Unexpected corpus size: {size} bytes"


@pytest.mark.slow
def test_phase11_minibatch_train_and_val_loss_drop() -> None:
    buf = io.StringIO()
    interp = Interpreter(stdout=buf)
    interp.run_file(ROOT / "examples" / "phase11_big_text.zen")
    lines = buf.getvalue().splitlines()

    init_val = float([l for l in lines if l.startswith("initial val_loss = ")][0].split("= ")[1])
    final_line = [l for l in lines if l.startswith("final train_loss:")][0]
    final_train = float(final_line.split("final train_loss: ")[1].split(" ")[0])
    final_val = float(final_line.split("final val_loss: ")[1])

    step0 = [l for l in lines if l.startswith("step 0 ")][0]
    init_train = float(step0.split("train_loss = ")[1].split(" ")[0])

    assert final_train < init_train
    assert final_val < init_val

    gen_idx = lines.index("generated:")
    generated = "\n".join(lines[gen_idx + 1 :])
    assert generated.startswith("The ")
    assert len(generated) >= 60
