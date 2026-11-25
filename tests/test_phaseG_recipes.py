"""Phase G tests: verify all 5 ZenNum/ZenNet recipes run cleanly."""

from __future__ import annotations

from io import StringIO
from pathlib import Path
from contextlib import redirect_stdout

import pytest

from zen.cli import main


REPO = Path(__file__).resolve().parent.parent


def _run_cli(path: Path) -> str:
    buf = StringIO()
    with redirect_stdout(buf):
        rc = main(["run", str(path)])
    assert rc == 0, f"Failed running {path}"
    return buf.getvalue().strip()


def test_recipe_zennum_basics():
    out = _run_cli(REPO / "examples" / "zennum_basics.zen")
    assert "Shape: [3, 4]" in out


def test_recipe_linear_regression():
    out = _run_cli(REPO / "examples" / "linear_regression.zen")
    assert "2.5" in out and "1.0" in out


def test_recipe_mlp_xor():
    p = REPO / "examples" / "mlp_xor.zen"
    lines = [ln for ln in p.read_text(encoding="utf-8").splitlines() if ln.strip()]
    assert len(lines) <= 18
    out = _run_cli(p)
    assert "XOR predictions:" in out


def test_recipe_tiny_classifier():
    out = _run_cli(REPO / "examples" / "tiny_classifier.zen")
    assert "Validation accuracy: 100.0%" in out


@pytest.mark.slow
def test_recipe_gpt_in_40_lines():
    p = REPO / "examples" / "gpt_in_40_lines.zen"
    lines = p.read_text(encoding="utf-8").splitlines()
    assert len(lines) <= 45
    out = _run_cli(p)
    assert "Start loss:" in out and "End loss:" in out and "Sample:" in out

    # Verify loss drops below 2.5
    import re

    m = re.search(r"Start loss:\s*([0-9.]+)\s*->\s*End loss:\s*([0-9.]+)", out)
    assert m is not None, f"Could not parse losses from:\n{out}"
    start_loss = float(m.group(1))
    end_loss = float(m.group(2))
    assert end_loss < start_loss, f"Loss did not decrease: {start_loss} -> {end_loss}"
    assert end_loss < 2.5, f"End loss {end_loss} is not below 2.5"

    # Verify generated sample contains at least one real word from corpus.txt
    sample_idx = out.find("Sample:")
    assert sample_idx != -1
    sample_text = out[sample_idx + len("Sample:") :].strip().lower()
    corpus_text = (REPO / "examples" / "data" / "corpus.txt").read_text(encoding="utf-8")[:4000].lower()
    corpus_words = {w for w in re.findall(r"[a-z]+", corpus_text) if len(w) >= 3}
    sample_words = set(re.findall(r"[a-z]+", sample_text))
    assert corpus_words & sample_words, (
        f"Sample {sample_text!r} contains no real word (>=3 chars) from corpus.txt"
    )


@pytest.mark.slow
def test_generate_memorization(tmp_path):
    """Train a tiny GPT to memorize one short sentence and verify generate() reproduces it."""
    script = tmp_path / "memorize.zen"
    script.write_text(
        '''import zennum as zn
import zennet as nn

zn.seed(42)
let text = "the cat sat on the mat. "
let tok = nn.CharTokenizer(text)
let ids = tok.encode(text)
let cfg = {"vocab_size": tok.vocab_size, "n_layer": 1, "n_head": 2, "d_model": 16, "block_size": 12}
let model = nn.GPT(cfg)
let opt = nn.AdamW(model.parameters(), 0.03)
let x = []
let y = []
let i = 0
while i + 12 < len(ids) {
    push(x, ids[i : i + 12])
    push(y, ids[i + 1 : i + 13])
    i = i + 1
}
let final_loss = 0.0
for step in range(1, 151) {
    opt.zero_grad()
    let loss = nn.cross_entropy(model(x), y)
    loss.backward()
    opt.step()
    final_loss = loss.item()
}
let out_greedy = nn.generate(model, tok.encode("the "), 20, 0.0)
let out_top_k = nn.generate(model, tok.encode("the "), 20, 0.5, 3)
say "loss=" + str(round(final_loss, 4))
say "greedy=" + tok.decode(out_greedy)
say "topk=" + tok.decode(out_top_k)
say "training=" + str(model.training)
''',
        encoding="utf-8",
    )
    out = _run_cli(script)
    assert "greedy=the cat sat on the mat. " in out, out
    assert "topk=the cat sat on the mat. " in out, out
    assert "training=true" in out, out
