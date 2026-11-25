"""Phase F tests: ZenNet DataLoader, Tokenizers, Trainer, generate, and unified import."""

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


def test_dataset_dataloader_and_tokenizers():
    code = """
import zennum as zn
import zennet as nn

zn.seed(3)
let X = [[0.0, 0.0], [0.0, 1.0], [1.0, 0.0], [1.0, 1.0], [2.0, 2.0]]
let Y = [[0.0], [1.0], [1.0], [0.0], [1.0]]
let splits = nn.train_val_split(X, Y, 0.4, true, 3)
let tr = splits[0]
let va = splits[1]
let dl = nn.DataLoader(tr, 2, false)
let bs = dl.batches()

let ctok = nn.CharTokenizer("hello world")
let cids = ctok.encode("hello")
let cdec = ctok.decode(cids)

let btok = nn.BPETokenizer(20)
btok.fit("abababab cdcdcd abab")
let bids = btok.encode("abab")
let bdec = btok.decode(bids)

let tb = nn.text_to_batches(ctok.encode("hello world hello world hello"), 4, 2)

say "{\\"tr_n\\": " + str(tr.size()) + ", \\"va_n\\": " + str(va.size()) + ", \\"n_batches\\": " + str(len(bs)) + ", \\"cdec\\": \\"" + cdec + "\\", \\"bdec\\": \\"" + bdec + "\\", \\"bpe_len\\": " + str(len(bids)) + ", \\"tb_shape\\": " + str(tb[0].shape) + "}"
"""
    o1, o2 = _run_both(code)
    assert o1 == o2
    d = json.loads(o1)
    assert d["tr_n"] == 3 and d["va_n"] == 2
    assert d["n_batches"] == 2
    assert d["cdec"] == "hello"
    assert d["bdec"] == "abab"
    assert d["bpe_len"] < 4
    assert d["tb_shape"] == [2, 4]


def test_trainer_checkpoint_resume_accuracy_perplexity_and_generate(tmp_path):
    ckpt = (tmp_path / "trainer_ckpt.json").as_posix()
    code = f"""
import zennum as zn
import zennet
import zennet as nn

zn.seed(42)

// Also verify unaliased `import zennet` works
let tok = zennet.CharTokenizer("abcabcabcabcabcabc")
let ids = tok.encode("abcabcabcabcabcabc")
let cfg = {{"vocab_size": tok.vocab_size, "n_layer": 1, "n_head": 2, "d_model": 8, "block_size": 6}}
let model = nn.GPT(cfg)
let tr = nn.Trainer(model, none, nn.cross_entropy, 0.08, 1.0, 0, "{ckpt}", 5)
let h1 = tr.fit(fn() {{ return nn.text_to_batches(ids, 6, 2) }}, 15, fn() {{ return nn.text_to_batches(ids, 6, 2) }})

// Resume into new model
let model2 = nn.GPT(cfg)
let tr2 = nn.Trainer(model2, none, nn.cross_entropy, 0.08, 1.0, 0, "{ckpt}", 0)
let h2 = tr2.fit(fn() {{ return nn.text_to_batches(ids, 6, 2) }}, 5, none, true)

let sample = nn.generate(model2, tok.encode("a"), 5, 0.7, 2)
let decoded = tok.decode(sample)
let acc = nn.accuracy([[0.1, 0.9], [0.8, 0.2], [0.2, 0.8]], [1, 0, 1])
let ppl = nn.perplexity(h1["final_loss"])

say "{{\\"loss_start\\": " + str(h1["loss_history"][0]) + ", \\"loss_end\\": " + str(h1["final_loss"]) + ", \\"val\\": " + str(h1["val_loss"]) + ", \\"resumed_steps\\": " + str(len(h2["loss_history"])) + ", \\"gen_len\\": " + str(len(sample)) + ", \\"acc\\": " + str(acc) + ", \\"ppl\\": " + str(ppl) + "}}"
"""
    o1, o2 = _run_both(code)
    for out in (o1, o2):
        d = json.loads(out)
        assert d["loss_end"] < d["loss_start"]
        assert d["val"] > 0.0
        assert d["resumed_steps"] == 5
        assert d["gen_len"] == 6
        assert abs(d["acc"] - 1.0) < 1e-6
        assert d["ppl"] >= 1.0


def test_phaseF_example_runs():
    assert main(["run", str(REPO / "examples" / "phaseF_trainer.zen")]) == 0
