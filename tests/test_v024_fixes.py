"""Tests for ZEN v0.2.4 fixes: shared RNG, concat/stack/tril/triu, none widening, strict mode, zen docs."""

from __future__ import annotations

from io import StringIO
import json
from pathlib import Path

import pytest

from zen.cli import main as cli_main
from zen.errors import ZenRuntimeError, ZenTypeError
from zen.interpreter import Interpreter
from zen.typechecker import type_check_source
from zen.vm import VM

REPO = Path(__file__).resolve().parent.parent


def _run_both(code: str) -> tuple[str, str]:
    b1 = StringIO()
    Interpreter(stdout=b1, base_dir=REPO).run(code, type_check=True)
    b2 = StringIO()
    VM(stdout=b2, base_dir=REPO).run(code, type_check=True)
    return b1.getvalue().strip(), b2.getvalue().strip()


def _assert_close(a, b, tol=1e-4) -> None:
    if isinstance(a, list):
        assert len(a) == len(b)
        for x, y in zip(a, b):
            _assert_close(x, y, tol)
    else:
        assert abs(a - b) <= tol, f"Mismatch: analytical={a} vs numerical={b}"


# 1. Process-wide shared RNG (seed & zn.seed across modules, VM, and tree-walker)


@pytest.mark.parametrize("seed_call", ["zn.seed(42)", "seed(42)"])
def test_shared_rng_text_to_batches_generate_and_full_gpt_training(seed_call: str) -> None:
    code = f"""
import zennum as zn
import zennet as nn

let raw = "the cat sat on the mat. the cat sat on the mat. "
let tok = nn.CharTokenizer(raw)
let ids = tok.encode(raw)

{seed_call}
let b1 = nn.text_to_batches(ids, 8, 2)
{seed_call}
let b2 = nn.text_to_batches(ids, 8, 2)

fn train_once() {{
    {seed_call}
    let cfg = {{"vocab_size": tok.vocab_size, "n_layer": 1, "n_head": 2, "d_model": 8, "block_size": 8}}
    let m = nn.GPT(cfg)
    let opt = nn.AdamW(m.parameters(), 0.02)
    let losses = []
    for s in range(6) {{
        let batch = nn.text_to_batches(ids, 8, 2)
        opt.zero_grad()
        let loss = nn.cross_entropy(m(batch[0]), batch[1])
        loss.backward()
        opt.step()
        push(losses, loss.item())
    }}
    return [losses, m]
}}

let r1 = train_once()
let r2 = train_once()

{seed_call}
let g1 = tok.decode(nn.generate(r1[1], tok.encode("the "), 12, 0.8, 4))
{seed_call}
let g2 = tok.decode(nn.generate(r2[1], tok.encode("the "), 12, 0.8, 4))

say "{{\\"b_eq\\": " + str(b1[0].tolist() == b2[0].tolist() and b1[1].tolist() == b2[1].tolist()) + ", \\"l_eq\\": " + str(r1[0] == r2[0]) + ", \\"g_eq\\": " + str(g1 == g2) + ", \\"g1\\": \\"" + g1 + "\\", \\"l1\\": " + str(r1[0]) + "}}"
"""
    o1, o2 = _run_both(code)
    assert o1 == o2
    d = json.loads(o1)
    assert d["b_eq"] is True
    assert d["l_eq"] is True
    assert d["g_eq"] is True


# 2. Autograd & ZenNum ops: concat, stack, tril, triu


def test_zennum_tril_and_triu() -> None:
    code = """
import zennum as zn

let a = zn.array([[1, 2, 3], [4, 5, 6], [7, 8, 9]])
say zn.tril(a).tolist()
say zn.tril(a, 1).tolist()
say zn.tril(a, -1).tolist()
say zn.triu(a).tolist()
say zn.triu(a, 1).tolist()
say zn.triu(zn.ones([3, 3], "bool"), 1).tolist()

let b = zn.arange(1, 9).reshape([2, 2, 2])
say zn.tril(b).tolist()
say zn.triu(b, 1).tolist()
"""
    o1, o2 = _run_both(code)
    assert o1 == o2
    lines = o1.splitlines()
    assert lines[0] == "[[1, 0, 0], [4, 5, 0], [7, 8, 9]]"
    assert lines[1] == "[[1, 2, 0], [4, 5, 6], [7, 8, 9]]"
    assert lines[2] == "[[0, 0, 0], [4, 0, 0], [7, 8, 0]]"
    assert lines[3] == "[[1, 2, 3], [0, 5, 6], [0, 0, 9]]"
    assert lines[4] == "[[0, 2, 3], [0, 0, 6], [0, 0, 0]]"
    assert lines[5] == "[[false, true, true], [false, false, true], [false, false, false]]"
    assert lines[6] == "[[[1, 0], [3, 4]], [[5, 0], [7, 8]]]"
    assert lines[7] == "[[[0, 2], [0, 0]], [[0, 6], [0, 0]]]"


def test_autograd_concat_stack_tril_triu_numerical_gradients() -> None:
    code = """
import zennum as zn
import zennet as nn

let w_cat = nn.tensor([[1.1, -0.5, 0.8, 1.4, -0.9], [0.3, 1.7, -1.2, 0.6, 0.2]], false)
let w_stk = nn.tensor([[[1.2, -0.4], [0.7, 1.5]], [[-0.8, 0.9], [0.4, -1.1]]], false)
let w_tri = nn.tensor([[1.3, -0.7, 2.1], [0.4, -1.2, 0.8], [0.5, 1.0, -0.6]], false)

fn loss_concat(a_arr, b_arr) {
    let a = nn.tensor(a_arr, true)
    let b = nn.tensor(b_arr, true)
    let y = nn.concat([a, b], 1)
    return [(y * w_cat).sum(), a, b]
}

fn loss_stack(a_arr, b_arr) {
    let a = nn.tensor(a_arr, true)
    let b = nn.tensor(b_arr, true)
    let y = nn.stack([a, b], 0)
    return [(y * w_stk).sum(), a, b]
}

fn loss_tri(x_arr) {
    let x = nn.tensor(x_arr, true)
    let y = nn.tril(x, 0) + nn.triu(x, 1) * 2.0
    return [(y * w_tri).sum(), x]
}

let a0 = zn.array([[0.5, -1.2], [0.8, 0.3]], "float64")
let b0 = zn.array([[1.0, 0.2, -0.6], [-0.4, 1.1, 0.7]], "float64")
let rc = loss_concat(a0, b0)
rc[0].backward()
let ana_ca = rc[1].grad.flatten().tolist()
let ana_cb = rc[2].grad.flatten().tolist()

let eps = 1e-5
let num_ca = []
let fa = a0.flatten().tolist()
for i in range(len(fa)) {
    let ap = zn.array(fa, "float64")
    let am = zn.array(fa, "float64")
    ap[i] = fa[i] + eps
    am[i] = fa[i] - eps
    let lp = nn.no_grad(fn() { return loss_concat(ap.reshape([2, 2]), b0)[0].item() })
    let lm = nn.no_grad(fn() { return loss_concat(am.reshape([2, 2]), b0)[0].item() })
    push(num_ca, (lp - lm) / (2.0 * eps))
}

let b1 = zn.array([[-0.3, 0.9], [1.4, -0.2]], "float64")
let rs = loss_stack(a0, b1)
rs[0].backward()
let ana_sa = rs[1].grad.flatten().tolist()
let num_sa = []
for i in range(len(fa)) {
    let ap = zn.array(fa, "float64")
    let am = zn.array(fa, "float64")
    ap[i] = fa[i] + eps
    am[i] = fa[i] - eps
    let lp = nn.no_grad(fn() { return loss_stack(ap.reshape([2, 2]), b1)[0].item() })
    let lm = nn.no_grad(fn() { return loss_stack(am.reshape([2, 2]), b1)[0].item() })
    push(num_sa, (lp - lm) / (2.0 * eps))
}

let x0 = zn.array([[0.2, -0.5, 1.1], [0.9, 0.4, -0.7], [-1.3, 0.6, 0.8]], "float64")
let rt = loss_tri(x0)
rt[0].backward()
let ana_tri = rt[1].grad.flatten().tolist()
let fx = x0.flatten().tolist()
let num_tri = []
for i in range(len(fx)) {
    let xp = zn.array(fx, "float64")
    let xm = zn.array(fx, "float64")
    xp[i] = fx[i] + eps
    xm[i] = fx[i] - eps
    let lp = nn.no_grad(fn() { return loss_tri(xp.reshape([3, 3]))[0].item() })
    let lm = nn.no_grad(fn() { return loss_tri(xm.reshape([3, 3]))[0].item() })
    push(num_tri, (lp - lm) / (2.0 * eps))
}

say "{\\"ana_ca\\": " + str(ana_ca) + ", \\"num_ca\\": " + str(num_ca) + ", \\"ana_cb\\": " + str(ana_cb) + ", \\"ana_sa\\": " + str(ana_sa) + ", \\"num_sa\\": " + str(num_sa) + ", \\"ana_tri\\": " + str(ana_tri) + ", \\"num_tri\\": " + str(num_tri) + "}"
"""
    o1, o2 = _run_both(code)
    assert o1 == o2
    d = json.loads(o1)
    _assert_close(d["ana_ca"], d["num_ca"])
    _assert_close(d["ana_sa"], d["num_sa"])
    _assert_close(d["ana_tri"], d["num_tri"])


# 3. Type checker: `let model = none` followed by reassignment


def test_type_checker_allows_let_none_reassignment() -> None:
    src = """
class Box {
    fn init(self, v) { self.v = v }
    fn get(self) { return self.v }
}
let model = none
model = Box(42)
say model.get()

let x = none
x = 123
x = "now a string"
say x
"""
    type_check_source(src)
    b = StringIO()
    VM(stdout=b, base_dir=REPO).run(src, type_check=True)
    assert b.getvalue().splitlines() == ["42", "now a string"]

    with pytest.raises(ZenTypeError, match="expected 'none', got 'int'"):
        type_check_source("let explicit_none: none = none\nexplicit_none = 1")

    with pytest.raises(ZenTypeError, match="constant"):
        type_check_source("const C = none\nC = 1")


# 4. Edge cases: warning on default strict=false, clear error on strict=true


def test_tokenizer_and_generate_strict_and_default_behavior() -> None:
    default_code = """
import zennum as zn
import zennet as nn

let ctok = nn.CharTokenizer("abc")
let e1 = ctok.encode("")
let e2 = ctok.encode("ab@c")

let btok = nn.BPETokenizer(10)
btok.fit("abc abc")
let b1 = btok.encode("")
let b2 = btok.encode("ab#")

zn.seed(1)
let m = nn.GPT({"vocab_size": 3, "n_layer": 1, "n_head": 1, "d_model": 4, "block_size": 4})
let g = nn.generate(m, [], 3, 0.0)
say "defaults=" + str(e1) + "|" + str(e2) + "|" + str(b1) + "|" + str(b2) + "|" + str(len(g))
"""
    o1, o2 = _run_both(default_code)
    assert o1 == o2
    assert "Warning: empty prompt" in o1
    assert "Warning: skipping unknown character(s)" in o1
    assert "Warning: empty prompt_ids" in o1
    assert "defaults=[]|[0, 1, 2]|[]|[4]|4" in o1

    strict_cases = [
        (
            'import zennet as nn\nlet t = nn.CharTokenizer("abc")\nt.encode("", true)',
            "empty prompt",
        ),
        (
            'import zennet as nn\nlet t = nn.CharTokenizer("abc")\nt.encode("a@b", true)',
            "unknown character",
        ),
        (
            'import zennet as nn\nlet b = nn.BPETokenizer(10)\nb.fit("abc abc")\nb.encode("", true)',
            "empty prompt",
        ),
        (
            'import zennet as nn\nlet b = nn.BPETokenizer(10)\nb.fit("abc abc")\nb.encode("a#", true)',
            "unknown character",
        ),
        (
            'import zennet as nn\nlet m = nn.GPT({"vocab_size": 4, "n_layer": 1, "n_head": 1, "d_model": 4, "block_size": 4})\nnn.generate(m, [], 3, 0.0, none, true)',
            "empty prompt_ids",
        ),
        (
            'import zennet as nn\nlet m = nn.GPT({"vocab_size": 4, "n_layer": 1, "n_head": 1, "d_model": 4, "block_size": 4})\nnn.generate(m, [9], 3, 0.0, none, true)',
            "out of range",
        ),
    ]
    for code, expected in strict_cases:
        for engine in (VM(base_dir=REPO), Interpreter(base_dir=REPO)):
            with pytest.raises(ZenRuntimeError, match=expected):
                engine.run(code, type_check=True)


# 5. Packaged docs & `zen docs` CLI


def test_zen_docs_cli_and_packaged_markdown(capsys: pytest.CaptureFixture[str]) -> None:
    assert cli_main(["docs"]) == 0
    out = capsys.readouterr().out
    assert "ZENNUM.md:" in out
    assert "ZENNET.md:" in out
    for line in out.strip().splitlines():
        _, path_str = line.split(": ", 1)
        assert Path(path_str).is_file()

    assert (REPO / "zen" / "docs" / "ZENNUM.md").is_file()
    assert (REPO / "zen" / "docs" / "ZENNET.md").is_file()


# 6. Native C wheel build: PyInit__zen_kernels export & loud failure on ZEN_BUILD_NATIVE=1


def test_native_kernel_exports_pyinit_and_setup_fails_loudly(monkeypatch: pytest.MonkeyPatch) -> None:
    import runpy
    from setuptools import Distribution
    from setuptools.command.build_ext import build_ext

    c_src = (REPO / "zen" / "native" / "zen_kernels.c").read_text(encoding="utf-8")
    assert "PyInit__zen_kernels" in c_src
    assert "ZEN_EXPORT void *PyInit__zen_kernels(void)" in c_src

    captured: dict[str, object] = {}
    monkeypatch.setattr("setuptools.setup", lambda **kw: captured.update(kw))
    monkeypatch.setenv("ZEN_BUILD_NATIVE", "1")
    monkeypatch.delenv("ZEN_SKIP_NATIVE_BUILD", raising=False)

    ns = runpy.run_path(str(REPO / "setup.py"))
    assert len(captured.get("ext_modules", [])) == 1
    opt_cls = ns["OptionalBuildExt"]

    def _boom(*args, **kwargs):
        raise RuntimeError("simulated compiler/linker failure")

    monkeypatch.setattr(build_ext, "build_extensions", _boom)
    monkeypatch.setattr(build_ext, "build_extension", _boom)

    cmd = opt_cls(Distribution())
    with pytest.raises(RuntimeError, match="simulated compiler/linker failure"):
        cmd.build_extensions()
    with pytest.raises(RuntimeError, match="simulated compiler/linker failure"):
        cmd.build_extension(captured["ext_modules"][0])

    monkeypatch.delenv("ZEN_BUILD_NATIVE", raising=False)
    cmd.build_extensions()
    cmd.build_extension(captured["ext_modules"][0])
