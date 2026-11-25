"""Tests for Phase 14: Save/load weights, checkpointing/resume, and tensor optimizations."""

import io
from pathlib import Path

from zen.tensor import ZenTensor
from zen.vm import VM

ROOT = Path(__file__).resolve().parent.parent


def test_save_and_load_weights_roundtrip(tmp_path: Path) -> None:
    ckpt = tmp_path / "weights.json"
    buf = io.StringIO()
    vm = VM(stdout=buf, base_dir=ROOT)
    vm.run(
        f"""
        import "stdlib/ai/tensor_autograd.zen" as ta
        let w = ta.from([[1.0, 2.5], [-3.0, 4.0]])
        let state = {{"step": 42, "w": w, "label": "ok"}}
        save_weights("{ckpt.as_posix()}", state)
        let loaded = load_weights("{ckpt.as_posix()}")
        say loaded["step"]
        say loaded["label"]
        say loaded["w"].tolist()
        """,
        type_check=True,
    )
    assert buf.getvalue().splitlines() == ["42", "ok", "[[1.0, 2.5], [-3.0, 4.0]]"]


def test_auto_char_gpt_checkpoint_save_and_resume(tmp_path: Path) -> None:
    ckpt = tmp_path / "gpt_ckpt.json"
    buf = io.StringIO()
    vm = VM(stdout=buf, base_dir=ROOT)
    vm.run(
        f"""
        import "stdlib/ai/tensor_nn.zen" as tnn
        import "stdlib/ai/tensor_gpt.zen" as lm

        seed(7)
        let text = read_file("examples/tiny.txt")
        let v = lm.build_vocab(text)
        let d = lm.encode(text, v["stoi"])

        let m1 = lm.AutoCharGPT(v["size"], 4, 8)
        let o1 = tnn.Adam(m1.parameters(), 0.05)
        let bx = [slice(d, 0, 4), slice(d, 4, 8)]
        let by = [slice(d, 1, 5), slice(d, 5, 9)]
        for step in range(3) {{
            let l = m1.compute_loss(bx, by)
            o1.zero_grad()
            l.backward()
            o1.step()
        }}
        m1.save_checkpoint("{ckpt.as_posix()}", 2, o1)
        let before = m1.forward_logits(bx).data.tolist()

        let m2 = lm.AutoCharGPT(v["size"], 4, 8)
        let o2 = tnn.Adam(m2.parameters(), 0.05)
        let resumed_step = m2.load_checkpoint("{ckpt.as_posix()}", o2)
        let after = m2.forward_logits(bx).data.tolist()

        say resumed_step
        say o2.t == o1.t
        say before == after
        """,
        type_check=True,
    )
    assert buf.getvalue().splitlines() == ["2", "true", "true"]


def test_tensor_optimizations_preserve_correctness() -> None:
    import random

    rng = random.Random(99)
    a = ZenTensor.randn([8, 5], rng)
    b = ZenTensor.randn([5, 4], rng)
    prod = a.matmul(b)
    assert prod.shape == (8, 4)
    t = prod.transpose()
    assert t.shape == (4, 8)
    assert abs(t.tolist()[2][3] - prod.tolist()[3][2]) < 1e-12
