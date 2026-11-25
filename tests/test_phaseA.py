"""Tests for Phase A: Native C kernels + pure-Python flat-buffer fallback."""

from array import array
import math

from zen import native_kernels


def test_detect_compiler_and_backend_status() -> None:
    # Should never crash even when no C compiler is present
    compiler = native_kernels.detect_c_compiler()
    assert compiler is None or isinstance(compiler, str)
    assert isinstance(native_kernels.NATIVE_BACKEND, str)


def test_flat_buffer_kernels_matmul_softmax_layernorm_embedding(monkeypatch) -> None:
    monkeypatch.setenv("ZEN_NO_NATIVE", "1")
    a = array("d", [1.0, 2.0, 3.0, 4.0, 5.0, 6.0])  # 2x3
    b = array("d", [7.0, 8.0, 9.0, 10.0, 11.0, 12.0])  # 3x2
    c = native_kernels.matmul_2d_buf(a, 0, 2, 3, b, 0, 2)
    assert list(c) == [58.0, 64.0, 139.0, 154.0]

    probs = native_kernels.softmax_2d_buf(array("d", [1.0, 2.0, 3.0]), 1, 3)
    assert abs(sum(probs) - 1.0) < 1e-12
    assert probs[2] > probs[1] > probs[0]

    ln = native_kernels.layernorm_2d_buf(array("d", [1.0, 2.0, 3.0, 4.0]), None, None, 1, 4, eps=1e-5)
    assert abs(sum(ln)) < 1e-12
    assert abs(sum(v * v for v in ln) / 4.0 - 1.0) < 1e-3

    table = array("d", [10.0, 11.0, 20.0, 21.0, 30.0, 31.0])
    emb = native_kernels.embedding_lookup_buf(table, [2, 0], 2)
    assert list(emb) == [30.0, 31.0, 10.0, 11.0]
    assert math.isfinite(emb[0])
