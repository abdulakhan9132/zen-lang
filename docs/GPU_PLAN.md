# GPU Support Plan (Design Note — Not Yet Implemented)

> **Status:** ZEN does **not** currently support GPU execution. All arrays, autograd, and models run strictly on the CPU (via optional C99/OpenMP kernels or the pure-Python flat-buffer fallback).

## Why GPUs Are Not Supported Yet

ZEN is built from scratch with zero third-party AI dependencies (`numpy`, `torch`, `cupy`, `jax`, etc. are intentionally disallowed). Adding GPU acceleration requires either shipping a small native C/C++ GPU runtime bridge or generating compute shaders directly, without pulling in external ML frameworks.

## Clean Architectural Seam

The codebase already isolates dense numeric buffers and kernel dispatch behind two Python boundaries:

1. **`zen/zennum_core.py` (`ZenArray`)** — stores contiguous typed flat buffers (`array('d')`, `array('f')`, `array('q')`, `array('B')`), shape, and strides, and delegates heavy 2D operations to `zen/native_kernels.py`.
2. **`zen/native_kernels.py`** — exposes `matmul_2d_buf`, `softmax_2d_buf`, `layernorm_2d_buf`, `embedding_lookup_buf`, and elementwise/reduction entry points with automatic CPU fallback when `ZEN_NO_NATIVE=1` or no native library is loaded.

Because `stdlib/zennum/` and `stdlib/zennet/` build all higher-level operations (`Tensor`, autograd, `Linear`, `MultiHeadAttention`, `GPT`) on top of `ZenArray`, `.zen` user programs will not need API changes when a GPU backend is added.

## Possible Future Backends

| Target Platform | Candidate Native API | Packaging Impact |
|---|---|---|
| Cross-platform (Linux/macOS/Windows) | **Vulkan Compute** or **WebGPU (`wgpu-native`)** | Single C ABI shader runner (`zen_gpu.c`), SPIR-V / WGSL kernels for matmul, softmax, layernorm, elementwise ops |
| Apple Silicon (macOS) | **Metal / MPS** | Optional `.m` bridge compiled only on `darwin` |
| NVIDIA (Linux/Windows) | **CUDA Driver API (`libcuda` via `ctypes`/PTX)** | Avoids bundling CUDA toolkit in wheels; loads driver dynamically if present |

## Phased Implementation Path

1. **Device abstraction on `ZenArray`:** Add `.device` (`"cpu"` today, `"gpu"` later) and explicit `.to("gpu")` / `.to("cpu")` transfers in `ZenArray` and `zennet.Tensor`.
2. **Kernel parity:** Implement tiled `matmul_f32` / `matmul_f64`, elementwise (`add`, `mul`, `exp`, `tanh`, `relu`), `softmax_2d`, `layernorm_2d`, and `embedding` compute shaders behind a C ABI matching `zen_kernels.c`.
3. **Buffer residency:** Keep intermediate tensors on device during `forward()`/`backward()` and copy back to CPU only on `.tolist()`, `.item()`, or `save_weights()`.
4. **Silent fallback:** Keep the universal `py3-none-any` wheel and pure-Python CPU fallback working identically on machines without a GPU or C/GPU SDK.
