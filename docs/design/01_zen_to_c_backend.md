# Design Memo 01: ZEN-to-C Backend, Differential Testing, BLAS/FFI & Microcontroller Target

> **Status:** Design Memo (not implemented in v0.3.0).

## 1. The Problem
ZEN's bytecode VM and pure-Python `ZenArray` flat-buffer engine are clean and portable, but Python bytecode dispatch plus loop-level arithmetic limits numeric throughput (`~183 ms/step` on a 1-layer `d_model=16` GPT) and cannot run standalone on embedded/microcontroller targets without Python. Compiling a strictly typed numeric subset of ZEN to C99 closes the speed gap while preserving VM semantics and zero-external-ML-library constraints.

## 2. Proposed Syntax (10 Lines of Examples)

```zen
// Compile a typed numeric kernel with `zen build --c` or `@native`
extern fn cblas_dgemm(m: int, n: int, k: int, a: array, b: array, c: array) -> none

fn matvec(w: array, x: array, out: array, rows: int, cols: int) -> none {
    for r in range(rows) {
        let acc: float = 0.0
        for c in range(cols) {
            acc = acc + w[r, c] * x[c]
        }
        out[r] = acc
    }
}
```

## 3. Error Messages
- `E020 [line L, col C] Unsupported feature in C backend: dynamic closure or 'any' variable 'v' cannot be lowered to C99 (annotate as int/float/bool/array).`
- `E021 [line L, col C] Unbounded recursion or non-static list growth in '@native' function 'f' is unsupported on no-alloc embedded target.`
- `E022 [line L, col C] C compiler invocation failed (no gcc/clang/cl found); use '--emit-c' to inspect generated '.c' only.`

## 4. Edge Cases
- **Overflow & division by zero:** C signed integer overflow is UB; emit `int64_t` helpers or explicit divisor checks (`if (b == 0) zen_abort("E007");`) so C matches VM semantics.
- **Broadcasting & slicing views:** Either restrict initial C subset to contiguous 1D/2D `ZenArray` buffers or pass `{double *data, int64_t *shape, int64_t *strides, int ndim}` descriptors.
- **No dynamic heap on microcontrollers:** Embedded profile (`--target mcu`) forbids `list.append`, strings, maps, and `malloc`, requiring caller-allocated static buffers or fixed-size arena pools.
- **Float precision:** Use IEEE-754 `double` (`float64`) by default to stay bit-compatible with VM `float`, with `--f32` opt-in for Cortex-M FPU targets.

## 5. Minimal Prototype Plan
1. Define `CBackendSubset` checker in `zen/c_lower.py` accepting functions with typed `int`, `float`, `bool`, fixed `array`, `for i in range(...)`, `if`/`else`, `while`, and arithmetic (`+ - * / div %`).
2. Emit standalone C99 `.c`/`.h` (`zen_runtime.h` + lowered functions) + Python `ctypes` wrapper (`zen build --c`).
3. Plug optional system BLAS (`cblas_dgemm` / Accelerate / OpenBLAS) and `extern fn` C FFI calls (`zen/ffi.py` parsing typed C signatures -> `dlsym`/`ctypes` in VM and direct `#include` prototypes in C).
4. Add `--target mcu` mode emitting allocation-free C99 with static tensor arenas suitable for `arm-none-eabi-gcc`.

## 6. Test Plan
- **Differential testing against VM:** Property/seed harness generating random typed numeric programs (`dot`, `matmul`, `conv1d`, `softmax`, `layernorm`, nested loops, negative `div` / `%`), running both `VM` and compiled C shared library, and asserting bit/ulp-exact equality (`<= 1e-12` for transcendental ops, exact `==` for integers).
- **FFI & BLAS tests:** Call a tiny compiled test `.so`/`.dll` via `extern fn`, verify error when library/symbol missing.
- **Microcontroller smoke test:** Compile generated `--target mcu` C output with `-std=c99 -ffreestanding -nostdlib -Wall -Werror` on CI when `gcc` is available (skip gracefully if no compiler).

## 7. Honest Difficulty and Risks
- **Alias & lifetime safety:** Mutating aliased `array` slices without a borrow checker or reference counting can create subtle bugs between VM and C; keep buffers caller-owned initially.
- **Dual maintenance:** Every VM semantic quirk (boolean-int distinction, `-7 div 2 == -4`) must be faithfully mirrored in C macros (`ZEN_FLOOR_DIV`), otherwise differential tests will diverge.
- **Cross-OS C toolchains:** Compiler/linker flags differ across Linux, macOS, and Windows MSVC; default wheel must stay pure Python.

## 8. Effort Estimate
- **12–16 steps:** AST subset validator (2), C99 emitter + `ZEN_FLOOR_DIV`/bounds checks (4), `ctypes` loader + `zen build --c` CLI (2), `extern fn` + optional BLAS hook (3), `--target mcu` static-arena mode (2), differential fuzzer test suite against VM (3).
