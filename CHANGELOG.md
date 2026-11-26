# Changelog

All notable changes to ZEN are documented in this file.

## [0.3.0] - 2026-11-26

### Added
- **Trustworthy Core (Stage 1):**
  - Explicit integer division operator `div` and `floor_div(a, b)` builtin, plus `zen check` warning on trailing `// <number|var_in_scope>` comments.
  - String interpolation (`f"hello {name}"`) with `zen check` warning when `{var_in_scope}` appears in a plain string.
  - Optional types (`T?`), `any` type, and flow narrowing (`x != none`, `x == none`, `type(x) == "..."`).
  - Structured error codes `E001`–`E014`, `zen explain [code]`, Damerau/Levenshtein `did_you_mean` hints, `--debug` flag (suppressing raw Python tracebacks by default with `"internal error, please report"`), and `--max-depth N` (default `10000`) backed by an iterative `_VMFrame` heap stack.
- **Everyday Language Ergonomics (Stage 2):**
  - Exception handling (`try { ... } catch e { ... }` and `raise`), keyword arguments (`fn(a, b = 2)`), membership (`in` / `not in` for strings, lists, and maps), `if`-expressions, and list comprehensions (`[expr for x in seq if cond]`).
  - String methods (`split`, `join`, `upper`, `lower`, `strip`, `replace`, `startswith`, `endswith`, `find`), string repetition (`"ab" * 3`), list methods (`append`, `pop`, `sort`, `reverse`, `index`), multiple returns & tuple destructuring (`return a, b`; `let x, y = f()`; `a, b = b, a`), and `input(prompt)`.
- **Standard Library & Scientific Safety (Stage 3):**
  - 10 packaged `.zen` standard library modules (`stdlib/*.zen` & `zen/stdlib/*.zen`): `json`, `files`, `path`, `regex`, `datetime`, `env`, `csv`, `process`, `http`, and `stats` (`statistics`).
  - Scientific NaN/Inf tracing (`zen run --nan-trace`), reproducible run manifests (`zen run --record run.json` and `zen replay run.json`), and executable doctests (`zen test --doc` via `//>` / `//`).
- **v1.0 Architecture Memos (Stage 4):** Added `docs/design/01_zen_to_c_backend.md` through `07_v1_language_freeze_checklist.md`.
- **Documentation & Packaging:** Added `docs/LANGUAGE.md` (packaged in `zen/docs/` and surfaced via `zen docs`) and Chapter 23 `"What's New in ZEN 0.3"` in `docs/Learn-ZEN.pdf`.

## [0.2.4] - 2026-11-26

### Added
- **Differentiable `concat`, `stack`, `tril`, and `triu`:** Added `nn.concat(tensors, axis)`, `nn.stack(tensors, axis)`, `nn.tril(t, k)` / `t.tril(k)`, and `nn.triu(t, k)` / `t.triu(k)` with full reverse-mode gradients in `zennet.autograd`, plus `zn.tril(a, k)` and `zn.triu(a, k)` in `zennum` (`zen/zennum_core.py`).
- **`strict` mode on tokenizers and `nn.generate()`:** Added optional `strict = false` to `CharTokenizer.encode`, `BPETokenizer.encode`, and `nn.generate`. Default (`strict = false`) preserves legacy behavior (empty prompt -> `[]` / token `0` fallback; unknown chars skipped) with clear warning messages; `strict = true` raises a descriptive `Assertion failed` runtime error.
- **Packaged reference docs & `zen docs` CLI:** Bundled `ZENNUM.md` and `ZENNET.md` inside `zen/docs/` and added `zen docs` to print their installed paths.

### Fixed
- **Process-wide shared RNG (`seed` / `zn.seed`):** Replaced per-interpreter `random.Random()` instances in `zen/builtins.py` with a single process-wide `_GLOBAL_RNG`, so `seed()` and `zn.seed()` deterministically reset `zennet.data.text_to_batches`, `zennet.train.generate`, `zennum`, and both the VM and tree-walk interpreter across all imported modules.
- **Type checker (`let x = none`):** Unannotated `let` bindings initialized to `none` now infer `any` (optional/unknown) in `zen/typechecker.py`, allowing subsequent reassignment (`let model = none; model = ...`) to pass `zen check`.
- **Guide PDF (`docs/Learn-ZEN.pdf`):** Fixed right-edge text clipping on pages 33, 39, 40, 43, and 47 by wrapping table cells and long example lines; documented `zen doctor`, `zen docs`, `clock()`, `file_exists()`, and `nn.Module.init(self)`; updated Chapters 15–17 and Exercise 10 to use `zennum` and `zennet` instead of legacy `stdlib/ai/`.

## [0.2.3] - 2026-11-25

### Fixed
- **CI Test Compatibility (`tests/test_vscode.py`):** Replaced top-level `from PIL import Image` with `pytest.importorskip("PIL")` inside `test_vscode_icon_dimensions()`, so `pytest` no longer fails collection when Pillow is not installed (no runtime dependency added; Pillow and tools moved to `[project.optional-dependencies] dev` in `pyproject.toml`).
- **CI User-Folder Extension Test:** Updated `test_vscode_extension_installed_for_user()` to skip cleanly when `CI` is set or when `~/.vscode/extensions/zen-lang-0.1.0` does not exist, instead of requiring Windows `USERPROFILE` or a preinstalled local extension.

## [0.2.2] - 2026-11-25

### Added
- **`zen doctor` CLI command:** Prints ZEN version, OS, Python version, active math backend (native kernels vs pure-Python fallback), resolved stdlib path, and guide PDF path.
- **Cross-Platform CI & Docs:** Added `.github/workflows/ci.yml` testing `ubuntu-latest`, `macos-latest`, and `windows-latest` across Python `3.10`–`3.13` (both default and `ZEN_NO_NATIVE=1`), switched `.github/workflows/wheels.yml` to manual `workflow_dispatch`, added `.gitattributes` (LF line endings for `.zen`/`.py`/`.c`), updated `README.md` with Ubuntu/Debian (PEP 668 `venv`/`pipx`), macOS, Windows, and WSL instructions, and added `docs/GPU_PLAN.md`.

### Fixed
- **Cross-Platform Lexer, Paths & Formatter:** Normalized CRLF (`\r\n`) line endings in `zen/lexer.py`, normalized `/` and `\` path handling in `zen/paths.py` (`resolve_import_path`, `resolve_stdlib_path`) and `zen/builtins.py`, enforced LF (`newline="\n"`) writes in `zen/formatter.py` and `zen/builtins.py`, and made `OptionalBuildExt` in `setup.py` fall back silently when C compilation fails.
- **Portable Native Kernel (`zen/native/zen_kernels.c`):** Guarded GCC/Clang visibility attributes so non-GCC compilers work, and added optional OpenMP parallelization (`#ifdef _OPENMP`) in `zen_matmul_f64` using canonical loop bounds compatible with both MSVC and GCC/Clang while preserving `ZEN_NO_NATIVE=1`.

## [0.2.1] - 2026-11-25

### Fixed
- **GPT Recipe (`examples/gpt_in_40_lines.zen`):** Tuned to train for 1000 steps in ~3 minutes on CPU (`183.31 ms/step`, loss `3.736 -> 1.973` < `2.5`), printing loss every 100 steps and generating real corpus words; added `QUICK = false` flag at the top for a ~19s 100-step preview.
- **`nn.generate()` (`stdlib/zennet/train.zen`):** Guarded `top_k > 0` to prevent out-of-bounds indexing, handled empty prompts safely, restored `model.training` state after generation, and verified exact sentence memorization across `block_size` boundaries.

### Changed
- **Test Suite (`slow` marker):** Marked slow GPT training tests (`test_recipe_gpt_in_40_lines`, `test_generate_memorization`) with `@pytest.mark.slow` and configured `pyproject.toml` (`addopts = "-m 'not slow'"`) so plain `pytest` finishes in ~38s while `pytest -m slow` runs the full training tests.
- **Universal Packaging:** Made the optional C kernel build opt-in (`ZEN_BUILD_NATIVE=1`) in `setup.py` so `python -m build` produces a universal `py3-none-any` wheel and sdist that install on any OS without a C compiler.
- **Guide PDF:** Updated `docs/Learn-ZEN.pdf` (v0.2.1) Chapter 22.5 with live-captured output for both `QUICK = true` and `QUICK = false` GPT runs.

## [0.2.0] - 2026-11-25

### Added
- **ZenNum (`import zennum as zn`):** NumPy-like N-dimensional arrays (`float64`, `float32`, `int64`, `bool`), multi-dimensional slicing (`a[1:3, :]`), boolean masking (`a[a > 0]`), broadcasting with clear shape-mismatch errors, reductions, math functions, and linear algebra (`@`, `matmul`, `dot`, `outer`, `norm`).
- **ZenNet (`import zennet as nn`):** PyTorch-like deep learning framework written in `.zen` — reverse-mode `Tensor` autograd (`no_grad`, `detach`, slicing/scatter gradients), `Module` (`parameters()`, `train()`, `eval()`, `save()`, `load()`, `__call__`), `Linear`, `Embedding`, `LayerNorm`, `Dropout`, `Sequential`, causal `MultiHeadAttention`, `TransformerBlock`, `GPT`, losses (`mse`, `cross_entropy`, `binary_cross_entropy`), optimizers (`SGD`, `Adam`, `AdamW`, `clip_grad_norm`), LR schedulers (`StepLR`, `CosineAnnealingLR`, `WarmupLR`), `Dataset`/`DataLoader`, `CharTokenizer`/`BPETokenizer`, `Trainer`/`nn.fit`, `generate`, `accuracy`, and `perplexity`.
- **Language Upgrades:** Default parameter values (`fn foo(x = 1)`), single class inheritance (`class Child(Parent)`), `__getitem__`/`__setitem__`/`__call__` dispatch, and unquoted package imports (`import zennum as zn`, `import zennet as nn`).
- **Optional Native C Kernels & Packaging:** `zen/native/zen_kernels.c` with pure-Python flat-buffer fallback (`ZEN_NO_NATIVE=1`), optional `setup.py` C extension, `.github/workflows/wheels.yml` (`cibuildwheel`), updated `docs/ZENNUM.md`, `docs/ZENNET.md`, `README.md`, and expanded 46-page `docs/Learn-ZEN.pdf` with Part A (ZenNum) & Part B (ZenNet).

## [0.1.1] - 2026-11-25

### Added
- **Beginner Guide PDF (`docs/Learn-ZEN.pdf`):** 28-page step-by-step guide with real runnable examples (`docs/guide_examples/`), real captured outputs, 10 exercises with verified answers, and `tools/build_guide.py`.
- **`zen guide` CLI command:** Opens `Learn-ZEN.pdf` in the default PDF viewer (`zen guide --path` prints the packaged PDF path).

## [0.1.0] - 2026-11-25

### Added
- **Core Language & Runtime:** Lexer, recursive-descent parser, AST, bytecode compiler, stack-based bytecode VM (`zen run`), and fallback tree-walking interpreter (`zen run --tree-walk`).
- **Syntax & Type System:** `let`, `const`, `say` (`print` alias), `assert`, `fn`, closures, `class` with operator dunders, `if`/`elif`/`else`, `while`, `for`-`in`, lists, maps, `import`, optional type annotations (`x: int`, `-> float`), static type inference/checker (`--no-check` to skip), and source-line + caret error reporting.
- **Standard Library & Pure-Python Tensors:** Math, seeded random, string/list helpers, file I/O, `save_weights`/`load_weights`, and `ZenTensor` (`tensor`, `zeros`, `ones`, `randn`, `+ - * /` with broadcasting, `@` matmul, `transpose`, `reshape`, `sum`, `mean`, `max`, `exp`, `log`, `tanh`, `sqrt`, `gather`, `scatter_add`).
- **AI Stack in `.zen` (`stdlib/ai/`):** Scalar `Value` autograd, matrix `Tensor` autograd, `Dense`/`MLP`/`TensorMLP`, `SGD` & `Adam` optimizers, and `AutoCharGPT` character-level transformer with checkpointing.
- **Developer Tools & Editor Support:** Interactive REPL (`zen`), formatter (`zen fmt`), type checker (`zen check`), test runner (`zen test` for `tests_zen/`), and VS Code extension (`tools/vscode-zen/`).
