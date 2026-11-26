# ZEN Progress Log

## Phases 1–15 + Beginner Guide (v0.1.1)
- **Status:** Done (`62` tests)

## Phase A: Native Kernels & Flat-Buffer Fallback
- **Status:** Done (`64` tests cumulative)
- **Notes:** `zen/native/zen_kernels.c` + `zen/native_kernels.py` ready; pure-Python flat-buffer fallback runs automatically when no C compiler is present.

## Phase B: ZenNum Core & Language Upgrades
- **Status:** Done (`70` tests cumulative)
- **Notes:** `stdlib/zennum/` implemented with `ZenArray` flat-buffer engine, multi-dim slicing, boolean masks, broadcasting, default args, class inheritance, and `import zennum as zn` / `import zennum`.

## Phase C: ZenNet Autograd Engine
- **Status:** Done (`83` tests cumulative)
- **Notes:** `stdlib/zennet/autograd.zen` written purely in ZEN; every op verified against numerical gradients.

## Phase D: ZenNet Layers, Losses & GPT
- **Status:** Done (`86` tests cumulative)
- **Notes:** `stdlib/zennet/nn.zen` implements `Module`, `Linear`, `Embedding`, `LayerNorm`, `Dropout`, `Sequential`, causal `MultiHeadAttention`, `TransformerBlock`, `GPT`, and losses.

## Phase E: ZenNet Optimizers & Schedulers
- **Status:** Done (`89` tests cumulative)
- **Notes:** `stdlib/zennet/optim.zen` implements `SGD`, `Adam`, `AdamW`, `clip_grad_norm`, `StepLR`, `CosineAnnealingLR`, and `WarmupLR`.

## Phase F: ZenNet Data, Tokenizers, Trainer & Unified Import
- **Status:** Done (`92` tests cumulative)
- **Notes:** `Dataset`, `DataLoader`, `train_val_split`, `CharTokenizer`, `BPETokenizer`, `text_to_batches`, `Trainer`, `fit`, `generate`, `accuracy`, `perplexity`, and unified `import zennet as nn`.

## Phase G: Recipes & GPT Benchmark
- **Status:** Done (`97` tests cumulative)
- **Notes:** Verified `zennum_basics.zen`, `linear_regression.zen`, `mlp_xor.zen`, `tiny_classifier.zen`, and `gpt_in_40_lines.zen` (`354.77 ms/step` pure-Python flat buffers on 2-layer `d_model=16` GPT).

## Phase H: Documentation & Expanded Learn-ZEN PDF
- **Status:** Done (`97` tests cumulative)
- **Notes:** Added `docs/ZENNUM.md`, `docs/ZENNET.md`, Part A & B runnable guide examples, cheat sheets, 16 new exercises, comparison table, and rebuilt 46-page `docs/Learn-ZEN.pdf`.

## Phase I: Packaging & Release Prep (`v0.2.0` – `v0.2.3`)
- **Status:** Complete
- **Notes:** Universal `py3-none-any` wheel & sdist, opt-in C extension (`ZEN_BUILD_NATIVE=1`), `zen doctor`, cross-platform CI, and CI test compatibility fixes.

## Release `v0.2.4` (Current)
- **Status:** Complete (committed & tagged locally as `v0.2.4`; `dist/zen_lang-0.2.4-py3-none-any.whl` and `dist/zen_lang-0.2.4.tar.gz` built and verified in clean venv)
- **Notes:**
  - Process-wide shared `_GLOBAL_RNG` in `zen/builtins.py` fixing `seed()` / `zn.seed()` determinism across `zennet.data`, `zennet.train`, VM, and tree-walk interpreter.
  - Added `zn.tril` / `zn.triu` and differentiable `nn.concat`, `nn.stack`, `nn.tril`, `nn.triu` with numerical gradient checks.
  - Type checker widening for unannotated `let x = none`.
  - Optional `strict` flag on `CharTokenizer.encode`, `BPETokenizer.encode`, and `nn.generate` (warning + legacy fallbacks by default; error when `strict = true`).
  - Packaged `ZENNUM.md` / `ZENNET.md` inside `zen/docs/`, added `zen docs`, fixed PDF clipping on pages 33/39/40/43/47, and updated Chapters 15–17 to use `zennum`/`zennet`.
  - Tests: `107` fast pytest tests (`2` slow deselected) + `5` `.zen` self-tests passing.

## Post-`v0.2.4` Cleanup (CI, Docs & Native Wheel Fix)
- **Status:** Complete (committed locally, not pushed)
- **Notes:**
  - Updated `.github/workflows/ci.yml` and `.github/workflows/wheels.yml` to `actions/checkout@v7` and `actions/setup-python@v7` (Node 24 native), added CI badge and `v0.2.4` Ubuntu `pipx` install instructions to `README.md`, and updated `PROGRESS.md`.
  - Fixed native wheel build: exported `PyInit__zen_kernels` stub in `zen/native/zen_kernels.c` for MSVC `/EXPORT:PyInit__zen_kernels` without Python headers, made `OptionalBuildExt` in `setup.py` fail loudly when `ZEN_BUILD_NATIVE=1`, added `CIBW_ENVIRONMENT: "ZEN_BUILD_NATIVE=1"` and `NATIVE_AVAILABLE` assertion in `.github/workflows/wheels.yml`, and added unit test coverage.

## Next Planned Items
- Push local commits and `v0.2.4` tag to GitHub and publish the `v0.2.4` GitHub release with `dist/zen_lang-0.2.4-py3-none-any.whl` and `dist/zen_lang-0.2.4.tar.gz`.
- Build and ship prebuilt multi-platform native C kernel wheels via `.github/workflows/wheels.yml`.
- Explore GPU/SIMD backends (see `docs/GPU_PLAN.md`), native AOT compilation, and LSP support.
