# Changelog

All notable changes to ZEN are documented in this file.

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
