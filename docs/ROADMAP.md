# ZEN Development Roadmap

## Phase 1: Spec, grammar, folder structure — **DONE**
- Created project structure (`zen/`, `docs/`, `examples/`, `tests/`).
- Defined language specification in `docs/SPEC.md` and EBNF grammar in `docs/GRAMMAR.md` (updated for `say` and `const`).
- Added canonical lexical constants in `zen/spec.py` and target milestone program in `examples/hello.zen`.

## Phase 2: Lexer, parser, AST — **DONE**
- Implemented `zen/tokens.py`, `zen/errors.py`, `zen/lexer.py`, `zen/ast.py`, and `zen/parser.py`.
- Added line/column tracking, comments, strings with escapes, numbers, operators with precedence, and clear `ZenSyntaxError` reporting.
- Added runnable AST example (`examples/phase2_ast.py`) and passing test suite (`tests/test_phase2.py`).

## Phase 3: Interpreter — **DONE**
- Implemented `zen/environment.py`, `zen/interpreter.py`, `zen/cli.py`, and `zen/__main__.py`.
- Supports `let`, `const`, `say` (plus `print` alias), `fn`, closures, `class` with `init`/methods, `if`/`elif`/`else`, `while`, `for`-`in`, `break`/`continue`, lists, maps, and `ZenRuntimeError` with line numbers.
- Milestone `examples/hello.zen` runs via real tokenizing, parsing, and execution (`zen run examples/hello.zen`).
- Added `examples/phase3_closures.zen` and passing test suite (`tests/test_phase3.py`).

## Phase 4: Loops, lists, math — **DONE**
- Implemented `zen/builtins.py` with `range`, `push`/`pop`, `map`/`filter`/`reduce`, math (`sqrt`, `exp`, `log`, `sin`, `cos`, `tanh`, `pow`, `abs`, `min`, `max`, `sum`, `seed`, `random`, `randint`, `uniform`), strings (`split`, `join`, `trim`, `upper`, `lower`, `contains`, `replace`, `chars`, `ord`, `chr`), file I/O (`read_file`, `write_file`), `clock`, and `import` for `.zen` modules.
- Added `examples/phase4_builtins.zen`, `examples/math_helper.zen`, and passing test suite (`tests/test_phase4.py`).

## Phase 5: Tensor engine — **DONE**
- Implemented pure-Python `ZenTensor` in `zen/tensor.py` using flat Python `list[float]` only (no numpy/torch).
- Supports `tensor`, `zeros`, `ones`, `tensor_rand`, `randn`, elementwise `+ - * /` with broadcasting, `@` (`matmul`), `transpose`/`.T`, `reshape`, `shape`, `tolist`, and `sum`/`mean`/`max` reductions.
- Added `examples/phase5_tensors.zen` and passing test suite (`tests/test_phase5.py`).

## Phase 6: Autograd in ZEN — **DONE**
- Implemented reverse-mode automatic differentiation (`Value`, `backward()`) completely in `.zen` at `stdlib/ai/autograd.zen`.
- Supports `+`, `-`, `*`, `/`, `**`, unary `-`, `relu`, `tanh`, `sigmoid`, `exp`, `log`, and topological reverse-mode gradient accumulation.
- Added `examples/phase6_autograd.zen` and passing test suite (`tests/test_phase6.py`) verified against hand-calculated derivatives.

## Phase 7: Neural network in `.zen` — **DONE**
- Implemented `Neuron`, `Dense`, `MLP`, `relu`, `sigmoid`, `softmax`, `mse_loss`, `cross_entropy_loss`, and `SGD` in `stdlib/ai/nn.zen`.
- Trained an MLP on XOR in `examples/phase7_xor.zen`, verifying loss drops cleanly across epochs and all 4 XOR inputs classify correctly.
- Added passing test suite (`tests/test_phase7.py`).

## Phase 8: Tiny character-level GPT in `.zen` — **DONE**
- Implemented `TinyCharGPT` (token/positional embeddings, causal self-attention, ReLU FFN, LM head, cross-entropy backward pass, SGD, and text generation) completely in `.zen` at `stdlib/ai/char_lm.zen`.
- Trained on `examples/tiny.txt` in `examples/phase8_char_lm.zen`, printing loss reduction, honest training time, and generated text.
- Added passing test suite (`tests/test_phase8.py`).

## Phase 9: Tensor autograd — **DONE**
- Added pure-math tensor `exp`, `log`, `tanh`, `clamp_min`, `gt`, `keepdims` reductions, batched N-D `matmul`, and `gather`/`scatter_add` in `zen/tensor.py` and `zen/builtins.py`.
- Implemented matrix/tensor autograd (`Tensor`, `backward()`) completely in `.zen` at `stdlib/ai/tensor_autograd.zen`, supporting `+`, `-`, `*`, `/`, `@`, `sum`, `mean`, `transpose`, `reshape`, `exp`, `log`, `relu`, `tanh`, `softmax`, and `gather`.
- Verified all operations against finite-difference numerical gradients in `tests/test_phase9.py` and added `examples/phase9_tensor_autograd.zen`.

## Phase 10: Rewrite models on tensor autograd — **DONE**
- Implemented `TensorLinear`, `TensorMLP`, `sigmoid`, `tensor_mse_loss`, `tensor_cross_entropy`, and `Adam` in `stdlib/ai/tensor_nn.zen`.
- Rewrote the character-level GPT (`AutoCharGPT`) using `Tensor` autograd in `stdlib/ai/tensor_gpt.zen` (zero hand-written gradients).
- Compared old vs new XOR & GPT training in `examples/phase10_compare.zen` with passing tests in `tests/test_phase10.py`.

## Phase 11: Bigger text test — **DONE**
- Added ~256 KB corpus at `examples/data/corpus.txt` and `slice` built-in in `zen/builtins.py`.
- Trained `AutoCharGPT` (`block_size=12, d_model=12, batch_size=8`) with 90/10 train/val split in `examples/phase11_big_text.zen` (`train_loss: 3.6967 -> 2.4734`, `val_loss: 3.6851 -> 2.4385`, `91.51 ms/step`, `4.58s` total).
- Added passing tests in `tests/test_phase11.py`.

## Phase 12: Speed (Bytecode VM, REPL, Error Formatting) — **DONE**
- Implemented bytecode compiler (`zen/bytecode.py`, `zen/compiler.py`) and stack-based bytecode VM (`zen/vm.py`), now the default runtime for `zen run`.
- Kept the tree-walking interpreter as a fallback via `zen run --tree-walk <file.zen>`.
- Added interactive REPL when running `zen` with no arguments (`zen` / `zen repl`), supporting multi-line blocks and error recovery.
- Enhanced `ZenSyntaxError` and `ZenRuntimeError` to print the offending source line and caret (`^`) alongside line/column.
- Added `examples/phase12_benchmark.zen` and passing tests in `tests/test_phase12.py`.

## Phase 13: Type checker — **DONE**
- Added optional type annotations (`let x: int`, `fn f(a: int) -> float`) in lexer/parser/AST and static semantic/type checker in `zen/typechecker.py`.
- Checks undefined names, argument counts, `const` reassignment, declared vs inferred types, return types, and operator mismatches (`"a" - 1`) with source-line + caret errors.
- Enabled in `zen run` by default with `--no-check` to skip; added `examples/phase13_types.zen` and `tests/test_phase13.py`.

## Phase 14: Save, load, and a real training run — **DONE**
- Added `save_weights`, `load_weights`, and `file_exists` built-ins in `zen/builtins.py` plus `save_checkpoint`/`load_checkpoint` on `AutoCharGPT` in `stdlib/ai/tensor_gpt.zen`.
- Optimized `zen/tensor.py` (`@lru_cache` strides/numel/broadcast shapes, strided 2D transpose, C-level `map(operator.mul, ...)` in `_matmul_2d` and `_elementwise`/`exp`/`tanh`), speeding up the tensor micro-benchmark from `0.326s` to `0.2429s` (`1.34x`).
- Ran `examples/phase14_train_gpt.zen` for 2000 steps (`d_model=16, block_size=12, batch_size=4`): `loss 3.6231 -> 1.9003` (`val_loss=2.0478`), `51.12 ms/step` (`102.23s` total), with checkpoints every 500 steps and resume verified.
- Added passing tests in `tests/test_phase14.py`.

## Phase 15: Developer tools — **DONE**
- Promoted `assert` to a first-class language keyword/statement (`assert cond, msg`) across lexer, parser, AST, compiler, VM, interpreter, and type checker.
- Implemented `zen/formatter.py` (`zen fmt file.zen` / `--check`), `zen check file.zen`, and `zen test` in `zen/cli.py`.
- Added 5 self-testing `.zen` suites in `tests_zen/` plus `examples/phase15_tools.zen` and `tests/test_phase15.py`.
