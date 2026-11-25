<p align="center">
  <img src="docs/zen-logo.png" alt="ZEN Logo" width="128" />
</p>

# ZEN (v0.2.1)

**ZEN is the language. ZenNum handles numbers. ZenNet builds neural networks.**

A minimal, readable programming language built from scratch for learning coding and building real AI models (from linear regression to Transformer GPTs) in pure `.zen` code — with zero external AI libraries (`numpy`/`torch` never used).

**[Beginner Guide: Learn ZEN (PDF)](docs/Learn-ZEN.pdf)** · **[ZenNum Reference](docs/ZENNUM.md)** · **[ZenNet Reference](docs/ZENNET.md)** — or run `zen guide` after installing.

## Installation

```bash
pip install zen-lang
```

Or install from source:

```bash
git clone https://github.com/abdulakhan9132/zen-lang.git
cd zen-lang
pip install -e .
```

## ZenNum Quickstart (`import zennum as zn`)

```zen
import zennum as zn

let a = zn.arange(1, 7).reshape([2, 3])
let b = zn.ones([1, 3]) * 10
say a + b            // Broadcast addition
say a[0:2, 1:3]      // Multi-dimensional slicing
say a[a > 3]         // Boolean masking
```

## Neural Network in 15 Lines (`examples/mlp_xor.zen`)

```zen
import zennum as zn
import zennet as nn

zn.seed(42)
let X = [[0.0, 0.0], [0.0, 1.0], [1.0, 0.0], [1.0, 1.0]]
let Y = [[0.0], [1.0], [1.0], [0.0]]

let model = nn.Sequential([nn.Linear(2, 8), nn.Tanh(), nn.Linear(8, 1), nn.Sigmoid()])
let opt = nn.Adam(model.parameters(), 0.08)
for step in range(150) {
    opt.zero_grad()
    let loss = nn.mse(model(X), Y)
    loss.backward()
    opt.step()
}
say "XOR predictions: " + str(map(model(X).data.flatten().tolist(), fn(v) { return round(v, 3) }))
```

## GPT Language Model in ~40 Lines (`examples/gpt_in_40_lines.zen`)

Train a multi-head causal Transformer GPT from scratch using `nn.CharTokenizer`, `nn.GPT`, `nn.AdamW`, `nn.cross_entropy`, and `nn.generate` — see [`examples/gpt_in_40_lines.zen`](examples/gpt_in_40_lines.zen).

## CLI Commands

| Command | Description |
|---|---|
| `zen` | Start the interactive REPL |
| `zen run file.zen` | Type-check and execute a `.zen` file on the bytecode VM (`--no-check`, `--tree-walk`) |
| `zen check file.zen` | Run the static type checker without executing |
| `zen fmt file.zen` | Format a `.zen` file in place (`--check` to verify) |
| `zen test` | Run all `.zen` test files in `tests_zen/` |
| `zen guide` | Open the `Learn-ZEN.pdf` beginner guide (`--path` prints its location) |

## Current Limitations

- **CPU-only execution:** Runs optional C native kernels (`zen/native/zen_kernels.c`) when compiled, falling back automatically to pure-Python flat buffers when no C compiler is present (`~0.63s` for `256x256` matmul; `~355 ms/step` on a 2-layer `d_model=16` GPT).
- **Small datasets & models:** Examples target low-RAM laptops (`examples/data/corpus.txt`) rather than GPU-scale training.
- **Bytecode VM only:** No LLVM/native AOT compiler, external package registry, or LSP server yet.

## License

MIT License — Copyright (c) 2026 Abdullah Khan Suri. See [LICENSE](LICENSE).
