<p align="center">
  <img src="docs/zen-logo.png" alt="ZEN Logo" width="128" />
</p>

# ZEN

A minimal, readable programming language built from scratch for learning and building AI models in pure `.zen` code without external ML libraries.

**[Read the Beginner Guide: Learn ZEN (PDF)](docs/Learn-ZEN.pdf)** — or run `zen guide` after installing.

## Installation

```bash
pip install zen-lang
```

Or install from source:

```bash
git clone https://github.com/<your-username>/zen-lang.git
cd zen-lang
pip install -e .
```

## Hello World

```zen
// hello.zen
let language: string = "ZEN"
let version: float = 0.1

fn introduce(name: string, ver: float) {
    say "I am " + name
    say ver
}

introduce(language, version)
```

## AI Example (XOR Neural Network in `.zen`)

```zen
import "stdlib/ai/nn.zen" as nn

seed(42)
let xs = [[0.0, 0.0], [0.0, 1.0], [1.0, 0.0], [1.0, 1.0]]
let ys = [0.0, 1.0, 1.0, 0.0]
let model = nn.MLP([nn.Dense(2, 4, "tanh"), nn.Dense(4, 1, "sigmoid")])
let opt = nn.SGD(model.parameters(), 0.8)

for epoch in range(200) {
    let preds = []
    for x in xs { push(preds, model.forward(x)) }
    let loss = nn.mse_loss(preds, ys)
    opt.zero_grad()
    loss.backward()
    opt.step()
}
say "final XOR loss = " + str(round(model.forward([1.0, 0.0]).data, 4))
```

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

- **Slow pure-Python tensors:** Matrix operations run on pure Python lists without C extensions, SIMD, or GPU acceleration (`~51 ms/step` on a `d_model=16`, `batch_size=4` character model).
- **Tiny GPT only:** The included character-level model (`AutoCharGPT`) is intentionally small for low-RAM machines and learns character/syllable patterns rather than fluent language.
- **No native compiler yet:** Execution uses a Python bytecode VM (no LLVM, C codegen, package manager, or LSP yet).

## License

MIT License — Copyright (c) 2026 Abdullah Khan Suri. See [LICENSE](LICENSE).
