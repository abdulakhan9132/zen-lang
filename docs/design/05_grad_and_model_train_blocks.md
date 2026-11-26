# Design Memo 05: `grad(f)` for Plain Functions & `model`/`train` Blocks

> **Status:** Design Memo (not implemented in v0.3.0).

## 1. The Problem
Today ZenNet uses PyTorch-style tape autograd (`Tensor`, `.backward()`, `.grad`, `Module`, `Trainer`). While familiar, it requires manual `opt.zero_grad()`, `loss.backward()`, `opt.step()`, and lacks JAX-style `grad(f)` on plain mathematical functions as well as the declarative `model` and `train` blocks envisioned in ZEN's original language spec.

## 2. Proposed Syntax (10 Lines of Examples)

```zen
// 1) Functional differentiation: `grad(f)` returns a gradient function
fn loss_fn(w, x, y) { return ((w * x) - y) ** 2 }
let df = grad(loss_fn)       // differentiates w.r.t. arg 0 (w)
say df(2.0, 3.0, 10.0)       // -24.0

// 2) Declarative `model` and `train` blocks (sugar over nn.Module & nn.Trainer)
model TinyMLP(in_dim: int, out_dim: int) {
    layer l1 = nn.Linear(in_dim, 8)
    layer act = nn.Tanh()
    layer l2 = nn.Linear(8, out_dim)
    forward(x) { return l2(act(l1(x))) }
}
train TinyMLP(2, 1) on (X, Y) with nn.Adam(lr = 0.08), nn.mse for 150 steps
```

## 3. Error Messages
- `E060 [line 3, col 10] Cannot differentiate 'loss_fn': argument 0 is non-numeric 'string' during trace.`
- `E061 [line 3, col 10] No gradient path from return value of 'f' to parameter 0 (did you convert through int() or lose Tensor wrapping?).`
- `E062 [line 12, col 1] 'model' body may only contain 'layer', 'param', and 'forward' declarations.`

## 4. Edge Cases
- **Plain floats/arrays in `grad(f)`:** Wrap selected argument(s) `argnums` in `nn.tensor(..., true)` transparently, run `f(...)`, require scalar output (`size == 1`), call `.backward()`, and return plain `float` or `ZenArray` matching input type!
- **Higher-order `grad(grad(f))`:** Tape-based `zennet.Tensor` today stores numeric `ZenArray` values rather than `Tensor` nodes inside `backward()`, so second derivatives would either need dual-number forward-AD or retaining grad-graph tensors. Document `v1` as first-order `grad(f)` with explicit error on nested `grad(grad(...))`.
- **`model` inheritance & methods:** Desugar `model Foo(...) { ... }` cleanly in `Parser`/`AST` lowering into a `class Foo(nn.Module)` with auto-generated `init` and `forward`!

## 5. Minimal Prototype Plan
1. Implement `nn.grad(fn_val, argnum = 0)` and `nn.value_and_grad(fn_val, argnum = 0)` purely in `stdlib/zennet/autograd.zen` by wrapping the `argnum`-th argument as a gradient-tracked `Tensor`, calling `fn_val(...)`, checking scalar output, calling `.backward()`, and unwrapping `.grad`.
2. Add `model` block parser desugaring to `ast.ClassDecl` subclassing `nn.Module` (auto-wiring `nn.Module.init(self)` and `self.<layer> = ...` assignments).
3. Add `train <model_expr> on <data> with <opt>, <loss> for <steps>` statement desugaring directly into `nn.Trainer(...).fit(...)`.

## 6. Test Plan
- Verify `nn.grad` on polynomial `3*x**2 + 2*x + 1` and vector dot product matches analytical & finite-difference gradients.
- Verify `model` and `train` blocks solve XOR in <= 15 lines and match manual `class ... (nn.Module)` + `Trainer` identically across VM and Interpreter.

## 7. Honest Difficulty and Risks
- **Non-differentiable control flow:** Branching inside `f` works for the taken path, while casting `int(x)` breaks gradients silently unless warned when `t.grad == none`.
- **Syntax footprint:** Adding new top-level keywords `model` / `train` could collide with existing user variable names (check repo first: e.g., `let model = nn.GPT(...)` is used in EVERY AI example!). Therefore `model` must be contextual (`model Name(...) {`) rather than a globally reserved keyword!

## 8. Effort Estimate
- **7–9 steps:** `nn.grad` / `nn.value_and_grad` in `zennet` (2), contextual `model` block desugaring (3), `train` statement desugaring (2), tests & docs (2).
