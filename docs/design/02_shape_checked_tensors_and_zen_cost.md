# Design Memo 02: Shape-Checked Tensors & Dry-Run `zen cost` Mode

> **Status:** Design Memo (not implemented in v0.3.0).

## 1. The Problem
Shape bugs (`[32, 16] @ [32, 64]`) in neural networks are currently caught only at runtime inside `forward()` or `backward()`, sometimes deep into a long training run. Users on low-RAM CPUs also have no way to see parameter counts, FLOPs, or peak activation memory before launching training.

## 2. Proposed Syntax (10 Lines of Examples)

```zen
// Optional symbolic dimension annotations on Tensor / Array parameters
fn attention(q: Tensor[B, H, T, D], k: Tensor[B, H, T, D], v: Tensor[B, H, T, D]) -> Tensor[B, H, T, D] {
    let scores = (q @ k.transpose([0, 1, 3, 2])) * (1.0 / sqrt(D))
    return scores.softmax(-1) @ v
}

let m = nn.GPT({"vocab_size": 42, "n_layer": 2, "n_head": 2, "d_model": 16, "block_size": 16})
// Dry-run CLI:
// $ zen cost examples/gpt_in_40_lines.zen
// Layer           Output Shape    Params   FLOPs/step   Activation Mem
// GPT.total       [4, 16, 42]     8,010    ~245,760     ~0.42 MB
```

## 3. Error Messages
- `E030 [line 4, col 21] Shape mismatch in '@': left Tensor[B, H, T, D] and right Tensor[B, H, T, D] cannot contract K=D with M=T (transpose right last two dims?).`
- `E031 [line 8, col 10] Return shape mismatch: declared Tensor[B, H, T, D], inferred Tensor[B, H, T, T].`
- `E032 [zen cost] Dynamic runtime shape in loop at line 19; pass '--batch 4 --seq 16' or symbolic constants to estimate cost.`

## 4. Edge Cases
- **Dynamic batch/sequence dimensions:** Dimension symbols (`B`, `T`) remain symbolic; unify equal names (`T == T`) and flag mismatches (`D != T`) unless resolved via `any`/`...`.
- **Broadcasting & reductions:** `sum(axis=1, keepdims=false)` drops an axis statically; `concat([A, B], axis)` sums dimension `axis` (`D1 + D2`).
- **Data-dependent control flow:** `if runtime_flag` branches returning different shapes join to `Tensor[...]` (dynamic shape) with a warning in `zen cost`.
- **Backward pass cost:** Training FLOPs ≈ 3x forward FLOPs (1x forward + 2x backward for matmul gradients); `zen cost` should label forward vs forward+backward clearly.

## 5. Minimal Prototype Plan
1. Parse `Tensor[...]` / `Array[...]` shape brackets in `Parser._parse_type_name` as `ShapeType(dims: tuple[int | str, ...])`.
2. Add symbolic dimension unification in `TypeChecker` for `@`, `+ - * /` broadcasting, `reshape`, `transpose`, `concat`, `stack`, and `sum(axis)`.
3. Implement `zen cost file.zen`: construct symbolic/shape-only `MetaArray`/`MetaTensor` stubs (zero numeric buffer allocation!) that run the AST/VM shape-transfer pass once, summing parameter counts, exact MAC/FLOPs per op, and live tensor byte footprints (`numel * 8` bytes for `float64`).

## 6. Test Plan
- Static shape tests checking valid Transformer attention compiles and mismatched `@` / `concat` raises `E030` in `zen check` without executing.
- `zen cost` tests verifying exact parameter count (`Linear(4, 3)` -> `4*3 + 3 = 15` params) and exact matmul FLOPs (`2 * M * K * N`) on `mlp_xor.zen` and `gpt_in_40_lines.zen`, completing in < 1 second without running training.

## 7. Honest Difficulty and Risks
- **Arithmetic on symbolic dims:** Expressions like `reshape([B, T, H, D_model / H])` need simple symbolic algebra (`D_model = H * D`) or else fall back to `?`.
- **Unannotated dynamic code:** Legacy `.zen` code without `Tensor[...]` annotations must keep checking cleanly with zero breakage.

## 8. Effort Estimate
- **8–10 steps:** `ShapeType` parser & AST (2), symbolic dim unifier & op transfer rules (3), zero-alloc `MetaTensor`/`MetaArray` cost tracer (3), `zen cost` CLI formatter + tests (2).
