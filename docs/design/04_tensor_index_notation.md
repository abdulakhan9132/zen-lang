# Design Memo 04: Tensor Index Notation (`einsum` / Named Axes)

> **Status:** Design Memo (not implemented in v0.3.0).

## 1. The Problem
Multi-head attention and batched linear algebra in ZEN currently require chaining `.reshape(...)`, `.transpose([0, 2, 1, 3])`, and `@` carefully by hand. Manual axis permutations are error-prone and obscure the mathematical equation being implemented.

## 2. Proposed Syntax (10 Lines of Examples)

```zen
import zennum as zn

// Batched multi-head attention scores in Einstein index notation
let q = zn.randn([2, 4, 8, 16])   // [B, H, T, D]
let k = zn.randn([2, 4, 8, 16])   // [B, H, S, D]

// Option A (expression form): contract shared index `d` across q and k
let scores = einstein("bhtd,bhsd->bhts", q, k)
// Option B (indexed tensor notation):
let scores2[b, h, t, s] = sum[d](q[b, h, t, d] * k[b, h, s, d])
```

## 3. Error Messages
- `E050 [line 8, col 14] Einsten index 'd' dimension mismatch: 'q' has size 16 on axis 3, 'k' has size 32 on axis 3.`
- `E051 [line 10, col 13] Unbound output index 'z' on LHS does not appear on RHS.`
- `E052 [line 8, col 14] Repeated index 'b' in 'bhtd' must refer to equal axis sizes (got 2 and 4).`

## 4. Edge Cases
- **Trace / diagonal (`"ii->i"` or `"ii->"`):** Output index repeated or contracted within a single tensor requires square axes on those positions.
- **Ellipsis (`"...ij,...jk->...ik"`):** Broadcast leading batch dimensions automatically just like `@`.
- **Autograd (`zennet.einsum`):** Backward pass of an `einsum` contraction is itself another `einsum` obtained by swapping the output label string with each input label string!
- **Performance:** Naive nested loops in pure Python are slow; lower pairwise contractions into `.transpose()` + `.reshape()` + `matmul_2d_buf()` whenever possible!

## 5. Minimal Prototype Plan
1. Build `zn.einsum(subscripts, *arrays)` first as a stdlib/core runtime function without touching parser grammar.
2. Compile `subscripts`: normalize dimensions, validate matching sizes, factor pairwise contractions into `transpose -> reshape -> matmul_2d_buf -> reshape` (with pure-Python loop fallback for >2-way or trace ops).
3. Add `nn.einsum(subscripts, *tensors)` in `stdlib/zennet/autograd.zen` where `backward()` generates the exact dual contraction strings for every tensor requiring gradients.
4. (Optional Phase 2 syntax): desugar `let c[i, j] = sum[k](a[i, k] * b[k, j])` in the parser directly into `einsum`!

## 6. Test Plan
- Verify `"ij,jk->ik"`, `"bhtd,bhsd->bhts"`, `"ij->ji"`, `"ii->"`, and `"...ij,...jk->...ik"` match explicit `transpose`/`@`/`sum` in `zennum`.
- Verify `nn.einsum` gradients numerically (central finite differences <= `1e-5`) for batched attention `q @ k.T` and outer products.

## 7. Honest Difficulty and Risks
- **Suboptimal contraction order:** For 3+ tensors (`"ij,jk,kl->il"`), greedy left-to-right can blow up memory; use cost estimation by smallest intermediate tensor size.
- **LHS syntax conflicts:** `let c[i, j] = ...` overlaps visually with index assignment `c[i, j] = ...`; prefer `zn.einsum` / `nn.einsum` first and keep indexed assignment unambiguous.

## 8. Effort Estimate
- **6–9 steps:** Subscript parser & validator (2), transpose+matmul lowering + fallback (3), `zennet` autograd `einsum` backward (2), optional syntax sugar + numerical grad tests (2).
