# Design Memo 03: Units of Measure

> **Status:** Design Memo (not implemented in v0.3.0).

## 1. The Problem
Scientific and robotics code frequently mixes physical quantities stored as plain `float`s (`meters` vs `kilometers`, `seconds` vs `milliseconds`, `radians` vs `degrees`), leading to silent unit mismatches that type checkers treating everything as `float` cannot detect.

## 2. Proposed Syntax (10 Lines of Examples)

```zen
unit m
unit s
unit kg
unit N = kg * m / s**2

let dist: float<m> = 100.0<m>
let time: float<s> = 9.58<s>
let speed: float<m / s> = dist / time
let force: float<N> = 70.0<kg> * (9.81<m / s**2>)
// Compile error: let bad = dist + time   -> E040: Cannot add 'float<m>' and 'float<s>'
```

## 3. Error Messages
- `E040 [line 9, col 11] Unit mismatch in '+': cannot add 'float<m>' and 'float<s>'.`
- `E041 [line 10, col 28] Return unit mismatch: expected 'float<m / s>', got 'float<m>'.`
- `E042 [line 11, col 15] Unit 'km' is not declared (declare with 'unit km' or 'unit km = 1000 * m').`

## 4. Edge Cases
- **Unit erasure at runtime:** Units are erased after `TypeChecker` passes (`float<m>` executes identically to `float` in VM/C with zero runtime overhead!), except scaled aliases (`km = 1000 * m`) which either forbid implicit cross-addition or require explicit conversion `to_m(x)` so numeric scaling is never silently forgotten.
- **Dimensionless vs unitful:** `x / x` produces dimensionless `float<1>` (`float`); `sin(theta)` requires `float<rad>` or dimensionless `float`.
- **Zero literals & polymorphism:** `0.0` or generic `fn norm(x: list<float<U>>) -> float<U>` needs unit polymorphism or explicit `0.0<m>`.
- **Arrays with units:** `Array<float<m>>` allows unit-safe physics simulations without per-element boxing.

## 5. Minimal Prototype Plan
1. Add `unit <name> [= <unit_expr>]` declaration and `<unit_expr>` suffix on numeric literals/types in `Lexer`/`Parser`.
2. Represent units canonically in `TypeChecker` as `dict[str, int]` exponent maps (`{"m": 1, "s": -1}`).
3. Propagate exponents through `*` (add), `/` (subtract), `** int` (multiply), and require exact exponent map equality for `+`, `-`, `==`, `<`, assignment, and return types; strip units during bytecode compilation.

## 6. Test Plan
- Positive tests: velocity, kinetic energy `0.5 * m * v**2` matching `J = kg * m**2 / s**2`, and zero runtime overhead verification.
- Negative tests: adding `m + s`, passing `s` where `m` is expected, non-integer exponent on unitful value (`x ** 0.5` only valid when all exponents are even -> `m**2 -> m`).

## 7. Honest Difficulty and Risks
- **Conversion factors vs pure dimensions:** Supporting `km = 1000 * m` silently inserts scale factors at runtime, violating zero-cost erasure; keep v1 conversions explicit (`km_to_m`) and units purely dimensional.
- **Syntax ambiguity with `<` comparison:** `a < b` vs `float<m>` — only parse `<...>` as a unit suffix when immediately attached to a type name or numeric literal with no intervening whitespace (`10.0<m>`).

## 8. Effort Estimate
- **6–8 steps:** Parser unit decl & `<unit>` suffix (2), `UnitMap` normalization & arithmetic rules in `TypeChecker` (3), polymorphism/generics helper + tests (2).
