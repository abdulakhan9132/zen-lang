# Design Memo 07: ZEN 1.0 Language Freeze Checklist

> **Status:** Design Memo (planning baseline for v1.0).

## 1. The Problem
Before declaring ZEN `1.0.0`, syntax, operator precedence, standard library APIs, bytecode/error semantics, and packaging contracts must be frozen so user code written for `1.0` keeps running across all `1.x` releases without breaking changes.

## 2. Proposed Freeze Policy (10 Lines of Examples)

```zen
// Core syntax frozen at 1.0:
let x: int = 7 div 2                 // immutable `const`, mutable `let`, `div` floor division
let tag: string? = none              // optional type `T?` + `if tag != none` narrowing
fn greet(name: string, n: int = 1) -> string {
    return f"hi {name}" * n
}
try {
    let a, b = [1, 2]                // destructuring & multiple returns
} catch e {
    say f"err {e.code}: {e.message}"
}
```

## 3. Error Messages (Stability Contract)
- Freeze error code numbering `E001`–`E014` (`zen explain E001`..`E014`) so tooling, IDE extensions, and tests can rely on stable error codes across `1.x`.
- Freeze `ZenSyntaxError` / `ZenTypeError` / `ZenRuntimeError` snippet formatting (`[line L, col C] [E###] ...`) and `Error` catch fields (`e.message`, `e.line`, `e.code`).
- Guarantee internal interpreter/VM bugs never emit raw Python tracebacks unless `--debug` is explicitly supplied.

## 4. Edge Cases & Checklist Items
1. **Lexical & Grammar Freeze:**
   - Reserved keywords (`let`, `const`, `say`, `assert`, `fn`, `class`, `return`, `if`, `elif`, `else`, `while`, `for`, `in`, `break`, `continue`, `and`, `or`, `not`, `true`, `false`, `none`, `import`, `as`, `div`, `try`, `catch`, `raise`) — contextual keywords (`model`, `train`, `unit`) kept non-breaking.
   - Line endings (CRLF/LF normalized everywhere), UTF-8 I/O, `/` path normalization.
2. **Runtime Parity:**
   - Differential test suite verifying `VM` (`zen run`) and `Interpreter` (`--tree-walk`) produce identical stdout, errors, and RNG outputs on every feature.
3. **Standard Library Freeze:**
   - Freeze module names and signatures (`zennum`, `zennet`, `json`, `files`, `path`, `regex`, `datetime`, `env`, `csv`, `process`, `http`, `stats`) and deprecate legacy `stdlib/ai/*` cleanly with migration notes.
4. **Tooling & Reproducibility:**
   - Freeze `zen` CLI subcommands (`run`, `repl`, `check`, `fmt`, `test`, `test --doc`, `replay`, `explain`, `doctor`, `guide`, `docs`), `--record` manifest schema version, and `.tmLanguage.json` grammar.
5. **Cross-Platform CI Matrix:**
   - Green CI on Ubuntu, macOS, and Windows across supported Python versions (`>=3.10`), both with C kernels and `ZEN_NO_NATIVE=1`.

## 5. Minimal Prototype Plan
1. Add explicit `"schema_version": 1` to `--record` manifests and `stdlib` deprecation warnings where needed.
2. Write a frozen "1.0 conformance suite" inside `tests_zen/` + `tests/test_conformance_v1.py` covering every syntax form, operator precedence edge case, and stdlib contract.
3. Run differential VM-vs-Interpreter checks across all conformance tests in CI before bumping to `1.0.0`.

## 6. Test Plan
- Every item on the checklist maps to at least one automated check in CI: CLI help/flags stability test, error code completeness test (`E001`–`E014` all documented by `zen explain`), VM==Interpreter parity test, `zen test --doc`, and clean-venv install test on all 3 OSes.

## 7. Honest Difficulty and Risks
- **Premature freeze:** Any feature shipped without clear semantics (e.g. `any` vs `T?` interaction, string `\"` escapes inside f-string braces) becomes permanent technical debt after `1.0`.
- **Two execution engines:** Keeping both the bytecode VM and tree-walk interpreter in sync doubles bug surface; consider declaring bytecode VM canonical and tree-walker a reference fallback.

## 8. Effort Estimate
- **5–7 steps:** Conformance test matrix (2), CLI/manifest version lock & `zen explain` audit (1), deprecate `stdlib/ai` notes (1), full 3-OS CI & native/pure-Python verification (2).
