# ZEN v0.3 Language Reference

Every core and v0.3 feature in ZEN, organized one section per feature with a small runnable example. Run any snippet with `zen run file.zen` or try lines in the REPL (`zen`).

---

## 1. Variables, Constants & Output

Use `let` for mutable bindings, `const` for immutable constants (`E005` on reassignment), and `say` (or `print`) for output.

```zen
const PI = 3.14159
let radius = 2.0
let area = PI * radius ** 2
say area
```

---

## 2. Numeric Types, `div`, and `floor_div`

`/` always returns true floating-point division (`float`), while `div` (and `floor_div(a, b)`) performs integer floor division. Division by zero raises `E007`. Note: `//` starts a comment in ZEN (`zen check` warns on trailing `// <number|var>` to catch C/Python floor-division accidents).

```zen
let a = 7
let b = 2
say a / b            // 3.5
say a div b          // 3
say floor_div(9, 4)  // 2
```

---

## 3. String Interpolation (`f"..."`) and String Operations

Prefix a string with `f` to interpolate expressions inside `{...}`. Plain strings keep `{...}` literal (`zen check` warns when `{var_in_scope}` appears in a non-`f` string). Repeat strings with `*` and check membership with `in` / `not in`.

```zen
let name = "ZEN"
let ver = 3
say f"{name} v0.{ver}"
say "literal {name}"
say "ha" * 3
say "zen" in "hello zen"
```

---

## 4. Optional Types (`T?`), `any`, and Flow Narrowing

Add `?` to any type (`int?`, `float?`, `string?`, `bool?`, `list?`, `map?`) to allow `none`, or use `any` for dynamic values. `zen check` narrows types inside `if x != none { ... }`, `if x == none { ... }`, and `if type(x) == "..." { ... }`. Unannotated `let x = none` widens cleanly on reassignment.

```zen
fn add_bonus(x: int?, extra: any) -> int {
    let total = 0
    if x != none {
        total = x + 10
    }
    if type(extra) == "list" {
        total = total + len(extra)
    }
    return total
}

say add_bonus(5, [1, 2])
say add_bonus(none, "ignored")
```

---

## 5. Control Flow & `if`-Expressions

Use `if` / `else if` / `else`, `while`, `for x in seq`, `break`, and `continue`. `if cond { a } else { b }` can also be used as an expression returning a value.

```zen
let score = 88
let grade = if score >= 90 { "A" } else { "B" }
say grade

let sum = 0
for i in range(1, 5) {
    sum = sum + i
}
say sum
```

---

## 6. Lists, Maps, Membership (`in`), and List Comprehensions

Build lists `[...]` and maps `{k: v}`, test membership with `in` / `not in`, and transform or filter lists concisely with `[expr for x in seq]` or `[expr for x in seq if cond]`.

```zen
let nums = [x * x for x in range(6) if x > 1]
say nums
say 9 in nums
say 5 not in nums

let cfg = {"epochs": 10, "lr": 0.01}
say "lr" in cfg
```

---

## 7. String & List Methods

Strings support `.split(sep)`, `.join(items)`, `.upper()`, `.lower()`, `.strip()`, `.replace(old, new)`, `.startswith(prefix)`, `.endswith(suffix)`, and `.find(sub)`. Lists support `.append(v)`, `.pop(idx)`, `.sort()`, `.reverse()`, and `.index(v)`.

```zen
let clean = "  hello,zen  ".strip().upper()
say clean.split(",")
say "-".join(["a", "b", "c"])

let arr = [3, 1, 2]
arr.append(4)
arr.sort()
arr.reverse()
say arr
```

---

## 8. Functions, Defaults, Keyword Arguments & Multiple Returns

Functions support default parameter values (`b: int = 5`), named keyword arguments (`pair(3, b = 4)`), multiple return values (`return a, b`), tuple-style destructuring (`let s, p = pair(...)`), and multi-assignment swaps (`a, b = b, a`).

```zen
fn stats_pair(a: int, b: int = 5) {
    return a + b, a * b
}

let s, p = stats_pair(3, b = 4)
s, p = p, s
say f"p={s}, s={p}"
```

---

## 9. Errors & Exceptions (`try` / `catch` / `raise`)

Raise custom errors with `raise expr` and handle runtime failures (`E001`–`E014`) with `try { ... } catch e { ... }`. The caught error object `e` exposes `e.message`, `e.code`, and `e.line`.

```zen
try {
    let x = 10 / 0
} catch e {
    say f"[{e.code}] line {e.line}: {e.message}"
}

try {
    raise "invalid state"
} catch err {
    say err.message
}
```

---

## 10. Classes & Single Inheritance

Define classes with `class Name`, `fn init(self, ...)`, methods, and single inheritance `class Child(Parent)`.

```zen
class Animal {
    fn init(self, name: string) {
        self.name = name
    }
    fn speak(self) -> string {
        return self.name + " makes a sound"
    }
}

class Dog(Animal) {
    fn speak(self) -> string {
        return self.name + " barks"
    }
}

let d = Dog("Rex")
say d.speak()
```

---

## 11. Standard Library Modules

Import built-in modules directly from anywhere without extra dependencies:
- `json`: `parse`, `stringify`
- `files`: `read`, `write`, `append`, `exists`, `is_file`, `is_dir`, `mkdir`, `remove`, `list_dir`, `copy`
- `path`: `join`, `basename`, `dirname`, `stem`, `ext`, `resolve`, `cwd`, `sep`
- `regex`: `match`, `search`, `find_all`, `replace`, `split`
- `datetime`: `now`, `today`, `timestamp`, `format`, `parse`, `iso`
- `env`: `argv`, `get`, `set`, `has`, `all`, `platform`, `os_name`
- `csv`: `parse`, `stringify`, `read`, `write`
- `process`: `run(cmd, cwd)` returning `{"code", "stdout", "stderr"}`
- `http`: `get`, `post` returning `{"status", "body", "headers"}`
- `stats` (alias `statistics`): `mean`, `median`, `std`, `var`, `min`, `max`, `sum`

```zen
import json
import stats
import regex

let data = json.parse("{\"scores\": [80, 90, 100]}")
say f"mean={stats.mean(data["scores"])}, cleaned={regex.replace("\\s+", "_", "zen std lib")}"
```

---

## 12. Scientific Tooling, Doctests & CLI Diagnostics

ZEN v0.3 includes built-in reliability and debugging tools:
- `zen explain [E001]` — Explain error codes (`E001`–`E014`) and fixes.
- `zen run --nan-trace file.zen` — Stop at the first `NaN`/`Inf` with operation, shapes, and ZEN line (`E014`).
- `zen run --record run.json file.zen` and `zen replay run.json` — Record seed, file SHA-256 hashes, and stdout hash, then verify `MATCH` / `MISMATCH`.
- `zen test --doc` — Execute embedded `//>` doctest assertions and check expected `//` output lines.
- `zen run --max-depth N file.zen` — Configure recursion depth (default `10000`, safe `E010` error without raw Python traceback unless `--debug` is passed).
- `seed(s)` / `zn.seed(s)` — Process-wide shared RNG across built-ins, `zennum`, and `zennet`.

```zen
//> say 6 div 2
// 3
import zennum as zn
zn.seed(42)
say zn.rand([2]).tolist()
```
