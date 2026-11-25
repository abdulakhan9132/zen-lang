# ZEN Language Specification (v0.1 — Phase 1 Draft)

ZEN is a clean, brace-delimited, dynamically typed language designed for building and training AI models from scratch.

> **Status:** This specification defines the target language design. None of these features are executable yet until Phases 2–5 are implemented.

---

## 1. Comments
Text after `//` on a line is ignored by ZEN and used for human notes.

```zen
// This is a comment
say "I am ZEN"
```

## 2. Values and Types
ZEN supports numbers, strings, booleans, `none`, lists, dictionaries, functions, classes, and (in Phase 5) tensors.

```zen
let count = 42          // integer
let lr = 0.01           // float
let name = "ZEN"        // string
let ready = true        // boolean (true / false)
let empty = none        // null / absence of a value
let nums = [1, 2, 3]    // list
let cfg = {"dim": 16}   // dictionary
```

## 3. Variables, Constants, and Output
`let` creates a mutable variable, `const` creates a read-only constant, and `=` updates an existing variable, list item, or map entry.
`say` outputs a value to the console (`print(...)` is kept as a function alias).

```zen
let x = 10
const pi = 3.14159
x = x + 5
nums[0] = 99
say x
print(x)
```

## 4. Operators and Precedence
From lowest precedence to highest:

| Precedence | Operators | Meaning |
|---|---|---|
| 1 | `or` | Logical OR |
| 2 | `and` | Logical AND |
| 3 | `==`, `!=` | Equal / Not equal |
| 4 | `<`, `<=`, `>`, `>=` | Comparisons |
| 5 | `+`, `-` | Add / Subtract |
| 6 | `*`, `/`, `%`, `@` | Multiply, Divide, Modulo, Matrix Multiply (Phase 5) |
| 7 | `**` | Power (right-associative) |
| 8 | `-x`, `not x` | Unary negation / Logical NOT |
| 9 | `f(x)`, `a[i]`, `obj.field` | Function call, Indexing, Member access |

```zen
let y = 2 * 3 + 1       // 7
let sq = 2 ** 3         // 8
```

## 5. Control Flow
Blocks are enclosed in `{ ... }` (no indentation sensitivity).

### Conditionals (`if` / `elif` / `else`)
Runs a block only when its condition is true.

```zen
if score > 90 {
    say "great"
} elif score > 50 {
    say "ok"
} else {
    say "try again"
}
```

### Loops (`while` and `for`)
`while` repeats while a condition holds; `for` iterates over each item in a list or `range(...)`.

```zen
let i = 0
while i < 3 {
    say i
    i = i + 1
}

for k in range(3) {
    say k
}
```

`break` exits the innermost loop; `continue` skips to the next iteration.

## 6. Functions (`fn`)
Functions group reusable steps under a name and optionally return a value with `return`.

```zen
fn add(a, b) {
    return a + b
}

say add(2, 3)
```

## 7. Classes (`class`)
Classes bundle data and methods together (used for autograd `Value`, layers, and models in Phases 6–8).

```zen
class Value {
    fn init(self, data) {
        self.data = data
        self.grad = 0.0
    }

    fn bump(self, amount) {
        self.data = self.data + amount
    }
}

let v = Value(2.0)
v.bump(1.0)
```

## 8. Modules (`import`)
Loads definitions from another `.zen` file under a short alias.

```zen
import "autograd.zen" as ag
```

## 9. Planned Built-in Functions
- **Phase 3:** `print` (alias for `say`), `len`, `str`, `type`
- **Phase 4:** `range`, `push`, `pop`, `map`, `filter`, `abs`, `sqrt`, `exp`, `log`, `sin`, `cos`, `pow`, `seed`, `random`, `min`, `max`, `sum`, `int`, `float`, `split`, `join`, `trim`, `upper`, `lower`, `contains`, `replace`, `chars`, `ord`, `chr`, `read_file`, `write_file`
- **Phase 5:** `tensor`, `zeros`, `ones`, `tensor_rand`, `randn`, `shape`, `reshape`, `transpose`, `matmul`, `mean`, `tensor_sum`, `tensor_mean`, `tensor_max`
