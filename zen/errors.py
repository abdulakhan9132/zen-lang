"""Custom error types, error codes, and helpful diagnostics for the ZEN language."""

from __future__ import annotations

from typing import Iterable


ERROR_CODES: dict[str, tuple[str, str, str]] = {
    "E001": (
        "Syntax error",
        "The lexer or parser encountered unexpected characters, an unclosed delimiter, or invalid syntax.",
        "Check matching quotes, braces `{ ... }`, parentheses `(...)`, and statement syntax.",
    ),
    "E002": (
        "Undefined variable or name",
        "You referenced a variable or function name that has not been declared in the current scope.",
        "Declare it with `let name = ...` or `fn name(...) { ... }`, or check spelling.",
    ),
    "E003": (
        "Unknown member, field, or method",
        "The module, class, instance, array, tensor, or value has no attribute with that name.",
        "Check the exported names or class methods and spelling.",
    ),
    "E004": (
        "Type mismatch",
        "An operation, assignment, annotation, or return value received a value of an incompatible type.",
        "Ensure operand and target types match, or convert explicitly with `int(...)`, `float(...)`, or `str(...)`.",
    ),
    "E005": (
        "Constant reassignment",
        "Variables declared with `const` cannot be reassigned after initialization.",
        "Use `let` instead of `const` if the variable needs to be updated.",
    ),
    "E006": (
        "Invalid function call arguments",
        "The function or method was called with the wrong number of arguments or an unknown keyword argument.",
        "Check the function's parameter list and default arguments.",
    ),
    "E007": (
        "Division or modulo by zero",
        "An arithmetic division (`/`, `div`, `floor_div`) or modulo (`%`) attempted to divide by zero.",
        "Guard the denominator (e.g. `if d == 0 { ... }`) before dividing.",
    ),
    "E008": (
        "Index or key out of range",
        "A list/string/array index was out of bounds, or a map lookup used a missing key.",
        "Check `len(...)` or `key in map` / `contains(map, key)` before indexing.",
    ),
    "E009": (
        "Import or file I/O error",
        "An imported module or data file could not be found or read.",
        "Use forward slashes `/` in paths and verify the file exists relative to the script or stdlib.",
    ),
    "E010": (
        "Maximum recursion depth exceeded",
        "A function called itself deeper than the configured recursion limit (`--max-depth N`).",
        "Check your base case, convert to a loop, or pass `--max-depth N` if deep recursion is intentional.",
    ),
    "E011": (
        "Invalid control-flow placement",
        "`return` was used outside a function, or `break`/`continue` was used outside a loop.",
        "Move `return` inside `fn` and `break`/`continue` inside `while` or `for`.",
    ),
    "E012": (
        "Runtime exception or assertion failure",
        "Program execution raised an error via `raise` or failed an `assert` check.",
        "Wrap fallible code in `try { ... } catch e { ... }` or fix the failing invariant.",
    ),
    "E013": (
        "Strict tokenizer or generator input error",
        "An empty prompt, unknown token character, or out-of-range token ID was passed with `strict = true`.",
        "Filter or validate prompt text/IDs before encoding or calling `nn.generate`.",
    ),
    "E014": (
        "NaN or Inf detected (--nan-trace)",
        "A numeric operation produced `NaN` or `Inf` while `--nan-trace` was active.",
        "Inspect the reported operation, tensor/array shapes, and learning rate or divisors at that line.",
    ),
}


def format_error_explanation(code: str | None = None) -> str:
    """Return human-readable help for an error code, or a summary table of all codes."""
    if code is None:
        lines = ["ZEN Error Codes:"]
        for k, (title, _, _) in sorted(ERROR_CODES.items()):
            lines.append(f"  {k}: {title}")
        lines.append("Run `zen explain E001` for details on a specific code.")
        return "\n".join(lines)
    norm = code.strip().upper()
    if norm not in ERROR_CODES:
        return f"Unknown error code '{code}'. Run `zen explain` to list known codes."
    title, desc, fix = ERROR_CODES[norm]
    return f"{norm}: {title}\n\nCause:\n  {desc}\n\nHow to fix:\n  {fix}"


def edit_distance(a: str, b: str) -> int:
    """Compute Levenshtein edit distance between two strings."""
    if a == b:
        return 0
    if not a:
        return len(b)
    if not b:
        return len(a)
    prev = list(range(len(b) + 1))
    for i, ca in enumerate(a, start=1):
        curr = [i]
        for j, cb in enumerate(b, start=1):
            cost = 0 if ca == cb else 1
            curr.append(min(curr[-1] + 1, prev[j] + 1, prev[j - 1] + cost))
        prev = curr
    return prev[-1]


def did_you_mean(name: str, candidates: Iterable[str]) -> str | None:
    """Return the closest candidate name within a small edit distance, or None."""
    if not name:
        return None
    max_dist = 1 if len(name) <= 3 else (2 if len(name) <= 7 else 3)
    best: str | None = None
    best_dist = max_dist + 1
    lower_name = name.lower()
    for cand in sorted(set(candidates)):
        if cand == name or cand.startswith("_"):
            continue
        d = edit_distance(lower_name, cand.lower())
        if d < best_dist:
            best_dist = d
            best = cand
    return best if best_dist <= max_dist else None


def format_source_snippet(source: str | None, line: int, column: int) -> str:
    """Render the offending source line with a caret pointing to the column."""
    if not source or line <= 0:
        return ""
    lines = source.replace("\r\n", "\n").replace("\r", "\n").splitlines()
    if line > len(lines):
        return ""
    text = lines[line - 1]
    gutter = f"{line:4d} | "
    col = max(1, column)
    pointer = " " * (len(gutter) + col - 1) + "^"
    return f"\n{gutter}{text}\n{pointer}"


class ZenError(Exception):
    """Base error for all ZEN compile-time and runtime errors."""

    default_code: str | None = None

    def __init__(
        self,
        message: str,
        line: int = 0,
        column: int = 0,
        source: str | None = None,
        code: str | None = None,
    ) -> None:
        self.message = message
        self.line = line
        self.column = column
        self.source = source
        self.code = code if code is not None else self.default_code
        location = f"[line {line}, col {column}] " if line > 0 else ""
        code_tag = f"[{self.code}] " if self.code else ""
        snippet = format_source_snippet(source, line, column)
        super().__init__(f"{location}{code_tag}{message}{snippet}")

    def with_source(self, source: str) -> "ZenError":
        if self.source:
            return self
        return type(self)(self.message, self.line, self.column, source, code=self.code)


class ZenSyntaxError(ZenError):
    """Raised by the lexer or parser when source code is malformed."""

    default_code = "E001"


class ZenTypeError(ZenError):
    """Raised by the semantic type checker before execution."""

    default_code = "E004"


class ZenRuntimeError(ZenError):
    """Raised by the interpreter or VM during program execution."""

    default_code = "E012"
