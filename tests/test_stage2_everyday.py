"""Stage 2 tests: Everyday language features (try/catch/raise, kwargs, in, if-expr, listcomp, methods, unpack, input)."""

from __future__ import annotations

from io import StringIO
from pathlib import Path

import pytest

from zen.errors import ZenRuntimeError, ZenTypeError
from zen.interpreter import Interpreter
from zen.typechecker import type_check_source
from zen.vm import VM

REPO = Path(__file__).resolve().parent.parent


def _run_both(code: str, stdin_text: str = "") -> tuple[str, str]:
    o1 = StringIO()
    Interpreter(stdout=o1, stdin=StringIO(stdin_text), base_dir=REPO).run(code, type_check=True)
    o2 = StringIO()
    VM(stdout=o2, stdin=StringIO(stdin_text), base_dir=REPO).run(code, type_check=True)
    return o1.getvalue().strip(), o2.getvalue().strip()


def test_try_catch_and_raise_and_div_zero() -> None:
    code = """
fn div_it(x) {
    return 10 / x
}
try {
    say div_it(0)
} catch e {
    say f"caught:{e.message}|line={e.line}"
}
try {
    raise "custom boom"
} catch err {
    say f"raised:{err.message}|line={err.line}"
}
"""
    o1, o2 = _run_both(code)
    assert o1 == o2
    lines = o1.splitlines()
    assert lines[0].startswith("caught:Division by zero|line=")
    assert "line=3" in lines[0]
    assert lines[1] == "raised:custom boom|line=11"


def test_keyword_arguments_and_errors() -> None:
    code = """
fn calc(a: int, b: int = 10, c: int = 100) -> int {
    return a + b * 2 + c * 3
}
say calc(1, b = 5)
say calc(a = 2, c = 4)
say calc(1, c = 2, b = 3)

class Adder {
    fn init(self, base: int, scale: int = 1) {
        self.base = base
        self.scale = scale
    }
    fn run(self, x: int, offset: int = 0) -> int {
        return (self.base + x) * self.scale + offset
    }
}
let ad = Adder(10, scale = 3)
say ad.run(2, offset = 5)
"""
    o1, o2 = _run_both(code)
    assert o1 == o2 == "311\n34\n13\n41"

    with pytest.raises(ZenTypeError, match=r"Did you mean 'scale'"):
        type_check_source("fn f(scale = 1) { return scale }\nf(scle = 2)\n")


def test_in_operator_if_expr_and_list_comprehension() -> None:
    code = """
let nums = [10, 20, 30]
let mp = {"k": 1, "m": 2}
say (20 in nums), (99 in nums), (99 not in nums)
say ("zen" in "hello zen!"), ("xyz" in "hello zen!")
say ("k" in mp), ("z" in mp)

let score = 85
let grade = if score >= 90 { "A" } elif score >= 80 { "B" } else { "C" }
say grade

let squares = [x * x for x in range(6) if x > 1]
say squares
"""
    o1, o2 = _run_both(code)
    assert o1 == o2 == "true false true\ntrue false\ntrue false\nB\n[4, 9, 16, 25]"


def test_string_and_list_methods_and_repetition() -> None:
    code = """
let s = "  Hello, ZEN World!  "
say s.strip().lower()
say s.strip().upper()
say s.strip().replace("ZEN", "AI")
say s.strip().startswith("Hello"), s.strip().endswith("!")
say s.strip().find("ZEN"), s.strip().find("missing")
let parts = "a,b,c".split(",")
say parts, "-".join(parts)
say "ab" * 3, 2 * "xy"

let xs = [3, 1, 2]
xs.append(4)
xs.sort()
say xs, xs.index(3)
let popped = xs.pop()
xs.reverse()
say popped, xs
"""
    o1, o2 = _run_both(code)
    assert o1 == o2
    lines = o1.splitlines()
    assert lines[0] == "hello, zen world!"
    assert lines[1] == "HELLO, ZEN WORLD!"
    assert lines[2] == "Hello, AI World!"
    assert lines[3] == "true true"
    assert lines[4] == "7 -1"
    assert lines[5] == "['a', 'b', 'c'] a-b-c"
    assert lines[6] == "ababab xyxy"
    assert lines[7] == "[1, 2, 3, 4] 2"
    assert lines[8] == "4 [3, 2, 1]"


def test_multiple_returns_destructuring_and_input() -> None:
    code = """
fn stats_pair(a, b) {
    return a + b, a * b
}
let s, p = stats_pair(3, 4)
say s, p
s, p = p, s
say s, p

let name = input("Name? ")
say f"hi {name}"
"""
    o1, o2 = _run_both(code, stdin_text="Ada\n")
    assert o1 == o2 == "7 12\n12 7\nName? hi Ada"
