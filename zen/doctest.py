"""Comment-based doctest runner (`//> ...` followed by `// ...`) for ZEN."""

from __future__ import annotations

from dataclasses import dataclass
import io
from pathlib import Path

from zen.errors import ZenError
from zen.vm import VM


@dataclass
class DocTestCase:
    file_path: Path
    line: int
    code: str
    expected: str


def extract_doctests(path: Path) -> list[DocTestCase]:
    """Extract `//> code` blocks followed by `// expected` comment lines."""
    lines = path.read_text(encoding="utf-8").splitlines()
    cases: list[DocTestCase] = []
    i = 0
    n = len(lines)
    while i < n:
        stripped = lines[i].lstrip()
        if stripped.startswith("//>"):
            start_line = i + 1
            code_lines: list[str] = []
            while i < n and lines[i].lstrip().startswith("//>"):
                part = lines[i].lstrip()[3:]
                if part.startswith(" "):
                    part = part[1:]
                code_lines.append(part)
                i += 1
            exp_lines: list[str] = []
            while i < n:
                s2 = lines[i].lstrip()
                if s2.startswith("//") and not s2.startswith("//>"):
                    exp = s2[2:]
                    if exp.startswith(" "):
                        exp = exp[1:]
                    exp_lines.append(exp)
                    i += 1
                else:
                    break
            cases.append(DocTestCase(path, start_line, "\n".join(code_lines), "\n".join(exp_lines)))
        else:
            i += 1
    return cases


def run_doctests(paths: list[Path]) -> tuple[int, int, list[str]]:
    """Run doctests across the given .zen files. Returns (passed, failed, messages)."""
    passed = 0
    failed = 0
    messages: list[str] = []
    for p in paths:
        cases = extract_doctests(p)
        if not cases:
            continue
        buf = io.StringIO()
        vm = VM(stdout=buf, base_dir=p.resolve().parent)
        for case in cases:
            buf.seek(0)
            buf.truncate(0)
            try:
                vm.run(case.code, file_path=p, type_check=True)
                actual = buf.getvalue().rstrip("\n")
                if actual == case.expected:
                    passed += 1
                    messages.append(f"PASS {p}:{case.line}")
                else:
                    failed += 1
                    messages.append(
                        f"FAIL {p}:{case.line}\n  expected: {case.expected!r}\n  actual:   {actual!r}"
                    )
            except ZenError as exc:
                failed += 1
                messages.append(f"FAIL {p}:{case.line}: {exc}")
    return passed, failed, messages
