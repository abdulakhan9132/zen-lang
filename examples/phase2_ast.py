"""Runnable Phase 2 example: tokenize and parse examples/hello.zen into an AST."""

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from zen.lexer import tokenize
from zen.parser import parse


def main() -> None:
    source = (ROOT / "examples" / "hello.zen").read_text(encoding="utf-8")
    tokens = tokenize(source)
    program = parse(source)
    print(f"Tokens ({len(tokens)}):")
    for tok in tokens:
        print(f"  line {tok.line}, col {tok.column}: {tok.type.name} ({tok.value!r})")
    print(f"\nParsed AST statements ({len(program.statements)}):")
    for stmt in program.statements:
        print(f"  {stmt}")


if __name__ == "__main__":
    main()
