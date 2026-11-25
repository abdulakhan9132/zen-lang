"""Runnable Phase 1 check script.

Verifies that the Phase 1 spec, grammar, and target example exist and reports
honestly that the ZEN lexer, parser, and interpreter are not built yet.
"""

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from zen.spec import KEYWORDS, OPERATORS, PLANNED_BUILTINS, VERSION


def main() -> None:
    root = ROOT
    hello_path = root / "examples" / "hello.zen"
    spec_path = root / "docs" / "SPEC.md"
    grammar_path = root / "docs" / "GRAMMAR.md"

    print(f"ZEN version: {VERSION}")
    print(f"Spec present: {spec_path.exists()} ({spec_path.name})")
    print(f"Grammar present: {grammar_path.exists()} ({grammar_path.name})")
    print(f"Defined keywords: {len(KEYWORDS)}")
    print(f"Defined operators: {len(OPERATORS)}")
    print(f"Planned built-in phases: {sorted(PLANNED_BUILTINS.keys())}")
    print("\nTarget Phase 3 program (examples/hello.zen):")
    print(hello_path.read_text(encoding="utf-8").strip())
    print("\nNote: ZEN interpreter is NOT built yet (scheduled for Phase 3).")


if __name__ == "__main__":
    main()
