"""Tests for Phase 1: Spec, grammar, folder structure, and examples."""

from pathlib import Path
from zen import VERSION
from zen.spec import DELIMITERS, KEYWORDS, OPERATORS, PLANNED_BUILTINS

ROOT = Path(__file__).resolve().parent.parent


def test_folder_structure_exists() -> None:
    for rel in ("zen", "docs", "examples", "tests"):
        assert (ROOT / rel).is_dir(), f"Missing directory: {rel}"


def test_required_docs_and_examples_exist() -> None:
    for rel in (
        "README.md",
        "pyproject.toml",
        "PROGRESS.md",
        "docs/SPEC.md",
        "docs/GRAMMAR.md",
        "docs/ROADMAP.md",
        "examples/hello.zen",
        "examples/check_phase1.py",
    ):
        path = ROOT / rel
        assert path.is_file(), f"Missing file: {rel}"
        assert path.read_text(encoding="utf-8").strip(), f"Empty file: {rel}"


def test_spec_and_grammar_cover_core_constructs() -> None:
    spec_text = (ROOT / "docs" / "SPEC.md").read_text(encoding="utf-8")
    grammar_text = (ROOT / "docs" / "GRAMMAR.md").read_text(encoding="utf-8")

    for section in ("Comments", "Variables", "Operators", "Control Flow", "Functions", "Classes"):
        assert section in spec_text, f"SPEC.md missing section: {section}"
    assert "say" in spec_text
    assert "print" in spec_text

    for rule in (
        "program",
        "statement",
        "let_stmt",
        "const_stmt",
        "say_stmt",
        "fn_decl",
        "class_decl",
        "if_stmt",
        "while_stmt",
        "for_stmt",
        "expression",
    ):
        assert rule in grammar_text, f"GRAMMAR.md missing rule: {rule}"


def test_zen_spec_constants_are_consistent() -> None:
    assert VERSION == "0.2.4"
    # Multi-character operators must precede single-character prefixes for maximal munch.
    assert OPERATORS.index("**") < OPERATORS.index("*")
    assert OPERATORS.index("==") < OPERATORS.index("=")
    assert OPERATORS.index("<=") < OPERATORS.index("<")
    assert OPERATORS.index(">=") < OPERATORS.index(">")

    assert {"let", "const", "say", "fn", "class", "if", "while", "for", "return"}.issubset(KEYWORDS)
    assert {"(", ")", "{", "}", "[", "]"}.issubset(DELIMITERS)
    assert "print" in PLANNED_BUILTINS[3]
    assert "range" in PLANNED_BUILTINS[4]
    assert "matmul" in PLANNED_BUILTINS[5]


def test_hello_zen_target_program() -> None:
    hello = (ROOT / "examples" / "hello.zen").read_text(encoding="utf-8")
    assert 'say "I am " + language' in hello
    assert "say version" in hello
    assert "introduce()" in hello
