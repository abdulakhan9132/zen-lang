"""Tests for VS Code ZEN extension and logo integration."""

import json
import os
from pathlib import Path

import pytest
from zen.interpreter import Interpreter
from zen.spec import KEYWORDS

ROOT = Path(__file__).resolve().parent.parent
EXT_DIR = ROOT / "tools" / "vscode-zen"


def test_vscode_extension_files_and_valid_json() -> None:
    pkg_path = EXT_DIR / "package.json"
    cfg_path = EXT_DIR / "language-configuration.json"
    grammar_path = EXT_DIR / "syntaxes" / "zen.tmLanguage.json"
    readme_path = EXT_DIR / "README.md"
    icon_path = EXT_DIR / "icons" / "zen.png"
    docs_logo_path = ROOT / "docs" / "zen-logo.png"

    for p in (pkg_path, cfg_path, grammar_path, readme_path, icon_path, docs_logo_path):
        assert p.is_file(), f"Missing file: {p}"

    pkg = json.loads(pkg_path.read_text(encoding="utf-8"))
    cfg = json.loads(cfg_path.read_text(encoding="utf-8"))
    grammar = json.loads(grammar_path.read_text(encoding="utf-8"))

    assert pkg["icon"] == "icons/zen.png"
    lang = pkg["contributes"]["languages"][0]
    assert lang["id"] == "zen"
    assert "ZEN" in lang["aliases"]
    assert ".zen" in lang["extensions"]
    assert lang["icon"]["light"] == "./icons/zen.png"
    assert lang["icon"]["dark"] == "./icons/zen.png"
    assert cfg["comments"]["lineComment"] == "//"
    assert grammar["scopeName"] == "source.zen"

    root_readme = (ROOT / "README.md").read_text(encoding="utf-8")
    assert "docs/zen-logo.png" in root_readme


def test_vscode_icon_dimensions() -> None:
    pytest.importorskip("PIL")
    from PIL import Image

    icon_path = EXT_DIR / "icons" / "zen.png"
    with Image.open(icon_path) as im:
        assert im.size == (128, 128)


def test_vscode_grammar_covers_all_keywords_and_builtins() -> None:
    grammar_text = (EXT_DIR / "syntaxes" / "zen.tmLanguage.json").read_text(encoding="utf-8")
    for kw in KEYWORDS:
        assert kw in grammar_text, f"Keyword '{kw}' missing from zen.tmLanguage.json"

    interp = Interpreter()
    for builtin_name in interp.globals.values:
        assert builtin_name in grammar_text, f"Built-in '{builtin_name}' missing from zen.tmLanguage.json"


def test_vscode_extension_installed_for_user() -> None:
    if os.environ.get("CI"):
        pytest.skip("User VS Code extension check skipped on CI")
    home = Path(os.environ.get("USERPROFILE") or Path.home())
    user_ext = home / ".vscode" / "extensions" / "zen-lang-0.1.0"
    if not user_ext.exists():
        pytest.skip(f"User VS Code extension not installed at {user_ext}")
    assert (user_ext / "package.json").is_file()
    assert (user_ext / "icons" / "zen.png").is_file()
    assert (user_ext / "syntaxes" / "zen.tmLanguage.json").is_file()
    assert (user_ext / "language-configuration.json").is_file()
