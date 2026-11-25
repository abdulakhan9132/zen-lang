"""Tests for the Learn ZEN beginner guide PDF, guide examples, and `zen guide` CLI."""

import io
from pathlib import Path

import pytest
from zen.cli import main as cli_main
from zen.vm import VM

ROOT = Path(__file__).resolve().parent.parent
GUIDE_DIR = ROOT / "docs" / "guide_examples"


def test_all_guide_examples_run_without_error() -> None:
    examples = sorted(GUIDE_DIR.glob("*.zen"))
    assert len(examples) >= 25
    for path in examples:
        buf = io.StringIO()
        VM(stdout=buf).run_file(path, type_check=True)


def test_guide_pdf_exists_and_is_packaged() -> None:
    repo_pdf = ROOT / "docs" / "Learn-ZEN.pdf"
    pkg_pdf = ROOT / "zen" / "docs" / "Learn-ZEN.pdf"
    for pdf in (repo_pdf, pkg_pdf):
        assert pdf.is_file(), f"Missing PDF: {pdf}"
        data = pdf.read_bytes()
        assert data.startswith(b"%PDF-"), f"Invalid PDF header in {pdf}"
        assert len(data) > 5000


def test_zen_guide_cli_path_and_fallback(monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]) -> None:
    assert cli_main(["guide", "--path"]) == 0
    out_path = Path(capsys.readouterr().out.strip())
    assert out_path.is_file()
    assert out_path.name == "Learn-ZEN.pdf"

    # If opening fails, `zen guide` must print the path instead of crashing
    def _boom(*args, **kwargs):
        raise OSError("no viewer")

    monkeypatch.setattr("zen.cli.os.startfile", _boom, raising=False)
    monkeypatch.setattr("zen.cli.subprocess.run", _boom)
    assert cli_main(["guide"]) == 0
    fallback_out = capsys.readouterr().out.strip()
    assert "Learn-ZEN.pdf" in fallback_out
