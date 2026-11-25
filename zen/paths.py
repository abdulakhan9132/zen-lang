"""Module/import path resolution for ZEN, including packaged stdlib."""

from __future__ import annotations

from pathlib import Path

_PKG_STDLIB = Path(__file__).resolve().parent / "stdlib"
_REPO_STDLIB = Path(__file__).resolve().parent.parent / "stdlib"
_PKG_GUIDE = Path(__file__).resolve().parent / "docs" / "Learn-ZEN.pdf"
_REPO_GUIDE = Path(__file__).resolve().parent.parent / "docs" / "Learn-ZEN.pdf"


def resolve_guide_pdf_path() -> Path | None:
    """Return the path to Learn-ZEN.pdf from the installed package or repo docs."""
    for candidate in (_PKG_GUIDE, _REPO_GUIDE):
        if candidate.is_file():
            return candidate.resolve()
    return None


def resolve_import_path(path_str: str, base_dir: Path) -> Path | None:
    """Resolve an imported .zen path relative to base_dir or packaged stdlib."""
    raw = Path(path_str)
    candidate = raw if raw.is_absolute() else (base_dir / raw)
    resolved = candidate.resolve()
    if resolved.is_file():
        return resolved

    # Fallback: resolve `pkg:...` (`import zennum`), `stdlib/...`, or `../stdlib/...`
    normalized = path_str.replace("\\", "/")
    if normalized.startswith("pkg:"):
        pkg_parts = normalized[4:].split(".")
        candidates = [
            "/".join(pkg_parts) + ".zen",
            "/".join(pkg_parts) + "/__init__.zen",
        ]
        for stdlib_root in (_PKG_STDLIB, _REPO_STDLIB):
            for rel in candidates:
                pkg_candidate = (stdlib_root / rel).resolve()
                if pkg_candidate.is_file():
                    return pkg_candidate
        return None

    if "stdlib/" in normalized:
        rel_sub = normalized.split("stdlib/", 1)[1]
    else:
        rel_sub = normalized.lstrip("./")

    for stdlib_root in (_PKG_STDLIB, _REPO_STDLIB):
        pkg_candidate = (stdlib_root / rel_sub).resolve()
        if pkg_candidate.is_file():
            return pkg_candidate
    return None
