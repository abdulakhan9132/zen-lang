"""Build docs/Learn-ZEN.pdf from real runnable .zen examples in docs/guide_examples/.

Dev-only script using reportlab (never a runtime dependency of zen-lang).
"""

from __future__ import annotations

import shutil
import subprocess
import sys
from pathlib import Path
from xml.sax.saxutils import escape

from reportlab.lib import colors
from reportlab.lib.enums import TA_CENTER, TA_LEFT
from reportlab.lib.pagesizes import LETTER
from reportlab.lib.styles import ParagraphStyle, getSampleStyleSheet
from reportlab.lib.units import inch
from reportlab.platypus import (
    Image,
    PageBreak,
    Paragraph,
    Preformatted,
    Spacer,
    Table,
    TableStyle,
)
from reportlab.platypus.doctemplate import BaseDocTemplate, PageTemplate
from reportlab.platypus.frames import Frame
from reportlab.platypus.tableofcontents import TableOfContents

ROOT = Path(__file__).resolve().parent.parent
GUIDE_DIR = ROOT / "docs" / "guide_examples"
PDF_OUT = ROOT / "docs" / "Learn-ZEN.pdf"
PKG_PDF_OUT = ROOT / "zen" / "docs" / "Learn-ZEN.pdf"
LOGO_PATH = ROOT / "docs" / "zen-logo.png"

PURPLE = colors.HexColor("#8A38F5")
PURPLE_DARK = colors.HexColor("#5B1FB8")
PURPLE_SOFT = colors.HexColor("#F4EEFF")
CODE_BG = colors.HexColor("#F6F7FB")
OUT_BG = colors.HexColor("#F0FDF4")
OUT_BORDER = colors.HexColor("#16A34A")
ERR_BG = colors.HexColor("#FEF2F2")
ERR_BORDER = colors.HexColor("#DC2626")
TEXT_DARK = colors.HexColor("#1F2937")
TEXT_MUTED = colors.HexColor("#6B7280")


def run_zen_path(path: Path, cmd: str = "run") -> tuple[str, str]:
    """Run an arbitrary .zen file using the zen CLI and return (code_text, output_text)."""
    code = path.read_text(encoding="utf-8").strip()
    proc = subprocess.run(
        [sys.executable, "-m", "zen", cmd, str(path)],
        cwd=ROOT,
        capture_output=True,
        text=True,
    )
    if cmd == "run" and proc.returncode != 0:
        raise RuntimeError(f"Example {path} failed:\n{proc.stderr}\n{proc.stdout}")
    output = (proc.stdout or "") + (proc.stderr or "")
    return code, output.strip()


def run_zen_example(rel_path: str, cmd: str = "run") -> tuple[str, str]:
    """Run a real .zen file using the zen CLI and return (code_text, output_text)."""
    return run_zen_path(GUIDE_DIR / rel_path, cmd=cmd)


class GuideDocTemplate(BaseDocTemplate):
    """Custom PDF template supporting Cover, TOC, and numbered content pages."""

    def __init__(self, filename: str, **kw) -> None:
        super().__init__(filename, **kw)
        frame = Frame(self.leftMargin, self.bottomMargin, self.width, self.height, id="normal")
        self.addPageTemplates([
            PageTemplate(id="cover", frames=[frame], onPage=self._draw_cover_bg),
            PageTemplate(id="content", frames=[frame], onPage=self._draw_chrome),
        ])

    def _draw_cover_bg(self, canvas, doc) -> None:
        canvas.saveState()
        w, h = LETTER
        canvas.setFillColor(PURPLE)
        canvas.rect(0, h - 0.45 * inch, w, 0.45 * inch, fill=1, stroke=0)
        canvas.rect(0, 0, w, 0.25 * inch, fill=1, stroke=0)
        canvas.restoreState()

    def _draw_chrome(self, canvas, doc) -> None:
        canvas.saveState()
        w, h = LETTER
        canvas.setStrokeColor(PURPLE_SOFT)
        canvas.setLineWidth(1)
        canvas.line(0.75 * inch, h - 0.6 * inch, w - 0.75 * inch, h - 0.6 * inch)
        canvas.setFont("Helvetica", 8.5)
        canvas.setFillColor(TEXT_MUTED)
        canvas.drawString(0.75 * inch, h - 0.5 * inch, "Learn ZEN (v0.2.4)")
        canvas.drawRightString(w - 0.75 * inch, 0.45 * inch, f"Page {doc.page}")
        canvas.restoreState()

    def afterFlowable(self, flowable) -> None:
        if isinstance(flowable, Paragraph) and flowable.style.name == "ChapterTitle":
            self.notify("TOCEntry", (0, flowable.getPlainText(), self.page))


def _code_box(title: str, text: str, bg=CODE_BG, border=PURPLE) -> Table:
    label = Paragraph(
        f"<b><font color='#4B5563' size='8'>{escape(title)}</font></b>",
        ParagraphStyle("CodeLabel", fontName="Helvetica", fontSize=8, leading=10),
    )
    pre = Preformatted(
        text,
        ParagraphStyle("CodeMono", fontName="Courier", fontSize=8.2, leading=11.0, textColor=TEXT_DARK),
    )
    tbl = Table([[label], [pre]], colWidths=[6.9 * inch])
    tbl.setStyle(
        TableStyle([
            ("BACKGROUND", (0, 0), (-1, -1), bg),
            ("LINEBEFORE", (0, 0), (0, -1), 3, border),
            ("BOX", (0, 0), (-1, -1), 0.5, colors.HexColor("#E5E7EB")),
            ("LEFTPADDING", (0, 0), (-1, -1), 10),
            ("RIGHTPADDING", (0, 0), (-1, -1), 8),
            ("TOPPADDING", (0, 0), (0, 0), 5),
            ("BOTTOMPADDING", (0, 0), (0, 0), 2),
            ("TOPPADDING", (0, 1), (0, 1), 2),
            ("BOTTOMPADDING", (0, 1), (0, 1), 6),
        ])
    )
    return tbl


def build_pdf() -> int:
    styles = getSampleStyleSheet()
    title_style = ParagraphStyle(
        "ChapterTitle",
        parent=styles["Heading1"],
        fontName="Helvetica-Bold",
        fontSize=18,
        leading=22,
        textColor=PURPLE_DARK,
        spaceAfter=6,
    )
    one_liner_style = ParagraphStyle(
        "OneLiner",
        parent=styles["Normal"],
        fontName="Helvetica",
        fontSize=10.5,
        leading=14.5,
        textColor=TEXT_DARK,
        spaceAfter=8,
    )
    note_style = ParagraphStyle(
        "Note",
        parent=styles["Normal"],
        fontName="Helvetica",
        fontSize=9.5,
        leading=13.5,
        textColor=colors.HexColor("#374151"),
        spaceAfter=6,
    )
    subhead_style = ParagraphStyle(
        "SubHead",
        parent=styles["Heading2"],
        fontName="Helvetica-Bold",
        fontSize=12,
        leading=15,
        textColor=PURPLE,
        spaceBefore=6,
        spaceAfter=4,
    )

    from reportlab.platypus.doctemplate import NextPageTemplate

    story: list = []

    # --- Cover Page ---
    story.append(Spacer(1, 1.1 * inch))
    if LOGO_PATH.is_file():
        img = Image(str(LOGO_PATH), width=1.6 * inch, height=1.6 * inch)
        img.hAlign = "CENTER"
        story.append(img)
        story.append(Spacer(1, 0.35 * inch))

    story.append(
        Paragraph(
            "Learn ZEN",
            ParagraphStyle("CoverTitle", fontName="Helvetica-Bold", fontSize=36, leading=42, textColor=PURPLE_DARK, alignment=TA_CENTER),
        )
    )
    story.append(Spacer(1, 0.15 * inch))
    story.append(
        Paragraph(
            "A Beginner's Step-by-Step Guide to Programming and AI in ZEN",
            ParagraphStyle("CoverSub", fontName="Helvetica", fontSize=14, leading=19, textColor=TEXT_DARK, alignment=TA_CENTER),
        )
    )
    story.append(Spacer(1, 0.55 * inch))
    story.append(
        Paragraph(
            "Version 0.2.4<br/>By Abdullah Khan Suri",
            ParagraphStyle("CoverMeta", fontName="Helvetica-Bold", fontSize=12, leading=18, textColor=PURPLE, alignment=TA_CENTER),
        )
    )
    story.append(NextPageTemplate("content"))
    story.append(PageBreak())

    # --- Table of Contents ---
    story.append(
        Paragraph(
            "Table of Contents",
            ParagraphStyle("TOCHead", fontName="Helvetica-Bold", fontSize=20, leading=24, textColor=PURPLE_DARK, spaceAfter=12),
        )
    )
    toc = TableOfContents()
    toc.levelStyles = [
        ParagraphStyle(fontName="Helvetica", fontSize=10, name="TOCHeading1", leftIndent=0, firstLineIndent=0, spaceBefore=2, leading=13, textColor=TEXT_DARK),
    ]
    story.append(toc)
    story.append(PageBreak())

    def add_chapter(
        title: str,
        one_liner: str,
        example_rel: str | None = None,
        extra_notes: list[str] | None = None,
        cmd: str = "run",
        output_is_error: bool = False,
        page_break: bool = True,
    ) -> None:
        story.append(Paragraph(escape(title), title_style))
        story.append(Paragraph(escape(one_liner), one_liner_style))
        if extra_notes:
            for n in extra_notes:
                story.append(Paragraph(escape(n), note_style))
        if example_rel is not None:
            code, out = run_zen_example(example_rel, cmd=cmd)
            story.append(Spacer(1, 4))
            story.append(_code_box(f"docs/guide_examples/{example_rel}", code, bg=CODE_BG, border=PURPLE))
            story.append(Spacer(1, 6))
            story.append(
                _code_box(
                    f"Output (zen {cmd} {example_rel})",
                    out if out else "(no output)",
                    bg=ERR_BG if output_is_error else OUT_BG,
                    border=ERR_BORDER if output_is_error else OUT_BORDER,
                )
            )
        if page_break:
            story.append(PageBreak())

    # Chapters 1-17 (1 idea per page)
    add_chapter(
        "1. What is ZEN, and how to install it",
        "ZEN is a small programming language designed from scratch to learn coding and build AI models in plain .zen files.",
        "01_install_hello.zen",
        [
            "Install ZEN from PyPI with Python 3.10 or newer: pip install zen-lang",
            "Check that the CLI works: zen --version",
        ],
    )
    add_chapter(
        "2. Your first program (say), running files, and the REPL",
        "Use `say` to display text or numbers on the screen, run saved files with `zen run file.zen`, or type `zen` for an interactive prompt.",
        "02_first_program.zen",
        [
            "Save code in a file ending with .zen, then run: zen run 02_first_program.zen",
            "Run `zen` with no arguments to open the REPL and test single lines immediately.",
        ],
    )
    add_chapter(
        "3. Variables, constants (let, const), and comments",
        "`let` creates a box whose value can change, `const` creates a fixed value that cannot change, and `//` writes notes ignored by ZEN.",
        "03_variables.zen",
    )
    add_chapter(
        "4. Data types: int, float, string, bool, none, list, map",
        "Every value in ZEN has a type: whole numbers (int), decimals (float), text (string), true/false (bool), empty (none), lists, or key-value maps.",
        "04_data_types.zen",
    )
    add_chapter(
        "5. Operators and string joining",
        "Operators combine values using math (+, -, *, /, %, **), comparisons (==, !=, <, >), logic (and, or, not), and `+` to join strings.",
        "05_operators.zen",
    )
    add_chapter(
        "6. Decisions: if / elif / else",
        "Conditionals let your program pick which block of `{ ... }` code to run depending on whether a test is true.",
        "06_decisions.zen",
    )
    add_chapter(
        "7. Loops: while, for, range, break, continue",
        "Loops repeat code: `for` visits every item in a `range` or list, `while` runs until false, `continue` skips, and `break` stops early.",
        "07_loops.zen",
    )
    add_chapter(
        "8. Functions, return values, and closures",
        "Functions (`fn`) group reusable steps under a name, send results back with `return`, and closures remember outer variables.",
        "08_functions.zen",
    )
    add_chapter(
        "9. Lists, maps, and built-in helpers",
        "Lists store ordered items (`push`, `len`, `map`, `filter`) and maps store named values looked up by key (`keys`, `contains`).",
        "09_lists_maps.zen",
    )
    add_chapter(
        "10. Classes",
        "Classes group data (`self.field`) and actions (`fn method(self)`) into reusable blueprints created with `init`.",
        "10_classes.zen",
    )

    # Chapter 11: valid typed example + real `zen check` caret error
    add_chapter(
        "11. Types: annotations, inference, zen check, and errors",
        "ZEN infers types automatically, supports optional annotations (`x: int`, `-> int`), and catches type mistakes before running.",
        "11_types.zen",
        page_break=False,
    )
    err_code, err_out = run_zen_example("errors/11_type_error.zen", cmd="check")
    story.append(Spacer(1, 6))
    story.append(Paragraph("<b>Catching a mismatch with `zen check`</b>", note_style))
    story.append(_code_box("docs/guide_examples/errors/11_type_error.zen", err_code, bg=CODE_BG, border=PURPLE))
    story.append(Spacer(1, 4))
    story.append(_code_box("Output (zen check errors/11_type_error.zen)", err_out, bg=ERR_BG, border=ERR_BORDER))
    story.append(PageBreak())

    add_chapter(
        "12. Testing & formatting: assert, zen test, zen fmt",
        "`assert` verifies a condition is true, `zen test` runs `.zen` test files in `tests_zen/`, and `zen fmt` formats code cleanly.",
        "12_assert_tools.zen",
        [
            "Format any file: zen fmt file.zen",
            "Run all self-tests in tests_zen/: zen test",
        ],
    )
    add_chapter(
        "13. Modules and files: import, read_file, write_file",
        "`import` loads functions and classes from another `.zen` file, while `read_file` and `write_file` read and save text files.",
        "13_modules_files.zen",
        [
            "Built-in helpers: `file_exists(path)` returns true/false before reading, and `clock()` returns high-resolution seconds for timing.",
        ],
    )
    add_chapter(
        "14. Tensors: create, add, and matrix multiply (@)",
        "A tensor holds grids of numbers (vectors and matrices) and supports fast elementwise math and `@` matrix multiplication.",
        "14_tensors.zen",
    )
    add_chapter(
        "15. Autograd: gradients in 8 lines",
        "ZenNet (`import zennet as nn`) tracks how numbers are combined and calls `backward()` to compute exact slopes (gradients).",
        "15_autograd.zen",
    )
    add_chapter(
        "16. Your first neural network (XOR, loss going down)",
        "By combining `nn.Sequential`, `nn.Linear`, activations, and `nn.Adam` in `.zen`, a neural network learns XOR as loss drops.",
        "16_xor_nn.zen",
    )
    add_chapter(
        "17. Tiny GPT: train, save weights, generate text",
        "`nn.GPT` (written in `.zen`) learns character patterns from text, saves weights to disk, reloads them, and generates new text.",
        "17_tiny_gpt.zen",
    )

    # Chapter 18: VS Code Setup
    story.append(Paragraph("18. VS Code setup (extension, logo, syntax colors)", title_style))
    story.append(
        Paragraph(
            "Install the bundled ZEN VS Code extension to get `.zen` file icons, bracket matching, comment toggling, and syntax highlighting.",
            one_liner_style,
        )
    )
    story.append(
        _code_box(
            "Install from .vsix or copy folder",
            "code --install-extension tools/vscode-zen/zen-lang-0.1.0.vsix\n"
            "# Or copy tools/vscode-zen to ~/.vscode/extensions/zen-lang-0.1.0\n"
            "# Then restart VS Code or run: Developer: Reload Window",
        )
    )
    story.append(Spacer(1, 8))
    story.append(Paragraph("Highlights keywords (`let`, `fn`, `class`), `say`, `assert`, all built-ins, strings, numbers, and `//` comments.", note_style))
    story.append(PageBreak())

    # Chapter 19: All Commands Table
    story.append(Paragraph("19. All commands in one table", title_style))
    story.append(Paragraph("Every command available in the `zen` CLI tool.", one_liner_style))
    cmd_rows = [
        ["Command", "What it does"],
        ["zen", "Start the interactive REPL"],
        ["zen run file.zen", "Type-check and run a .zen file on the bytecode VM"],
        ["zen run --no-check file.zen", "Run without static type checking"],
        ["zen run --tree-walk file.zen", "Run using the fallback tree-walking interpreter"],
        ["zen check file.zen", "Run the type checker only (no execution)"],
        ["zen fmt file.zen", "Format a .zen file cleanly (--check to verify only)"],
        ["zen test", "Run all .zen test files in tests_zen/"],
        ["zen guide", "Open this Learn-ZEN.pdf guide (--path prints its path)"],
        ["zen docs", "Print paths to packaged ZENNUM.md and ZENNET.md references"],
        ["zen doctor", "Print ZEN version, OS, Python, math backend, stdlib & PDF paths"],
        ["zen --version", "Print the installed ZEN version"],
    ]
    cmd_tbl = Table(cmd_rows, colWidths=[2.6 * inch, 4.3 * inch])
    cmd_tbl.setStyle(
        TableStyle([
            ("BACKGROUND", (0, 0), (-1, 0), PURPLE),
            ("TEXTCOLOR", (0, 0), (-1, 0), colors.white),
            ("FONTNAME", (0, 0), (-1, 0), "Helvetica-Bold"),
            ("FONTNAME", (0, 1), (0, -1), "Courier"),
            ("FONTNAME", (1, 1), (1, -1), "Helvetica"),
            ("FONTSIZE", (0, 0), (-1, -1), 9),
            ("ROWBACKGROUNDS", (0, 1), (-1, -1), [colors.white, PURPLE_SOFT]),
            ("GRID", (0, 0), (-1, -1), 0.5, colors.HexColor("#D1D5DB")),
            ("LEFTPADDING", (0, 0), (-1, -1), 8),
            ("RIGHTPADDING", (0, 0), (-1, -1), 8),
            ("TOPPADDING", (0, 0), (-1, -1), 6),
            ("BOTTOMPADDING", (0, 0), (-1, -1), 6),
        ])
    )
    story.append(cmd_tbl)
    story.append(PageBreak())

    # Chapter 20: 10 Practice Exercises + Verified Answers
    exercises = [
        ("Exercise 1 (Hello)", "Print your name and favorite number using `say`.", "ex01_hello.zen"),
        ("Exercise 2 (Variables)", "Compute the area of a 5 x 8 rectangle and check it with `assert`.", "ex02_variables.zen"),
        ("Exercise 3 (Decisions)", "Check whether `n = 14` is even or odd and print the label.", "ex03_even_odd.zen"),
        ("Exercise 4 (Loops)", "Sum the numbers from 1 to 10 using a `for` loop and `range`.", "ex04_sum_loop.zen"),
        ("Exercise 5 (Functions & Types)", "Write a typed function `max_two(a: int, b: int) -> int`.", "ex05_max_fn.zen"),
        ("Exercise 6 (Lists & Filter)", "Filter `[3, 12, 7, 19, 10]` to keep only numbers >= 10.", "ex06_filter_list.zen"),
        ("Exercise 7 (Maps & Strings)", "Count how many times each word appears in `\"zen is fast and zen is fun\"`.", "ex07_word_map.zen"),
        ("Exercise 8 (Classes)", "Build a `Counter` class with `inc()` and `value()` methods.", "ex08_counter_class.zen"),
        ("Exercise 9 (Tensors)", "Multiply two 2x2 tensors with `@` and print their `mean`.", "ex09_tensor_matmul.zen"),
        ("Exercise 10 (Autograd)", "Find `df/dx` at `x = 4` for `f(x) = (x - 1) ** 2` using `zennet`.", "ex10_autograd_grad.zen"),
    ]

    story.append(Paragraph("20. 10 Practice Exercises", title_style))
    story.append(Paragraph("Try writing each program yourself before checking the runnable answers on the following pages.", one_liner_style))
    for ex_title, prompt, _ in exercises:
        story.append(Paragraph(f"<b>{escape(ex_title)}:</b> {escape(prompt)}", note_style))
    story.append(PageBreak())

    story.append(Paragraph("Exercise Answers (1 - 4)", subhead_style))
    for idx, (ex_title, _, fname) in enumerate(exercises):
        if idx == 4:
            story.append(PageBreak())
            story.append(Paragraph("Exercise Answers (5 - 7)", subhead_style))
        elif idx == 7:
            story.append(PageBreak())
            story.append(Paragraph("Exercise Answers (8 - 10)", subhead_style))
        code, out = run_zen_example(fname)
        story.append(Paragraph(f"<b>{escape(ex_title)}</b> ({escape(fname)})", note_style))
        story.append(_code_box(f"docs/guide_examples/{fname}", code, bg=CODE_BG, border=PURPLE))
        story.append(Spacer(1, 2))
        story.append(_code_box("Output", out, bg=OUT_BG, border=OUT_BORDER))
        story.append(Spacer(1, 6))
    story.append(PageBreak())

    # --- Part A: ZenNum (Numbers and Arrays) ---
    story.append(Paragraph("21. Part A — ZenNum: Numbers and Arrays", title_style))
    story.append(
        Paragraph(
            "ZenNum gives ZEN fast N-dimensional arrays (`import zennum as zn`) with slicing, broadcasting, and matrix math.",
            one_liner_style,
        )
    )
    add_chapter(
        "21.1 Creating Arrays with ZenNum",
        "Create multi-dimensional grids of numbers with `array`, `zeros`, `ones`, `arange`, `linspace`, `eye`, and `randn`.",
        "22_zn_intro.zen",
    )
    add_chapter(
        "21.2 Shapes, Reshaping & Transpose",
        "Inspect dimensions with `.shape`, `.ndim`, `.size`, and rearrange data without loops via `.reshape` and `.T`.",
        "23_zn_shapes.zen",
    )
    add_chapter(
        "21.3 Multi-Dimensional Slicing & Boolean Masks",
        "Pick sub-matrices with `a[rows, cols]` slices or filter elements matching a condition like `a[a > 7]`.",
        "24_zn_slicing.zen",
    )
    add_chapter(
        "21.4 Broadcasting, Reductions & Matrix Multiply",
        "Smaller arrays expand to match larger ones automatically, and `sum`, `mean`, `@` work across entire arrays.",
        "25_zn_broadcasting.zen",
    )

    cell_hdr = ParagraphStyle("TblHdr", fontName="Helvetica-Bold", fontSize=8.2, leading=10.5, textColor=colors.white)
    cell_key = ParagraphStyle("TblKey", fontName="Helvetica-Bold", fontSize=8.0, leading=10.2, textColor=TEXT_DARK)
    cell_mono = ParagraphStyle("TblMono", fontName="Courier", fontSize=7.7, leading=10.0, textColor=TEXT_DARK)

    def wrap_sheet(rows: list[list[str]]) -> list[list[Paragraph]]:
        out: list[list[Paragraph]] = []
        for r_idx, row in enumerate(rows):
            if r_idx == 0:
                out.append([Paragraph(escape(c), cell_hdr) for c in row])
            else:
                out.append([
                    Paragraph(escape(row[0]), cell_key),
                    *[Paragraph(escape(c), cell_mono) for c in row[1:]],
                ])
        return out

    # 1-Page ZenNum Cheat Sheet
    story.append(Paragraph("21.5 ZenNum 1-Page Cheat Sheet", title_style))
    story.append(Paragraph("All core ZenNum functions in one place (`import zennum as zn`).", one_liner_style))
    zn_sheet = [
        ["Category", "Functions / Syntax"],
        ["Import", "import zennum as zn   (or: import zennum)"],
        ["Create", "zn.array(data, dtype), zn.zeros(shape), zn.ones(shape), zn.full(shape, val), zn.arange(a, b, s), zn.linspace(a, b, n), zn.eye(n), zn.rand(shape), zn.randn(shape), zn.seed(s)"],
        ["Inspect", "a.shape, a.ndim, a.size, a.dtype, a.T, a.tolist(), a.copy(), a.astype(dtype)"],
        ["Slice / Mask", "a[1:3, :], a[:, -1], a[a > 0], a[0, 0] = 9, zn.where(cond, x, y), zn.clip(a, lo, hi), zn.tril(a, k), zn.triu(a, k)"],
        ["Shape", "zn.reshape(a, s), zn.transpose(a, axes), zn.flatten(a), zn.squeeze(a), zn.unsqueeze(a, ax), zn.concat(list, ax), zn.stack(list, ax), zn.split(a, n, ax)"],
        ["Math & Stats", "zn.abs, zn.sqrt, zn.exp, zn.log, zn.sin, zn.cos, zn.tanh, zn.pow, zn.sum, zn.mean, zn.std, zn.var, zn.min, zn.max, zn.argmax, zn.argmin, zn.sort, zn.cumsum"],
        ["Linear Algebra", "a @ b, zn.matmul(a, b), zn.dot(a, b), zn.outer(a, b), zn.norm(a)"],
    ]
    zn_tbl = Table(wrap_sheet(zn_sheet), colWidths=[1.35 * inch, 5.55 * inch])
    zn_tbl.setStyle(
        TableStyle([
            ("BACKGROUND", (0, 0), (-1, 0), PURPLE),
            ("ROWBACKGROUNDS", (0, 1), (-1, -1), [colors.white, PURPLE_SOFT]),
            ("GRID", (0, 0), (-1, -1), 0.5, colors.HexColor("#D1D5DB")),
            ("VALIGN", (0, 0), (-1, -1), "TOP"),
            ("LEFTPADDING", (0, 0), (-1, -1), 6),
            ("RIGHTPADDING", (0, 0), (-1, -1), 6),
            ("TOPPADDING", (0, 0), (-1, -1), 5),
            ("BOTTOMPADDING", (0, 0), (-1, -1), 5),
        ])
    )
    story.append(zn_tbl)
    story.append(PageBreak())

    # 8 ZenNum Exercises + Verified Answers
    zn_exercises = [
        ("ZN Ex 1", "Create a 3x3 array filled with 5s using `zn.ones`.", "ex11_zn_ones.zen"),
        ("ZN Ex 2", "Create 5 evenly spaced floats from 0 to 10 with `zn.linspace`.", "ex12_zn_linspace.zen"),
        ("ZN Ex 3", "Reshape `1..9` into `[3, 3]` and slice the bottom-right 2x2 block.", "ex13_zn_slice.zen"),
        ("ZN Ex 4", "Keep only positive numbers from `[-3, 2, -1, 5, 8]` using a boolean mask.", "ex14_zn_mask.zen"),
        ("ZN Ex 5", "Compute L2 `norm` and `sum` of `[3.0, 4.0]`.", "ex15_zn_norm.zen"),
        ("ZN Ex 6", "Find the maximum value and its index in `[10, 50, 20, 40]`.", "ex16_zn_argmax.zen"),
        ("ZN Ex 7", "Stack two 2x2 matrices vertically with `zn.concat`.", "ex17_zn_concat.zen"),
        ("ZN Ex 8", "Multiply `[[1, 2], [3, 4]]` by `zn.eye(2)` using `@`.", "ex18_zn_matmul.zen"),
    ]
    story.append(Paragraph("21.6 8 ZenNum Exercises & Answers", title_style))
    for i, (etitle, prompt, fname) in enumerate(zn_exercises):
        if i == 4:
            story.append(PageBreak())
        code, out = run_zen_example(fname)
        story.append(Paragraph(f"<b>{escape(etitle)}:</b> {escape(prompt)} ({escape(fname)})", note_style))
        story.append(_code_box(f"docs/guide_examples/{fname}", code, bg=CODE_BG, border=PURPLE))
        story.append(Spacer(1, 2))
        story.append(_code_box("Output", out, bg=OUT_BG, border=OUT_BORDER))
        story.append(Spacer(1, 4))
    story.append(PageBreak())

    # --- Part B: ZenNet (Build AI) ---
    story.append(Paragraph("22. Part B — ZenNet: Build AI", title_style))
    story.append(
        Paragraph(
            "ZenNet (`import zennet as nn`) gives you PyTorch-style autograd, layers, optimizers, datasets, tokenizers, and GPT written purely in `.zen`.",
            one_liner_style,
        )
    )
    add_chapter(
        "22.1 Autograd Tensors & Gradients",
        "Set `requires_grad = true`, do math cleanly, call `.backward()`, and read `.grad` (skip tracking with `nn.no_grad`).",
        "26_nn_autograd.zen",
    )
    add_chapter(
        "22.2 Neural Network Layers & Losses",
        "Stack `Linear`, `ReLU`, `GELU`, `LayerNorm`, or `Dropout` inside `nn.Sequential`, then score predictions with `mse` or `cross_entropy`.",
        "27_nn_layers.zen",
        [
            "Custom modules: when subclassing `nn.Module`, always call `nn.Module.init(self)` inside `fn init(self, ...)` so `parameters()` finds your layers.",
        ],
    )
    add_chapter(
        "22.3 Training XOR in 15 Lines",
        "Combine `nn.Sequential`, `nn.Adam`, and `nn.mse` in a short training loop until XOR predictions approach `[0, 1, 1, 0]`.",
        "28_nn_xor.zen",
    )
    add_chapter(
        "22.4 Datasets, DataLoaders & Tokenizers",
        "Split data with `train_val_split`, shuffle mini-batches via `DataLoader`, and convert text to IDs using `CharTokenizer` or `BPETokenizer`.",
        "29_nn_data.zen",
        [
            "Prompt safety: `encode(text)` and `nn.generate(...)` warn and keep default fallbacks (`[]` / skipped unknown chars / token 0); pass `strict = true` to raise an error instead.",
        ],
    )
    add_chapter(
        "22.5 Building & Training a Mini GPT",
        "Configure `nn.GPT`, train cleanly with `nn.fit` or a custom loop, inspect `nn.perplexity`, and generate text with `nn.generate`.",
        "30_nn_gpt.zen",
        page_break=False,
    )
    story.append(Spacer(1, 6))
    story.append(
        Paragraph(
            "<b>Full Corpus Recipe (`examples/gpt_in_40_lines.zen`):</b> Set `QUICK = true` at the top for a ~20-second 100-step preview, or `QUICK = false` (default) for 1000 steps (~3 minutes on CPU, loss &lt; 2.5).",
            note_style,
        )
    )
    gpt_recipe = ROOT / "examples" / "gpt_in_40_lines.zen"
    gpt_code, gpt_full_out = run_zen_path(gpt_recipe)
    import tempfile

    with tempfile.NamedTemporaryFile("w", suffix=".zen", delete=False, encoding="utf-8") as tf:
        tf.write(gpt_code.replace("let QUICK = false", "let QUICK = true", 1))
        quick_path = Path(tf.name)
    try:
        _, gpt_quick_out = run_zen_path(quick_path)
    finally:
        quick_path.unlink(missing_ok=True)
    story.append(_code_box("examples/gpt_in_40_lines.zen", gpt_code, bg=CODE_BG, border=PURPLE))
    story.append(Spacer(1, 4))
    story.append(_code_box("Output (QUICK = true, 100 steps)", gpt_quick_out, bg=OUT_BG, border=OUT_BORDER))
    story.append(Spacer(1, 4))
    story.append(_code_box("Output (QUICK = false, 1000 steps)", gpt_full_out, bg=OUT_BG, border=OUT_BORDER))
    story.append(PageBreak())

    # 1-Page ZenNet Cheat Sheet
    story.append(Paragraph("22.6 ZenNet 1-Page Cheat Sheet", title_style))
    story.append(Paragraph("Every building block in `import zennet as nn`.", one_liner_style))
    nn_sheet = [
        ["Category", "API"],
        ["Autograd", "nn.tensor(d, req_grad), t.backward(), t.grad, t.zero_grad(), t.detach(), t.item(), nn.no_grad(fn)"],
        ["Ops & Act.", "+ - * / ** @, reshape, transpose, concat, stack, tril, triu, sum, mean, relu, gelu, sigmoid, tanh, softmax, log_softmax, masked_fill, gather, layernorm"],
        ["Layers", "nn.Module (call nn.Module.init(self)), nn.Linear, nn.Embedding, nn.LayerNorm, nn.Dropout, nn.ReLU, nn.GELU, nn.Tanh, nn.Sigmoid, nn.Sequential, nn.MultiHeadAttention, nn.TransformerBlock, nn.GPT"],
        ["Losses", "nn.mse(pred, y), nn.cross_entropy(logits, targets), nn.binary_cross_entropy(p, y)"],
        ["Optimizers", "nn.SGD(params, lr, momentum), nn.Adam(params, lr), nn.AdamW(params, lr, weight_decay), nn.clip_grad_norm(params, max)"],
        ["Schedulers", "nn.StepLR(opt, step, gamma), nn.CosineAnnealingLR(opt, total, min_lr), nn.WarmupLR(opt, steps)"],
        ["Data & Tok.", "nn.Dataset, nn.DataLoader, nn.train_val_split, nn.CharTokenizer, nn.BPETokenizer, nn.text_to_batches"],
        ["Train & Eval", "nn.Trainer, nn.fit, nn.generate(model, prompt, n, temp, top_k, strict), nn.accuracy, nn.perplexity, model.save / load"],
    ]
    nn_tbl = Table(wrap_sheet(nn_sheet), colWidths=[1.25 * inch, 5.65 * inch])
    nn_tbl.setStyle(
        TableStyle([
            ("BACKGROUND", (0, 0), (-1, 0), PURPLE),
            ("ROWBACKGROUNDS", (0, 1), (-1, -1), [colors.white, PURPLE_SOFT]),
            ("GRID", (0, 0), (-1, -1), 0.5, colors.HexColor("#D1D5DB")),
            ("VALIGN", (0, 0), (-1, -1), "TOP"),
            ("LEFTPADDING", (0, 0), (-1, -1), 6),
            ("RIGHTPADDING", (0, 0), (-1, -1), 6),
            ("TOPPADDING", (0, 0), (-1, -1), 5),
            ("BOTTOMPADDING", (0, 0), (-1, -1), 5),
        ])
    )
    story.append(nn_tbl)
    story.append(PageBreak())

    # 8 ZenNet Exercises + Verified Answers
    nn_exercises = [
        ("NN Ex 1", "Find `d/dw` of `(w - 5)^2` at `w = 3`.", "ex19_nn_grad.zen"),
        ("NN Ex 2", "Pass `[1, 0, -1, 2]` through `nn.Linear(4, 2)` and check `.shape`.", "ex20_nn_linear.zen"),
        ("NN Ex 3", "Build a `Sequential([Linear(2, 3), GELU(), Linear(3, 1)])` and count parameters.", "ex21_nn_seq.zen"),
        ("NN Ex 4", "Compute `cross_entropy` and `accuracy` for 2 logit rows.", "ex22_nn_loss.zen"),
        ("NN Ex 5", "Use `SGD` to minimize `(w - 1)^2` starting from `w = 4`.", "ex23_nn_sgd.zen"),
        ("NN Ex 6", "Run a calculation inside `nn.no_grad` and confirm `requires_grad` is `false`.", "ex24_nn_nograd.zen"),
        ("NN Ex 7", "Train `BPETokenizer` on a repeated phrase, encode it, and decode back.", "ex25_nn_bpe.zen"),
        ("NN Ex 8", "Create a tiny `nn.GPT` and generate 4 tokens greedily (`temp = 0.0`).", "ex26_nn_gpt.zen"),
    ]
    story.append(Paragraph("22.7 8 ZenNet Exercises & Answers", title_style))
    for i, (etitle, prompt, fname) in enumerate(nn_exercises):
        if i == 4:
            story.append(PageBreak())
        code, out = run_zen_example(fname)
        story.append(Paragraph(f"<b>{escape(etitle)}:</b> {escape(prompt)} ({escape(fname)})", note_style))
        story.append(_code_box(f"docs/guide_examples/{fname}", code, bg=CODE_BG, border=PURPLE))
        story.append(Spacer(1, 2))
        story.append(_code_box("Output", out, bg=OUT_BG, border=OUT_BORDER))
        story.append(Spacer(1, 4))
    story.append(PageBreak())

    # Comparison page: ZenNum and ZenNet vs NumPy and PyTorch
    story.append(Paragraph("23. ZenNum and ZenNet vs NumPy and PyTorch", title_style))
    story.append(Paragraph("If you already know Python's AI stack, here is how ZEN compares side-by-side.", one_liner_style))
    cmp_rows = [
        ["Concept", "Python (NumPy / PyTorch)", "ZEN (ZenNum / ZenNet)"],
        ["Import arrays", "import numpy as np", "import zennum as zn"],
        ["Import deep learning", "import torch.nn as nn", "import zennet as nn"],
        ["Create / reshape array", "np.arange(6).reshape(2, 3)", "zn.arange(6).reshape([2, 3])"],
        ["2D slice & mask", "a[1:3, :], a[a > 0]", "a[1:3, :], a[a > 0]"],
        ["Matrix multiply", "A @ B", "A @ B"],
        ["Autograd tensor", "torch.tensor(x, requires_grad=True)", "nn.tensor(x, true)"],
        ["Sequential MLP", "nn.Sequential(nn.Linear(2, 8), nn.Tanh())", "nn.Sequential([nn.Linear(2, 8), nn.Tanh()])"],
        ["Custom Model", "class Net(nn.Module): def forward(self, x)", "class Net(nn.Module) { fn forward(self, x) }"],
        ["Optimizer + step", "opt = torch.optim.AdamW(m.parameters())", "let opt = nn.AdamW(m.parameters(), 0.01)"],
        ["One-line training", "custom loop or Lightning Trainer", "nn.fit(model, data, steps)"],
        ["Runtime / GPU", "C / CUDA", "Pure ZEN + C/Python CPU fallback (no GPU yet)"],
    ]
    cmp_tbl = Table(wrap_sheet(cmp_rows), colWidths=[1.5 * inch, 2.65 * inch, 2.75 * inch])
    cmp_tbl.setStyle(
        TableStyle([
            ("BACKGROUND", (0, 0), (-1, 0), PURPLE),
            ("ROWBACKGROUNDS", (0, 1), (-1, -1), [colors.white, PURPLE_SOFT]),
            ("GRID", (0, 0), (-1, -1), 0.5, colors.HexColor("#D1D5DB")),
            ("VALIGN", (0, 0), (-1, -1), "TOP"),
            ("LEFTPADDING", (0, 0), (-1, -1), 5),
            ("RIGHTPADDING", (0, 0), (-1, -1), 5),
            ("TOPPADDING", (0, 0), (-1, -1), 5),
            ("BOTTOMPADDING", (0, 0), (-1, -1), 5),
        ])
    )
    story.append(cmp_tbl)
    story.append(PageBreak())

    # Chapter 24: Current Limitations and Roadmap
    story.append(Paragraph("24. Current limitations and roadmap", title_style))
    story.append(Paragraph("What ZEN v0.2.4 can do today, and what is still planned for future versions.", one_liner_style))
    limits = [
        ("CPU-only execution:", "Pure-Python flat buffers run when no C compiler is present (~0.63s for 256x256 matmul; ~180–260 ms/step on a 1-layer d_model=16 GPT), with optional native C kernels (`zen/native/zen_kernels.c`). No GPU/CUDA yet."),
        ("Tiny GPT scale:", "Models and examples target low-RAM laptops using character/small-BPE corpora rather than large foundation models."),
        ("No native compiler yet:", "Programs run on the Python bytecode VM (`zen run`). LLVM/AOT compilation, external package manager, and LSP remain future work."),
    ]
    for head, body in limits:
        story.append(Paragraph(f"<b>{escape(head)}</b> {escape(body)}", note_style))
    story.append(Spacer(1, 8))
    story.append(Paragraph("Roadmap", subhead_style))
    story.append(Paragraph("v0.2.2-v0.2.4: Bytecode VM, static type checker (`let x = none` widening), VS Code extension, ZenNum N-D arrays (`tril`/`triu`), ZenNet autograd (`concat`/`stack`/`tril`/`triu`), strict tokenizer/generate mode, process-wide shared RNG, `zen doctor`, `zen docs`, and Learn-ZEN guide.", note_style))
    story.append(Paragraph("Future: Prebuilt multi-platform native C wheels, GPU/SIMD backends, native compiler, and LSP.", note_style))

    PDF_OUT.parent.mkdir(parents=True, exist_ok=True)
    doc = GuideDocTemplate(
        str(PDF_OUT),
        pagesize=LETTER,
        leftMargin=0.75 * inch,
        rightMargin=0.75 * inch,
        topMargin=0.75 * inch,
        bottomMargin=0.75 * inch,
        title="Learn ZEN",
        author="Abdullah Khan Suri",
    )
    doc.multiBuild(story)

    PKG_PDF_OUT.parent.mkdir(parents=True, exist_ok=True)
    shutil.copyfile(PDF_OUT, PKG_PDF_OUT)
    return doc.page


if __name__ == "__main__":
    pages = build_pdf()
    print(f"Built {PDF_OUT} ({pages} pages)")
