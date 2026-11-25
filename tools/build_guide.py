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


def run_zen_example(rel_path: str, cmd: str = "run") -> tuple[str, str]:
    """Run a real .zen file using the zen CLI and return (code_text, output_text)."""
    path = GUIDE_DIR / rel_path
    code = path.read_text(encoding="utf-8").strip()
    proc = subprocess.run(
        [sys.executable, "-m", "zen", cmd, str(path)],
        cwd=ROOT,
        capture_output=True,
        text=True,
    )
    if cmd == "run" and proc.returncode != 0:
        raise RuntimeError(f"Guide example {rel_path} failed:\n{proc.stderr}\n{proc.stdout}")
    output = (proc.stdout or "") + (proc.stderr or "")
    return code, output.strip()


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
        canvas.drawString(0.75 * inch, h - 0.5 * inch, "Learn ZEN (v0.1.1)")
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
        ParagraphStyle("CodeMono", fontName="Courier", fontSize=8.7, leading=11.5, textColor=TEXT_DARK),
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
            "Version 0.1.1<br/>By Abdullah Khan Suri",
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
    )
    add_chapter(
        "14. Tensors: create, add, and matrix multiply (@)",
        "A tensor holds grids of numbers (vectors and matrices) and supports fast elementwise math and `@` matrix multiplication.",
        "14_tensors.zen",
    )
    add_chapter(
        "15. Autograd: gradients in 10 lines",
        "Autograd (`stdlib/ai/autograd.zen`) tracks how numbers are combined and calls `backward()` to compute exact slopes (gradients).",
        "15_autograd.zen",
    )
    add_chapter(
        "16. Your first neural network (XOR, loss going down)",
        "By combining linear layers, activations, and the Adam optimizer in `.zen`, a neural network learns the XOR pattern as loss drops.",
        "16_xor_nn.zen",
    )
    add_chapter(
        "17. Tiny GPT: train, save checkpoint, generate text",
        "`AutoCharGPT` (written in `.zen`) learns character patterns from text, saves weights to disk, reloads them, and generates new text.",
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
        ("Exercise 10 (Autograd)", "Find `df/dx` at `x = 4` for `f(x) = (x - 1) ** 2` using `autograd.zen`.", "ex10_autograd_grad.zen"),
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

    # Chapter 21: Current Limitations and Roadmap
    story.append(Paragraph("21. Current limitations and roadmap", title_style))
    story.append(Paragraph("What ZEN can do today, and what is still planned for future versions.", one_liner_style))
    limits = [
        ("Slow pure-Python tensors:", "Matrix math uses pure Python lists without C extensions, SIMD, or GPU support (~51 ms/step on a tiny d_model=16 GPT)."),
        ("Tiny GPT only:", "AutoCharGPT trains on small text files for low-RAM laptops and learns character/syllable patterns rather than fluent prose."),
        ("No native compiler yet:", "Programs run on a Python bytecode VM (`zen run`). There is no LLVM, C codegen, package manager, or LSP server yet."),
    ]
    for head, body in limits:
        story.append(Paragraph(f"<b>{escape(head)}</b> {escape(body)}", note_style))
    story.append(Spacer(1, 8))
    story.append(Paragraph("Roadmap", subhead_style))
    story.append(Paragraph("v0.1.x: Language, bytecode VM, type checker, developer tools, pure-Python tensors, .zen autograd/MLP/GPT, and this guide.", note_style))
    story.append(Paragraph("Future: Faster native tensor kernels, IR/native compilation, and richer standard library modules.", note_style))

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
