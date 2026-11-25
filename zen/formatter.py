"""Source code formatter for ZEN (`zen fmt`)."""

from __future__ import annotations

from zen.lexer import tokenize
from zen.parser import parse
from zen.tokens import Token, TokenType

_BINARY_OPS = frozenset({
    TokenType.PLUS,
    TokenType.MINUS,
    TokenType.STAR,
    TokenType.SLASH,
    TokenType.PERCENT,
    TokenType.AT,
    TokenType.POW,
    TokenType.EQ,
    TokenType.EQ_EQ,
    TokenType.BANG_EQ,
    TokenType.LT,
    TokenType.LT_EQ,
    TokenType.GT,
    TokenType.GT_EQ,
    TokenType.AND,
    TokenType.OR,
    TokenType.IN,
    TokenType.AS,
    TokenType.ARROW,
})

_UNARY_PREFIX_TRIGGERS = frozenset({
    TokenType.LPAREN,
    TokenType.LBRACKET,
    TokenType.LBRACE,
    TokenType.COMMA,
    TokenType.COLON,
    TokenType.RETURN,
    TokenType.SAY,
    TokenType.ASSERT,
    TokenType.NOT,
    TokenType.ELIF,
    TokenType.IF,
    TokenType.WHILE,
}) | _BINARY_OPS

_KEYWORD_SPACE_AFTER = frozenset({
    TokenType.LET,
    TokenType.CONST,
    TokenType.SAY,
    TokenType.ASSERT,
    TokenType.CLASS,
    TokenType.IF,
    TokenType.ELIF,
    TokenType.ELSE,
    TokenType.WHILE,
    TokenType.FOR,
    TokenType.IN,
    TokenType.IMPORT,
    TokenType.AS,
    TokenType.AND,
    TokenType.OR,
    TokenType.NOT,
})


def _extract_comments(source: str) -> dict[int, tuple[str, bool]]:
    """Map 1-based line number -> (comment_text, is_standalone)."""
    comments: dict[int, tuple[str, bool]] = {}
    for line_no, line in enumerate(source.splitlines(), start=1):
        in_str = False
        esc = False
        i = 0
        n = len(line)
        while i < n:
            ch = line[i]
            if in_str:
                if esc:
                    esc = False
                elif ch == "\\":
                    esc = True
                elif ch == '"':
                    in_str = False
            else:
                if ch == '"':
                    in_str = True
                elif ch == "/" and i + 1 < n and line[i + 1] == "/":
                    prefix = line[:i].strip()
                    comments[line_no] = (line[i:].rstrip(), len(prefix) == 0)
                    break
            i += 1
    return comments


def _render_token(tok: Token) -> str:
    if tok.type == TokenType.STRING:
        escaped = (
            tok.value.replace("\\", "\\\\")
            .replace('"', '\\"')
            .replace("\n", "\\n")
            .replace("\t", "\\t")
            .replace("\r", "\\r")
        )
        return f'"{escaped}"'
    if tok.type == TokenType.TRUE:
        return "true"
    if tok.type == TokenType.FALSE:
        return "false"
    if tok.type == TokenType.NONE:
        return "none"
    return str(tok.value)


def _find_matching_brace(tokens: list[Token], start: int) -> int:
    depth = 0
    for i in range(start, len(tokens)):
        if tokens[i].type == TokenType.LBRACE:
            depth += 1
        elif tokens[i].type == TokenType.RBRACE:
            depth -= 1
            if depth == 0:
                return i
    return len(tokens) - 1


def _find_matching_bracket(tokens: list[Token], start: int) -> int:
    depth = 0
    for i in range(start, len(tokens)):
        if tokens[i].type == TokenType.LBRACKET:
            depth += 1
        elif tokens[i].type == TokenType.RBRACKET:
            depth -= 1
            if depth == 0:
                return i
    return len(tokens) - 1


def _is_map_literal(tokens: list[Token], lbrace_idx: int) -> bool:
    prev = tokens[lbrace_idx - 1] if lbrace_idx > 0 else None
    if prev is not None and prev.type in _UNARY_PREFIX_TRIGGERS:
        return True
    end = _find_matching_brace(tokens, lbrace_idx)
    if end == lbrace_idx + 1:
        # Empty `{}` after block header (`if cond {}`) is a block, otherwise map
        return prev is not None and prev.type in _UNARY_PREFIX_TRIGGERS
    return False


def format_source(source: str) -> str:
    """Format valid ZEN source code with consistent 4-space indentation and spacing."""
    parse(source)  # Validate syntax first
    tokens = tokenize(source)[:-1]  # Exclude EOF
    comments = _extract_comments(source)

    if not tokens and not comments:
        return ""

    out_lines: list[str] = []
    cur_parts: list[str] = []
    indent = 0
    container_stack: list[tuple[str, bool]] = []  # ("block"|"map"|"list", is_multiline)
    prev_tok: Token | None = None
    prev_was_unary = False
    last_src_line = 0

    def _emit_line(text: str) -> None:
        out_lines.append(text)

    def _flush_line(trailing_line: int | None = None) -> None:
        nonlocal cur_parts
        line_str = ("    " * indent + "".join(cur_parts)).rstrip() if cur_parts else ""
        if trailing_line is not None and trailing_line in comments and not comments[trailing_line][1]:
            c_text = comments.pop(trailing_line)[0]
            if line_str:
                line_str = f"{line_str}  {c_text}"
            else:
                line_str = ("    " * indent) + c_text
        if line_str:
            _emit_line(line_str)
        cur_parts = []

    def _emit_standalone_comments_up_to(target_line: int) -> None:
        nonlocal last_src_line
        pending = sorted(ln for ln in comments if comments[ln][1] and ln <= target_line)
        for ln in pending:
            c_text, _ = comments.pop(ln)
            if last_src_line > 0 and ln - last_src_line > 1 and out_lines and out_lines[-1] != "":
                _emit_line("")
            _emit_line(("    " * indent) + c_text)
            last_src_line = ln

    def _needs_space(prev: Token | None, cur: Token, prev_unary: bool) -> bool:
        if prev is None:
            return False
        if prev_unary:
            return False
        if cur.type in (TokenType.COMMA, TokenType.COLON, TokenType.RPAREN, TokenType.RBRACKET, TokenType.DOT, TokenType.SEMICOLON):
            return False
        if prev.type == TokenType.COLON and container_stack and container_stack[-1][0] == "list":
            return False
        if prev.type in (TokenType.LPAREN, TokenType.LBRACKET, TokenType.DOT):
            return False
        if prev.type == TokenType.LBRACE:
            return True
        if cur.type == TokenType.RBRACE:
            return True
        if cur.type == TokenType.LPAREN:
            return prev.type in (_KEYWORD_SPACE_AFTER | _BINARY_OPS | {TokenType.COMMA, TokenType.COLON, TokenType.RETURN})
        if cur.type == TokenType.LBRACKET:
            return prev.type in (_KEYWORD_SPACE_AFTER | _BINARY_OPS | {TokenType.COMMA, TokenType.COLON, TokenType.RETURN})
        if prev.type == TokenType.FN:
            return cur.type == TokenType.IDENT
        if prev.type == TokenType.RETURN:
            return True
        if prev.type in (TokenType.COMMA, TokenType.COLON):
            return True
        if prev.type in _KEYWORD_SPACE_AFTER or prev.type in _BINARY_OPS or cur.type in _BINARY_OPS:
            return True
        if cur.type == TokenType.LBRACE:
            return True
        return True

    n = len(tokens)
    for idx, tok in enumerate(tokens):
        if tok.type == TokenType.SEMICOLON:
            continue

        _emit_standalone_comments_up_to(tok.line)

        is_new_stmt_line = prev_tok is not None and tok.line > prev_tok.line
        in_multiline_container = container_stack and container_stack[-1][1]

        # Statement boundary on new line (not inside single-line parens/lists, or inside block/multiline container)
        if prev_tok is not None and not cur_parts:
            pass
        elif is_new_stmt_line and (not container_stack or in_multiline_container):
            # Avoid breaking mid-expression if previous line ended in an infix operator or '('
            if prev_tok is not None and prev_tok.type not in (
                _BINARY_OPS | {TokenType.LPAREN, TokenType.COMMA, TokenType.COLON}
            ):
                _flush_line(prev_tok.line)
                if tok.line - max(prev_tok.line, last_src_line) > 1 and out_lines and out_lines[-1] != "":
                    _emit_line("")
                prev_tok = None

        if tok.type == TokenType.LBRACE:
            is_map = _is_map_literal(tokens, idx)
            end_idx = _find_matching_brace(tokens, idx)
            if is_map:
                is_multi = tokens[end_idx].line > tok.line
                if cur_parts and _needs_space(prev_tok, tok, prev_was_unary):
                    cur_parts.append(" ")
                cur_parts.append("{")
                container_stack.append(("map", is_multi))
                if is_multi:
                    _flush_line(tok.line)
                    indent += 1
                    prev_tok = None
                else:
                    prev_tok = tok
                prev_was_unary = False
                last_src_line = tok.line
                continue
            else:
                if cur_parts and _needs_space(prev_tok, tok, prev_was_unary):
                    cur_parts.append(" ")
                cur_parts.append("{")
                is_empty = end_idx == idx + 1
                container_stack.append(("block", not is_empty))
                if not is_empty:
                    _flush_line(tok.line)
                    indent += 1
                    prev_tok = None
                else:
                    prev_tok = tok
                prev_was_unary = False
                last_src_line = tok.line
                continue

        if tok.type == TokenType.RBRACE:
            kind, is_multi = container_stack.pop() if container_stack else ("block", True)
            if is_multi:
                if cur_parts:
                    _flush_line(prev_tok.line if prev_tok else tok.line)
                indent = max(0, indent - 1)
                cur_parts.append("}")
                # Check if next token is `elif` or `else` on same block
                nxt = tokens[idx + 1] if idx + 1 < n else None
                if nxt is not None and nxt.type in (TokenType.ELIF, TokenType.ELSE):
                    prev_tok = tok
                else:
                    _flush_line(tok.line)
                    prev_tok = None
            else:
                cur_parts.append("}")
                prev_tok = tok
            prev_was_unary = False
            last_src_line = tok.line
            continue

        if tok.type == TokenType.LBRACKET:
            end_idx = _find_matching_bracket(tokens, idx)
            is_multi = tokens[end_idx].line > tok.line
            if cur_parts and _needs_space(prev_tok, tok, prev_was_unary):
                cur_parts.append(" ")
            cur_parts.append("[")
            container_stack.append(("list", is_multi))
            if is_multi:
                _flush_line(tok.line)
                indent += 1
                prev_tok = None
            else:
                prev_tok = tok
            prev_was_unary = False
            last_src_line = tok.line
            continue

        if tok.type == TokenType.RBRACKET:
            _, is_multi = container_stack.pop() if container_stack else ("list", False)
            if is_multi:
                if cur_parts:
                    _flush_line(prev_tok.line if prev_tok else tok.line)
                indent = max(0, indent - 1)
            cur_parts.append("]")
            prev_tok = tok
            prev_was_unary = False
            last_src_line = tok.line
            continue

        if tok.type == TokenType.COMMA and container_stack and container_stack[-1][1]:
            cur_parts.append(",")
            _flush_line(tok.line)
            prev_tok = None
            prev_was_unary = False
            last_src_line = tok.line
            continue

        # Normal token spacing
        if not cur_parts and last_src_line > 0 and tok.line - last_src_line > 1 and out_lines and out_lines[-1] != "":
            _emit_line("")

        if cur_parts and _needs_space(prev_tok, tok, prev_was_unary):
            cur_parts.append(" ")

        cur_parts.append(_render_token(tok))
        is_unary_minus = tok.type == TokenType.MINUS and (
            prev_tok is None or prev_tok.type in _UNARY_PREFIX_TRIGGERS
        )
        prev_was_unary = is_unary_minus
        prev_tok = tok
        last_src_line = tok.line

    if cur_parts:
        _flush_line(prev_tok.line if prev_tok else None)
    _emit_standalone_comments_up_to(10**9)

    while out_lines and out_lines[-1] == "":
        out_lines.pop()
    return "\n".join(out_lines) + "\n"


def format_file(path: str, *, check_only: bool = False) -> bool:
    """Format a .zen file in place (or return whether it changed in check_only mode)."""
    from pathlib import Path

    p = Path(path)
    original = p.read_text(encoding="utf-8")
    formatted = format_source(original)
    changed = original != formatted
    if changed and not check_only:
        p.write_text(formatted, encoding="utf-8")
    return changed
