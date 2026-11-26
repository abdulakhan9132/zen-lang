"""Lexer (tokenizer) for the ZEN language."""

from zen.errors import ZenSyntaxError
from zen.tokens import (
    KEYWORD_TOKENS,
    SINGLE_CHAR_TOKENS,
    TWO_CHAR_OPERATORS,
    Token,
    TokenType,
)

ESCAPE_MAP = {
    "n": "\n",
    "t": "\t",
    "r": "\r",
    '"': '"',
    "\\": "\\",
}


class Lexer:
    """Scans ZEN source text into a sequence of Tokens with line and column info."""

    def __init__(self, source: str) -> None:
        source = source.replace("\r\n", "\n").replace("\r", "\n")
        self.source = source
        self.length = len(source)
        self.pos = 0
        self.line = 1
        self.column = 1
        self.trailing_comments: list[tuple[int, str]] = []
        self._last_token_line = 0

    def _peek(self, offset: int = 0) -> str:
        idx = self.pos + offset
        if idx >= self.length:
            return ""
        return self.source[idx]

    def _advance(self) -> str:
        ch = self.source[self.pos]
        self.pos += 1
        if ch == "\n":
            self.line += 1
            self.column = 1
        else:
            self.column += 1
        return ch

    def _skip_whitespace_and_comments(self) -> None:
        while self.pos < self.length:
            ch = self._peek()
            if ch in " \t\r\n":
                self._advance()
            elif ch == "/" and self._peek(1) == "/":
                comment_line = self.line
                self._advance()
                self._advance()
                comment_chars: list[str] = []
                while self.pos < self.length and self._peek() != "\n":
                    comment_chars.append(self._advance())
                if self._last_token_line == comment_line:
                    self.trailing_comments.append((comment_line, "".join(comment_chars).strip()))
            else:
                break

    def _scan_number(self, start_line: int, start_col: int) -> Token:
        chars: list[str] = []
        is_float = False
        while self.pos < self.length and self._peek().isdigit():
            chars.append(self._advance())

        if self._peek() == "." and self._peek(1).isdigit():
            is_float = True
            chars.append(self._advance())  # consume '.'
            while self.pos < self.length and self._peek().isdigit():
                chars.append(self._advance())

        if self._peek() in ("e", "E"):
            next1 = self._peek(1)
            next2 = self._peek(2)
            if next1.isdigit() or (next1 in ("+", "-") and next2.isdigit()):
                is_float = True
                chars.append(self._advance())  # consume 'e'/'E'
                if self._peek() in ("+", "-"):
                    chars.append(self._advance())
                while self.pos < self.length and self._peek().isdigit():
                    chars.append(self._advance())

        if self._peek() == ".":
            raise ZenSyntaxError("Invalid number format (unexpected '.')", self.line, self.column)

        raw = "".join(chars)
        value: int | float = float(raw) if is_float else int(raw)
        return Token(TokenType.NUMBER, value, start_line, start_col)

    def _scan_string(self, start_line: int, start_col: int) -> Token:
        self._advance()  # opening quote
        chars: list[str] = []
        while self.pos < self.length:
            ch = self._peek()
            if ch == '"':
                self._advance()
                return Token(TokenType.STRING, "".join(chars), start_line, start_col)
            if ch == "\n":
                raise ZenSyntaxError("Unterminated string literal", start_line, start_col)
            if ch == "\\":
                self._advance()
                if self.pos >= self.length:
                    raise ZenSyntaxError("Unterminated escape sequence in string", start_line, start_col)
                esc = self._advance()
                if esc not in ESCAPE_MAP:
                    raise ZenSyntaxError(f"Unknown escape sequence '\\{esc}'", self.line, self.column - 1)
                chars.append(ESCAPE_MAP[esc])
            else:
                chars.append(self._advance())
        raise ZenSyntaxError("Unterminated string literal", start_line, start_col)

    def _scan_fstring(self, start_line: int, start_col: int) -> Token:
        self._advance()  # 'f'
        self._advance()  # opening '"'
        chars: list[str] = []
        brace_depth = 0
        in_inner_str = False
        while self.pos < self.length:
            ch = self._peek()
            if ch == '"' and brace_depth == 0:
                self._advance()
                return Token(TokenType.FSTRING, "".join(chars), start_line, start_col)
            if ch == "\n":
                raise ZenSyntaxError("Unterminated f-string literal", start_line, start_col)
            if ch == "\\" and brace_depth == 0:
                self._advance()
                if self.pos >= self.length:
                    raise ZenSyntaxError("Unterminated escape sequence in f-string", start_line, start_col)
                esc = self._advance()
                if esc not in ESCAPE_MAP:
                    raise ZenSyntaxError(f"Unknown escape sequence '\\{esc}'", self.line, self.column - 1)
                chars.append(ESCAPE_MAP[esc])
                continue
            if brace_depth > 0:
                if ch == '"' and not in_inner_str:
                    in_inner_str = True
                elif ch == '"' and in_inner_str:
                    in_inner_str = False
                elif ch == "\\" and in_inner_str:
                    chars.append(self._advance())
                    if self.pos < self.length:
                        chars.append(self._advance())
                    continue
                elif not in_inner_str:
                    if ch == "{":
                        brace_depth += 1
                    elif ch == "}":
                        brace_depth -= 1
                chars.append(self._advance())
            else:
                if ch == "{":
                    if self._peek(1) == "{":
                        self._advance()
                        self._advance()
                        chars.append("{{")
                    else:
                        brace_depth = 1
                        chars.append(self._advance())
                elif ch == "}" and self._peek(1) == "}":
                    self._advance()
                    self._advance()
                    chars.append("}}")
                elif ch == "}":
                    raise ZenSyntaxError("Single '}' is not allowed in f-string", self.line, self.column)
                else:
                    chars.append(self._advance())
        raise ZenSyntaxError("Unterminated f-string literal", start_line, start_col)

    def _scan_ident_or_keyword(self, start_line: int, start_col: int) -> Token:
        chars: list[str] = []
        while self.pos < self.length and (self._peek().isalnum() or self._peek() == "_"):
            chars.append(self._advance())
        word = "".join(chars)
        tok_type = KEYWORD_TOKENS.get(word, TokenType.IDENT)
        if tok_type == TokenType.TRUE:
            return Token(tok_type, True, start_line, start_col)
        if tok_type == TokenType.FALSE:
            return Token(tok_type, False, start_line, start_col)
        if tok_type == TokenType.NONE:
            return Token(tok_type, None, start_line, start_col)
        return Token(tok_type, word, start_line, start_col)

    def tokenize(self) -> list[Token]:
        tokens: list[Token] = []
        while True:
            self._skip_whitespace_and_comments()
            if self.pos >= self.length:
                tokens.append(Token(TokenType.EOF, None, self.line, self.column))
                return tokens

            start_line = self.line
            start_col = self.column
            ch = self._peek()

            if ch == "f" and self._peek(1) == '"':
                tok = self._scan_fstring(start_line, start_col)
                self._last_token_line = tok.line
                tokens.append(tok)
                continue

            if ch.isdigit():
                tok = self._scan_number(start_line, start_col)
                self._last_token_line = tok.line
                tokens.append(tok)
                continue

            if ch == '"':
                tok = self._scan_string(start_line, start_col)
                self._last_token_line = tok.line
                tokens.append(tok)
                continue

            if ch.isalpha() or ch == "_":
                tok = self._scan_ident_or_keyword(start_line, start_col)
                self._last_token_line = tok.line
                tokens.append(tok)
                continue

            two = ch + self._peek(1)
            if two in TWO_CHAR_OPERATORS:
                self._advance()
                self._advance()
                tok = Token(TWO_CHAR_OPERATORS[two], two, start_line, start_col)
                self._last_token_line = tok.line
                tokens.append(tok)
                continue

            if ch in SINGLE_CHAR_TOKENS:
                self._advance()
                tok = Token(SINGLE_CHAR_TOKENS[ch], ch, start_line, start_col)
                self._last_token_line = tok.line
                tokens.append(tok)
                continue

            raise ZenSyntaxError(f"Unexpected character {ch!r}", start_line, start_col)


def tokenize_with_comments(source: str) -> tuple[list[Token], list[tuple[int, str]]]:
    """Lex source code and also return trailing `//` comments that followed code on the same line."""
    lx = Lexer(source)
    return lx.tokenize(), lx.trailing_comments


def tokenize(source: str) -> list[Token]:
    """Convenience helper to lex source code into tokens."""
    return Lexer(source).tokenize()
