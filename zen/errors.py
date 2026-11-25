"""Custom error types for the ZEN language."""


def format_source_snippet(source: str | None, line: int, column: int) -> str:
    """Render the offending source line with a caret pointing to the column."""
    if not source or line <= 0:
        return ""
    lines = source.splitlines()
    if line > len(lines):
        return ""
    text = lines[line - 1]
    gutter = f"{line:4d} | "
    col = max(1, column)
    pointer = " " * (len(gutter) + col - 1) + "^"
    return f"\n{gutter}{text}\n{pointer}"


class ZenError(Exception):
    """Base error for all ZEN compile-time and runtime errors."""

    def __init__(
        self,
        message: str,
        line: int = 0,
        column: int = 0,
        source: str | None = None,
    ) -> None:
        self.message = message
        self.line = line
        self.column = column
        self.source = source
        location = f"[line {line}, col {column}] " if line > 0 else ""
        snippet = format_source_snippet(source, line, column)
        super().__init__(f"{location}{message}{snippet}")

    def with_source(self, source: str) -> "ZenError":
        if self.source:
            return self
        return type(self)(self.message, self.line, self.column, source)


class ZenSyntaxError(ZenError):
    """Raised by the lexer or parser when source code is malformed."""


class ZenTypeError(ZenError):
    """Raised by the semantic type checker before execution."""


class ZenRuntimeError(ZenError):
    """Raised by the interpreter or VM during program execution."""
