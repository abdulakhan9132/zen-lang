"""Token types and Token dataclass for the ZEN lexer."""

from dataclasses import dataclass
from enum import Enum, auto
from typing import Any


class TokenType(Enum):
    # Literals & identifiers
    NUMBER = auto()
    STRING = auto()
    FSTRING = auto()
    IDENT = auto()

    # Keywords
    LET = auto()
    CONST = auto()
    SAY = auto()
    ASSERT = auto()
    FN = auto()
    CLASS = auto()
    RETURN = auto()
    IF = auto()
    ELIF = auto()
    ELSE = auto()
    WHILE = auto()
    FOR = auto()
    IN = auto()
    BREAK = auto()
    CONTINUE = auto()
    AND = auto()
    OR = auto()
    NOT = auto()
    TRUE = auto()
    FALSE = auto()
    NONE = auto()
    IMPORT = auto()
    AS = auto()
    DIV = auto()
    TRY = auto()
    CATCH = auto()
    RAISE = auto()

    # Operators
    PLUS = auto()       # +
    MINUS = auto()      # -
    STAR = auto()       # *
    SLASH = auto()      # /
    PERCENT = auto()    # %
    AT = auto()         # @
    POW = auto()        # **
    EQ = auto()         # =
    EQ_EQ = auto()      # ==
    BANG_EQ = auto()    # !=
    LT = auto()         # <
    LT_EQ = auto()      # <=
    GT = auto()         # >
    GT_EQ = auto()      # >=

    # Delimiters
    LPAREN = auto()     # (
    RPAREN = auto()     # )
    LBRACE = auto()     # {
    RBRACE = auto()     # }
    LBRACKET = auto()   # [
    RBRACKET = auto()   # ]
    COMMA = auto()      # ,
    DOT = auto()        # .
    COLON = auto()      # :
    SEMICOLON = auto()  # ;
    QUESTION = auto()   # ?
    ARROW = auto()      # ->

    EOF = auto()


KEYWORD_TOKENS: dict[str, TokenType] = {
    "let": TokenType.LET,
    "const": TokenType.CONST,
    "say": TokenType.SAY,
    "assert": TokenType.ASSERT,
    "fn": TokenType.FN,
    "class": TokenType.CLASS,
    "return": TokenType.RETURN,
    "if": TokenType.IF,
    "elif": TokenType.ELIF,
    "else": TokenType.ELSE,
    "while": TokenType.WHILE,
    "for": TokenType.FOR,
    "in": TokenType.IN,
    "break": TokenType.BREAK,
    "continue": TokenType.CONTINUE,
    "and": TokenType.AND,
    "or": TokenType.OR,
    "not": TokenType.NOT,
    "true": TokenType.TRUE,
    "false": TokenType.FALSE,
    "none": TokenType.NONE,
    "import": TokenType.IMPORT,
    "as": TokenType.AS,
    "div": TokenType.DIV,
    "try": TokenType.TRY,
    "catch": TokenType.CATCH,
    "raise": TokenType.RAISE,
}

TWO_CHAR_OPERATORS: dict[str, TokenType] = {
    "**": TokenType.POW,
    "==": TokenType.EQ_EQ,
    "!=": TokenType.BANG_EQ,
    "<=": TokenType.LT_EQ,
    ">=": TokenType.GT_EQ,
    "->": TokenType.ARROW,
}

SINGLE_CHAR_TOKENS: dict[str, TokenType] = {
    "+": TokenType.PLUS,
    "-": TokenType.MINUS,
    "*": TokenType.STAR,
    "/": TokenType.SLASH,
    "%": TokenType.PERCENT,
    "@": TokenType.AT,
    "=": TokenType.EQ,
    "<": TokenType.LT,
    ">": TokenType.GT,
    "(": TokenType.LPAREN,
    ")": TokenType.RPAREN,
    "{": TokenType.LBRACE,
    "}": TokenType.RBRACE,
    "[": TokenType.LBRACKET,
    "]": TokenType.RBRACKET,
    ",": TokenType.COMMA,
    ".": TokenType.DOT,
    ":": TokenType.COLON,
    ";": TokenType.SEMICOLON,
    "?": TokenType.QUESTION,
}


@dataclass(frozen=True)
class Token:
    type: TokenType
    value: Any
    line: int
    column: int
