"""Tests for Phase 2: Lexer, Parser, and AST."""

import pytest
from zen import ast
from zen.errors import ZenSyntaxError
from zen.lexer import tokenize
from zen.parser import parse
from zen.tokens import TokenType


def test_lexer_tokens_line_and_column() -> None:
    src = 'let x = 42 // comment\nsay "hi\\n"'
    tokens = tokenize(src)
    assert [(t.type, t.value, t.line, t.column) for t in tokens[:-1]] == [
        (TokenType.LET, "let", 1, 1),
        (TokenType.IDENT, "x", 1, 5),
        (TokenType.EQ, "=", 1, 7),
        (TokenType.NUMBER, 42, 1, 9),
        (TokenType.SAY, "say", 2, 1),
        (TokenType.STRING, "hi\n", 2, 5),
    ]
    assert tokens[-1].type == TokenType.EOF


def test_lexer_numbers_operators_and_errors() -> None:
    tokens = tokenize("3.14 ** 2 == 1 != 0 <= >= @")
    types = [t.type for t in tokens[:-1]]
    assert types == [
        TokenType.NUMBER,
        TokenType.POW,
        TokenType.NUMBER,
        TokenType.EQ_EQ,
        TokenType.NUMBER,
        TokenType.BANG_EQ,
        TokenType.NUMBER,
        TokenType.LT_EQ,
        TokenType.GT_EQ,
        TokenType.AT,
    ]
    assert tokens[0].value == 3.14

    with pytest.raises(ZenSyntaxError) as exc:
        tokenize('let s = "unterminated')
    assert exc.value.line == 1

    with pytest.raises(ZenSyntaxError) as exc2:
        tokenize("let a = 1\nlet b = $")
    assert exc2.value.line == 2


def test_operator_precedence() -> None:
    prog = parse("1 + 2 * 3 ** 2")
    expr = prog.statements[0].expression
    # 1 + (2 * (3 ** 2))
    assert isinstance(expr, ast.BinaryExpr) and expr.op == "+"
    assert isinstance(expr.left, ast.Literal) and expr.left.value == 1
    assert isinstance(expr.right, ast.BinaryExpr) and expr.right.op == "*"
    assert isinstance(expr.right.right, ast.BinaryExpr) and expr.right.right.op == "**"

    # Right-associative power & unary minus vs power
    prog2 = parse("-2 ** 2")
    expr2 = prog2.statements[0].expression
    assert isinstance(expr2, ast.UnaryExpr) and expr2.op == "-"
    assert isinstance(expr2.operand, ast.BinaryExpr) and expr2.operand.op == "**"

    prog3 = parse("2 ** 3 ** 2")
    expr3 = prog3.statements[0].expression
    assert isinstance(expr3, ast.BinaryExpr) and expr3.op == "**"
    assert isinstance(expr3.right, ast.BinaryExpr) and expr3.right.op == "**"


def test_statements_control_flow_and_literals() -> None:
    src = """
    const PI = 3.14
    let items = [1, 2, 3]
    let m = {"a": 1, "b": 2}
    fn add(x, y) {
        return x + y
    }
    class Box {
        fn init(self, v) {
            self.v = v
        }
    }
    if true and not false {
        say items[0]
    } elif false {
        say 0
    } else {
        say 1
    }
    for i in items {
        while i > 0 {
            break
        }
    }
    """
    prog = parse(src)
    assert isinstance(prog.statements[0], ast.ConstStmt)
    assert isinstance(prog.statements[1], ast.LetStmt)
    assert isinstance(prog.statements[2], ast.LetStmt)
    assert isinstance(prog.statements[3], ast.FnDecl)
    assert isinstance(prog.statements[4], ast.ClassDecl)
    assert isinstance(prog.statements[5], ast.IfStmt)
    assert len(prog.statements[5].branches) == 2
    assert isinstance(prog.statements[6], ast.ForStmt)


def test_parser_syntax_error_reports_line() -> None:
    with pytest.raises(ZenSyntaxError) as exc:
        parse("let x = 1\nlet y = ")
    assert exc.value.line == 2

    with pytest.raises(ZenSyntaxError) as exc2:
        parse("1 + 2 = 3")
    assert exc2.value.line == 1
