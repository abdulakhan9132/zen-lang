"""Recursive-descent parser for the ZEN language."""

from zen import ast
from zen.errors import ZenSyntaxError
from zen.lexer import tokenize
from zen.tokens import Token, TokenType


class Parser:
    """Parses a list of Tokens into a ZEN AST (ast.Program)."""

    def __init__(self, tokens: list[Token]) -> None:
        self.tokens = tokens
        self.pos = 0

    def _peek(self, offset: int = 0) -> Token:
        idx = self.pos + offset
        if idx >= len(self.tokens):
            return self.tokens[-1]
        return self.tokens[idx]

    def _previous(self) -> Token:
        return self.tokens[self.pos - 1]

    def _is_at_end(self) -> bool:
        return self._peek().type == TokenType.EOF

    def _advance(self) -> Token:
        tok = self._peek()
        if not self._is_at_end():
            self.pos += 1
        return tok

    def _check(self, tok_type: TokenType) -> bool:
        return self._peek().type == tok_type

    def _match(self, *types: TokenType) -> Token | None:
        if self._peek().type in types:
            return self._advance()
        return None

    def _expect(self, tok_type: TokenType, message: str) -> Token:
        tok = self._peek()
        if tok.type == tok_type:
            return self._advance()
        raise ZenSyntaxError(f"{message} (got {tok.type.name} {tok.value!r})", tok.line, tok.column)

    def _consume_optional_semicolon(self) -> None:
        self._match(TokenType.SEMICOLON)

    def parse_program(self) -> ast.Program:
        start = self._peek()
        statements: list[ast.Node] = []
        while not self._is_at_end():
            if self._match(TokenType.SEMICOLON):
                continue
            statements.append(self._parse_statement())
        return ast.Program(start.line, start.column, statements)

    def _parse_block(self) -> ast.Block:
        lbrace = self._expect(TokenType.LBRACE, "Expected '{' to start block")
        statements: list[ast.Node] = []
        while not self._check(TokenType.RBRACE) and not self._is_at_end():
            if self._match(TokenType.SEMICOLON):
                continue
            statements.append(self._parse_statement())
        self._expect(TokenType.RBRACE, "Expected '}' to close block")
        return ast.Block(lbrace.line, lbrace.column, statements)

    def _parse_statement(self) -> ast.Node:
        tok = self._peek()
        if tok.type == TokenType.LET:
            return self._parse_let()
        if tok.type == TokenType.CONST:
            return self._parse_const()
        if tok.type == TokenType.SAY:
            return self._parse_say()
        if tok.type == TokenType.ASSERT:
            return self._parse_assert()
        if tok.type == TokenType.FN and self._peek(1).type == TokenType.IDENT:
            return self._parse_fn_decl()
        if tok.type == TokenType.CLASS:
            return self._parse_class_decl()
        if tok.type == TokenType.IMPORT:
            return self._parse_import()
        if tok.type == TokenType.IF:
            return self._parse_if()
        if tok.type == TokenType.WHILE:
            return self._parse_while()
        if tok.type == TokenType.FOR:
            return self._parse_for()
        if tok.type == TokenType.RETURN:
            return self._parse_return()
        if tok.type == TokenType.BREAK:
            self._advance()
            self._consume_optional_semicolon()
            return ast.BreakStmt(tok.line, tok.column)
        if tok.type == TokenType.CONTINUE:
            self._advance()
            self._consume_optional_semicolon()
            return ast.ContinueStmt(tok.line, tok.column)
        return self._parse_expr_or_assign()

    def _parse_type_name(self) -> str:
        tok = self._peek()
        if tok.type == TokenType.IDENT:
            self._advance()
            return tok.value
        if tok.type == TokenType.NONE:
            self._advance()
            return "none"
        raise ZenSyntaxError("Expected type name", tok.line, tok.column)

    def _parse_let(self) -> ast.LetStmt:
        kw = self._advance()
        name = self._expect(TokenType.IDENT, "Expected variable name after 'let'")
        type_ann = self._parse_type_name() if self._match(TokenType.COLON) else None
        self._expect(TokenType.EQ, "Expected '=' after variable name")
        value = self._parse_expression()
        self._consume_optional_semicolon()
        return ast.LetStmt(kw.line, kw.column, name.value, value, type_ann)

    def _parse_const(self) -> ast.ConstStmt:
        kw = self._advance()
        name = self._expect(TokenType.IDENT, "Expected constant name after 'const'")
        type_ann = self._parse_type_name() if self._match(TokenType.COLON) else None
        self._expect(TokenType.EQ, "Expected '=' after constant name")
        value = self._parse_expression()
        self._consume_optional_semicolon()
        return ast.ConstStmt(kw.line, kw.column, name.value, value, type_ann)

    def _parse_say(self) -> ast.SayStmt:
        kw = self._advance()
        values = [self._parse_expression()]
        while self._match(TokenType.COMMA):
            values.append(self._parse_expression())
        self._consume_optional_semicolon()
        return ast.SayStmt(kw.line, kw.column, values)

    def _parse_assert(self) -> ast.AssertStmt:
        kw = self._advance()
        cond = self._parse_expression()
        message = self._parse_expression() if self._match(TokenType.COMMA) else None
        self._consume_optional_semicolon()
        return ast.AssertStmt(kw.line, kw.column, cond, message)

    def _parse_params(self) -> tuple[list[str], list[str | None], list[ast.Node | None]]:
        self._expect(TokenType.LPAREN, "Expected '(' before parameters")
        params: list[str] = []
        param_types: list[str | None] = []
        defaults: list[ast.Node | None] = []
        seen_default = False

        def _one_param() -> None:
            nonlocal seen_default
            p_tok = self._expect(TokenType.IDENT, "Expected parameter name")
            p_type = self._parse_type_name() if self._match(TokenType.COLON) else None
            p_def: ast.Node | None = None
            if self._match(TokenType.EQ):
                p_def = self._parse_expression()
                seen_default = True
            elif seen_default:
                raise ZenSyntaxError("Non-default parameter follows default parameter", p_tok.line, p_tok.column)
            params.append(p_tok.value)
            param_types.append(p_type)
            defaults.append(p_def)

        if not self._check(TokenType.RPAREN):
            _one_param()
            while self._match(TokenType.COMMA):
                if self._check(TokenType.RPAREN):
                    break
                _one_param()
        self._expect(TokenType.RPAREN, "Expected ')' after parameters")
        return params, param_types, defaults

    def _parse_fn_decl(self) -> ast.FnDecl:
        kw = self._advance()
        name = self._expect(TokenType.IDENT, "Expected function name after 'fn'")
        params, param_types, defaults = self._parse_params()
        ret_type = self._parse_type_name() if self._match(TokenType.ARROW) else None
        body = self._parse_block()
        p_types_opt = param_types if any(t is not None for t in param_types) else None
        defs_opt = defaults if any(d is not None for d in defaults) else None
        return ast.FnDecl(kw.line, kw.column, name.value, params, body, p_types_opt, ret_type, defs_opt)

    def _parse_class_decl(self) -> ast.ClassDecl:
        kw = self._advance()
        name = self._expect(TokenType.IDENT, "Expected class name after 'class'")
        superclass: ast.Node | None = None
        if self._match(TokenType.LPAREN):
            base_tok = self._expect(TokenType.IDENT, "Expected base class name")
            superclass = ast.VarExpr(base_tok.line, base_tok.column, base_tok.value)
            while dot := self._match(TokenType.DOT):
                mem = self._expect(TokenType.IDENT, "Expected base class name after '.'")
                superclass = ast.MemberExpr(dot.line, dot.column, superclass, mem.value)
            self._expect(TokenType.RPAREN, "Expected ')' after base class")
        self._expect(TokenType.LBRACE, "Expected '{' after class name")
        methods: list[ast.FnDecl] = []
        while not self._check(TokenType.RBRACE) and not self._is_at_end():
            if not self._check(TokenType.FN):
                bad = self._peek()
                raise ZenSyntaxError("Expected 'fn' method declaration inside class", bad.line, bad.column)
            methods.append(self._parse_fn_decl())
        self._expect(TokenType.RBRACE, "Expected '}' after class body")
        return ast.ClassDecl(kw.line, kw.column, name.value, methods, superclass)

    def _parse_import(self) -> ast.ImportStmt:
        kw = self._advance()
        if self._check(TokenType.STRING):
            path_tok = self._advance()
            alias: str | None = None
            if self._match(TokenType.AS):
                alias_tok = self._expect(TokenType.IDENT, "Expected alias identifier after 'as'")
                alias = alias_tok.value
            self._consume_optional_semicolon()
            return ast.ImportStmt(kw.line, kw.column, path_tok.value, alias)

        first = self._expect(TokenType.IDENT, "Expected module name or path string after 'import'")
        parts = [first.value]
        while self._match(TokenType.DOT):
            part = self._expect(TokenType.IDENT, "Expected module name after '.'")
            parts.append(part.value)
        if self._match(TokenType.AS):
            alias = self._expect(TokenType.IDENT, "Expected alias identifier after 'as'").value
        else:
            alias = parts[-1]
        self._consume_optional_semicolon()
        return ast.ImportStmt(kw.line, kw.column, "pkg:" + ".".join(parts), alias)

    def _parse_if(self) -> ast.IfStmt:
        kw = self._advance()
        branches: list[tuple[ast.Node, ast.Block]] = []
        cond = self._parse_expression()
        block = self._parse_block()
        branches.append((cond, block))

        while self._match(TokenType.ELIF):
            elif_cond = self._parse_expression()
            elif_block = self._parse_block()
            branches.append((elif_cond, elif_block))

        else_block: ast.Block | None = None
        if self._match(TokenType.ELSE):
            else_block = self._parse_block()

        return ast.IfStmt(kw.line, kw.column, branches, else_block)

    def _parse_while(self) -> ast.WhileStmt:
        kw = self._advance()
        cond = self._parse_expression()
        body = self._parse_block()
        return ast.WhileStmt(kw.line, kw.column, cond, body)

    def _parse_for(self) -> ast.ForStmt:
        kw = self._advance()
        var_tok = self._expect(TokenType.IDENT, "Expected loop variable name after 'for'")
        self._expect(TokenType.IN, "Expected 'in' after loop variable")
        iterable = self._parse_expression()
        body = self._parse_block()
        return ast.ForStmt(kw.line, kw.column, var_tok.value, iterable, body)

    def _parse_return(self) -> ast.ReturnStmt:
        kw = self._advance()
        next_tok = self._peek()
        has_no_value = (
            next_tok.type in (TokenType.RBRACE, TokenType.SEMICOLON, TokenType.EOF)
            or next_tok.line > kw.line
        )
        value = None if has_no_value else self._parse_expression()
        self._consume_optional_semicolon()
        return ast.ReturnStmt(kw.line, kw.column, value)

    def _parse_expr_or_assign(self) -> ast.Node:
        expr = self._parse_expression()
        if eq := self._match(TokenType.EQ):
            if not isinstance(expr, (ast.VarExpr, ast.IndexExpr, ast.MemberExpr)):
                raise ZenSyntaxError("Invalid assignment target", eq.line, eq.column)
            value = self._parse_expression()
            self._consume_optional_semicolon()
            return ast.AssignStmt(expr.line, expr.column, expr, value)
        self._consume_optional_semicolon()
        return ast.ExprStmt(expr.line, expr.column, expr)

    # --- Expressions (precedence climbing) ---

    def _parse_expression(self) -> ast.Node:
        return self._parse_or()

    def _parse_or(self) -> ast.Node:
        expr = self._parse_and()
        while op := self._match(TokenType.OR):
            right = self._parse_and()
            expr = ast.BinaryExpr(op.line, op.column, expr, "or", right)
        return expr

    def _parse_and(self) -> ast.Node:
        expr = self._parse_equality()
        while op := self._match(TokenType.AND):
            right = self._parse_equality()
            expr = ast.BinaryExpr(op.line, op.column, expr, "and", right)
        return expr

    def _parse_equality(self) -> ast.Node:
        expr = self._parse_comparison()
        while op := self._match(TokenType.EQ_EQ, TokenType.BANG_EQ):
            right = self._parse_comparison()
            expr = ast.BinaryExpr(op.line, op.column, expr, op.value, right)
        return expr

    def _parse_comparison(self) -> ast.Node:
        expr = self._parse_term()
        while op := self._match(TokenType.LT, TokenType.LT_EQ, TokenType.GT, TokenType.GT_EQ):
            right = self._parse_term()
            expr = ast.BinaryExpr(op.line, op.column, expr, op.value, right)
        return expr

    def _parse_term(self) -> ast.Node:
        expr = self._parse_factor()
        while op := self._match(TokenType.PLUS, TokenType.MINUS):
            right = self._parse_factor()
            expr = ast.BinaryExpr(op.line, op.column, expr, op.value, right)
        return expr

    def _parse_factor(self) -> ast.Node:
        expr = self._parse_unary()
        while op := self._match(TokenType.STAR, TokenType.SLASH, TokenType.PERCENT, TokenType.AT):
            right = self._parse_unary()
            expr = ast.BinaryExpr(op.line, op.column, expr, op.value, right)
        return expr

    def _parse_unary(self) -> ast.Node:
        if op := self._match(TokenType.MINUS, TokenType.NOT):
            operand = self._parse_unary()
            return ast.UnaryExpr(op.line, op.column, op.value, operand)
        return self._parse_power()

    def _parse_power(self) -> ast.Node:
        expr = self._parse_postfix()
        if op := self._match(TokenType.POW):
            right = self._parse_unary()
            return ast.BinaryExpr(op.line, op.column, expr, "**", right)
        return expr

    def _parse_postfix(self) -> ast.Node:
        expr = self._parse_primary()
        while True:
            if self._check(TokenType.LPAREN) and self._peek().line == self._previous().line:
                lparen = self._advance()
                args: list[ast.Node] = []
                if not self._check(TokenType.RPAREN):
                    args.append(self._parse_expression())
                    while self._match(TokenType.COMMA):
                        if self._check(TokenType.RPAREN):
                            break
                        args.append(self._parse_expression())
                self._expect(TokenType.RPAREN, "Expected ')' after arguments")
                expr = ast.CallExpr(lparen.line, lparen.column, expr, args)
            elif self._check(TokenType.LBRACKET) and self._peek().line == self._previous().line:
                lbrack = self._advance()
                items = [self._parse_subscript_item()]
                while self._match(TokenType.COMMA):
                    if self._check(TokenType.RBRACKET):
                        break
                    items.append(self._parse_subscript_item())
                self._expect(TokenType.RBRACKET, "Expected ']' after index")
                index: ast.Node = items[0] if len(items) == 1 else ast.TupleExpr(lbrack.line, lbrack.column, items)
                expr = ast.IndexExpr(lbrack.line, lbrack.column, expr, index)
            elif dot := self._match(TokenType.DOT):
                member = self._expect(TokenType.IDENT, "Expected member name after '.'")
                expr = ast.MemberExpr(dot.line, dot.column, expr, member.value)
            else:
                break
        return expr

    def _parse_primary(self) -> ast.Node:
        tok = self._peek()
        if tok.type in (TokenType.NUMBER, TokenType.STRING, TokenType.TRUE, TokenType.FALSE, TokenType.NONE):
            self._advance()
            return ast.Literal(tok.line, tok.column, tok.value)
        if tok.type == TokenType.IDENT:
            self._advance()
            return ast.VarExpr(tok.line, tok.column, tok.value)
        if tok.type == TokenType.FN:
            kw = self._advance()
            params, param_types, defaults = self._parse_params()
            ret_type = self._parse_type_name() if self._match(TokenType.ARROW) else None
            body = self._parse_block()
            p_types_opt = param_types if any(t is not None for t in param_types) else None
            defs_opt = defaults if any(d is not None for d in defaults) else None
            return ast.FnExpr(kw.line, kw.column, params, body, p_types_opt, ret_type, defs_opt)
        if tok.type == TokenType.LPAREN:
            self._advance()
            expr = self._parse_expression()
            self._expect(TokenType.RPAREN, "Expected ')' after grouped expression")
            return expr
        if tok.type == TokenType.LBRACKET:
            return self._parse_list()
        if tok.type == TokenType.LBRACE:
            return self._parse_map()
        raise ZenSyntaxError(f"Expected expression, got {tok.type.name} ({tok.value!r})", tok.line, tok.column)

    def _parse_subscript_item(self) -> ast.Node:
        tok = self._peek()
        if self._check(TokenType.COLON):
            start: ast.Node | None = None
        else:
            start = self._parse_expression()
        if self._match(TokenType.COLON):
            stop: ast.Node | None = None
            if not (self._check(TokenType.RBRACKET) or self._check(TokenType.COMMA) or self._check(TokenType.COLON)):
                stop = self._parse_expression()
            step: ast.Node | None = None
            if self._match(TokenType.COLON):
                if not (self._check(TokenType.RBRACKET) or self._check(TokenType.COMMA)):
                    step = self._parse_expression()
            return ast.SliceExpr(tok.line, tok.column, start, stop, step)
        return start

    def _parse_list(self) -> ast.ListExpr:
        lbrack = self._advance()
        elements: list[ast.Node] = []
        if not self._check(TokenType.RBRACKET):
            elements.append(self._parse_expression())
            while self._match(TokenType.COMMA):
                if self._check(TokenType.RBRACKET):
                    break
                elements.append(self._parse_expression())
        self._expect(TokenType.RBRACKET, "Expected ']' after list elements")
        return ast.ListExpr(lbrack.line, lbrack.column, elements)

    def _parse_map(self) -> ast.MapExpr:
        lbrace = self._advance()
        entries: list[tuple[ast.Node, ast.Node]] = []
        if not self._check(TokenType.RBRACE):
            key = self._parse_expression()
            self._expect(TokenType.COLON, "Expected ':' after map key")
            val = self._parse_expression()
            entries.append((key, val))
            while self._match(TokenType.COMMA):
                if self._check(TokenType.RBRACE):
                    break
                key = self._parse_expression()
                self._expect(TokenType.COLON, "Expected ':' after map key")
                val = self._parse_expression()
                entries.append((key, val))
        self._expect(TokenType.RBRACE, "Expected '}' after map literal")
        return ast.MapExpr(lbrace.line, lbrace.column, entries)


def parse(source: str) -> ast.Program:
    """Lex and parse ZEN source code into an AST Program."""
    return Parser(tokenize(source)).parse_program()
