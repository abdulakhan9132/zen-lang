"""Static semantic pass and optional type inference/checker for ZEN."""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
import re
import sys
from typing import Any, TextIO

from zen import ast
from zen.errors import ZenError, ZenTypeError, did_you_mean
from zen.lexer import tokenize_with_comments
from zen.parser import parse
from zen.paths import resolve_import_path
from zen.spec import KEYWORDS


@dataclass
class ZType:
    name: str

    def __str__(self) -> str:
        return self.name


ANY = ZType("any")
INT = ZType("int")
FLOAT = ZType("float")
STRING = ZType("string")
BOOL = ZType("bool")
NONE = ZType("none")
LIST = ZType("list")
MAP = ZType("map")
TENSOR = ZType("tensor")
ARRAY = ZType("array")
SLICE = ZType("slice")
TUPLE = ZType("tuple")

_PRIMITIVE_TYPES: dict[str, ZType] = {
    "any": ANY,
    "int": INT,
    "float": FLOAT,
    "string": STRING,
    "str": STRING,
    "bool": BOOL,
    "none": NONE,
    "null": NONE,
    "list": LIST,
    "map": MAP,
    "tensor": TENSOR,
    "array": ARRAY,
}


@dataclass
class OptionalType(ZType):
    inner: ZType = field(default_factory=lambda: ANY)

    def __str__(self) -> str:
        return f"{self.inner}?"


@dataclass
class FnType(ZType):
    params: tuple[ZType, ...] | None = None
    ret: ZType = field(default_factory=lambda: ANY)
    min_args: int = 0
    max_args: int | None = None
    is_method: bool = False


@dataclass
class ClassType(ZType):
    methods: dict[str, FnType] = field(default_factory=dict)
    fields: dict[str, ZType] = field(default_factory=dict)
    parent: "ClassType | None" = None

    def find_method(self, name: str) -> FnType | None:
        cur: ClassType | None = self
        seen: set[int] = set()
        while cur is not None and id(cur) not in seen:
            seen.add(id(cur))
            if name in cur.methods:
                return cur.methods[name]
            cur = cur.parent
        return None

    def find_field(self, name: str) -> ZType | None:
        cur: ClassType | None = self
        seen: set[int] = set()
        while cur is not None and id(cur) not in seen:
            seen.add(id(cur))
            if name in cur.fields:
                return cur.fields[name]
            cur = cur.parent
        return None


@dataclass
class InstanceType(ZType):
    klass: ClassType = field(default_factory=lambda: ClassType("object"))


@dataclass
class ModuleType(ZType):
    exports: dict[str, "_Binding"] = field(default_factory=dict)


@dataclass
class _Binding:
    ztype: ZType
    is_const: bool = False


class _Scope:
    def __init__(self, parent: "_Scope | None" = None) -> None:
        self.parent = parent
        self.bindings: dict[str, _Binding] = {}

    def define(self, name: str, ztype: ZType, is_const: bool = False) -> None:
        self.bindings[name] = _Binding(ztype, is_const)

    def lookup(self, name: str) -> _Binding | None:
        if name in self.bindings:
            return self.bindings[name]
        if self.parent is not None:
            return self.parent.lookup(name)
        return None

    def all_names(self) -> set[str]:
        names = set(self.bindings.keys())
        if self.parent is not None:
            names.update(self.parent.all_names())
        return names


def _is_numeric(t: ZType) -> bool:
    return t == INT or t == FLOAT


def _is_assignable(target: ZType, value: ZType) -> bool:
    if target is ANY or value is ANY:
        return True
    if isinstance(target, OptionalType):
        return value == NONE or _is_assignable(target.inner, value)
    if isinstance(value, OptionalType):
        return target is ANY
    if target == value:
        return True
    if target == FLOAT and value == INT:
        return True
    return False


def _join_types(t1: ZType, t2: ZType) -> ZType:
    if t1 == t2:
        return t1
    if _is_numeric(t1) and _is_numeric(t2):
        return FLOAT
    return ANY


def _builtin_bindings() -> dict[str, _Binding]:
    def fn(min_a: int, max_a: int | None, ret: ZType = ANY) -> _Binding:
        return _Binding(FnType("fn", None, ret, min_a, max_a), is_const=True)

    return {
        "print": fn(0, None, NONE),
        "len": fn(1, 1, INT),
        "str": fn(1, 1, STRING),
        "type": fn(1, 1, STRING),
        "range": fn(1, 3, LIST),
        "push": fn(2, 2, LIST),
        "append": fn(2, 2, LIST),
        "pop": fn(1, 2, ANY),
        "map": fn(2, 2, LIST),
        "filter": fn(2, 2, LIST),
        "reduce": fn(3, 3, ANY),
        "sqrt": fn(1, 1, ANY),
        "exp": fn(1, 1, ANY),
        "log": fn(1, 2, ANY),
        "sin": fn(1, 1, FLOAT),
        "cos": fn(1, 1, FLOAT),
        "tanh": fn(1, 1, ANY),
        "pow": fn(2, 2, FLOAT),
        "abs": fn(1, 1, ANY),
        "floor": fn(1, 1, INT),
        "floor_div": fn(2, 2, ANY),
        "ceil": fn(1, 1, INT),
        "round": fn(1, 2, FLOAT),
        "min": fn(1, None, ANY),
        "max": fn(1, None, ANY),
        "sum": fn(1, 3, ANY),
        "mean": fn(1, 3, ANY),
        "tensor": fn(1, 1, TENSOR),
        "zeros": fn(1, 1, TENSOR),
        "ones": fn(1, 1, TENSOR),
        "tensor_rand": fn(1, 1, TENSOR),
        "randn": fn(1, 1, TENSOR),
        "shape": fn(1, 1, LIST),
        "reshape": fn(2, 2, TENSOR),
        "transpose": fn(1, 2, TENSOR),
        "matmul": fn(2, 2, ANY),
        "tolist": fn(1, 1, ANY),
        "tensor_sum": fn(1, 3, ANY),
        "tensor_mean": fn(1, 3, ANY),
        "tensor_max": fn(1, 3, ANY),
        "tensor_exp": fn(1, 1, TENSOR),
        "tensor_log": fn(1, 1, TENSOR),
        "tensor_tanh": fn(1, 1, TENSOR),
        "tensor_gt": fn(1, 2, TENSOR),
        "tensor_clamp_min": fn(1, 2, TENSOR),
        "tensor_gather": fn(2, 2, TENSOR),
        "tensor_scatter_add": fn(3, 3, TENSOR),
        "seed": fn(1, 1, NONE),
        "random": fn(0, 0, FLOAT),
        "randint": fn(2, 2, INT),
        "uniform": fn(2, 2, FLOAT),
        "int": fn(1, 1, INT),
        "float": fn(1, 1, FLOAT),
        "split": fn(1, 2, LIST),
        "join": fn(1, 2, STRING),
        "trim": fn(1, 1, STRING),
        "upper": fn(1, 1, STRING),
        "lower": fn(1, 1, STRING),
        "contains": fn(2, 2, BOOL),
        "replace": fn(3, 3, STRING),
        "chars": fn(1, 1, LIST),
        "ord": fn(1, 1, INT),
        "chr": fn(1, 1, STRING),
        "keys": fn(1, 1, LIST),
        "values": fn(1, 1, LIST),
        "slice": fn(2, 3, ANY),
        "read_file": fn(1, 1, STRING),
        "write_file": fn(2, 2, NONE),
        "file_exists": fn(1, 1, BOOL),
        "save_weights": fn(2, 2, NONE),
        "load_weights": fn(1, 1, ANY),
        "clock": fn(0, 0, FLOAT),
        "fields": fn(1, 1, MAP),
        "_zennum_rt": _Binding(ANY, is_const=True),
    }


_DUNDER_MAP: dict[str, tuple[str, str]] = {
    "+": ("__add__", "__radd__"),
    "-": ("__sub__", "__rsub__"),
    "*": ("__mul__", "__rmul__"),
    "/": ("__truediv__", "__rtruediv__"),
    "%": ("__mod__", "__rmod__"),
    "**": ("__pow__", "__rpow__"),
    "@": ("__matmul__", "__rmatmul__"),
}


class TypeChecker:
    """Static semantic & type checker for parsed ZEN ASTs."""

    def __init__(self, base_dir: Path | None = None) -> None:
        self.base_dir = (base_dir or Path.cwd()).resolve()
        self.global_scope = _Scope()
        self._initial_builtins = _builtin_bindings()
        self.global_scope.bindings.update(self._initial_builtins)
        self._module_cache: dict[Path, ModuleType] = {}
        self.warnings: list[str] = []
        self._declared_vars: list[tuple[int, str]] = []

    def check_program(self, program: ast.Program, source: str | None = None) -> None:
        try:
            self._predeclare_decls(program.statements, self.global_scope)
            for stmt in program.statements:
                self._check_stmt(stmt, self.global_scope, expected_ret=None, in_loop=False)
            if source:
                self._scan_comment_warnings(source)
        except ZenError as exc:
            if source:
                raise exc.with_source(source) from exc
            raise

    def _scan_comment_warnings(self, source: str) -> None:
        _, trailing = tokenize_with_comments(source)
        for line_no, comment in trailing:
            if not comment:
                continue
            if re.fullmatch(r"\d+(?:\.\d+)?", comment):
                self.warnings.append(
                    f"Warning [line {line_no}]: '// {comment}' is a comment, not integer division; use 'div' or 'floor_div(...)'."
                )
            elif re.fullmatch(r"[A-Za-z_][A-Za-z0-9_]*", comment):
                in_scope = any(ln <= line_no and nm == comment for ln, nm in self._declared_vars)
                if in_scope:
                    self.warnings.append(
                        f"Warning [line {line_no}]: '// {comment}' is a comment referring to variable '{comment}', not division; use 'div' or 'floor_div(...)'."
                    )

    def _predeclare_decls(self, stmts: list[ast.Node], scope: _Scope) -> None:
        for stmt in stmts:
            if isinstance(stmt, ast.ClassDecl):
                klass = ClassType(stmt.name)
                scope.define(stmt.name, klass, is_const=False)
                for m in stmt.methods:
                    klass.methods[m.name] = self._build_fn_signature(m, scope, is_method=True)
                    self._collect_self_fields(m.body, klass)
            elif isinstance(stmt, ast.FnDecl):
                scope.define(stmt.name, self._build_fn_signature(stmt, scope, is_method=False))

    def _resolve_type_name(self, name: str | None, scope: _Scope, node: ast.Node) -> ZType:
        if name is None:
            return ANY
        if name.endswith("?"):
            inner_name = name[:-1]
            inner_t = self._resolve_type_name(inner_name, scope, node)
            return OptionalType(inner_t.name, inner=inner_t)
        if name in _PRIMITIVE_TYPES:
            return _PRIMITIVE_TYPES[name]
        binding = scope.lookup(name)
        if binding is not None and isinstance(binding.ztype, ClassType):
            return InstanceType(binding.ztype.name, binding.ztype)
        class_names = [k for k in scope.all_names() if isinstance(scope.lookup(k).ztype, ClassType)]  # type: ignore[union-attr]
        hint = did_you_mean(name, list(_PRIMITIVE_TYPES.keys()) + class_names)
        suffix = f". Did you mean '{hint}'?" if hint else ""
        raise ZenTypeError(f"Unknown type '{name}'{suffix}", node.line, node.column, code="E004")

    def _always_returns(self, block: ast.Block) -> bool:
        for stmt in block.statements:
            if isinstance(stmt, ast.ReturnStmt):
                return True
            if isinstance(stmt, ast.IfStmt):
                if stmt.else_block is not None and all(
                    self._always_returns(b) for _, b in stmt.branches
                ) and self._always_returns(stmt.else_block):
                    return True
        return False

    def _check_block(
        self,
        block: ast.Block,
        parent_scope: _Scope,
        expected_ret: ZType | None,
        in_loop: bool,
        *,
        collected_returns: list[ZType] | None = None,
    ) -> _Scope:
        scope = _Scope(parent_scope)
        for stmt in block.statements:
            self._check_stmt(stmt, scope, expected_ret, in_loop, collected_returns=collected_returns)
        return scope

    def _check_stmt(
        self,
        node: ast.Node,
        scope: _Scope,
        expected_ret: ZType | None,
        in_loop: bool,
        *,
        collected_returns: list[ZType] | None = None,
    ) -> None:
        if isinstance(node, (ast.LetStmt, ast.ConstStmt)):
            is_const = isinstance(node, ast.ConstStmt)
            self._declared_vars.append((node.line, node.name))
            val_type = self._check_expr(node.value, scope)
            if node.type_ann is not None:
                ann_type = self._resolve_type_name(node.type_ann, scope, node)
                if not _is_assignable(ann_type, val_type):
                    raise ZenTypeError(
                        f"Type mismatch in '{node.name}': cannot assign '{val_type}' to '{ann_type}'",
                        node.line,
                        node.column,
                        code="E004",
                    )
                scope.define(node.name, ann_type, is_const=is_const)
            else:
                inferred = ANY if (not is_const and val_type == NONE) else val_type
                scope.define(node.name, inferred, is_const=is_const)
            return

        if isinstance(node, ast.AssignStmt):
            val_type = self._check_expr(node.value, scope)
            target = node.target
            if isinstance(target, ast.VarExpr):
                binding = scope.lookup(target.name)
                if binding is None:
                    hint = did_you_mean(target.name, scope.all_names())
                    suffix = f". Did you mean '{hint}'?" if hint else ""
                    raise ZenTypeError(
                        f"Cannot assign to undefined variable '{target.name}'{suffix}",
                        target.line,
                        target.column,
                        code="E002",
                    )
                if binding.is_const:
                    raise ZenTypeError(
                        f"Cannot reassign constant '{target.name}'",
                        target.line,
                        target.column,
                        code="E005",
                    )
                if not _is_assignable(binding.ztype, val_type):
                    raise ZenTypeError(
                        f"Type mismatch in assignment to '{target.name}': expected '{binding.ztype}', got '{val_type}'",
                        target.line,
                        target.column,
                        code="E004",
                    )
                return
            if isinstance(target, ast.IndexExpr):
                self._check_expr(target.target, scope)
                self._check_expr(target.index, scope)
                return
            if isinstance(target, ast.MemberExpr):
                obj_type = self._check_expr(target.target, scope)
                if isinstance(obj_type, InstanceType):
                    obj_type.klass.fields.setdefault(target.member, val_type)
                return
            raise ZenTypeError("Invalid assignment target", target.line, target.column)

        if isinstance(node, ast.SayStmt):
            for v in node.values:
                self._check_expr(v, scope)
            return

        if isinstance(node, ast.AssertStmt):
            self._check_expr(node.condition, scope)
            if node.message is not None:
                self._check_expr(node.message, scope)
            return

        if isinstance(node, ast.FnDecl):
            fn_type = self._build_fn_signature(node, scope, is_method=False)
            scope.define(node.name, fn_type, is_const=False)
            self._check_fn_body(node, fn_type, scope)
            return

        if isinstance(node, ast.ClassDecl):
            existing = scope.lookup(node.name)
            if existing is not None and isinstance(existing.ztype, ClassType):
                klass = existing.ztype
            else:
                klass = ClassType(node.name)
                scope.define(node.name, klass, is_const=False)
            if node.superclass is not None:
                super_t = self._check_expr(node.superclass, scope)
                if isinstance(super_t, ClassType):
                    klass.parent = super_t
                elif super_t is not ANY:
                    raise ZenTypeError(
                        f"Superclass must be a class, got '{super_t}'",
                        node.superclass.line,
                        node.superclass.column,
                    )
            inst_type = InstanceType(node.name, klass)

            for m in node.methods:
                m_type = self._build_fn_signature(m, scope, is_method=True)
                klass.methods[m.name] = m_type
                self._collect_self_fields(m.body, klass)

            for m in node.methods:
                m_type = klass.methods[m.name]
                self._check_fn_body(m, m_type, scope, self_type=inst_type)
            return

        if isinstance(node, ast.ImportStmt):
            self._check_import(node, scope)
            return

        if isinstance(node, ast.IfStmt):
            restores: list[tuple[_Binding, ZType]] = []
            try:
                for cond, block in node.branches:
                    self._check_expr(cond, scope)
                    then_n, else_n = self._extract_narrowings(cond, scope)
                    saved_then = self._apply_narrowings(then_n, scope)
                    try:
                        self._check_block(block, scope, expected_ret, in_loop, collected_returns=collected_returns)
                    finally:
                        self._restore_narrowings(saved_then)
                    restores.extend(self._apply_narrowings(else_n, scope))
                if node.else_block is not None:
                    self._check_block(node.else_block, scope, expected_ret, in_loop, collected_returns=collected_returns)
            finally:
                self._restore_narrowings(restores)
            return

        if isinstance(node, ast.WhileStmt):
            self._check_expr(node.condition, scope)
            self._check_block(node.body, scope, expected_ret, True, collected_returns=collected_returns)
            return

        if isinstance(node, ast.ForStmt):
            self._declared_vars.append((node.line, node.var_name))
            iter_type = self._check_expr(node.iterable, scope)
            if iter_type not in (ANY, LIST, MAP, STRING, ARRAY):
                raise ZenTypeError(
                    f"Cannot iterate over value of type '{iter_type}'",
                    node.iterable.line,
                    node.iterable.column,
                )
            loop_scope = _Scope(scope)
            loop_scope.define(node.var_name, STRING if iter_type == STRING else ANY)
            for s in node.body.statements:
                self._check_stmt(s, loop_scope, expected_ret, True, collected_returns=collected_returns)
            return

        if isinstance(node, ast.ReturnStmt):
            if expected_ret is None:
                raise ZenTypeError("'return' used outside of function", node.line, node.column, code="E011")
            ret_val_type = NONE if node.value is None else self._check_expr(node.value, scope)
            if expected_ret is not ANY and not _is_assignable(expected_ret, ret_val_type):
                raise ZenTypeError(
                    f"Return type mismatch: expected '{expected_ret}', got '{ret_val_type}'",
                    node.line,
                    node.column,
                    code="E004",
                )
            if collected_returns is not None:
                collected_returns.append(ret_val_type)
            return

        if isinstance(node, (ast.BreakStmt, ast.ContinueStmt)):
            if not in_loop:
                kw = "break" if isinstance(node, ast.BreakStmt) else "continue"
                raise ZenTypeError(f"'{kw}' used outside of loop", node.line, node.column, code="E011")
            return

        if isinstance(node, ast.ExprStmt):
            self._check_expr(node.expression, scope)
            return

        raise ZenTypeError(f"Unsupported statement: {type(node).__name__}", node.line, node.column)

    def _extract_narrowings(self, cond: ast.Node, scope: _Scope) -> tuple[dict[str, ZType], dict[str, ZType]]:
        then_n: dict[str, ZType] = {}
        else_n: dict[str, ZType] = {}

        def _non_none(t: ZType) -> ZType:
            return t.inner if isinstance(t, OptionalType) else t

        if isinstance(cond, ast.UnaryExpr) and cond.op == "not":
            t_inner, e_inner = self._extract_narrowings(cond.operand, scope)
            return e_inner, t_inner

        if isinstance(cond, ast.BinaryExpr):
            if cond.op == "and":
                t1, _ = self._extract_narrowings(cond.left, scope)
                t2, _ = self._extract_narrowings(cond.right, scope)
                then_n.update(t1)
                then_n.update(t2)
                return then_n, else_n
            if cond.op in ("!=", "=="):
                var_name: str | None = None
                is_none_cmp = False
                if isinstance(cond.left, ast.VarExpr) and isinstance(cond.right, ast.Literal) and cond.right.value is None:
                    var_name = cond.left.name
                    is_none_cmp = True
                elif isinstance(cond.right, ast.VarExpr) and isinstance(cond.left, ast.Literal) and cond.left.value is None:
                    var_name = cond.right.name
                    is_none_cmp = True
                if is_none_cmp and var_name is not None:
                    b = scope.lookup(var_name)
                    if b is not None and isinstance(b.ztype, OptionalType):
                        if cond.op == "!=":
                            then_n[var_name] = _non_none(b.ztype)
                            else_n[var_name] = NONE
                        else:
                            then_n[var_name] = NONE
                            else_n[var_name] = _non_none(b.ztype)
                        return then_n, else_n

                # `type(x) == "list"` narrowing
                type_var: str | None = None
                type_lit: str | None = None
                if (
                    isinstance(cond.left, ast.CallExpr)
                    and isinstance(cond.left.callee, ast.VarExpr)
                    and cond.left.callee.name == "type"
                    and len(cond.left.args) == 1
                    and isinstance(cond.left.args[0], ast.VarExpr)
                    and isinstance(cond.right, ast.Literal)
                    and isinstance(cond.right.value, str)
                ):
                    type_var = cond.left.args[0].name
                    type_lit = cond.right.value
                elif (
                    isinstance(cond.right, ast.CallExpr)
                    and isinstance(cond.right.callee, ast.VarExpr)
                    and cond.right.callee.name == "type"
                    and len(cond.right.args) == 1
                    and isinstance(cond.right.args[0], ast.VarExpr)
                    and isinstance(cond.left, ast.Literal)
                    and isinstance(cond.left.value, str)
                ):
                    type_var = cond.right.args[0].name
                    type_lit = cond.left.value
                if type_var is not None and type_lit in _PRIMITIVE_TYPES:
                    if cond.op == "==":
                        then_n[type_var] = _PRIMITIVE_TYPES[type_lit]
                    else:
                        else_n[type_var] = _PRIMITIVE_TYPES[type_lit]
        return then_n, else_n

    def _apply_narrowings(self, narrow: dict[str, ZType], scope: _Scope) -> list[tuple[_Binding, ZType]]:
        saved: list[tuple[_Binding, ZType]] = []
        for name, new_t in narrow.items():
            b = scope.lookup(name)
            if b is not None:
                saved.append((b, b.ztype))
                b.ztype = new_t
        return saved

    def _restore_narrowings(self, saved: list[tuple[_Binding, ZType]]) -> None:
        for b, old_t in reversed(saved):
            b.ztype = old_t

    def _collect_self_fields(self, block: ast.Block, klass: ClassType) -> None:
        for stmt in block.statements:
            if isinstance(stmt, ast.AssignStmt) and isinstance(stmt.target, ast.MemberExpr):
                if isinstance(stmt.target.target, ast.VarExpr) and stmt.target.target.name == "self":
                    klass.fields.setdefault(stmt.target.member, ANY)
            elif isinstance(stmt, ast.IfStmt):
                for _, b in stmt.branches:
                    self._collect_self_fields(b, klass)
                if stmt.else_block is not None:
                    self._collect_self_fields(stmt.else_block, klass)
            elif isinstance(stmt, (ast.WhileStmt, ast.ForStmt)):
                self._collect_self_fields(stmt.body, klass)

    def _build_fn_signature(self, node: ast.FnDecl | ast.FnExpr, scope: _Scope, *, is_method: bool) -> FnType:
        p_types: list[ZType] = []
        raw_pt = node.param_types or [None] * len(node.params)
        for i, pt in enumerate(raw_pt):
            if is_method and i == 0 and pt is None:
                p_types.append(ANY)
            else:
                p_types.append(self._resolve_type_name(pt, scope, node))
        if node.defaults:
            for d in node.defaults:
                if d is not None:
                    self._check_expr(d, scope)
        ret_t = self._resolve_type_name(node.return_type, scope, node) if node.return_type is not None else ANY
        max_arity = len(node.params) - 1 if is_method and node.params else len(node.params)
        num_defaults = sum(1 for d in (node.defaults or []) if d is not None)
        min_arity = max(0, max_arity - num_defaults)
        fn_name = node.name if isinstance(node, ast.FnDecl) else "<lambda>"
        return FnType(fn_name, tuple(p_types), ret_t, min_arity, max_arity, is_method=is_method)

    def _check_fn_body(
        self,
        node: ast.FnDecl | ast.FnExpr,
        fn_type: FnType,
        parent_scope: _Scope,
        *,
        self_type: InstanceType | None = None,
    ) -> FnType:
        fn_scope = _Scope(parent_scope)
        assert fn_type.params is not None
        for i, p_name in enumerate(node.params):
            self._declared_vars.append((node.line, p_name))
            if i == 0 and self_type is not None:
                fn_scope.define(p_name, self_type)
            else:
                fn_scope.define(p_name, fn_type.params[i])

        declared_ret = fn_type.ret if node.return_type is not None else ANY
        collected: list[ZType] = []
        for s in node.body.statements:
            self._check_stmt(s, fn_scope, declared_ret, False, collected_returns=collected)

        if node.return_type is not None and declared_ret not in (ANY, NONE):
            if not self._always_returns(node.body):
                raise ZenTypeError(
                    f"Function '{fn_type.name}' declares return type '{declared_ret}' but may not return a value",
                    node.line,
                    node.column,
                )

        if node.return_type is None and isinstance(node, ast.FnDecl):
            # Infer return type only when safe (single consistent return type)
            if not collected:
                inferred = NONE
            else:
                inferred = collected[0]
                for rt in collected[1:]:
                    inferred = _join_types(inferred, rt)
                if not self._always_returns(node.body) and inferred != NONE:
                    inferred = ANY
            # Update signature in place if not a method with dynamic fields
            updated = FnType(
                fn_type.name,
                fn_type.params,
                inferred,
                fn_type.min_args,
                fn_type.max_args,
                is_method=fn_type.is_method,
            )
            if self_type is not None:
                self_type.klass.methods[node.name] = updated
            else:
                binding = parent_scope.bindings.get(node.name)
                if binding is not None and isinstance(binding.ztype, FnType):
                    binding.ztype = updated
            return updated
        return fn_type

    def _check_import(self, node: ast.ImportStmt, scope: _Scope) -> None:
        target = resolve_import_path(node.path, self.base_dir)
        if target is None:
            raise ZenTypeError(f"Imported file not found: '{node.path}'", node.line, node.column, code="E009")

        if target in self._module_cache:
            mod = self._module_cache[target]
        else:
            mod_src = target.read_text(encoding="utf-8")
            mod_prog = parse(mod_src)
            mod_checker = TypeChecker(base_dir=target.parent)
            mod_checker._module_cache = self._module_cache
            mod_checker.check_program(mod_prog, source=mod_src)
            exports = {
                k: b
                for k, b in mod_checker.global_scope.bindings.items()
                if b is not mod_checker._initial_builtins.get(k)
            }
            mod_name = target.parent.name if target.stem == "__init__" else target.stem
            mod = ModuleType(mod_name, exports)
            self._module_cache[target] = mod

        if node.alias:
            scope.define(node.alias, mod)
        else:
            for k, b in mod.exports.items():
                scope.bindings[k] = b

    def _check_expr(self, node: ast.Node, scope: _Scope) -> ZType:
        if isinstance(node, ast.Literal):
            v = node.value
            if v is None:
                return NONE
            if isinstance(v, bool):
                return BOOL
            if isinstance(v, int):
                return INT
            if isinstance(v, float):
                return FLOAT
            if isinstance(v, str):
                for m in re.finditer(r"\{([A-Za-z_][A-Za-z0-9_]*)\}", v):
                    var_name = m.group(1)
                    if scope.lookup(var_name) is not None:
                        self.warnings.append(
                            f"Warning [line {node.line}]: plain string contains '{{{var_name}}}' matching variable '{var_name}' in scope; did you mean f\"...\"?"
                        )
                return STRING
            return ANY

        if isinstance(node, ast.FStringExpr):
            for part in node.parts:
                if not isinstance(part, str):
                    self._check_expr(part, scope)
            return STRING

        if isinstance(node, ast.VarExpr):
            binding = scope.lookup(node.name)
            if binding is None:
                hint = did_you_mean(node.name, list(scope.all_names()) + list(KEYWORDS))
                suffix = f". Did you mean '{hint}'?" if hint else ""
                raise ZenTypeError(f"Undefined variable '{node.name}'{suffix}", node.line, node.column, code="E002")
            return binding.ztype

        if isinstance(node, ast.ListExpr):
            for el in node.elements:
                self._check_expr(el, scope)
            return LIST

        if isinstance(node, ast.MapExpr):
            for k, v in node.entries:
                kt = self._check_expr(k, scope)
                if kt in (LIST, MAP):
                    raise ZenTypeError("Map key cannot be a list or map", k.line, k.column)
                self._check_expr(v, scope)
            return MAP

        if isinstance(node, ast.FnExpr):
            fn_type = self._build_fn_signature(node, scope, is_method=False)
            return self._check_fn_body(node, fn_type, scope)

        if isinstance(node, ast.UnaryExpr):
            op_t = self._check_expr(node.operand, scope)
            if node.op == "not":
                return BOOL
            if node.op == "-":
                if op_t in (ANY, INT, FLOAT, TENSOR, ARRAY):
                    return op_t
                if isinstance(op_t, InstanceType) and "__neg__" in op_t.klass.methods:
                    return op_t.klass.methods["__neg__"].ret
                raise ZenTypeError(f"Cannot negate value of type '{op_t}'", node.line, node.column)
            return ANY

        if isinstance(node, ast.BinaryExpr):
            return self._check_binary(node, scope)

        if isinstance(node, ast.CallExpr):
            callee_t = self._check_expr(node.callee, scope)
            arg_types = [self._check_expr(a, scope) for a in node.args]
            return self._check_call(callee_t, arg_types, node)

        if isinstance(node, ast.SliceExpr):
            if node.start is not None:
                self._check_expr(node.start, scope)
            if node.stop is not None:
                self._check_expr(node.stop, scope)
            if node.step is not None:
                self._check_expr(node.step, scope)
            return SLICE

        if isinstance(node, ast.TupleExpr):
            for el in node.elements:
                self._check_expr(el, scope)
            return TUPLE

        if isinstance(node, ast.IndexExpr):
            target_t = self._check_expr(node.target, scope)
            idx_t = self._check_expr(node.index, scope)
            if isinstance(target_t, InstanceType):
                get_m = target_t.klass.find_method("__getitem__")
                if get_m is not None:
                    return get_m.ret
                raise ZenTypeError(f"Cannot index into type '{target_t}'", node.line, node.column)
            if target_t in (LIST, STRING) and idx_t not in (ANY, INT, SLICE):
                raise ZenTypeError(f"Index must be 'int', got '{idx_t}'", node.index.line, node.index.column)
            if target_t == TENSOR and idx_t not in (ANY, INT):
                raise ZenTypeError(f"Index must be 'int', got '{idx_t}'", node.index.line, node.index.column)
            if target_t not in (ANY, LIST, MAP, STRING, TENSOR, ARRAY):
                raise ZenTypeError(f"Cannot index into type '{target_t}'", node.line, node.column)
            return STRING if target_t == STRING else (LIST if target_t == LIST and idx_t == SLICE else (ARRAY if target_t == ARRAY else ANY))

        if isinstance(node, ast.MemberExpr):
            obj_t = self._check_expr(node.target, scope)
            if obj_t is ANY:
                return ANY
            if isinstance(obj_t, ModuleType):
                if node.member not in obj_t.exports:
                    hint = did_you_mean(node.member, obj_t.exports.keys())
                    suffix = f". Did you mean '{hint}'?" if hint else ""
                    raise ZenTypeError(
                        f"Module '{obj_t.name}' has no member '{node.member}'{suffix}",
                        node.line,
                        node.column,
                        code="E003",
                    )
                return obj_t.exports[node.member].ztype
            if isinstance(obj_t, ClassType):
                m = obj_t.find_method(node.member)
                if m is not None:
                    extra = 1 if m.is_method and m.params else 0
                    return FnType(
                        m.name,
                        m.params,
                        m.ret,
                        m.min_args + extra,
                        None if m.max_args is None else m.max_args + extra,
                        is_method=False,
                    )
                hint = did_you_mean(node.member, obj_t.methods.keys())
                suffix = f". Did you mean '{hint}'?" if hint else ""
                raise ZenTypeError(
                    f"Class '{obj_t.name}' has no method '{node.member}'{suffix}",
                    node.line,
                    node.column,
                    code="E003",
                )
            if isinstance(obj_t, InstanceType):
                m = obj_t.klass.find_method(node.member)
                if m is not None:
                    return m
                f = obj_t.klass.find_field(node.member)
                if f is not None:
                    return f
                hint = did_you_mean(
                    node.member,
                    list(obj_t.klass.methods.keys()) + list(obj_t.klass.fields.keys()),
                )
                suffix = f". Did you mean '{hint}'?" if hint else ""
                raise ZenTypeError(
                    f"'{obj_t.klass.name}' instance has no field or method '{node.member}'{suffix}",
                    node.line,
                    node.column,
                    code="E003",
                )
            if obj_t == ARRAY:
                array_members: dict[str, ZType] = {
                    "shape": LIST,
                    "ndim": INT,
                    "size": INT,
                    "dtype": STRING,
                    "T": ARRAY,
                    "reshape": FnType("reshape", None, ARRAY, 1, None),
                    "transpose": FnType("transpose", None, ARRAY, 0, None),
                    "sum": FnType("sum", None, ANY, 0, 2),
                    "mean": FnType("mean", None, ANY, 0, 2),
                    "std": FnType("std", None, ANY, 0, 2),
                    "var": FnType("var", None, ANY, 0, 2),
                    "min": FnType("min", None, ANY, 0, 2),
                    "max": FnType("max", None, ANY, 0, 2),
                    "argmax": FnType("argmax", None, ANY, 0, 1),
                    "argmin": FnType("argmin", None, ANY, 0, 1),
                    "tolist": FnType("tolist", None, ANY, 0, 0),
                    "copy": FnType("copy", None, ARRAY, 0, 0),
                    "astype": FnType("astype", None, ARRAY, 1, 1),
                    "clip": FnType("clip", None, ARRAY, 2, 2),
                    "flatten": FnType("flatten", None, ARRAY, 0, 0),
                    "squeeze": FnType("squeeze", None, ARRAY, 0, 1),
                    "unsqueeze": FnType("unsqueeze", None, ARRAY, 1, 1),
                }
                if node.member not in array_members:
                    raise ZenTypeError(f"Array has no member '{node.member}'", node.line, node.column)
                return array_members[node.member]
            if obj_t == TENSOR:
                tensor_members: dict[str, ZType] = {
                    "shape": LIST,
                    "T": TENSOR,
                    "reshape": FnType("reshape", None, TENSOR, 1, 1),
                    "transpose": FnType("transpose", None, TENSOR, 0, 1),
                    "sum": FnType("sum", None, ANY, 0, 2),
                    "mean": FnType("mean", None, ANY, 0, 2),
                    "max": FnType("max", None, ANY, 0, 2),
                    "tolist": FnType("tolist", None, ANY, 0, 0),
                    "exp": FnType("exp", None, TENSOR, 0, 0),
                    "log": FnType("log", None, TENSOR, 0, 0),
                    "tanh": FnType("tanh", None, TENSOR, 0, 0),
                }
                if node.member not in tensor_members:
                    raise ZenTypeError(f"Tensor has no member '{node.member}'", node.line, node.column)
                return tensor_members[node.member]
            raise ZenTypeError(
                f"Cannot access member '{node.member}' on type '{obj_t}'",
                node.line,
                node.column,
            )

        return ANY

    def _check_call(self, callee_t: ZType, arg_types: list[ZType], node: ast.CallExpr) -> ZType:
        if callee_t is ANY:
            return ANY
        if isinstance(callee_t, ClassType):
            init_m = callee_t.find_method("init")
            if init_m is not None:
                self._check_fn_args(init_m, arg_types, node)
            elif arg_types:
                raise ZenTypeError(
                    f"Class '{callee_t.name}' takes no constructor arguments, got {len(arg_types)}",
                    node.line,
                    node.column,
                )
            return InstanceType(callee_t.name, callee_t)
        if isinstance(callee_t, FnType):
            self._check_fn_args(callee_t, arg_types, node)
            return callee_t.ret
        if isinstance(callee_t, InstanceType):
            call_m = callee_t.klass.find_method("__call__")
            if call_m is not None:
                self._check_fn_args(call_m, arg_types, node)
                return call_m.ret
            raise ZenTypeError(f"Value of type '{callee_t}' is not callable", node.line, node.column)
        raise ZenTypeError(f"Value of type '{callee_t}' is not callable", node.line, node.column)

    def _check_fn_args(self, fn_t: FnType, arg_types: list[ZType], node: ast.CallExpr) -> None:
        argc = len(arg_types)
        if argc < fn_t.min_args or (fn_t.max_args is not None and argc > fn_t.max_args):
            if fn_t.max_args == fn_t.min_args:
                exp = f"{fn_t.min_args}"
            elif fn_t.max_args is None:
                exp = f"at least {fn_t.min_args}"
            else:
                exp = f"{fn_t.min_args} to {fn_t.max_args}"
            raise ZenTypeError(
                f"Function '{fn_t.name}' expected {exp} argument(s), got {argc}",
                node.line,
                node.column,
            )
        if fn_t.params is not None:
            offset = 1 if fn_t.is_method and fn_t.params else 0
            for i, arg_t in enumerate(arg_types):
                param_t = fn_t.params[offset + i]
                if not _is_assignable(param_t, arg_t):
                    raise ZenTypeError(
                        f"Argument {i + 1} of '{fn_t.name}' expected '{param_t}', got '{arg_t}'",
                        node.args[i].line,
                        node.args[i].column,
                    )

    def _check_binary(self, node: ast.BinaryExpr, scope: _Scope) -> ZType:
        lt = self._check_expr(node.left, scope)
        rt = self._check_expr(node.right, scope)
        op = node.op

        if op in ("==", "!="):
            return BOOL
        if op in ("and", "or"):
            return lt if lt == rt else ANY

        if op in _DUNDER_MAP:
            d, rd = _DUNDER_MAP[op]
            if isinstance(lt, InstanceType) and d in lt.klass.methods:
                return lt.klass.methods[d].ret
            if isinstance(rt, InstanceType) and rd in rt.klass.methods:
                return rt.klass.methods[rd].ret

        if lt is ANY or rt is ANY:
            if op in ("<", "<=", ">", ">="):
                return BOOL
            return ANY

        if op == "+":
            if lt == INT and rt == INT:
                return INT
            if _is_numeric(lt) and _is_numeric(rt):
                return FLOAT
            if lt == STRING and rt == STRING:
                return STRING
            if lt == LIST and rt == LIST:
                return LIST
            if (lt == ARRAY and (rt == ARRAY or _is_numeric(rt))) or (rt == ARRAY and _is_numeric(lt)):
                return ARRAY
            if (lt == TENSOR and (rt == TENSOR or _is_numeric(rt))) or (rt == TENSOR and _is_numeric(lt)):
                return TENSOR
            raise ZenTypeError(f"Unsupported '+' between '{lt}' and '{rt}'", node.line, node.column)

        if op in ("-", "*", "/", "div", "%", "**"):
            if (lt == ARRAY and (rt == ARRAY or _is_numeric(rt))) or (rt == ARRAY and _is_numeric(lt)):
                return ARRAY
            if op in ("-", "*", "/") and (
                (lt == TENSOR and (rt == TENSOR or _is_numeric(rt)))
                or (rt == TENSOR and _is_numeric(lt))
            ):
                return TENSOR
            if _is_numeric(lt) and _is_numeric(rt):
                if op == "/":
                    return FLOAT
                if op == "div":
                    return INT if (lt == INT and rt == INT) else FLOAT
                return INT if (lt == INT and rt == INT) else FLOAT
            raise ZenTypeError(
                f"Operator '{op}' not supported between '{lt}' and '{rt}'",
                node.line,
                node.column,
                code="E004",
            )

        if op in ("<", "<=", ">", ">="):
            if (lt == ARRAY and (rt == ARRAY or _is_numeric(rt))) or (rt == ARRAY and _is_numeric(lt)):
                return ARRAY
            if (_is_numeric(lt) and _is_numeric(rt)) or (lt == STRING and rt == STRING):
                return BOOL
            raise ZenTypeError(
                f"Comparison '{op}' not supported between '{lt}' and '{rt}'",
                node.line,
                node.column,
            )

        if op == "@":
            if lt == ARRAY and rt == ARRAY:
                return ARRAY
            if lt == TENSOR and rt == TENSOR:
                return ANY
            raise ZenTypeError(
                f"Matrix multiply '@' requires tensors, got '{lt}' and '{rt}'",
                node.line,
                node.column,
            )

        return ANY


def type_check_program(
    program: ast.Program,
    base_dir: Path | None = None,
    source: str | None = None,
    *,
    stderr: TextIO | None = None,
) -> list[str]:
    """Run static type & semantic checks on a parsed ZEN AST, returning any warnings."""
    checker = TypeChecker(base_dir=base_dir)
    checker.check_program(program, source=source)
    out = stderr if stderr is not None else sys.stderr
    for w in checker.warnings:
        out.write(w + "\n")
    return checker.warnings


def type_check_source(
    source: str,
    file_path: Path | None = None,
    *,
    stderr: TextIO | None = None,
) -> ast.Program:
    """Parse and type-check ZEN source text, returning the AST."""
    try:
        program = parse(source)
        base_dir = file_path.resolve().parent if file_path is not None else Path.cwd()
        type_check_program(program, base_dir=base_dir, source=source, stderr=stderr)
        return program
    except ZenError as exc:
        raise exc.with_source(source) from exc
