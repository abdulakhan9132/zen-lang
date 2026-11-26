"""Abstract Syntax Tree (AST) node definitions for ZEN."""

from dataclasses import dataclass
from typing import Any


@dataclass(frozen=True)
class Node:
    line: int
    column: int


# --- Statements ---

@dataclass(frozen=True)
class Program(Node):
    statements: list[Node]


@dataclass(frozen=True)
class Block(Node):
    statements: list[Node]


@dataclass(frozen=True)
class LetStmt(Node):
    name: str
    value: Node
    type_ann: str | None = None


@dataclass(frozen=True)
class ConstStmt(Node):
    name: str
    value: Node
    type_ann: str | None = None


@dataclass(frozen=True)
class UnpackStmt(Node):
    names: list[str]
    value: Node
    is_const: bool = False


@dataclass(frozen=True)
class UnpackAssignStmt(Node):
    targets: list[Node]
    value: Node


@dataclass(frozen=True)
class AssignStmt(Node):
    target: Node
    value: Node


@dataclass(frozen=True)
class SayStmt(Node):
    values: list[Node]


@dataclass(frozen=True)
class AssertStmt(Node):
    condition: Node
    message: Node | None = None


@dataclass(frozen=True)
class FnDecl(Node):
    name: str
    params: list[str]
    body: Block
    param_types: list[str | None] | None = None
    return_type: str | None = None
    defaults: list[Node | None] | None = None


@dataclass(frozen=True)
class ClassDecl(Node):
    name: str
    methods: list[FnDecl]
    superclass: Node | None = None


@dataclass(frozen=True)
class ImportStmt(Node):
    path: str
    alias: str | None


@dataclass(frozen=True)
class IfStmt(Node):
    branches: list[tuple[Node, Block]]
    else_block: Block | None


@dataclass(frozen=True)
class WhileStmt(Node):
    condition: Node
    body: Block


@dataclass(frozen=True)
class ForStmt(Node):
    var_name: str
    iterable: Node
    body: Block


@dataclass(frozen=True)
class ReturnStmt(Node):
    value: Node | None


@dataclass(frozen=True)
class BreakStmt(Node):
    pass


@dataclass(frozen=True)
class ContinueStmt(Node):
    pass


@dataclass(frozen=True)
class ExprStmt(Node):
    expression: Node


@dataclass(frozen=True)
class TryStmt(Node):
    try_block: Block
    catch_var: str
    catch_block: Block


@dataclass(frozen=True)
class RaiseStmt(Node):
    value: Node


# --- Expressions ---

@dataclass(frozen=True)
class Literal(Node):
    value: Any


@dataclass(frozen=True)
class FStringExpr(Node):
    parts: list[str | Node]


@dataclass(frozen=True)
class VarExpr(Node):
    name: str


@dataclass(frozen=True)
class ListExpr(Node):
    elements: list[Node]


@dataclass(frozen=True)
class ListCompExpr(Node):
    element: Node
    var_name: str
    iterable: Node
    condition: Node | None = None


@dataclass(frozen=True)
class IfExpr(Node):
    branches: list[tuple[Node, Block]]
    else_block: Block


@dataclass(frozen=True)
class MapExpr(Node):
    entries: list[tuple[Node, Node]]


@dataclass(frozen=True)
class FnExpr(Node):
    params: list[str]
    body: Block
    param_types: list[str | None] | None = None
    return_type: str | None = None
    defaults: list[Node | None] | None = None


@dataclass(frozen=True)
class UnaryExpr(Node):
    op: str
    operand: Node


@dataclass(frozen=True)
class BinaryExpr(Node):
    left: Node
    op: str
    right: Node


@dataclass(frozen=True)
class CallExpr(Node):
    callee: Node
    args: list[Node]
    kwargs: list[tuple[str, Node]] | None = None


@dataclass(frozen=True)
class IndexExpr(Node):
    target: Node
    index: Node


@dataclass(frozen=True)
class SliceExpr(Node):
    start: Node | None
    stop: Node | None
    step: Node | None


@dataclass(frozen=True)
class TupleExpr(Node):
    elements: list[Node]


@dataclass(frozen=True)
class MemberExpr(Node):
    target: Node
    member: str
