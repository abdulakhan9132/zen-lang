"""Bytecode opcodes and compiled CodeObject structures for the ZEN VM."""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import IntEnum, auto
from typing import Any


class Op(IntEnum):
    # Constants & stack
    LOAD_CONST = auto()
    POP = auto()
    DUP = auto()

    # Globals
    DEFINE_GLOBAL = auto()   # arg: (name, is_const)
    LOAD_GLOBAL = auto()     # arg: name
    STORE_GLOBAL = auto()    # arg: name

    # Locals & cells (closures)
    LOAD_LOCAL = auto()      # arg: slot
    STORE_LOCAL = auto()     # arg: (slot, is_const)
    DEFINE_CELL = auto()     # arg: (slot, is_const)
    LOAD_CELL = auto()       # arg: slot
    STORE_CELL = auto()      # arg: slot
    LOAD_UPVALUE = auto()    # arg: up_idx
    STORE_UPVALUE = auto()   # arg: up_idx

    # Containers
    BUILD_LIST = auto()      # arg: count
    BUILD_MAP = auto()       # arg: count
    GET_INDEX = auto()
    SET_INDEX = auto()
    GET_MEMBER = auto()      # arg: name
    SET_MEMBER = auto()      # arg: name

    # Operations
    UNARY_OP = auto()        # arg: op str ("-" | "not")
    BINARY_OP = auto()       # arg: op str ("+", "-", "*", "/", "%", "**", "@", "==", "!=", "<", "<=", ">", ">=")

    # Control flow
    JUMP = auto()            # arg: target ip
    JUMP_IF_FALSE = auto()   # arg: target ip (pops condition)
    JUMP_IF_FALSE_OR_POP = auto()  # arg: target ip (peeks; pops if truthy)
    JUMP_IF_TRUE_OR_POP = auto()   # arg: target ip (peeks; pops if falsy)
    GET_ITER = auto()
    FOR_ITER = auto()        # arg: target ip (pushes next item, or pops iter & jumps)

    # Functions, classes, modules, output
    MAKE_CLOSURE = auto()    # arg: CodeObject
    CALL = auto()            # arg: argc
    RETURN = auto()
    BUILD_CLASS = auto()     # arg: (class_name, method_names_tuple)
    IMPORT = auto()          # arg: (path, alias)
    SAY = auto()             # arg: count
    ASSERT = auto()          # arg: has_message (bool)


@dataclass
class Instruction:
    op: Op
    arg: Any = None
    line: int = 0
    column: int = 0


@dataclass
class CodeObject:
    name: str
    params: list[str]
    instructions: list[Instruction] = field(default_factory=list)
    num_locals: int = 0
    cell_slots: tuple[bool, ...] = ()
    upvalues: tuple[tuple[bool, int], ...] = ()
    is_module: bool = False
    ops: tuple[int, ...] = ()
    args: tuple[Any, ...] = ()
    lines: tuple[int, ...] = ()
    cols: tuple[int, ...] = ()

    def finalize_flat(self) -> "CodeObject":
        self.ops = tuple(int(i.op) for i in self.instructions)
        self.args = tuple(i.arg for i in self.instructions)
        self.lines = tuple(i.line for i in self.instructions)
        self.cols = tuple(i.column for i in self.instructions)
        return self
