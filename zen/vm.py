"""Stack-based bytecode Virtual Machine for the ZEN language."""

from __future__ import annotations

import io
import re
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from zen import ast
from zen import nan_trace
from zen.builtins import register_phase4_builtins
from zen.bytecode import CodeObject, Op
from zen.compiler import compile_program
from zen.environment import Environment
from zen.errors import ZenError, ZenRuntimeError, did_you_mean
from zen.recorder import record_zen_access
from zen.interpreter import (
    ZenBuiltin,
    ZenClass,
    ZenInstance,
    ZenModule,
    _is_number,
    bind_call_args,
    is_truthy,
    list_member,
    make_error_instance,
    str_member,
    zen_format,
    zen_type,
)
from zen.lexer import tokenize_with_comments
from zen.parser import parse
from zen.paths import resolve_import_path
from zen.tensor import ZenTensor
from zen.typechecker import type_check_program
from zen.zennum_core import ZenArray


@dataclass
class _Cell:
    value: Any
    is_const: bool = False


class VMClosure:
    """Compiled ZEN function with captured closure cells and defining module globals."""

    def __init__(
        self,
        code: CodeObject,
        upvalues: tuple[_Cell, ...],
        globals_env: Environment,
        defaults: tuple[Any, ...] = (),
    ) -> None:
        self.code = code
        self.name = code.name
        self.upvalues = upvalues
        self.globals_env = globals_env
        self.defaults = defaults

    def call(
        self,
        vm: "VM",
        args: list[Any],
        line: int,
        col: int,
        bound_self: ZenInstance | None = None,
        kwargs: list[tuple[str, Any]] | None = None,
    ) -> Any:
        actual_args = bind_call_args(
            self.name, self.code.params, self.defaults, args, kwargs, bound_self, line, col
        )
        return vm._execute_code(self.code, self.upvalues, actual_args, self.globals_env)


@dataclass
class VMBoundMethod:
    instance: ZenInstance
    method: VMClosure

    @property
    def name(self) -> str:
        return self.method.name

    def call(
        self,
        vm: "VM",
        args: list[Any],
        line: int,
        col: int,
        kwargs: list[tuple[str, Any]] | None = None,
    ) -> Any:
        return self.method.call(vm, args, line, col, bound_self=self.instance, kwargs=kwargs)


class VMClass(ZenClass):
    """Class representation using VMClosures for methods."""

    def __init__(
        self,
        name: str,
        methods: dict[str, VMClosure],
        parent: ZenClass | None = None,
    ) -> None:  # type: ignore[override]
        self.name = name
        self.methods = methods  # type: ignore[assignment]
        self.parent = parent

    def call(
        self,
        vm: "VM",
        args: list[Any],
        line: int,
        col: int,
        kwargs: list[tuple[str, Any]] | None = None,
    ) -> ZenInstance:  # type: ignore[override]
        instance = _VMInstance(self)
        init_method = self.find_method("init")
        if init_method is not None:
            init_method.call(vm, args, line, col, bound_self=instance, kwargs=kwargs)  # type: ignore[arg-type]
        elif args or kwargs:
            raise ZenRuntimeError(
                f"Class '{self.name}' takes no constructor arguments, got {len(args) + len(kwargs or [])}",
                line,
                col,
                code="E006",
            )
        return instance


class _VMInstance(ZenInstance):
    def get(self, name: str, line: int, col: int) -> Any:
        if name in self.fields:
            return self.fields[name]
        method = self.klass.find_method(name)
        if method is not None:
            return VMBoundMethod(self, method)  # type: ignore[arg-type]
        hint = did_you_mean(name, list(self.fields.keys()) + list(self.klass.methods.keys()))
        suffix = f". Did you mean '{hint}'?" if hint else ""
        raise ZenRuntimeError(
            f"'{self.klass.name}' instance has no field or method '{name}'{suffix}",
            line,
            col,
            code="E003",
        )


class _VMFrame:
    __slots__ = ("code", "upvalues", "locals_arr", "globals_env", "stack", "ip", "try_stack")

    def __init__(
        self,
        code: CodeObject,
        upvalues: tuple[_Cell, ...],
        args: list[Any],
        globals_env: Environment,
    ) -> None:
        self.code = code
        self.upvalues = upvalues
        locals_arr: list[Any] = [None] * code.num_locals
        cell_slots = code.cell_slots
        for i, arg_val in enumerate(args):
            locals_arr[i] = _Cell(arg_val, False) if cell_slots[i] else arg_val
        self.locals_arr = locals_arr
        self.globals_env = globals_env
        self.stack: list[Any] = []
        self.ip = 0
        self.try_stack: list[tuple[int, int]] = []


def _vm_class_call(
    self: VMClass,
    vm: "VM",
    args: list[Any],
    line: int,
    col: int,
    kwargs: list[tuple[str, Any]] | None = None,
) -> ZenInstance:
    instance = _VMInstance(self)
    init_method = self.find_method("init")
    if init_method is not None:
        init_method.call(vm, args, line, col, bound_self=instance, kwargs=kwargs)  # type: ignore[arg-type]
    elif args or kwargs:
        raise ZenRuntimeError(
            f"Class '{self.name}' takes no constructor arguments, got {len(args) + len(kwargs or [])}",
            line,
            col,
            code="E006",
        )
    return instance


VMClass.call = _vm_class_call  # type: ignore[method-assign]


_DUNDER_MAP = {
    "+": ("__add__", "__radd__"),
    "-": ("__sub__", "__rsub__"),
    "*": ("__mul__", "__rmul__"),
    "/": ("__truediv__", "__rtruediv__"),
    "**": ("__pow__", "__rpow__"),
    "@": ("__matmul__", "__rmatmul__"),
}


_OP_LOAD_CONST = int(Op.LOAD_CONST)
_OP_POP = int(Op.POP)
_OP_DUP = int(Op.DUP)
_OP_DEFINE_GLOBAL = int(Op.DEFINE_GLOBAL)
_OP_LOAD_GLOBAL = int(Op.LOAD_GLOBAL)
_OP_STORE_GLOBAL = int(Op.STORE_GLOBAL)
_OP_LOAD_LOCAL = int(Op.LOAD_LOCAL)
_OP_STORE_LOCAL = int(Op.STORE_LOCAL)
_OP_DEFINE_CELL = int(Op.DEFINE_CELL)
_OP_LOAD_CELL = int(Op.LOAD_CELL)
_OP_STORE_CELL = int(Op.STORE_CELL)
_OP_LOAD_UPVALUE = int(Op.LOAD_UPVALUE)
_OP_STORE_UPVALUE = int(Op.STORE_UPVALUE)
_OP_BUILD_LIST = int(Op.BUILD_LIST)
_OP_BUILD_MAP = int(Op.BUILD_MAP)
_OP_BUILD_TUPLE = int(Op.BUILD_TUPLE)
_OP_BUILD_FSTRING = int(Op.BUILD_FSTRING)
_OP_BUILD_SLICE = int(Op.BUILD_SLICE)
_OP_LIST_APPEND = int(Op.LIST_APPEND)
_OP_UNPACK_SEQ = int(Op.UNPACK_SEQ)
_OP_GET_INDEX = int(Op.GET_INDEX)
_OP_SET_INDEX = int(Op.SET_INDEX)
_OP_GET_MEMBER = int(Op.GET_MEMBER)
_OP_SET_MEMBER = int(Op.SET_MEMBER)
_OP_UNARY_OP = int(Op.UNARY_OP)
_OP_BINARY_OP = int(Op.BINARY_OP)
_OP_JUMP = int(Op.JUMP)
_OP_JUMP_IF_FALSE = int(Op.JUMP_IF_FALSE)
_OP_JUMP_IF_FALSE_OR_POP = int(Op.JUMP_IF_FALSE_OR_POP)
_OP_JUMP_IF_TRUE_OR_POP = int(Op.JUMP_IF_TRUE_OR_POP)
_OP_GET_ITER = int(Op.GET_ITER)
_OP_FOR_ITER = int(Op.FOR_ITER)
_OP_MAKE_CLOSURE = int(Op.MAKE_CLOSURE)
_OP_CALL = int(Op.CALL)
_OP_CALL_KW = int(Op.CALL_KW)
_OP_RETURN = int(Op.RETURN)
_OP_PUSH_TRY = int(Op.PUSH_TRY)
_OP_POP_TRY = int(Op.POP_TRY)
_OP_RAISE = int(Op.RAISE)
_OP_BUILD_CLASS = int(Op.BUILD_CLASS)
_OP_IMPORT = int(Op.IMPORT)
_OP_SAY = int(Op.SAY)
_OP_ASSERT = int(Op.ASSERT)
_NUM_TYPES = (int, float)


class VM:
    """Bytecode Virtual Machine executing compiled ZEN CodeObjects."""

    def __init__(
        self,
        stdout: io.TextIOBase | None = None,
        base_dir: Path | None = None,
        *,
        stderr: io.TextIOBase | None = None,
        stdin: io.TextIOBase | None = None,
        max_depth: int = 10000,
    ) -> None:
        self.stdout = stdout if stdout is not None else sys.stdout
        self.stderr = stderr if stderr is not None else sys.stderr
        self.stdin = stdin if stdin is not None else sys.stdin
        self.max_depth = max(1, int(max_depth))
        self._frame_count = 0
        self.current_line = 0
        self.base_dir = (base_dir or Path.cwd()).resolve()
        self.globals = Environment()
        self._module_cache: dict[Path, ZenModule] = {}
        self._register_builtins()

    def _register_builtins(self) -> None:
        self.globals.define("print", ZenBuiltin("print", self._builtin_print))
        self.globals.define("len", ZenBuiltin("len", self._builtin_len))
        self.globals.define("str", ZenBuiltin("str", lambda x: zen_format(x)))
        self.globals.define("type", ZenBuiltin("type", zen_type))
        register_phase4_builtins(self, ZenBuiltin)  # type: ignore[arg-type]
        self._builtin_names = set(self.globals.values.keys())
        self._builtin_values = {k: self.globals.values[k].value for k in self._builtin_names}

    def _builtin_print(self, *args: Any) -> None:
        self.stdout.write(" ".join(zen_format(a) for a in args) + "\n")

    def _builtin_len(self, obj: Any) -> int:
        if isinstance(obj, (list, dict, str)):
            return len(obj)
        if isinstance(obj, (ZenTensor, ZenArray)):
            return obj.shape[0] if obj.shape else 1
        raise ZenRuntimeError(f"Object of type '{zen_type(obj)}' has no len()")

    def _emit_comment_warnings(self, source: str) -> None:
        _, trailing = tokenize_with_comments(source)
        for line_no, comment in trailing:
            if not comment:
                continue
            if re.fullmatch(r"\d+(?:\.\d+)?", comment):
                self.stderr.write(
                    f"Warning [line {line_no}]: '// {comment}' is a comment, not integer division; use 'div' or 'floor_div(...)'.\n"
                )
            elif re.fullmatch(r"[A-Za-z_][A-Za-z0-9_]*", comment) and comment in self.globals.values:
                self.stderr.write(
                    f"Warning [line {line_no}]: '// {comment}' is a comment referring to variable '{comment}', not division; use 'div' or 'floor_div(...)'.\n"
                )

    def run(
        self,
        source: str,
        file_path: Path | None = None,
        *,
        return_last_expr: bool = False,
        type_check: bool = False,
    ) -> Any:
        prev_dir = self.base_dir
        if file_path is not None:
            self.base_dir = file_path.resolve().parent
        try:
            program = parse(source)
            if type_check:
                type_check_program(program, base_dir=self.base_dir, source=source, stderr=self.stderr)
            else:
                self._emit_comment_warnings(source)
            last_is_expr = return_last_expr and len(program.statements) > 0 and isinstance(program.statements[-1], ast.ExprStmt)
            if last_is_expr:
                expr_stmt = program.statements[-1]
                assert isinstance(expr_stmt, ast.ExprStmt)
                prefix_prog = ast.Program(program.line, program.column, program.statements[:-1])
                if prefix_prog.statements:
                    self._execute_code(compile_program(prefix_prog), (), [], self.globals)
                # Compile single expression to return its value
                ret_prog = ast.Program(
                    expr_stmt.line,
                    expr_stmt.column,
                    [ast.ReturnStmt(expr_stmt.line, expr_stmt.column, expr_stmt.expression)],
                )
                return self._execute_code(compile_program(ret_prog), (), [], self.globals)
            code = compile_program(program)
            return self._execute_code(code, (), [], self.globals)
        except ZenError as exc:
            raise exc.with_source(source) from exc
        finally:
            self.base_dir = prev_dir

    def run_file(self, path: str | Path, *, type_check: bool = False) -> Environment:
        p = Path(path).resolve()
        record_zen_access(p)
        source = p.read_text(encoding="utf-8")
        self.run(source, file_path=p, type_check=type_check)
        return self.globals

    def _call_value(
        self,
        callee: Any,
        args: list[Any],
        line: int,
        col: int,
        kwargs: list[tuple[str, Any]] | None = None,
    ) -> Any:
        if isinstance(callee, ZenInstance):
            m = callee.klass.find_method("__call__")
            if m is not None:
                return m.call(self, args, line, col, bound_self=callee, kwargs=kwargs)  # type: ignore[arg-type]
        if isinstance(callee, ZenBuiltin):
            kw_dict = dict(kwargs) if kwargs else None
            return callee.call(self, args, line, col, kwargs=kw_dict)  # type: ignore[arg-type]
        if isinstance(callee, (VMClosure, VMBoundMethod, VMClass)):
            return callee.call(self, args, line, col, kwargs=kwargs)  # type: ignore[arg-type]
        raise ZenRuntimeError(f"Value of type '{zen_type(callee)}' is not callable", line, col, code="E004")

    def _execute_code(
        self,
        code: CodeObject,
        upvalues: tuple[_Cell, ...],
        args: list[Any],
        globals_env: Environment,
    ) -> Any:
        if self._frame_count + 1 > self.max_depth:
            err_line = code.lines[0] if code.lines else self.current_line
            raise ZenRuntimeError(
                f"Maximum recursion depth ({self.max_depth}) exceeded in '{code.name}' at depth {self._frame_count + 1}",
                err_line,
                1,
                code="E010",
            )
        base_depth = self._frame_count
        frames: list[_VMFrame] = [_VMFrame(code, upvalues, args, globals_env)]
        self._frame_count = base_depth + 1
        try:
            while frames:
                frame = frames[-1]
                f_code = frame.code
                ops = f_code.ops
                cargs = f_code.args
                lines = f_code.lines
                cols = f_code.cols
                n_instr = len(ops)
                stack = frame.stack
                push = stack.append
                pop = stack.pop
                locals_arr = frame.locals_arr
                f_upvalues = frame.upvalues
                f_globals = frame.globals_env
                g_vals = f_globals.values
                ip = frame.ip
                cur_ip = ip
                switched = False

                try:
                    while ip < n_instr:
                        op = ops[ip]
                        arg = cargs[ip]
                        cur_ip = ip
                        ip += 1
                        self.current_line = lines[cur_ip]
                        if nan_trace.ENABLED:
                            nan_trace.set_location(lines[cur_ip], cols[cur_ip])

                        if op == _OP_LOAD_CONST:
                            push(arg)
                        elif op == _OP_POP:
                            pop()
                        elif op == _OP_DUP:
                            push(stack[-1])
                        elif op == _OP_LOAD_LOCAL:
                            push(locals_arr[arg])
                        elif op == _OP_STORE_LOCAL:
                            locals_arr[arg[0]] = pop()
                        elif op == _OP_DEFINE_CELL:
                            locals_arr[arg[0]] = _Cell(pop(), arg[1])
                        elif op == _OP_LOAD_CELL:
                            push(locals_arr[arg].value)
                        elif op == _OP_STORE_CELL:
                            cell = locals_arr[arg]
                            if cell.is_const:
                                raise ZenRuntimeError("Cannot reassign constant", lines[cur_ip], cols[cur_ip], code="E005")
                            cell.value = pop()
                        elif op == _OP_LOAD_UPVALUE:
                            push(f_upvalues[arg].value)
                        elif op == _OP_STORE_UPVALUE:
                            cell = f_upvalues[arg]
                            if cell.is_const:
                                raise ZenRuntimeError("Cannot reassign constant", lines[cur_ip], cols[cur_ip], code="E005")
                            cell.value = pop()
                        elif op == _OP_DEFINE_GLOBAL:
                            f_globals.define(arg[0], pop(), is_const=arg[1])
                        elif op == _OP_LOAD_GLOBAL:
                            binding = g_vals.get(arg)
                            if binding is None:
                                hint = did_you_mean(arg, g_vals.keys())
                                suffix = f". Did you mean '{hint}'?" if hint else ""
                                raise ZenRuntimeError(f"Undefined variable '{arg}'{suffix}", lines[cur_ip], cols[cur_ip], code="E002")
                            push(binding.value)
                        elif op == _OP_STORE_GLOBAL:
                            f_globals.assign(arg, pop(), lines[cur_ip], cols[cur_ip])
                        elif op == _OP_BUILD_LIST:
                            if arg == 0:
                                push([])
                            else:
                                items = stack[-arg:]
                                del stack[-arg:]
                                push(items)
                        elif op == _OP_BUILD_MAP:
                            m: dict[Any, Any] = {}
                            if arg > 0:
                                items = stack[-2 * arg :]
                                del stack[-2 * arg :]
                                for i in range(0, 2 * arg, 2):
                                    k = items[i]
                                    if isinstance(k, (list, dict)):
                                        raise ZenRuntimeError("Map key cannot be a list or map", lines[cur_ip], cols[cur_ip])
                                    m[k] = items[i + 1]
                            push(m)
                        elif op == _OP_BUILD_TUPLE:
                            if arg == 0:
                                push(())
                            else:
                                items = tuple(stack[-arg:])
                                del stack[-arg:]
                                push(items)
                        elif op == _OP_BUILD_FSTRING:
                            if arg == 0:
                                push("")
                            else:
                                parts = stack[-arg:]
                                del stack[-arg:]
                                push("".join(p if isinstance(p, str) else zen_format(p) for p in parts))
                        elif op == _OP_BUILD_SLICE:
                            step = pop()
                            stop = pop()
                            start = pop()
                            push(slice(start, stop, step))
                        elif op == _OP_LIST_APPEND:
                            val = pop()
                            stack[-arg].append(val)
                        elif op == _OP_UNPACK_SEQ:
                            seq = pop()
                            if not isinstance(seq, (list, tuple)):
                                raise ZenRuntimeError(f"Cannot unpack value of type '{zen_type(seq)}'", lines[cur_ip], cols[cur_ip], code="E004")
                            if len(seq) != arg:
                                raise ZenRuntimeError(f"Expected {arg} values to unpack, got {len(seq)}", lines[cur_ip], cols[cur_ip], code="E008")
                            for item in reversed(seq):
                                push(item)
                        elif op == _OP_GET_INDEX:
                            idx = pop()
                            container = pop()
                            push(self._get_index(container, idx, lines[cur_ip], cols[cur_ip]))
                        elif op == _OP_SET_INDEX:
                            val = pop()
                            idx = pop()
                            container = pop()
                            self._set_index(container, idx, val, lines[cur_ip], cols[cur_ip])
                        elif op == _OP_GET_MEMBER:
                            obj = pop()
                            if isinstance(obj, (ZenInstance, ZenModule, ZenClass)):
                                push(obj.get(arg, lines[cur_ip], cols[cur_ip]))
                            elif isinstance(obj, ZenTensor):
                                push(self._tensor_member(obj, arg, lines[cur_ip], cols[cur_ip]))
                            elif isinstance(obj, ZenArray):
                                push(self._array_member(obj, arg, lines[cur_ip], cols[cur_ip]))
                            elif isinstance(obj, str):
                                push(str_member(obj, arg, ZenBuiltin, lines[cur_ip], cols[cur_ip]))
                            elif isinstance(obj, list):
                                push(list_member(obj, arg, ZenBuiltin, lines[cur_ip], cols[cur_ip]))
                            else:
                                raise ZenRuntimeError(
                                    f"Cannot access member '{arg}' on '{zen_type(obj)}'",
                                    lines[cur_ip],
                                    cols[cur_ip],
                                    code="E003",
                                )
                        elif op == _OP_SET_MEMBER:
                            val = pop()
                            obj = pop()
                            if isinstance(obj, ZenInstance):
                                obj.fields[arg] = val
                            else:
                                raise ZenRuntimeError(
                                    f"Cannot set property '{arg}' on '{zen_type(obj)}'",
                                    lines[cur_ip],
                                    cols[cur_ip],
                                    code="E003",
                                )
                        elif op == _OP_UNARY_OP:
                            val = pop()
                            if arg == "not":
                                push(not is_truthy(val))
                            elif arg == "-":
                                if isinstance(val, ZenInstance):
                                    neg_m = val.klass.find_method("__neg__")
                                    if neg_m is not None:
                                        push(neg_m.call(self, [], lines[cur_ip], cols[cur_ip], bound_self=val))  # type: ignore[arg-type]
                                        continue
                                if isinstance(val, ZenTensor):
                                    push(val.mul(-1.0))
                                elif isinstance(val, ZenArray):
                                    push(val.neg())
                                elif _is_number(val):
                                    push(-val)
                                else:
                                    raise ZenRuntimeError(f"Cannot negate '{zen_type(val)}'", lines[cur_ip], cols[cur_ip], code="E004")
                        elif op == _OP_BINARY_OP:
                            right = pop()
                            left = pop()
                            tl = type(left)
                            tr = type(right)
                            if not nan_trace.ENABLED and (tl is int or tl is float) and (tr is int or tr is float):
                                if arg == "+":
                                    push(left + right)
                                    continue
                                if arg == "-":
                                    push(left - right)
                                    continue
                                if arg == "*":
                                    push(left * right)
                                    continue
                                if arg == "<":
                                    push(left < right)
                                    continue
                                if arg == "<=":
                                    push(left <= right)
                                    continue
                                if arg == ">":
                                    push(left > right)
                                    continue
                                if arg == ">=":
                                    push(left >= right)
                                    continue
                                if arg == "==":
                                    push(left == right)
                                    continue
                                if arg == "!=":
                                    push(left != right)
                                    continue
                            push(self._binary_op(left, arg, right, lines[cur_ip], cols[cur_ip]))
                        elif op == _OP_JUMP:
                            ip = arg
                        elif op == _OP_JUMP_IF_FALSE:
                            if not is_truthy(pop()):
                                ip = arg
                        elif op == _OP_JUMP_IF_FALSE_OR_POP:
                            if not is_truthy(stack[-1]):
                                ip = arg
                            else:
                                pop()
                        elif op == _OP_JUMP_IF_TRUE_OR_POP:
                            if is_truthy(stack[-1]):
                                ip = arg
                            else:
                                pop()
                        elif op == _OP_GET_ITER:
                            iterable = pop()
                            if isinstance(iterable, str):
                                push(iter(list(iterable)))
                            elif isinstance(iterable, (list, range)):
                                push(iter(iterable))
                            elif isinstance(iterable, ZenArray):
                                push(iter(iterable.get_item(i) for i in range(iterable.shape[0])))
                            elif isinstance(iterable, dict):
                                push(iter(iterable.keys()))
                            else:
                                raise ZenRuntimeError(
                                    f"Cannot iterate over '{zen_type(iterable)}'",
                                    lines[cur_ip],
                                    cols[cur_ip],
                                    code="E004",
                                )
                        elif op == _OP_FOR_ITER:
                            it = stack[-1]
                            try:
                                push(next(it))
                            except StopIteration:
                                pop()
                                ip = arg
                        elif op == _OP_MAKE_CLOSURE:
                            nd = arg.num_defaults
                            if nd:
                                def_vals = tuple(stack[-nd:])
                                del stack[-nd:]
                            else:
                                def_vals = ()
                            ups = tuple(
                                locals_arr[idx] if is_local else f_upvalues[idx]
                                for is_local, idx in arg.upvalues
                            )
                            push(VMClosure(arg, ups, f_globals, def_vals))
                        elif op == _OP_CALL:
                            call_args = stack[-arg:] if arg else []
                            if arg:
                                del stack[-arg:]
                            callee = pop()
                            if type(callee) is VMClosure:
                                c_code = callee.code
                                total = len(c_code.params)
                                min_total = total - c_code.num_defaults
                                if arg < min_total or arg > total:
                                    exp = f"{min_total}" if min_total == total else f"{min_total} to {total}"
                                    raise ZenRuntimeError(
                                        f"Function '{callee.name}' expected {exp} argument(s), got {arg}",
                                        lines[cur_ip],
                                        cols[cur_ip],
                                        code="E006",
                                    )
                                if arg < total:
                                    call_args = call_args + list(callee.defaults[arg - min_total :])
                                new_depth = self._frame_count + 1
                                if new_depth > self.max_depth:
                                    raise ZenRuntimeError(
                                        f"Maximum recursion depth ({self.max_depth}) exceeded in '{callee.name}' at depth {new_depth}",
                                        lines[cur_ip],
                                        cols[cur_ip],
                                        code="E010",
                                    )
                                frame.ip = ip
                                frames.append(_VMFrame(c_code, callee.upvalues, call_args, callee.globals_env))
                                self._frame_count = new_depth
                                switched = True
                                break
                            push(self._call_value(callee, call_args, lines[cur_ip], cols[cur_ip]))
                        elif op == _OP_CALL_KW:
                            pos_c, kw_names = arg
                            kw_c = len(kw_names)
                            kw_vals = stack[-kw_c:] if kw_c else []
                            if kw_c:
                                del stack[-kw_c:]
                            pos_args = stack[-pos_c:] if pos_c else []
                            if pos_c:
                                del stack[-pos_c:]
                            callee = pop()
                            kw_pairs = list(zip(kw_names, kw_vals))
                            if type(callee) is VMClosure:
                                bound_args = bind_call_args(
                                    callee.name,
                                    callee.code.params,
                                    callee.defaults,
                                    pos_args,
                                    kw_pairs,
                                    None,
                                    lines[cur_ip],
                                    cols[cur_ip],
                                )
                                new_depth = self._frame_count + 1
                                if new_depth > self.max_depth:
                                    raise ZenRuntimeError(
                                        f"Maximum recursion depth ({self.max_depth}) exceeded in '{callee.name}' at depth {new_depth}",
                                        lines[cur_ip],
                                        cols[cur_ip],
                                        code="E010",
                                    )
                                frame.ip = ip
                                frames.append(_VMFrame(callee.code, callee.upvalues, bound_args, callee.globals_env))
                                self._frame_count = new_depth
                                switched = True
                                break
                            push(self._call_value(callee, pos_args, lines[cur_ip], cols[cur_ip], kwargs=kw_pairs))
                        elif op == _OP_RETURN:
                            ret_val = pop() if stack else None
                            frames.pop()
                            self._frame_count -= 1
                            if not frames:
                                return ret_val
                            frames[-1].stack.append(ret_val)
                            switched = True
                            break
                        elif op == _OP_PUSH_TRY:
                            frame.try_stack.append((arg, len(stack)))
                        elif op == _OP_POP_TRY:
                            if frame.try_stack:
                                frame.try_stack.pop()
                        elif op == _OP_RAISE:
                            val = pop()
                            raise ZenRuntimeError(zen_format(val), lines[cur_ip], cols[cur_ip], code="E012")
                        elif op == _OP_BUILD_CLASS:
                            cls_name, method_names, has_super = arg
                            sup_cls: ZenClass | None = None
                            if has_super:
                                sup_val = pop()
                                if not isinstance(sup_val, ZenClass):
                                    raise ZenRuntimeError("Superclass must be a class", lines[cur_ip], cols[cur_ip], code="E004")
                                sup_cls = sup_val
                            n_m = len(method_names)
                            closures = stack[-n_m:] if n_m else []
                            if n_m:
                                del stack[-n_m:]
                            push(VMClass(cls_name, dict(zip(method_names, closures)), sup_cls))
                        elif op == _OP_IMPORT:
                            path_str, alias = arg
                            self._exec_import(path_str, alias, lines[cur_ip], cols[cur_ip])
                        elif op == _OP_SAY:
                            vals = stack[-arg:]
                            del stack[-arg:]
                            self.stdout.write(" ".join(zen_format(v) for v in vals) + "\n")
                        elif op == _OP_ASSERT:
                            msg_val = pop() if arg else None
                            cond_val = pop()
                            if not is_truthy(cond_val):
                                if arg:
                                    raise ZenRuntimeError(f"Assertion failed: {zen_format(msg_val)}", lines[cur_ip], cols[cur_ip], code="E012")
                                raise ZenRuntimeError("Assertion failed", lines[cur_ip], cols[cur_ip], code="E012")
                except ZenRuntimeError as exc:
                    if not exc.line and cur_ip < n_instr:
                        exc = ZenRuntimeError(exc.message, lines[cur_ip], cols[cur_ip], exc.source, code=exc.code)
                    caught = False
                    while frames:
                        top = frames[-1]
                        if top.try_stack:
                            catch_ip, saved_len = top.try_stack.pop()
                            del top.stack[saved_len:]
                            top.stack.append(make_error_instance(exc, exc.line or self.current_line))
                            top.ip = catch_ip
                            caught = True
                            break
                        frames.pop()
                        self._frame_count -= 1
                    if not caught:
                        raise
                    continue

                if not switched:
                    frames.pop()
                    self._frame_count -= 1
                    if not frames:
                        return None
                    frames[-1].stack.append(None)
            return None
        finally:
            self._frame_count = base_depth

    def _exec_import(self, path_str: str, alias: str | None, line: int, col: int) -> None:
        target = resolve_import_path(path_str, self.base_dir)
        if target is None:
            raise ZenRuntimeError(f"Imported file not found: '{path_str}'", line, col)

        if target in self._module_cache:
            module = self._module_cache[target]
        else:
            mod_vm = VM(stdout=self.stdout, base_dir=target.parent, stderr=self.stderr, stdin=self.stdin, max_depth=self.max_depth)
            mod_vm._module_cache = self._module_cache
            for k, bval in self._builtin_values.items():
                mod_vm.globals.define(k, bval)
            mod_env = mod_vm.run_file(target)
            exports = {
                k: b.value
                for k, b in mod_env.values.items()
                if b.value is not self._builtin_values.get(k)
            }
            mod_name = target.parent.name if target.stem == "__init__" else target.stem
            module = ZenModule(mod_name, exports)
            self._module_cache[target] = module

        if alias:
            self.globals.define(alias, module)
        else:
            for k, v in module.exports.items():
                self.globals.define(k, v)

    def _get_index(self, container: Any, idx: Any, line: int, col: int) -> Any:
        if isinstance(container, ZenInstance):
            m = container.klass.find_method("__getitem__")
            if m is not None:
                return m.call(self, [idx], line, col, bound_self=container)  # type: ignore[arg-type]
        if isinstance(container, ZenArray):
            return container.get_item(idx)
        if isinstance(container, ZenTensor):
            if not isinstance(idx, int) or isinstance(idx, bool):
                raise ZenRuntimeError("Tensor index must be an integer", line, col)
            return container.get_index(idx)
        if isinstance(container, (list, str)):
            if isinstance(idx, slice):
                return container[idx]
            if not isinstance(idx, int) or isinstance(idx, bool):
                raise ZenRuntimeError("Index must be an integer", line, col)
            try:
                return container[idx]
            except IndexError as exc:
                raise ZenRuntimeError(f"Index {idx} out of range", line, col) from exc
        if isinstance(container, dict):
            if idx not in container:
                raise ZenRuntimeError(f"Key {idx!r} not found in map", line, col)
            return container[idx]
        raise ZenRuntimeError(f"Cannot index into '{zen_type(container)}'", line, col)

    def _set_index(self, container: Any, idx: Any, value: Any, line: int, col: int) -> None:
        if isinstance(container, ZenInstance):
            m = container.klass.find_method("__setitem__")
            if m is not None:
                m.call(self, [idx, value], line, col, bound_self=container)  # type: ignore[arg-type]
                return
        if isinstance(container, ZenArray):
            container.set_item(idx, value)
            return
        if isinstance(container, list):
            if isinstance(idx, slice):
                container[idx] = value
                return
            if not isinstance(idx, int) or isinstance(idx, bool):
                raise ZenRuntimeError("List index must be an integer", line, col)
            try:
                container[idx] = value
            except IndexError as exc:
                raise ZenRuntimeError(f"List index {idx} out of range", line, col) from exc
            return
        if isinstance(container, dict):
            container[idx] = value
            return
        raise ZenRuntimeError(f"Cannot assign by index on '{zen_type(container)}'", line, col)

    def _array_member(self, a: ZenArray, member: str, line: int, col: int) -> Any:
        if member == "shape":
            return list(a.shape)
        if member == "dtype":
            return a.dtype
        if member == "ndim":
            return a.ndim
        if member == "size":
            return a.size
        if member == "T":
            return a.transpose()
        if member == "tolist":
            return ZenBuiltin("tolist", a.tolist)
        if member == "copy":
            return ZenBuiltin("copy", a.copy)
        if member == "astype":
            return ZenBuiltin("astype", a.astype)
        if member == "reshape":
            return ZenBuiltin("reshape", a.reshape)
        if member == "transpose":
            return ZenBuiltin("transpose", a.transpose)
        if member == "flatten":
            return ZenBuiltin("flatten", a.flatten)
        if member == "clip":
            return ZenBuiltin("clip", a.clip)
        if member in ("sum", "mean", "std", "var", "min", "max", "argmax", "argmin", "norm"):
            return ZenBuiltin(member, lambda axis=None, keepdims=False, _m=member: a.reduce(_m, axis, bool(keepdims)))
        if member == "sort":
            return ZenBuiltin("sort", a.sort)
        if member == "cumsum":
            return ZenBuiltin("cumsum", a.cumsum)
        if member == "dot":
            return ZenBuiltin("dot", lambda b: a.matmul(ZenArray.from_list(b)))
        if member == "all":
            return ZenBuiltin("all", lambda: all(a.data))
        if member == "any":
            return ZenBuiltin("any", lambda: any(a.data))
        raise ZenRuntimeError(f"Array has no member '{member}'", line, col)

    def _tensor_member(self, t: ZenTensor, member: str, line: int, col: int) -> Any:
        if member == "shape":
            return list(t.shape)
        if member == "T":
            return t.transpose()
        if member == "reshape":
            return ZenBuiltin("reshape", t.reshape)
        if member == "transpose":
            return ZenBuiltin("transpose", t.transpose)
        if member == "sum":
            return ZenBuiltin("sum", t.sum)
        if member == "mean":
            return ZenBuiltin("mean", t.mean)
        if member == "max":
            return ZenBuiltin("max", t.max)
        if member == "tolist":
            return ZenBuiltin("tolist", t.tolist)
        if member == "exp":
            return ZenBuiltin("exp", t.exp)
        if member == "log":
            return ZenBuiltin("log", t.log)
        if member == "tanh":
            return ZenBuiltin("tanh", t.tanh)
        raise ZenRuntimeError(f"Tensor has no member '{member}'", line, col)

    def _binary_op(self, left: Any, op: str, right: Any, line: int, col: int) -> Any:
        res = self._binary_op_impl(left, op, right, line, col)
        if nan_trace.ENABLED:
            nan_trace.check_value(res, op, left, right, line, col)
        return res

    def _binary_op_impl(self, left: Any, op: str, right: Any, line: int, col: int) -> Any:
        if op in _DUNDER_MAP:
            d, rd = _DUNDER_MAP[op]
            if isinstance(left, ZenInstance):
                m = left.klass.find_method(d)
                if m is not None:
                    return m.call(self, [right], line, col, bound_self=left)  # type: ignore[arg-type]
            if isinstance(right, ZenInstance):
                rm = right.klass.find_method(rd)
                if rm is not None:
                    return rm.call(self, [left], line, col, bound_self=right)  # type: ignore[arg-type]

        if isinstance(left, ZenArray) or isinstance(right, ZenArray):
            if op in ("==", "!=") and (left is None or right is None or isinstance(left, (str, dict, ZenInstance, VMClosure, ZenModule, ZenClass, ZenBuiltin)) or isinstance(right, (str, dict, ZenInstance, VMClosure, ZenModule, ZenClass, ZenBuiltin))):
                return False if op == "==" else True
            if op in ("==", "!=", "<", "<=", ">", ">="):
                if isinstance(left, ZenArray):
                    return left.cmp(op, right)
                flip = {"<": ">", "<=": ">=", ">": "<", ">=": "<=", "==": "==", "!=": "!="}[op]
                return right.cmp(flip, left)
            if op == "@":
                if isinstance(left, ZenArray) and isinstance(right, ZenArray):
                    return left.matmul(right)
                raise ZenRuntimeError("Matrix multiply '@' requires two arrays", line, col)
            if op == "+":
                return left.add(right) if isinstance(left, ZenArray) else right.add(left)
            if op == "-":
                return left.sub(right) if isinstance(left, ZenArray) else right.rsub(left)
            if op == "*":
                return left.mul(right) if isinstance(left, ZenArray) else right.mul(left)
            if op == "/":
                return left.div(right) if isinstance(left, ZenArray) else right.rdiv(left)
            if op == "**":
                return left.pow(right) if isinstance(left, ZenArray) else ZenArray.from_list(left).pow(right)

        if op == "==":
            if type(left) is bool or type(right) is bool:
                return type(left) is type(right) and left == right
            return left == right
        if op == "!=":
            if type(left) is bool or type(right) is bool:
                return not (type(left) is type(right) and left == right)
            return left != right

        if isinstance(left, ZenTensor) or isinstance(right, ZenTensor):
            if op == "@":
                if isinstance(left, ZenTensor) and isinstance(right, ZenTensor):
                    return left.matmul(right)
                raise ZenRuntimeError("Matrix multiply '@' requires two Tensors", line, col)
            if op == "+":
                return left.add(right) if isinstance(left, ZenTensor) else right.add(left)
            if op == "-":
                return left.sub(right) if isinstance(left, ZenTensor) else right.rsub(left)
            if op == "*":
                return left.mul(right) if isinstance(left, ZenTensor) else right.mul(left)
            if op == "/":
                return left.div(right) if isinstance(left, ZenTensor) else right.rdiv(left)

        if op in ("in", "not in"):
            if isinstance(right, str):
                if not isinstance(left, str):
                    raise ZenRuntimeError(
                        f"Operator '{op}' on string requires string on left, got '{zen_type(left)}'",
                        line,
                        col,
                        code="E004",
                    )
                found = left in right
            elif isinstance(right, list):
                found = left in right
            elif isinstance(right, dict):
                found = False if isinstance(left, (list, dict)) else (left in right)
            else:
                raise ZenRuntimeError(
                    f"Operator '{op}' requires list, string, or map on right, got '{zen_type(right)}'",
                    line,
                    col,
                    code="E004",
                )
            return found if op == "in" else not found

        if op == "*":
            if isinstance(left, str) and isinstance(right, int) and not isinstance(right, bool):
                return left * right
            if isinstance(right, str) and isinstance(left, int) and not isinstance(left, bool):
                return right * left

        if op == "+":
            if isinstance(left, str) and isinstance(right, str):
                return left + right
            if isinstance(left, list) and isinstance(right, list):
                return left + right
            if _is_number(left) and _is_number(right):
                return left + right
            raise ZenRuntimeError(
                f"Unsupported '+' between '{zen_type(left)}' and '{zen_type(right)}'",
                line,
                col,
                code="E004",
            )

        if op in ("-", "*", "/", "div", "%", "**", "<", "<=", ">", ">="):
            if not (_is_number(left) and _is_number(right)):
                raise ZenRuntimeError(
                    f"Operator '{op}' requires numbers, got '{zen_type(left)}' and '{zen_type(right)}'",
                    line,
                    col,
                    code="E004",
                )
            if op == "-":
                return left - right
            if op == "*":
                return left * right
            if op == "/":
                if right == 0:
                    raise ZenRuntimeError("Division by zero", line, col, code="E007")
                return left / right
            if op == "div":
                if right == 0:
                    raise ZenRuntimeError("Division by zero", line, col, code="E007")
                return left // right
            if op == "%":
                if right == 0:
                    raise ZenRuntimeError("Modulo by zero", line, col, code="E007")
                return left % right
            if op == "**":
                return left ** right
            if op == "<":
                return left < right
            if op == "<=":
                return left <= right
            if op == ">":
                return left > right
            if op == ">=":
                return left >= right

        raise ZenRuntimeError(
            f"Unsupported operator '{op}' for '{zen_type(left)}' and '{zen_type(right)}'",
            line,
            col,
        )
