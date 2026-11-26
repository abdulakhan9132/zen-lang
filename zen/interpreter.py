"""Tree-walking interpreter for the ZEN language."""

from __future__ import annotations

import io
import sys
from dataclasses import dataclass
from pathlib import Path
import re
from typing import Any, Callable

from zen import ast
from zen import nan_trace
from zen.builtins import register_phase4_builtins
from zen.environment import Environment
from zen.errors import ZenError, ZenRuntimeError, did_you_mean
from zen.lexer import tokenize_with_comments
from zen.parser import parse
from zen.paths import resolve_import_path
from zen.recorder import record_zen_access
from zen.tensor import ZenTensor
from zen.typechecker import type_check_program
from zen.zennum_core import ZenArray


class _ReturnSignal(Exception):
    def __init__(self, value: Any) -> None:
        self.value = value


class _BreakSignal(Exception):
    pass


class _ContinueSignal(Exception):
    pass


def _is_number(val: Any) -> bool:
    return isinstance(val, (int, float)) and not isinstance(val, bool)


def is_truthy(val: Any) -> bool:
    if val is None or val is False:
        return False
    if _is_number(val) and val == 0:
        return False
    if isinstance(val, (str, list, dict)) and len(val) == 0:
        return False
    return True


def zen_format(val: Any, *, nested: bool = False) -> str:
    if val is None:
        return "none"
    if isinstance(val, bool):
        return "true" if val else "false"
    if isinstance(val, str):
        return repr(val) if nested else val
    if isinstance(val, (int, float)):
        return str(val)
    if isinstance(val, list):
        return "[" + ", ".join(zen_format(x, nested=True) for x in val) + "]"
    if isinstance(val, dict):
        items = ", ".join(
            f"{zen_format(k, nested=True)}: {zen_format(v, nested=True)}"
            for k, v in val.items()
        )
        return "{" + items + "}"
    if isinstance(val, ZenInstance):
        return f"<{val.klass.name} instance>"
    if isinstance(val, ZenClass):
        return f"<class {val.name}>"
    if isinstance(val, (ZenFunction, ZenBuiltin)):
        return f"<fn {val.name}>"
    if isinstance(val, ZenModule):
        return f"<module {val.name}>"
    if isinstance(val, ZenTensor):
        return f"Tensor({val.tolist()})"
    if isinstance(val, ZenArray):
        return f"Array({val.tolist()}, dtype={val.dtype})"
    return str(val)


def zen_type(val: Any) -> str:
    if val is None:
        return "none"
    if isinstance(val, bool):
        return "bool"
    if isinstance(val, int):
        return "int"
    if isinstance(val, float):
        return "float"
    if isinstance(val, str):
        return "string"
    if isinstance(val, list):
        return "list"
    if isinstance(val, dict):
        return "map"
    if isinstance(val, ZenTensor):
        return "tensor"
    if isinstance(val, ZenArray):
        return "array"
    if isinstance(val, ZenInstance):
        return val.klass.name
    if isinstance(val, ZenClass):
        return "class"
    if isinstance(val, (ZenFunction, ZenBuiltin, ZenBoundMethod)):
        return "fn"
    if isinstance(val, ZenModule):
        return "module"
    return type(val).__name__.lower()


@dataclass
class ZenBuiltin:
    name: str
    fn: Callable[..., Any]

    def call(
        self,
        interpreter: "Interpreter",
        args: list[Any],
        line: int,
        col: int,
        kwargs: dict[str, Any] | None = None,
    ) -> Any:
        try:
            if kwargs:
                res = self.fn(*args, **kwargs)
            else:
                res = self.fn(*args)
            if nan_trace.ENABLED:
                nan_trace.check_value(res, self.name, args[0] if args else None, args[1] if len(args) > 1 else None, line=line, col=col)
            return res
        except OverflowError as exc:
            if nan_trace.ENABLED:
                raise ZenRuntimeError(f"Inf produced by '{self.name}' (overflow)", line, col, code="E014") from exc
            raise ZenRuntimeError(f"Built-in '{self.name}' overflow error: {exc}", line, col, code="E004") from exc
        except ZenRuntimeError:
            raise
        except TypeError as exc:
            raise ZenRuntimeError(f"Built-in '{self.name}' argument error: {exc}", line, col, code="E006") from exc
        except ValueError as exc:
            raise ZenRuntimeError(f"Built-in '{self.name}' value error: {exc}", line, col, code="E004") from exc
        except OSError as exc:
            raise ZenRuntimeError(f"Built-in '{self.name}' I/O error: {exc}", line, col, code="E009") from exc


def bind_call_args(
    fn_name: str,
    params: list[str],
    defaults: tuple[Any, ...],
    pos_args: list[Any],
    kw_pairs: list[tuple[str, Any]] | None,
    bound_self: Any,
    line: int,
    col: int,
) -> list[Any]:
    """Bind positional, keyword, and default arguments onto a ZEN parameter list."""
    actual_pos = ([bound_self] + pos_args) if bound_self is not None else list(pos_args)
    total = len(params)
    min_total = total - len(defaults)
    if not kw_pairs:
        if len(actual_pos) < min_total or len(actual_pos) > total:
            offset = 1 if bound_self is not None and total > 0 else 0
            shown_min = max(0, min_total - offset)
            shown_max = max(0, total - offset)
            exp = f"{shown_min}" if shown_min == shown_max else f"{shown_min} to {shown_max}"
            raise ZenRuntimeError(
                f"Function '{fn_name}' expected {exp} argument(s), got {len(pos_args)}",
                line,
                col,
                code="E006",
            )
        if len(actual_pos) < total:
            actual_pos = actual_pos + list(defaults[len(actual_pos) - min_total :])
        return actual_pos

    if len(actual_pos) > total:
        offset = 1 if bound_self is not None and total > 0 else 0
        raise ZenRuntimeError(
            f"Function '{fn_name}' expected at most {total - offset} argument(s), got {len(pos_args) + len(kw_pairs)}",
            line,
            col,
            code="E006",
        )
    slots: list[Any] = [None] * total
    assigned = [False] * total
    for i, v in enumerate(actual_pos):
        slots[i] = v
        assigned[i] = True

    offset = 1 if bound_self is not None and total > 0 else 0
    param_idx = {p: i for i, p in enumerate(params)}
    valid_kw = params[offset:]
    for kname, kval in kw_pairs:
        idx = param_idx.get(kname, -1)
        if idx < offset:
            hint = did_you_mean(kname, valid_kw)
            suffix = f". Did you mean '{hint}'?" if hint else ""
            raise ZenRuntimeError(
                f"Function '{fn_name}' got unexpected keyword argument '{kname}'{suffix}",
                line,
                col,
                code="E006",
            )
        if assigned[idx]:
            raise ZenRuntimeError(
                f"Function '{fn_name}' got multiple values for argument '{kname}'",
                line,
                col,
                code="E006",
            )
        slots[idx] = kval
        assigned[idx] = True

    for i in range(total):
        if not assigned[i]:
            if i >= min_total:
                slots[i] = defaults[i - min_total]
            else:
                raise ZenRuntimeError(
                    f"Function '{fn_name}' missing required argument '{params[i]}'",
                    line,
                    col,
                    code="E006",
                )
    return slots


def make_error_instance(exc: ZenRuntimeError, fallback_line: int = 0) -> "ZenInstance":
    """Wrap a ZenRuntimeError as a ZEN Error object with .message, .line, and .code."""
    inst = ZenInstance(ZenClass("Error", {}))
    inst.fields["message"] = exc.message
    inst.fields["line"] = exc.line if exc.line else fallback_line
    inst.fields["code"] = exc.code or "E012"
    return inst


def str_member(s: str, member: str, make_builtin_cls: type[ZenBuiltin], line: int, col: int) -> Any:
    """Resolve built-in string methods."""
    if member == "split":
        return make_builtin_cls("split", lambda sep=None: s.split() if sep is None else (list(s) if sep == "" else s.split(sep)))
    if member == "join":
        return make_builtin_cls("join", lambda items: s.join(zen_format(x) for x in items))
    if member == "upper":
        return make_builtin_cls("upper", s.upper)
    if member == "lower":
        return make_builtin_cls("lower", s.lower)
    if member == "strip":
        return make_builtin_cls("strip", lambda chars=None: s.strip() if chars is None else s.strip(chars))
    if member == "replace":
        return make_builtin_cls("replace", lambda old, new, count=-1: s.replace(old, new) if count < 0 else s.replace(old, new, count))
    if member == "startswith":
        return make_builtin_cls("startswith", s.startswith)
    if member == "endswith":
        return make_builtin_cls("endswith", s.endswith)
    if member == "find":
        return make_builtin_cls("find", lambda sub, start=0, end=None: s.find(sub, start) if end is None else s.find(sub, start, end))
    hint = did_you_mean(member, ("split", "join", "upper", "lower", "strip", "replace", "startswith", "endswith", "find"))
    suffix = f". Did you mean '{hint}'?" if hint else ""
    raise ZenRuntimeError(f"String has no method '{member}'{suffix}", line, col, code="E003")


def list_member(lst: list[Any], member: str, make_builtin_cls: type[ZenBuiltin], line: int, col: int) -> Any:
    """Resolve built-in list methods."""
    if member == "append":
        return make_builtin_cls("append", lambda item: lst.append(item))
    if member == "pop":
        def _pop(idx: int | None = None) -> Any:
            if not lst:
                raise ZenRuntimeError("Cannot pop from an empty list", line, col, code="E008")
            try:
                return lst.pop() if idx is None else lst.pop(idx)
            except IndexError as exc:
                raise ZenRuntimeError(f"pop index {idx} out of range", line, col, code="E008") from exc
        return make_builtin_cls("pop", _pop)
    if member == "sort":
        def _sort(reverse: bool = False) -> None:
            try:
                lst.sort(reverse=bool(reverse))
            except TypeError as exc:
                raise ZenRuntimeError(f"Cannot sort list with mixed types: {exc}", line, col, code="E004") from exc
        return make_builtin_cls("sort", _sort)
    if member == "reverse":
        return make_builtin_cls("reverse", lst.reverse)
    if member == "index":
        def _index(val: Any) -> int:
            try:
                return lst.index(val)
            except ValueError as exc:
                raise ZenRuntimeError(f"Value {zen_format(val)!r} is not in list", line, col, code="E008") from exc
        return make_builtin_cls("index", _index)
    hint = did_you_mean(member, ("append", "pop", "sort", "reverse", "index"))
    suffix = f". Did you mean '{hint}'?" if hint else ""
    raise ZenRuntimeError(f"List has no method '{member}'{suffix}", line, col, code="E003")


class ZenFunction:
    """User-defined ZEN function or method with lexical closure."""

    def __init__(
        self,
        name: str,
        params: list[str],
        body: ast.Block,
        closure: Environment,
        defaults: tuple[Any, ...] = (),
    ) -> None:
        self.name = name
        self.params = params
        self.body = body
        self.closure = closure
        self.defaults = defaults

    def call(
        self,
        interpreter: "Interpreter",
        args: list[Any],
        line: int,
        col: int,
        bound_self: "ZenInstance | None" = None,
        kwargs: list[tuple[str, Any]] | None = None,
    ) -> Any:
        actual_args = bind_call_args(self.name, self.params, self.defaults, args, kwargs, bound_self, line, col)
        if interpreter._call_depth + 1 > interpreter.max_depth:
            raise ZenRuntimeError(
                f"Maximum recursion depth ({interpreter.max_depth}) exceeded in '{self.name}' at depth {interpreter._call_depth + 1}",
                line,
                col,
                code="E010",
            )
        local_env = Environment(self.closure)
        for param, arg_val in zip(self.params, actual_args):
            local_env.define(param, arg_val)
        prev_fn = interpreter._current_fn
        interpreter._current_fn = self.name
        interpreter._call_depth += 1
        try:
            interpreter.execute_block(self.body, local_env)
        except _ReturnSignal as ret:
            return ret.value
        except RecursionError as exc:
            raise ZenRuntimeError(
                f"Maximum recursion depth exceeded in '{self.name}' at depth {interpreter._call_depth}",
                interpreter.current_line or line,
                col,
                code="E010",
            ) from exc
        finally:
            interpreter._call_depth -= 1
            interpreter._current_fn = prev_fn
        return None


@dataclass
class ZenBoundMethod:
    instance: "ZenInstance"
    method: ZenFunction

    @property
    def name(self) -> str:
        return self.method.name

    def call(
        self,
        interpreter: "Interpreter",
        args: list[Any],
        line: int,
        col: int,
        kwargs: list[tuple[str, Any]] | None = None,
    ) -> Any:
        return self.method.call(interpreter, args, line, col, bound_self=self.instance, kwargs=kwargs)


class ZenClass:
    """Runtime representation of a ZEN class."""

    def __init__(
        self,
        name: str,
        methods: dict[str, ZenFunction],
        parent: "ZenClass | None" = None,
    ) -> None:
        self.name = name
        self.methods = methods
        self.parent = parent

    def find_method(self, name: str) -> ZenFunction | None:
        if name in self.methods:
            return self.methods[name]
        if self.parent is not None:
            return self.parent.find_method(name)
        return None

    def get(self, name: str, line: int, col: int) -> Any:
        m = self.find_method(name)
        if m is not None:
            return m
        hint = did_you_mean(name, self.methods.keys())
        suffix = f". Did you mean '{hint}'?" if hint else ""
        raise ZenRuntimeError(f"Class '{self.name}' has no method '{name}'{suffix}", line, col, code="E003")

    def call(
        self,
        interpreter: "Interpreter",
        args: list[Any],
        line: int,
        col: int,
        kwargs: list[tuple[str, Any]] | None = None,
    ) -> "ZenInstance":
        instance = ZenInstance(self)
        init_method = self.find_method("init")
        if init_method is not None:
            init_method.call(interpreter, args, line, col, bound_self=instance, kwargs=kwargs)
        elif args or kwargs:
            raise ZenRuntimeError(
                f"Class '{self.name}' takes no constructor arguments, got {len(args) + len(kwargs or [])}",
                line,
                col,
                code="E006",
            )
        return instance


class ZenInstance:
    """Runtime instance of a ZEN class."""

    def __init__(self, klass: ZenClass) -> None:
        self.klass = klass
        self.fields: dict[str, Any] = {}

    def get(self, name: str, line: int, col: int) -> Any:
        if name in self.fields:
            return self.fields[name]
        method = self.klass.find_method(name)
        if method is not None:
            return ZenBoundMethod(self, method)
        hint = did_you_mean(name, list(self.fields.keys()) + list(self.klass.methods.keys()))
        suffix = f". Did you mean '{hint}'?" if hint else ""
        raise ZenRuntimeError(
            f"'{self.klass.name}' instance has no field or method '{name}'{suffix}",
            line,
            col,
            code="E003",
        )


@dataclass
class ZenModule:
    name: str
    exports: dict[str, Any]

    def get(self, member: str, line: int, col: int) -> Any:
        if member in self.exports:
            return self.exports[member]
        hint = did_you_mean(member, self.exports.keys())
        suffix = f". Did you mean '{hint}'?" if hint else ""
        raise ZenRuntimeError(f"Module '{self.name}' has no member '{member}'{suffix}", line, col, code="E003")


class Interpreter:
    """Executes a parsed ZEN AST."""

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
        self._call_depth = 0
        self._current_fn = "<module>"
        self.current_line = 0
        self.base_dir = (base_dir or Path.cwd()).resolve()
        self.globals = Environment()
        self._module_cache: dict[Path, ZenModule] = {}
        self._register_core_builtins()

    def _register_core_builtins(self) -> None:
        self.globals.define("print", ZenBuiltin("print", self._builtin_print))
        self.globals.define("len", ZenBuiltin("len", self._builtin_len))
        self.globals.define("str", ZenBuiltin("str", lambda x: zen_format(x)))
        self.globals.define("type", ZenBuiltin("type", zen_type))
        register_phase4_builtins(self, ZenBuiltin)
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

    def run(self, source: str, file_path: Path | None = None, *, type_check: bool = False) -> Environment:
        prev_dir = self.base_dir
        prev_rec = sys.getrecursionlimit()
        if file_path is not None:
            self.base_dir = file_path.resolve().parent
        try:
            target_rec = max(prev_rec, min(self.max_depth * 6 + 1000, 30000))
            if target_rec > prev_rec:
                try:
                    sys.setrecursionlimit(target_rec)
                except (ValueError, RecursionError):
                    pass
            program = parse(source)
            if type_check:
                type_check_program(program, base_dir=self.base_dir, source=source, stderr=self.stderr)
            else:
                self._emit_comment_warnings(source)
            for stmt in program.statements:
                self.execute(stmt, self.globals)
            return self.globals
        except ZenError as exc:
            raise exc.with_source(source) from exc
        except RecursionError as exc:
            raise ZenRuntimeError(
                f"Maximum recursion depth exceeded in '{self._current_fn}' at depth {self._call_depth}",
                self.current_line,
                1,
                source=source,
                code="E010",
            ) from exc
        finally:
            self.base_dir = prev_dir
            if sys.getrecursionlimit() != prev_rec:
                try:
                    sys.setrecursionlimit(prev_rec)
                except (ValueError, RecursionError):
                    pass

    def run_file(self, path: str | Path, *, type_check: bool = False) -> Environment:
        p = Path(path).resolve()
        record_zen_access(p)
        source = p.read_text(encoding="utf-8")
        return self.run(source, file_path=p, type_check=type_check)

    def execute_block(self, block: ast.Block, env: Environment) -> None:
        for stmt in block.statements:
            self.execute(stmt, env)

    def execute(self, node: ast.Node, env: Environment) -> None:
        if node.line:
            self.current_line = node.line
            if nan_trace.ENABLED:
                nan_trace.set_location(node.line, node.column)
        if isinstance(node, ast.LetStmt):
            val = self.evaluate(node.value, env)
            env.define(node.name, val, is_const=False)
            return
        if isinstance(node, ast.ConstStmt):
            val = self.evaluate(node.value, env)
            env.define(node.name, val, is_const=True)
            return
        if isinstance(node, ast.UnpackStmt):
            seq = self.evaluate(node.value, env)
            if not isinstance(seq, (list, tuple)):
                raise ZenRuntimeError(f"Cannot unpack value of type '{zen_type(seq)}'", node.line, node.column, code="E004")
            if len(seq) != len(node.names):
                raise ZenRuntimeError(f"Expected {len(node.names)} values to unpack, got {len(seq)}", node.line, node.column, code="E008")
            for name, item in zip(node.names, seq):
                env.define(name, item, is_const=node.is_const)
            return
        if isinstance(node, ast.UnpackAssignStmt):
            seq = self.evaluate(node.value, env)
            if not isinstance(seq, (list, tuple)):
                raise ZenRuntimeError(f"Cannot unpack value of type '{zen_type(seq)}'", node.line, node.column, code="E004")
            if len(seq) != len(node.targets):
                raise ZenRuntimeError(f"Expected {len(node.targets)} values to unpack, got {len(seq)}", node.line, node.column, code="E008")
            for target, item in zip(node.targets, seq):
                self._assign_target(target, item, env)
            return
        if isinstance(node, ast.AssignStmt):
            val = self.evaluate(node.value, env)
            self._assign_target(node.target, val, env)
            return
        if isinstance(node, ast.SayStmt):
            rendered = [zen_format(self.evaluate(v, env)) for v in node.values]
            self.stdout.write(" ".join(rendered) + "\n")
            return
        if isinstance(node, ast.AssertStmt):
            cond = self.evaluate(node.condition, env)
            if not is_truthy(cond):
                if node.message is not None:
                    msg = zen_format(self.evaluate(node.message, env))
                    raise ZenRuntimeError(f"Assertion failed: {msg}", node.line, node.column)
                raise ZenRuntimeError("Assertion failed", node.line, node.column)
            return
        if isinstance(node, ast.FnDecl):
            defs = tuple(self.evaluate(d, env) for d in (node.defaults or []) if d is not None)
            fn_obj = ZenFunction(node.name, node.params, node.body, env, defs)
            env.define(node.name, fn_obj)
            return
        if isinstance(node, ast.ClassDecl):
            parent_cls: ZenClass | None = None
            if node.superclass is not None:
                sup_val = self.evaluate(node.superclass, env)
                if not isinstance(sup_val, ZenClass):
                    raise ZenRuntimeError("Superclass must be a class", node.line, node.column)
                parent_cls = sup_val
            methods = {
                m.name: ZenFunction(
                    m.name,
                    m.params,
                    m.body,
                    env,
                    tuple(self.evaluate(d, env) for d in (m.defaults or []) if d is not None),
                )
                for m in node.methods
            }
            env.define(node.name, ZenClass(node.name, methods, parent_cls))
            return
        if isinstance(node, ast.ImportStmt):
            self._execute_import(node, env)
            return
        if isinstance(node, ast.IfStmt):
            for cond_node, block in node.branches:
                if is_truthy(self.evaluate(cond_node, env)):
                    self.execute_block(block, Environment(env))
                    return
            if node.else_block is not None:
                self.execute_block(node.else_block, Environment(env))
            return
        if isinstance(node, ast.WhileStmt):
            while is_truthy(self.evaluate(node.condition, env)):
                try:
                    self.execute_block(node.body, Environment(env))
                except _ContinueSignal:
                    continue
                except _BreakSignal:
                    break
            return
        if isinstance(node, ast.ForStmt):
            iterable = self.evaluate(node.iterable, env)
            if isinstance(iterable, str):
                items = list(iterable)
            elif isinstance(iterable, (list, range)):
                items = iterable
            elif isinstance(iterable, ZenArray):
                items = [iterable.get_item(i) for i in range(iterable.shape[0])]
            elif isinstance(iterable, dict):
                items = list(iterable.keys())
            else:
                raise ZenRuntimeError(
                    f"Cannot iterate over value of type '{zen_type(iterable)}'",
                    node.line,
                    node.column,
                )
            for item in items:
                loop_env = Environment(env)
                loop_env.define(node.var_name, item)
                try:
                    self.execute_block(node.body, loop_env)
                except _ContinueSignal:
                    continue
                except _BreakSignal:
                    break
            return
        if isinstance(node, ast.TryStmt):
            try:
                self.execute_block(node.try_block, Environment(env))
            except ZenRuntimeError as exc:
                catch_env = Environment(env)
                catch_env.define(node.catch_var, make_error_instance(exc, node.line))
                self.execute_block(node.catch_block, catch_env)
            return
        if isinstance(node, ast.RaiseStmt):
            val = self.evaluate(node.value, env)
            raise ZenRuntimeError(zen_format(val), node.line, node.column, code="E012")
        if isinstance(node, ast.ReturnStmt):
            val = None if node.value is None else self.evaluate(node.value, env)
            raise _ReturnSignal(val)
        if isinstance(node, ast.BreakStmt):
            raise _BreakSignal()
        if isinstance(node, ast.ContinueStmt):
            raise _ContinueSignal()
        if isinstance(node, ast.ExprStmt):
            self.evaluate(node.expression, env)
            return
        raise ZenRuntimeError(f"Unsupported statement node: {type(node).__name__}", node.line, node.column)

    def _execute_import(self, node: ast.ImportStmt, env: Environment) -> None:
        target = resolve_import_path(node.path, self.base_dir)
        if target is None:
            raise ZenRuntimeError(f"Imported file not found: '{node.path}'", node.line, node.column)

        if target in self._module_cache:
            module = self._module_cache[target]
        else:
            mod_interp = Interpreter(stdout=self.stdout, base_dir=target.parent, stderr=self.stderr, stdin=self.stdin, max_depth=self.max_depth)
            # Share core/registered globals builtins & module cache
            mod_interp._module_cache = self._module_cache
            for k, bval in self._builtin_values.items():
                mod_interp.globals.define(k, bval)
            mod_env = mod_interp.run_file(target)
            exports = {
                k: b.value
                for k, b in mod_env.values.items()
                if b.value is not self._builtin_values.get(k)
            }
            mod_name = target.parent.name if target.stem == "__init__" else target.stem
            module = ZenModule(mod_name, exports)
            self._module_cache[target] = module

        if node.alias:
            env.define(node.alias, module)
        else:
            for k, v in module.exports.items():
                env.define(k, v)

    def _assign_target(self, target: ast.Node, value: Any, env: Environment) -> None:
        if isinstance(target, ast.VarExpr):
            env.assign(target.name, value, target.line, target.column)
            return
        if isinstance(target, ast.IndexExpr):
            container = self.evaluate(target.target, env)
            idx = self.evaluate(target.index, env)
            if isinstance(container, ZenInstance):
                m = container.klass.find_method("__setitem__")
                if m is not None:
                    m.call(self, [idx, value], target.line, target.column, bound_self=container)
                    return
            if isinstance(container, ZenArray):
                container.set_item(idx, value)
                return
            if isinstance(container, list):
                if isinstance(idx, slice):
                    container[idx] = value
                    return
                if not isinstance(idx, int) or isinstance(idx, bool):
                    raise ZenRuntimeError("List index must be an integer", target.line, target.column)
                try:
                    container[idx] = value
                except IndexError as exc:
                    raise ZenRuntimeError(f"List index {idx} out of range", target.line, target.column) from exc
                return
            if isinstance(container, dict):
                container[idx] = value
                return
            raise ZenRuntimeError(
                f"Cannot assign by index on '{zen_type(container)}'",
                target.line,
                target.column,
            )
        if isinstance(target, ast.MemberExpr):
            obj = self.evaluate(target.target, env)
            if isinstance(obj, ZenInstance):
                obj.fields[target.member] = value
                return
            raise ZenRuntimeError(
                f"Cannot set property '{target.member}' on '{zen_type(obj)}'",
                target.line,
                target.column,
            )
        raise ZenRuntimeError("Invalid assignment target", target.line, target.column)

    def evaluate(self, node: ast.Node, env: Environment) -> Any:
        if node.line:
            self.current_line = node.line
        if isinstance(node, ast.Literal):
            return node.value
        if isinstance(node, ast.FStringExpr):
            return "".join(
                part if isinstance(part, str) else zen_format(self.evaluate(part, env))
                for part in node.parts
            )
        if isinstance(node, ast.VarExpr):
            return env.get(node.name, node.line, node.column)
        if isinstance(node, ast.ListExpr):
            return [self.evaluate(el, env) for el in node.elements]
        if isinstance(node, ast.ListCompExpr):
            iterable = self.evaluate(node.iterable, env)
            if isinstance(iterable, str):
                items = list(iterable)
            elif isinstance(iterable, (list, range)):
                items = iterable
            elif isinstance(iterable, ZenArray):
                items = [iterable.get_item(i) for i in range(iterable.shape[0])]
            elif isinstance(iterable, dict):
                items = list(iterable.keys())
            else:
                raise ZenRuntimeError(f"Cannot iterate over value of type '{zen_type(iterable)}'", node.line, node.column, code="E004")
            out_list: list[Any] = []
            for item in items:
                comp_env = Environment(env)
                comp_env.define(node.var_name, item)
                if node.condition is None or is_truthy(self.evaluate(node.condition, comp_env)):
                    out_list.append(self.evaluate(node.element, comp_env))
            return out_list
        if isinstance(node, ast.IfExpr):
            for cond_node, block in node.branches:
                if is_truthy(self.evaluate(cond_node, env)):
                    return self._eval_expr_block(block, Environment(env))
            return self._eval_expr_block(node.else_block, Environment(env))
        if isinstance(node, ast.MapExpr):
            result: dict[Any, Any] = {}
            for k_node, v_node in node.entries:
                key = self.evaluate(k_node, env)
                if isinstance(key, (list, dict)):
                    raise ZenRuntimeError("Map key cannot be a list or map", k_node.line, k_node.column)
                result[key] = self.evaluate(v_node, env)
            return result
        if isinstance(node, ast.FnExpr):
            defs = tuple(self.evaluate(d, env) for d in (node.defaults or []) if d is not None)
            return ZenFunction("<lambda>", node.params, node.body, env, defs)
        if isinstance(node, ast.SliceExpr):
            s = None if node.start is None else self.evaluate(node.start, env)
            e = None if node.stop is None else self.evaluate(node.stop, env)
            st = None if node.step is None else self.evaluate(node.step, env)
            return slice(s, e, st)
        if isinstance(node, ast.TupleExpr):
            return tuple(self.evaluate(el, env) for el in node.elements)
        if isinstance(node, ast.UnaryExpr):
            return self._eval_unary(node, env)
        if isinstance(node, ast.BinaryExpr):
            return self._eval_binary(node, env)
        if isinstance(node, ast.CallExpr):
            callee = self.evaluate(node.callee, env)
            args = [self.evaluate(a, env) for a in node.args]
            kw = [(k, self.evaluate(v, env)) for k, v in node.kwargs] if node.kwargs else None
            return self._call_value(callee, args, node.line, node.column, kwargs=kw)
        if isinstance(node, ast.IndexExpr):
            container = self.evaluate(node.target, env)
            idx = self.evaluate(node.index, env)
            return self._eval_index(container, idx, node.line, node.column)
        if isinstance(node, ast.MemberExpr):
            obj = self.evaluate(node.target, env)
            if isinstance(obj, (ZenInstance, ZenModule, ZenClass)):
                return obj.get(node.member, node.line, node.column)
            if isinstance(obj, ZenTensor):
                return self._tensor_member(obj, node.member, node.line, node.column)
            if isinstance(obj, ZenArray):
                return self._array_member(obj, node.member, node.line, node.column)
            if isinstance(obj, str):
                return str_member(obj, node.member, ZenBuiltin, node.line, node.column)
            if isinstance(obj, list):
                return list_member(obj, node.member, ZenBuiltin, node.line, node.column)
            raise ZenRuntimeError(
                f"Cannot access member '{node.member}' on '{zen_type(obj)}'",
                node.line,
                node.column,
                code="E003",
            )
        raise ZenRuntimeError(f"Unsupported expression node: {type(node).__name__}", node.line, node.column)

    def _eval_expr_block(self, block: ast.Block, env: Environment) -> Any:
        stmts = block.statements
        if stmts and isinstance(stmts[-1], ast.ExprStmt):
            for s in stmts[:-1]:
                self.execute(s, env)
            return self.evaluate(stmts[-1].expression, env)
        for s in stmts:
            self.execute(s, env)
        return None

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
                return m.call(self, args, line, col, bound_self=callee, kwargs=kwargs)
        if isinstance(callee, ZenBuiltin):
            kw_dict = dict(kwargs) if kwargs else None
            return callee.call(self, args, line, col, kwargs=kw_dict)
        if isinstance(callee, (ZenFunction, ZenBoundMethod, ZenClass)):
            return callee.call(self, args, line, col, kwargs=kwargs)
        raise ZenRuntimeError(f"Value of type '{zen_type(callee)}' is not callable", line, col, code="E004")

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

    def _eval_index(self, container: Any, idx: Any, line: int, col: int) -> Any:
        if isinstance(container, ZenInstance):
            m = container.klass.find_method("__getitem__")
            if m is not None:
                return m.call(self, [idx], line, col, bound_self=container)
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

    def _eval_unary(self, node: ast.UnaryExpr, env: Environment) -> Any:
        val = self.evaluate(node.operand, env)
        if node.op == "not":
            return not is_truthy(val)
        if node.op == "-":
            if isinstance(val, ZenInstance):
                neg_m = val.klass.find_method("__neg__")
                if neg_m is not None:
                    return neg_m.call(self, [], node.line, node.column, bound_self=val)
            if isinstance(val, ZenTensor):
                return val.mul(-1.0)
            if isinstance(val, ZenArray):
                return val.neg()
            if _is_number(val):
                return -val
            raise ZenRuntimeError(f"Cannot negate value of type '{zen_type(val)}'", node.line, node.column)
        raise ZenRuntimeError(f"Unknown unary operator '{node.op}'", node.line, node.column)

    def _try_dunder(self, left: Any, right: Any, dunder: str, rdunder: str, line: int, col: int) -> tuple[bool, Any]:
        if isinstance(left, ZenInstance):
            m = left.klass.find_method(dunder)
            if m is not None:
                return True, m.call(self, [right], line, col, bound_self=left)
        if isinstance(right, ZenInstance):
            rm = right.klass.find_method(rdunder)
            if rm is not None:
                return True, rm.call(self, [left], line, col, bound_self=right)
        return False, None

    def _eval_binary(self, node: ast.BinaryExpr, env: Environment) -> Any:
        if nan_trace.ENABLED and node.line:
            nan_trace.set_location(node.line, node.column)
        res = self._eval_binary_impl(node, env)
        if nan_trace.ENABLED:
            nan_trace.check_value(res, node.op, line=node.line, col=node.column)
        return res

    def _eval_binary_impl(self, node: ast.BinaryExpr, env: Environment) -> Any:
        if node.op == "and":
            left = self.evaluate(node.left, env)
            return self.evaluate(node.right, env) if is_truthy(left) else left
        if node.op == "or":
            left = self.evaluate(node.left, env)
            return left if is_truthy(left) else self.evaluate(node.right, env)

        left = self.evaluate(node.left, env)
        right = self.evaluate(node.right, env)
        op = node.op

        dunder_map = {
            "+": ("__add__", "__radd__"),
            "-": ("__sub__", "__rsub__"),
            "*": ("__mul__", "__rmul__"),
            "/": ("__truediv__", "__rtruediv__"),
            "**": ("__pow__", "__rpow__"),
            "@": ("__matmul__", "__rmatmul__"),
        }
        if op in dunder_map:
            handled, res = self._try_dunder(left, right, *dunder_map[op], node.line, node.column)
            if handled:
                return res

        if isinstance(left, ZenArray) or isinstance(right, ZenArray):
            if op in ("==", "!=") and (left is None or right is None or isinstance(left, (str, dict, ZenInstance, ZenFunction, ZenModule, ZenClass, ZenBuiltin)) or isinstance(right, (str, dict, ZenInstance, ZenFunction, ZenModule, ZenClass, ZenBuiltin))):
                return False if op == "==" else True
            if op in ("==", "!=", "<", "<=", ">", ">="):
                if isinstance(left, ZenArray):
                    return left.cmp(op, right)
                flip = {"<": ">", "<=": ">=", ">": "<", ">=": "<=", "==": "==", "!=": "!="}[op]
                return right.cmp(flip, left)
            if op == "@":
                if isinstance(left, ZenArray) and isinstance(right, ZenArray):
                    return left.matmul(right)
                raise ZenRuntimeError("Matrix multiply '@' requires two arrays", node.line, node.column)
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
                raise ZenRuntimeError("Matrix multiply '@' requires two Tensors", node.line, node.column)
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
                        node.line,
                        node.column,
                        code="E004",
                    )
                found = left in right
            elif isinstance(right, list):
                found = left in right
            elif isinstance(right, dict):
                if isinstance(left, (list, dict)):
                    found = False
                else:
                    found = left in right
            else:
                raise ZenRuntimeError(
                    f"Operator '{op}' requires list, string, or map on right, got '{zen_type(right)}'",
                    node.line,
                    node.column,
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
                node.line,
                node.column,
            )

        if op in ("-", "*", "/", "div", "%", "**", "<", "<=", ">", ">="):
            if not (_is_number(left) and _is_number(right)):
                raise ZenRuntimeError(
                    f"Operator '{op}' requires numbers, got '{zen_type(left)}' and '{zen_type(right)}'",
                    node.line,
                    node.column,
                    code="E004",
                )
            if op == "-":
                return left - right
            if op == "*":
                return left * right
            if op == "/":
                if right == 0:
                    raise ZenRuntimeError("Division by zero", node.line, node.column, code="E007")
                return left / right
            if op == "div":
                if right == 0:
                    raise ZenRuntimeError("Division by zero", node.line, node.column, code="E007")
                return left // right
            if op == "%":
                if right == 0:
                    raise ZenRuntimeError("Modulo by zero", node.line, node.column, code="E007")
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
            node.line,
            node.column,
        )
