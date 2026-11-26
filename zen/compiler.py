"""AST-to-Bytecode compiler for the ZEN VM."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from zen import ast
from zen.bytecode import CodeObject, Instruction, Op
from zen.errors import ZenRuntimeError


@dataclass
class _VarInfo:
    kind: str       # "local" | "upvalue" | "global"
    index: int | str
    is_const: bool


class _FunctionCompiler:
    def __init__(
        self,
        name: str,
        params: list[str],
        parent: "_FunctionCompiler | None" = None,
        is_module: bool = False,
        num_defaults: int = 0,
    ) -> None:
        self.name = name
        self.params = params
        self.parent = parent
        self.is_module = is_module
        self.num_defaults = num_defaults

        self.instructions: list[Instruction] = []
        self.scopes: list[dict[str, tuple[int, bool]]] = [{}]
        self.slot_is_const: list[bool] = []
        self.slot_is_cell: list[bool] = []
        self.upvalues: list[tuple[bool, int]] = []
        self.upvalue_is_const: list[bool] = []
        self.loops: list[tuple[int, list[int], int, int]] = []  # (continue_ip, break_patches, stack_depth_delta, try_depth)
        self._try_depth: int = 0

        # Register parameters as local slots in scope 0
        for p in params:
            self._alloc_local(p, is_const=False)

    def _emit(self, op: Op, arg: Any = None, node: ast.Node | None = None) -> int:
        line = node.line if node else 0
        col = node.column if node else 0
        self.instructions.append(Instruction(op, arg, line, col))
        return len(self.instructions) - 1

    def _patch(self, instr_idx: int, target: int) -> None:
        self.instructions[instr_idx].arg = target

    def _push_scope(self) -> None:
        self.scopes.append({})

    def _pop_scope(self) -> None:
        self.scopes.pop()

    def _alloc_local(self, name: str, is_const: bool) -> int:
        # Allow re-declaring in same block scope or allocate fresh slot
        cur = self.scopes[-1]
        if name in cur:
            slot, _ = cur[name]
            self.slot_is_const[slot] = is_const
            cur[name] = (slot, is_const)
            return slot
        slot = len(self.slot_is_const)
        self.slot_is_const.append(is_const)
        self.slot_is_cell.append(False)
        cur[name] = (slot, is_const)
        return slot

    def _find_local(self, name: str) -> tuple[int, bool] | None:
        for scope in reversed(self.scopes):
            if name in scope:
                return scope[name]
        return None

    def _resolve_upvalue(self, name: str) -> tuple[int, bool] | None:
        if self.parent is None:
            return None
        local = self.parent._find_local(name)
        if local is not None:
            slot, is_const = local
            self.parent.slot_is_cell[slot] = True
            return self._add_upvalue(True, slot, is_const)

        up = self.parent._resolve_upvalue(name)
        if up is not None:
            up_idx, is_const = up
            return self._add_upvalue(False, up_idx, is_const)
        return None

    def _add_upvalue(self, is_local: bool, index: int, is_const: bool) -> tuple[int, bool]:
        pair = (is_local, index)
        for i, existing in enumerate(self.upvalues):
            if existing == pair:
                return i, self.upvalue_is_const[i]
        self.upvalues.append(pair)
        self.upvalue_is_const.append(is_const)
        return len(self.upvalues) - 1, is_const

    def _resolve_var(self, name: str) -> _VarInfo:
        local = self._find_local(name)
        if local is not None:
            return _VarInfo("local", local[0], local[1])
        up = self._resolve_upvalue(name)
        if up is not None:
            return _VarInfo("upvalue", up[0], up[1])
        return _VarInfo("global", name, False)

    def _is_module_top_level(self) -> bool:
        return self.is_module and len(self.scopes) == 1

    def compile_program(self, program: ast.Program) -> CodeObject:
        for stmt in program.statements:
            self.compile_stmt(stmt)
        self._emit(Op.LOAD_CONST, None)
        self._emit(Op.RETURN)
        self._finalize_cells()
        return self._to_code_object()

    def _to_code_object(self) -> CodeObject:
        return CodeObject(
            name=self.name,
            params=self.params,
            instructions=self.instructions,
            num_locals=len(self.slot_is_const),
            cell_slots=tuple(self.slot_is_cell),
            upvalues=tuple(self.upvalues),
            is_module=self.is_module,
            num_defaults=self.num_defaults,
        ).finalize_flat()

    def compile_block(self, block: ast.Block) -> None:
        self._push_scope()
        for stmt in block.statements:
            self.compile_stmt(stmt)
        self._pop_scope()

    def compile_stmt(self, node: ast.Node) -> None:
        if isinstance(node, (ast.LetStmt, ast.ConstStmt)):
            is_const = isinstance(node, ast.ConstStmt)
            self.compile_expr(node.value)
            if self._is_module_top_level():
                self._emit(Op.DEFINE_GLOBAL, (node.name, is_const), node)
            else:
                slot = self._alloc_local(node.name, is_const)
                self._emit(Op.STORE_LOCAL, (slot, is_const, True), node)  # True = is_decl
            return

        if isinstance(node, ast.UnpackStmt):
            self.compile_expr(node.value)
            self._emit(Op.UNPACK_SEQ, len(node.names), node)
            for name in node.names:
                if self._is_module_top_level():
                    self._emit(Op.DEFINE_GLOBAL, (name, node.is_const), node)
                else:
                    slot = self._alloc_local(name, node.is_const)
                    self._emit(Op.STORE_LOCAL, (slot, node.is_const, True), node)
            return

        if isinstance(node, ast.UnpackAssignStmt):
            self.compile_expr(node.value)
            self._emit(Op.UNPACK_SEQ, len(node.targets), node)
            for target in node.targets:
                if isinstance(target, ast.VarExpr):
                    info = self._resolve_var(target.name)
                    if info.is_const:
                        raise ZenRuntimeError(f"Cannot reassign constant '{target.name}'", target.line, target.column, code="E005")
                    if info.kind == "local":
                        self._emit(Op.STORE_LOCAL, (info.index, info.is_const, False), target)
                    elif info.kind == "upvalue":
                        self._emit(Op.STORE_UPVALUE, info.index, target)
                    else:
                        self._emit(Op.STORE_GLOBAL, info.index, target)
                else:
                    tmp_name = f"$unp_{id(target)}_{len(self.instructions)}"
                    slot = self._alloc_local(tmp_name, False)
                    self._emit(Op.STORE_LOCAL, (slot, False, True), target)
                    if isinstance(target, ast.IndexExpr):
                        self.compile_expr(target.target)
                        self.compile_expr(target.index)
                        self._emit(Op.LOAD_LOCAL, slot, target)
                        self._emit(Op.SET_INDEX, None, target)
                    elif isinstance(target, ast.MemberExpr):
                        self.compile_expr(target.target)
                        self._emit(Op.LOAD_LOCAL, slot, target)
                        self._emit(Op.SET_MEMBER, target.member, target)
                    else:
                        raise ZenRuntimeError("Invalid assignment target", target.line, target.column)
            return

        if isinstance(node, ast.AssignStmt):
            target = node.target
            if isinstance(target, ast.VarExpr):
                self.compile_expr(node.value)
                info = self._resolve_var(target.name)
                if info.is_const:
                    raise ZenRuntimeError(f"Cannot reassign constant '{target.name}'", target.line, target.column)
                if info.kind == "local":
                    self._emit(Op.STORE_LOCAL, (info.index, info.is_const, False), target)
                elif info.kind == "upvalue":
                    self._emit(Op.STORE_UPVALUE, info.index, target)
                else:
                    self._emit(Op.STORE_GLOBAL, info.index, target)
                return
            if isinstance(target, ast.IndexExpr):
                self.compile_expr(target.target)
                self.compile_expr(target.index)
                self.compile_expr(node.value)
                self._emit(Op.SET_INDEX, None, target)
                return
            if isinstance(target, ast.MemberExpr):
                self.compile_expr(target.target)
                self.compile_expr(node.value)
                self._emit(Op.SET_MEMBER, target.member, target)
                return
            raise ZenRuntimeError("Invalid assignment target", target.line, target.column)

        if isinstance(node, ast.SayStmt):
            for val in node.values:
                self.compile_expr(val)
            self._emit(Op.SAY, len(node.values), node)
            return

        if isinstance(node, ast.AssertStmt):
            self.compile_expr(node.condition)
            if node.message is not None:
                self.compile_expr(node.message)
                self._emit(Op.ASSERT, True, node)
            else:
                self._emit(Op.ASSERT, False, node)
            return

        if isinstance(node, ast.FnDecl):
            defs = [d for d in (node.defaults or []) if d is not None]
            if self._is_module_top_level():
                for d in defs:
                    self.compile_expr(d)
                fn_code = self._compile_function(node.name, node.params, node.body, len(defs))
                self._emit(Op.MAKE_CLOSURE, fn_code, node)
                self._emit(Op.DEFINE_GLOBAL, (node.name, False), node)
            else:
                slot = self._alloc_local(node.name, False)
                # Pre-allocate local slot/cell so recursive calls inside body capture the Cell
                self._emit(Op.LOAD_CONST, None, node)
                self._emit(Op.STORE_LOCAL, (slot, False, True), node)
                for d in defs:
                    self.compile_expr(d)
                fn_code = self._compile_function(node.name, node.params, node.body, len(defs))
                self._emit(Op.MAKE_CLOSURE, fn_code, node)
                self._emit(Op.STORE_LOCAL, (slot, False, False), node)
            return

        if isinstance(node, ast.ClassDecl):
            method_names: list[str] = []
            for m in node.methods:
                m_defs = [d for d in (m.defaults or []) if d is not None]
                for d in m_defs:
                    self.compile_expr(d)
                m_code = self._compile_function(m.name, m.params, m.body, len(m_defs))
                self._emit(Op.MAKE_CLOSURE, m_code, m)
                method_names.append(m.name)
            has_super = node.superclass is not None
            if node.superclass is not None:
                self.compile_expr(node.superclass)
            self._emit(Op.BUILD_CLASS, (node.name, tuple(method_names), has_super), node)
            if self._is_module_top_level():
                self._emit(Op.DEFINE_GLOBAL, (node.name, False), node)
            else:
                slot = self._alloc_local(node.name, False)
                self._emit(Op.STORE_LOCAL, (slot, False, True), node)
            return

        if isinstance(node, ast.ImportStmt):
            self._emit(Op.IMPORT, (node.path, node.alias), node)
            return

        if isinstance(node, ast.IfStmt):
            end_jumps: list[int] = []
            for cond, block in node.branches:
                self.compile_expr(cond)
                jf = self._emit(Op.JUMP_IF_FALSE, 0, cond)
                self.compile_block(block)
                end_jumps.append(self._emit(Op.JUMP, 0, block))
                self._patch(jf, len(self.instructions))
            if node.else_block is not None:
                self.compile_block(node.else_block)
            end_ip = len(self.instructions)
            for j in end_jumps:
                self._patch(j, end_ip)
            return

        if isinstance(node, ast.WhileStmt):
            loop_start = len(self.instructions)
            self.compile_expr(node.condition)
            exit_jump = self._emit(Op.JUMP_IF_FALSE, 0, node.condition)
            self.loops.append((loop_start, [], 0, self._try_depth))
            self.compile_block(node.body)
            self._emit(Op.JUMP, loop_start, node)
            loop_end = len(self.instructions)
            self._patch(exit_jump, loop_end)
            _, breaks, _, _ = self.loops.pop()
            for br in breaks:
                self._patch(br, loop_end)
            return

        if isinstance(node, ast.ForStmt):
            self.compile_expr(node.iterable)
            self._emit(Op.GET_ITER, None, node)
            loop_start = len(self.instructions)
            for_iter = self._emit(Op.FOR_ITER, 0, node)
            self._push_scope()
            slot = self._alloc_local(node.var_name, False)
            self._emit(Op.STORE_LOCAL, (slot, False, True), node)
            self.loops.append((loop_start, [], 1, self._try_depth))  # 1 iterator on stack
            for stmt in node.body.statements:
                self.compile_stmt(stmt)
            self._pop_scope()
            self._emit(Op.JUMP, loop_start, node)
            loop_end = len(self.instructions)
            self._patch(for_iter, loop_end)
            _, breaks, _, _ = self.loops.pop()
            if breaks:
                # Normal loop exit skips POP; break jumps directly to POP to clean up iterator
                skip_pop = self._emit(Op.JUMP, 0, node)
                pop_ip = len(self.instructions)
                for br in breaks:
                    self._patch(br, pop_ip)
                self._emit(Op.POP, None, node)
                self._patch(skip_pop, len(self.instructions))
            return

        if isinstance(node, ast.TryStmt):
            push_try = self._emit(Op.PUSH_TRY, 0, node)
            self._try_depth += 1
            self.compile_block(node.try_block)
            self._try_depth -= 1
            self._emit(Op.POP_TRY, None, node)
            end_jump = self._emit(Op.JUMP, 0, node)
            catch_ip = len(self.instructions)
            self._patch(push_try, catch_ip)
            self._push_scope()
            slot = self._alloc_local(node.catch_var, False)
            self._emit(Op.STORE_LOCAL, (slot, False, True), node)
            for stmt in node.catch_block.statements:
                self.compile_stmt(stmt)
            self._pop_scope()
            self._patch(end_jump, len(self.instructions))
            return

        if isinstance(node, ast.RaiseStmt):
            self.compile_expr(node.value)
            self._emit(Op.RAISE, None, node)
            return

        if isinstance(node, ast.ReturnStmt):
            if node.value is None:
                self._emit(Op.LOAD_CONST, None, node)
            else:
                self.compile_expr(node.value)
            self._emit(Op.RETURN, None, node)
            return

        if isinstance(node, ast.BreakStmt):
            if not self.loops:
                raise ZenRuntimeError("'break' used outside of loop", node.line, node.column, code="E011")
            for _ in range(self._try_depth - self.loops[-1][3]):
                self._emit(Op.POP_TRY, None, node)
            br = self._emit(Op.JUMP, 0, node)
            self.loops[-1][1].append(br)
            return

        if isinstance(node, ast.ContinueStmt):
            if not self.loops:
                raise ZenRuntimeError("'continue' used outside of loop", node.line, node.column, code="E011")
            for _ in range(self._try_depth - self.loops[-1][3]):
                self._emit(Op.POP_TRY, None, node)
            cont_ip = self.loops[-1][0]
            self._emit(Op.JUMP, cont_ip, node)
            return

        if isinstance(node, ast.ExprStmt):
            self.compile_expr(node.expression)
            self._emit(Op.POP, None, node)
            return

        raise ZenRuntimeError(f"Unsupported statement in compiler: {type(node).__name__}", node.line, node.column)

    def _compile_block_expr(self, block: ast.Block) -> None:
        self._push_scope()
        stmts = block.statements
        if stmts and isinstance(stmts[-1], ast.ExprStmt):
            for s in stmts[:-1]:
                self.compile_stmt(s)
            self.compile_expr(stmts[-1].expression)
        else:
            for s in stmts:
                self.compile_stmt(s)
            self._emit(Op.LOAD_CONST, None, block)
        self._pop_scope()

    def _compile_function(self, name: str, params: list[str], body: ast.Block, num_defaults: int = 0) -> CodeObject:
        child = _FunctionCompiler(name, params, parent=self, is_module=False, num_defaults=num_defaults)
        for stmt in body.statements:
            child.compile_stmt(stmt)
        child._emit(Op.LOAD_CONST, None, body)
        child._emit(Op.RETURN, None, body)
        child._finalize_cells()
        return child._to_code_object()

    def _finalize_cells(self) -> None:
        # Patch LOAD_LOCAL/STORE_LOCAL on slots that turned out to be captured as cells
        for instr in self.instructions:
            if instr.op == Op.LOAD_LOCAL and self.slot_is_cell[instr.arg]:
                instr.op = Op.LOAD_CELL
            elif instr.op == Op.STORE_LOCAL:
                slot, is_const, is_decl = instr.arg
                if self.slot_is_cell[slot]:
                    if is_decl:
                        instr.op = Op.DEFINE_CELL
                        instr.arg = (slot, is_const)
                    else:
                        instr.op = Op.STORE_CELL
                        instr.arg = slot
                else:
                    instr.arg = (slot, is_const)

    def compile_expr(self, node: ast.Node) -> None:
        if isinstance(node, ast.Literal):
            self._emit(Op.LOAD_CONST, node.value, node)
            return

        if isinstance(node, ast.FStringExpr):
            for part in node.parts:
                if isinstance(part, str):
                    self._emit(Op.LOAD_CONST, part, node)
                else:
                    self.compile_expr(part)
            self._emit(Op.BUILD_FSTRING, len(node.parts), node)
            return

        if isinstance(node, ast.VarExpr):
            info = self._resolve_var(node.name)
            if info.kind == "local":
                self._emit(Op.LOAD_LOCAL, info.index, node)
            elif info.kind == "upvalue":
                self._emit(Op.LOAD_UPVALUE, info.index, node)
            else:
                self._emit(Op.LOAD_GLOBAL, info.index, node)
            return

        if isinstance(node, ast.ListExpr):
            for el in node.elements:
                self.compile_expr(el)
            self._emit(Op.BUILD_LIST, len(node.elements), node)
            return

        if isinstance(node, ast.ListCompExpr):
            self._emit(Op.BUILD_LIST, 0, node)
            self.compile_expr(node.iterable)
            self._emit(Op.GET_ITER, None, node)
            loop_start = len(self.instructions)
            for_iter = self._emit(Op.FOR_ITER, 0, node)
            self._push_scope()
            slot = self._alloc_local(node.var_name, False)
            self._emit(Op.STORE_LOCAL, (slot, False, True), node)
            if node.condition is not None:
                self.compile_expr(node.condition)
                skip = self._emit(Op.JUMP_IF_FALSE, 0, node.condition)
                self.compile_expr(node.element)
                self._emit(Op.LIST_APPEND, 2, node)
                self._patch(skip, len(self.instructions))
            else:
                self.compile_expr(node.element)
                self._emit(Op.LIST_APPEND, 2, node)
            self._pop_scope()
            self._emit(Op.JUMP, loop_start, node)
            self._patch(for_iter, len(self.instructions))
            return

        if isinstance(node, ast.IfExpr):
            end_jumps: list[int] = []
            for cond, block in node.branches:
                self.compile_expr(cond)
                jf = self._emit(Op.JUMP_IF_FALSE, 0, cond)
                self._compile_block_expr(block)
                end_jumps.append(self._emit(Op.JUMP, 0, block))
                self._patch(jf, len(self.instructions))
            self._compile_block_expr(node.else_block)
            end_ip = len(self.instructions)
            for j in end_jumps:
                self._patch(j, end_ip)
            return

        if isinstance(node, ast.MapExpr):
            for k, v in node.entries:
                self.compile_expr(k)
                self.compile_expr(v)
            self._emit(Op.BUILD_MAP, len(node.entries), node)
            return

        if isinstance(node, ast.FnExpr):
            defs = [d for d in (node.defaults or []) if d is not None]
            for d in defs:
                self.compile_expr(d)
            fn_code = self._compile_function("<lambda>", node.params, node.body, len(defs))
            self._emit(Op.MAKE_CLOSURE, fn_code, node)
            return

        if isinstance(node, ast.SliceExpr):
            if node.start is not None:
                self.compile_expr(node.start)
            else:
                self._emit(Op.LOAD_CONST, None, node)
            if node.stop is not None:
                self.compile_expr(node.stop)
            else:
                self._emit(Op.LOAD_CONST, None, node)
            if node.step is not None:
                self.compile_expr(node.step)
            else:
                self._emit(Op.LOAD_CONST, None, node)
            self._emit(Op.BUILD_SLICE, None, node)
            return

        if isinstance(node, ast.TupleExpr):
            for el in node.elements:
                self.compile_expr(el)
            self._emit(Op.BUILD_TUPLE, len(node.elements), node)
            return

        if isinstance(node, ast.UnaryExpr):
            self.compile_expr(node.operand)
            self._emit(Op.UNARY_OP, node.op, node)
            return

        if isinstance(node, ast.BinaryExpr):
            if node.op == "and":
                self.compile_expr(node.left)
                jf = self._emit(Op.JUMP_IF_FALSE_OR_POP, 0, node)
                self.compile_expr(node.right)
                self._patch(jf, len(self.instructions))
                return
            if node.op == "or":
                self.compile_expr(node.left)
                jt = self._emit(Op.JUMP_IF_TRUE_OR_POP, 0, node)
                self.compile_expr(node.right)
                self._patch(jt, len(self.instructions))
                return
            self.compile_expr(node.left)
            self.compile_expr(node.right)
            self._emit(Op.BINARY_OP, node.op, node)
            return

        if isinstance(node, ast.CallExpr):
            self.compile_expr(node.callee)
            for a in node.args:
                self.compile_expr(a)
            if node.kwargs:
                for _, kv in node.kwargs:
                    self.compile_expr(kv)
                self._emit(Op.CALL_KW, (len(node.args), tuple(k for k, _ in node.kwargs)), node)
            else:
                self._emit(Op.CALL, len(node.args), node)
            return

        if isinstance(node, ast.IndexExpr):
            self.compile_expr(node.target)
            self.compile_expr(node.index)
            self._emit(Op.GET_INDEX, None, node)
            return

        if isinstance(node, ast.MemberExpr):
            self.compile_expr(node.target)
            self._emit(Op.GET_MEMBER, node.member, node)
            return

        raise ZenRuntimeError(f"Unsupported expression in compiler: {type(node).__name__}", node.line, node.column)


def compile_program(program: ast.Program, name: str = "<module>") -> CodeObject:
    """Compile an AST Program into a top-level CodeObject."""
    comp = _FunctionCompiler(name, [], parent=None, is_module=True)
    return comp.compile_program(program)
