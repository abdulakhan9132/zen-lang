"""Lexical environment (scopes and bindings) for the ZEN interpreter."""

from dataclasses import dataclass
from typing import Any
from zen.errors import ZenRuntimeError


@dataclass
class _Binding:
    value: Any
    is_const: bool = False


class Environment:
    """Nested lexical scope supporting mutable (`let`) and immutable (`const`) bindings."""

    def __init__(self, parent: "Environment | None" = None) -> None:
        self.parent = parent
        self.values: dict[str, _Binding] = {}

    def define(self, name: str, value: Any, *, is_const: bool = False) -> None:
        self.values[name] = _Binding(value, is_const)

    def get(self, name: str, line: int = 0, column: int = 0) -> Any:
        if name in self.values:
            return self.values[name].value
        if self.parent is not None:
            return self.parent.get(name, line, column)
        raise ZenRuntimeError(f"Undefined variable '{name}'", line, column)

    def assign(self, name: str, value: Any, line: int = 0, column: int = 0) -> None:
        if name in self.values:
            binding = self.values[name]
            if binding.is_const:
                raise ZenRuntimeError(f"Cannot reassign constant '{name}'", line, column)
            binding.value = value
            return
        if self.parent is not None:
            self.parent.assign(name, value, line, column)
            return
        raise ZenRuntimeError(f"Cannot assign to undefined variable '{name}'", line, column)
