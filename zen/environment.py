"""Lexical environment (scopes and bindings) for the ZEN interpreter."""

from dataclasses import dataclass
from typing import Any
from zen.errors import ZenRuntimeError, did_you_mean


@dataclass
class _Binding:
    value: Any
    is_const: bool = False


class Environment:
    """Nested lexical scope supporting mutable (`let`) and immutable (`const`) bindings."""

    def __init__(self, parent: "Environment | None" = None) -> None:
        self.parent = parent
        self.values: dict[str, _Binding] = {}

    def all_names(self) -> set[str]:
        names = set(self.values.keys())
        if self.parent is not None:
            names.update(self.parent.all_names())
        return names

    def define(self, name: str, value: Any, *, is_const: bool = False) -> None:
        self.values[name] = _Binding(value, is_const)

    def get(self, name: str, line: int = 0, column: int = 0) -> Any:
        if name in self.values:
            return self.values[name].value
        if self.parent is not None:
            return self.parent.get(name, line, column)
        hint = did_you_mean(name, self.all_names())
        suffix = f" Did you mean '{hint}'?" if hint else ""
        raise ZenRuntimeError(f"Undefined variable '{name}'.{suffix}" if suffix else f"Undefined variable '{name}'", line, column, code="E002")

    def assign(self, name: str, value: Any, line: int = 0, column: int = 0) -> None:
        if name in self.values:
            binding = self.values[name]
            if binding.is_const:
                raise ZenRuntimeError(f"Cannot reassign constant '{name}'", line, column, code="E005")
            binding.value = value
            return
        if self.parent is not None:
            self.parent.assign(name, value, line, column)
            return
        hint = did_you_mean(name, self.all_names())
        suffix = f" Did you mean '{hint}'?" if hint else ""
        raise ZenRuntimeError(f"Cannot assign to undefined variable '{name}'.{suffix}" if suffix else f"Cannot assign to undefined variable '{name}'", line, column, code="E002")
