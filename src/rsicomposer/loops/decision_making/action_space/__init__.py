"""Decision action definitions and validation, independent of tool execution."""

from collections.abc import Iterable
from copy import copy, deepcopy
from typing import Protocol

from rsicomposer.loops.contracts.decision import OperationCall, OperationSpec

from .access import ActionPermissions


class DecisionActionSpace(Protocol):
    def describe(self) -> tuple[OperationSpec, ...]: ...

    def validate(self, call: OperationCall) -> OperationCall: ...




class DecisionActionDefinition(Protocol):
    """Describe and normalize an operation without executing tools."""

    spec: OperationSpec

    def validate(self, arguments: dict[str, object]) -> OperationCall: ...


class ModularDecisionActionSpace:
    def __init__(self, definitions: Iterable[DecisionActionDefinition], *,
                 permissions: ActionPermissions | None = None):
        self.permissions = permissions if permissions is not None else ActionPermissions()
        if not isinstance(self.permissions, ActionPermissions):
            raise TypeError('Expected ActionPermissions')
        self._definitions: dict[str, DecisionActionDefinition] = {}
        self._add(definitions)

    def with_definitions(self, definitions):
        """Extend this endpoint's catalog without mutating another endpoint's space."""
        extended = copy(self)
        extended._definitions = self._definitions.copy()
        extended._add(definitions)
        return extended

    def _add(self, definitions):
        for definition in definitions:
            if not isinstance(definition.spec, OperationSpec):
                raise TypeError("Decision definitions require an OperationSpec")
            name = definition.spec.name
            if not isinstance(name, str) or not name:
                raise ValueError("Decision operation names must be nonempty strings")
            if name in self._definitions:
                raise ValueError("Decision operation names must be unique")
            if not callable(getattr(definition, "validate", None)):
                raise TypeError("Decision definitions require a validate method")
            self._definitions[name] = definition

    def describe(self) -> tuple[OperationSpec, ...]:
        return tuple(deepcopy(definition.spec) for definition in self._definitions.values())

    def validate(self, call: OperationCall) -> OperationCall:
        if not isinstance(call, OperationCall) or not isinstance(call.arguments, dict):
            raise TypeError("Expected an OperationCall with an argument mapping")
        if not isinstance(call.name, str) or call.name not in self._definitions:
            raise ValueError(f"Unknown decision operation: {call.name}")
        validated = self._definitions[call.name].validate(deepcopy(call.arguments))
        if not isinstance(validated, OperationCall) or not isinstance(validated.arguments, dict):
            raise TypeError("Definition must return an OperationCall with an argument mapping")
        if validated.name != call.name:
            raise ValueError("Validated operation name must match its definition")
        return deepcopy(validated)
