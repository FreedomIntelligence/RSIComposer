"""Decision action definitions and validation, independent of tool execution."""

from collections.abc import Iterable
from copy import deepcopy
from typing import Protocol

from rsisys.loops.contracts.directive import Directive
from rsisys.loops.contracts.coordination import OperationResult
from rsisys.loops.contracts.decision import OperationCall, OperationSpec
from .directives import DirectiveDefinitions, DEFAULT_DIRECTIVE_MODES
from .access import ActionPermissions


class DecisionActionSpace(Protocol):
    def describe(self) -> tuple[OperationSpec, ...]: ...

    def validate(self, call: OperationCall) -> OperationCall: ...

    def directive_modes(self) -> tuple[str, ...]: ...

    def validate_directive(self, directive: Directive) -> None: ...


class DecisionActionDefinition(Protocol):
    """Describe and normalize an operation without executing tools."""

    spec: OperationSpec

    def validate(self, arguments: dict[str, object]) -> OperationCall: ...


class ModularDecisionActionSpace:
    def __init__(self, definitions: Iterable[DecisionActionDefinition], *,
                 directive_modes: tuple[str, ...] | None = None,
                 directives: DirectiveDefinitions | None = None,
                 permissions: ActionPermissions | None = None):
        self.permissions = permissions if permissions is not None else ActionPermissions()
        if not isinstance(self.permissions, ActionPermissions):
            raise TypeError('Expected ActionPermissions')
        self._definitions: dict[str, DecisionActionDefinition] = {}
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
        self.directives = directives if directives is not None else DirectiveDefinitions()
        if not isinstance(self.directives, DirectiveDefinitions):
            raise TypeError("Expected a DirectiveDefinitions catalog")
        if directive_modes is None:
            directive_modes = self.directives.modes()
        if isinstance(directive_modes, str):
            raise TypeError("directive_modes must be a collection of mode names")
        self._directive_modes = tuple(directive_modes)
        if any(not isinstance(mode, str) or mode not in self.directives.modes()
               for mode in self._directive_modes):
            raise ValueError("Unsupported directive mode")
        if len(set(self._directive_modes)) != len(self._directive_modes):
            raise ValueError("Directive modes must be unique")

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

    def directive_modes(self) -> tuple[str, ...]:
        return self._directive_modes

    def validate_directive(self, directive: Directive) -> None:
        # Exploration owns payload validation for each supported mode.
        if directive.mode not in self._directive_modes:
            raise ValueError(f"Unsupported directive mode: {directive.mode}")
        if directive.input_refs and not self.directives.describe((directive.mode,))[0].accepts_input_refs:
            raise ValueError(f"{directive.mode} does not accept input references")


class _StringArgumentDefinition:
    def __init__(self, name: str, description: str, argument: str, *, nonempty: bool = False):
        self.argument, self.nonempty = argument, nonempty
        self.spec = OperationSpec(name, description, {
            "type": "object", "properties": {argument: {
                "type": "string", **({"minLength": 1, "pattern": r'\S'} if nonempty else {})}},
            "required": [argument], "additionalProperties": False,
        })

    def validate(self, arguments: dict[str, object]) -> OperationCall:
        if (not isinstance(arguments, dict) or set(arguments) != {self.argument}
                or not isinstance(arguments[self.argument], str)):
            raise ValueError(f"{self.spec.name} requires exactly one string argument: {self.argument}")
        if self.nonempty and not arguments[self.argument].strip():
            raise ValueError(f"{self.spec.name} requires a nonempty {self.argument}")
        return OperationCall(self.spec.name, dict(arguments))


class DemoDecisionActionSpace(ModularDecisionActionSpace):
    def __init__(self, *, directive_modes: tuple[str, ...] | None = None,
                 directives: DirectiveDefinitions | None = None,
                 extra_definitions: Iterable[DecisionActionDefinition] = (),
                 permissions: ActionPermissions | None = None):
        super().__init__((_StringArgumentDefinition(
            "prepare_data", "Prepare a demonstration input artifact", "text",
        ), *extra_definitions), directive_modes=directive_modes, directives=directives, permissions=permissions)


class LocalDecisionActionSpace(ModularDecisionActionSpace):
    def __init__(self, *, directive_modes: tuple[str, ...] | None = None,
                 directives: DirectiveDefinitions | None = None,
                 extra_definitions: Iterable[DecisionActionDefinition] = (),
                 permissions: ActionPermissions | None = None):
        super().__init__((
            _StringArgumentDefinition("prepare_data", "Store text as a reusable input artifact", "text"),
            _StringArgumentDefinition("search_literature", "Search explicitly imported text literature",
                                      "query", nonempty=True),
            _StringArgumentDefinition("store_strategy", "Save an explicit strategy as an artifact",
                                      "strategy", nonempty=True),
            *extra_definitions,
        ), directive_modes=directive_modes, directives=directives, permissions=permissions)
