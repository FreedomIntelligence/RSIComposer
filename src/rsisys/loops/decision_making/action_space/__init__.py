"""Decision action definitions and validation, independent of tool execution."""

from collections.abc import Iterable
from copy import deepcopy
from dataclasses import dataclass
from typing import Protocol

from rsisys.loops.contracts.artifacts import ArtifactRef
from rsisys.loops.contracts.directive import Directive
from rsisys.loops.contracts.lifecycle import ExecutionStatus


@dataclass(frozen=True)
class OperationSpec:
    name: str
    description: str
    input_schema: dict[str, object]


@dataclass(frozen=True)
class OperationCall:
    name: str
    arguments: dict[str, object]


@dataclass(frozen=True)
class OperationResult:
    summary: str
    artifacts: tuple[ArtifactRef, ...]
    simulated: bool
    status: ExecutionStatus = ExecutionStatus.COMPLETED
    error: str | None = None

    def __post_init__(self):
        if self.status not in (ExecutionStatus.COMPLETED, ExecutionStatus.FAILED, ExecutionStatus.CANCELLED):
            raise ValueError("An operation result must be terminal")
        if self.status == ExecutionStatus.FAILED and not self.error:
            raise ValueError("A failed operation requires an error")
        if self.status == ExecutionStatus.COMPLETED and self.error:
            raise ValueError("A completed operation cannot carry an error")


DEFAULT_DIRECTIVE_MODES = ("execute", "sft", "rl", "evaluate", "configure_prompt")


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
                 directive_modes: tuple[str, ...] = DEFAULT_DIRECTIVE_MODES):
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
        if isinstance(directive_modes, str):
            raise TypeError("directive_modes must be a collection of mode names")
        self._directive_modes = tuple(directive_modes)
        if any(not isinstance(mode, str) or mode not in DEFAULT_DIRECTIVE_MODES
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


class _StringArgumentDefinition:
    def __init__(self, name: str, description: str, argument: str, *, nonempty: bool = False):
        self.argument, self.nonempty = argument, nonempty
        self.spec = OperationSpec(name, description, {
            "type": "object", "properties": {argument: {"type": "string"}},
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
    def __init__(self, *, directive_modes: tuple[str, ...] = DEFAULT_DIRECTIVE_MODES):
        super().__init__((_StringArgumentDefinition(
            "prepare_data", "Prepare a demonstration input artifact", "text",
        ),), directive_modes=directive_modes)


class LocalDecisionActionSpace(ModularDecisionActionSpace):
    def __init__(self, *, directive_modes: tuple[str, ...] = DEFAULT_DIRECTIVE_MODES):
        super().__init__((
            _StringArgumentDefinition("prepare_data", "Store text as a reusable input artifact", "text"),
            _StringArgumentDefinition("search_literature", "Search explicitly imported text literature",
                                      "query", nonempty=True),
            _StringArgumentDefinition("store_strategy", "Save an explicit strategy as an artifact",
                                      "strategy", nonempty=True),
        ), directive_modes=directive_modes)
