"""Concrete artifact and literature operations for the bundled decision harness."""
from collections.abc import Iterable

from rsisys.loops.contracts.decision import OperationCall, OperationSpec
from rsisys.loops.decision_making.action_space import (
    DecisionActionDefinition,
    ModularDecisionActionSpace,
)
from rsisys.loops.decision_making.action_space.access import ActionPermissions


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
    def __init__(self, *,
                 extra_definitions: Iterable[DecisionActionDefinition] = (),
                 permissions: ActionPermissions | None = None):
        super().__init__((_StringArgumentDefinition(
            "prepare_data", "Prepare a demonstration input artifact", "text",
        ), *extra_definitions), permissions=permissions)


class LocalDecisionActionSpace(ModularDecisionActionSpace):
    def __init__(self, *,
                 extra_definitions: Iterable[DecisionActionDefinition] = (),
                 permissions: ActionPermissions | None = None):
        super().__init__((
            _StringArgumentDefinition("prepare_data", "Store text as a reusable input artifact", "text"),
            _StringArgumentDefinition("search_literature", "Search explicitly imported text literature",
                                      "query", nonempty=True),
            _StringArgumentDefinition("store_strategy", "Save an explicit strategy as an artifact",
                                      "strategy", nonempty=True),
            *extra_definitions,
        ), permissions=permissions)
