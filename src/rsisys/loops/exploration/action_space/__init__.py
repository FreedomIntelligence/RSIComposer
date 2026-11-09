"""Actor action contracts and catalog validation; no environment implementation."""
from dataclasses import dataclass, field
from typing import Protocol

from rsisys.loops.contracts.model_service import ModelRef, TokenUsage


@dataclass(frozen=True)
class ActionSpec:
    name: str
    description: str
    input_schema: dict[str, object]


@dataclass(frozen=True)
class ActionProposal:
    name: str
    arguments: dict[str, object]
    simulated: bool
    model: ModelRef | None = None
    usage: TokenUsage | None = None


@dataclass(frozen=True)
class Action:
    name: str
    text: str = ""
    arguments: dict[str, object] = field(default_factory=dict)


class ActionSpace(Protocol):
    def describe(self) -> tuple[ActionSpec, ...]: ...

    def validate(self, proposal: ActionProposal, allowed_actions: tuple[str, ...]) -> Action: ...


class ActionDefinition(Protocol):
    spec: ActionSpec
    def validate(self, arguments: dict[str, object]) -> Action: ...


class ModularActionSpace:
    """Validates configured actions; execution remains exclusively in Environment."""

    def __init__(self, definitions: tuple[ActionDefinition, ...]):
        self._definitions = {definition.spec.name: definition for definition in definitions}
        if len(self._definitions) != len(definitions):
            raise ValueError('Action names must be unique')

    def describe(self) -> tuple[ActionSpec, ...]:
        from copy import deepcopy
        return tuple(deepcopy(definition.spec) for definition in self._definitions.values())

    def validate(self, proposal: ActionProposal, allowed_actions: tuple[str, ...]) -> Action:
        if not isinstance(proposal, ActionProposal) or not isinstance(proposal.arguments, dict):
            raise TypeError('Expected an ActionProposal with an argument mapping')
        if proposal.name not in allowed_actions or proposal.name not in self._definitions:
            raise ValueError(f'Unsupported or disallowed action: {proposal.name}')
        action = self._definitions[proposal.name].validate(proposal.arguments)
        if not isinstance(action, Action) or action.name != proposal.name:
            raise ValueError('Validated action must preserve its proposed identity')
        return action
