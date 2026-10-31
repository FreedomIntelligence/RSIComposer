from dataclasses import dataclass, field
from typing import Protocol
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


class EchoActionSpace:
    def describe(self) -> tuple[ActionSpec, ...]:
        return (ActionSpec("echo", "Echo a string in the task environment", {
            "type": "object", "properties": {"text": {"type": "string"}},
            "required": ["text"], "additionalProperties": False,
        }),)

    def validate(self, proposal: ActionProposal, allowed_actions: tuple[str, ...]) -> Action:
        if proposal.name != "echo" or proposal.name not in allowed_actions:
            raise ValueError(f"Unsupported or disallowed action: {proposal.name}")
        if set(proposal.arguments) != {"text"} or not isinstance(proposal.arguments["text"], str):
            raise ValueError("echo requires exactly one string argument: text")
        return Action(name=proposal.name, text=proposal.arguments["text"])


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
