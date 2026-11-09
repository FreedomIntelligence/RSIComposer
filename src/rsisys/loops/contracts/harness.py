"""Loop endpoint ports. Concrete policy, memory and tool choices belong to harnesses."""
from dataclasses import dataclass
from typing import TYPE_CHECKING, Literal, Protocol

from .coordination import ExplorationProgress, LoopStepResult
from .directive import Directive
from .feedback import Feedback

if TYPE_CHECKING:
    from rsisys.loops.decision_making.action_space import DecisionActionSpace
    from rsisys.loops.exploration.action_space import ActionSpace


class DecisionHarness(Protocol):
    def step(self, objective: str, action_space: 'DecisionActionSpace') -> LoopStepResult: ...
    def pending(self) -> Directive | None: ...
    def receive_feedback(self, directive: Directive, feedback: Feedback) -> None: ...


class ExplorationHarness(Protocol):
    def accept(self, directive: Directive) -> ExplorationProgress: ...
    def advance(self, directive_id: str, action_space: 'ActionSpace') -> ExplorationProgress: ...
    def inspect(self, directive_id: str) -> ExplorationProgress: ...
    def cancel(self, directive_id: str) -> ExplorationProgress: ...
    def acknowledge(self, directive_id: str) -> None: ...


@dataclass(frozen=True)
class DecisionRunResult:
    status: Literal['dispatch', 'waiting', 'finished', 'step_limit']
    steps: tuple[LoopStepResult, ...]
