"""Public lifecycle boundaries; no concrete loop or backend imports."""

from dataclasses import dataclass
from typing import Protocol, TYPE_CHECKING

if TYPE_CHECKING:
    from rsisys.loops.decision_making.decision import LoopStepResult

from .directive import Directive
from .feedback import Feedback
from .jobs import JobRef


@dataclass(frozen=True)
class ExplorationProgress:
    directive_id: str
    phase: str
    directive: Directive
    job: JobRef | None = None
    feedback: Feedback | None = None

    @property
    def terminal(self) -> bool:
        return self.feedback is not None


class ExplorationEndpoint(Protocol):
    def accept(self, directive: Directive) -> ExplorationProgress: ...
    def advance(self, directive_id: str) -> ExplorationProgress: ...
    def inspect(self, directive_id: str) -> ExplorationProgress: ...
    def cancel(self, directive_id: str) -> ExplorationProgress: ...
    def acknowledge(self, directive_id: str) -> None: ...


class DecisionEndpoint(Protocol):
    def step(self, objective: str) -> 'LoopStepResult': ...
    def pending(self) -> Directive | None: ...
    def receive_feedback(self, directive: Directive, feedback: Feedback) -> None: ...
