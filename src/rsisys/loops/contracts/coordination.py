"""Public lifecycle boundaries; no concrete loop or backend imports."""

from dataclasses import dataclass
from typing import Protocol, Literal

from .artifacts import ArtifactRef
from .lifecycle import ExecutionStatus

from .directive import Directive
from .feedback import Feedback
from .jobs import JobRef


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


@dataclass(frozen=True)
class LoopStepResult:
    kind: Literal["progressed", "dispatch", "waiting", "finished"]
    directive: Directive | None = None
    job: JobRef | None = None
    operation_result: OperationResult | None = None
    reason: str = ""


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
    def step(self, objective: str) -> LoopStepResult: ...
    def pending(self) -> Directive | None: ...
    def receive_feedback(self, directive: Directive, feedback: Feedback) -> None: ...
