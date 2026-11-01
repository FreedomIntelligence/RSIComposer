"""Public lifecycle boundaries; no concrete loop or backend imports."""

from dataclasses import dataclass, field
import math
from typing import Protocol, Literal

from .artifacts import ArtifactRef
from .lifecycle import ExecutionStatus

from .directive import Directive
from .feedback import Feedback
from .jobs import JobRef, JobProgress
from .model_service import ModelRef, TokenUsage


@dataclass(frozen=True)
class OperationResult:
    summary: str
    artifacts: tuple[ArtifactRef, ...]
    simulated: bool
    status: ExecutionStatus = ExecutionStatus.COMPLETED
    error: str | None = None
    data: dict[str, object] = field(default_factory=dict)

    def __post_init__(self):
        if not isinstance(self.data, dict):
            raise TypeError('Operation result data must be a dict')
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

    def __post_init__(self):
        if self.kind not in ('progressed', 'dispatch', 'waiting', 'finished'):
            raise ValueError('Unknown decision step kind')
        if (self.kind == 'dispatch') != (self.directive is not None):
            raise ValueError('Only dispatch steps must carry a directive')


@dataclass(frozen=True)
class ExplorationProgress:
    directive_id: str
    phase: str
    directive: Directive
    job: JobRef | None = None
    feedback: Feedback | None = None
    metrics: dict[str, float] = field(default_factory=dict)
    elapsed_seconds: float | None = None
    job_progress: JobProgress | None = None

    def __post_init__(self):
        if self.directive_id != self.directive.directive_id:
            raise ValueError('Progress identity does not match its directive')
        if self.feedback is not None:
            if not isinstance(self.feedback, Feedback) or self.feedback.directive_id != self.directive_id:
                raise ValueError('Progress feedback must match its directive')
            self.feedback.__post_init__()

    @property
    def terminal(self) -> bool:
        return self.feedback is not None


@dataclass(frozen=True)
class ExecutionView:
    """Current exploration facts, not a second history or memory system."""
    model: ModelRef | None = None
    progress: ExplorationProgress | None = None


@dataclass(frozen=True)
class ReviewDecision:
    action: Literal['continue', 'cancel'] = 'continue'
    next_review_seconds: float | None = None
    reason: str = ''
    simulated: bool = False
    usage: TokenUsage | None = None

    def __post_init__(self):
        if self.action not in ('continue', 'cancel'):
            raise ValueError('A review may only continue or cancel the current directive')
        if self.next_review_seconds is not None and (
                type(self.next_review_seconds) not in (int, float)
                or not math.isfinite(self.next_review_seconds) or self.next_review_seconds <= 0):
            raise ValueError('next_review_seconds must be finite and positive or None')


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


class RecoverableExplorationEndpoint(ExplorationEndpoint, Protocol):
    """Optional discovery of the command still owned by the exploration side."""
    def pending(self) -> ExplorationProgress | None: ...


class ObservableExplorationEndpoint(ExplorationEndpoint, Protocol):
    def execution_view(self) -> ExecutionView: ...


class ReviewingDecisionEndpoint(DecisionEndpoint, Protocol):
    def update_execution(self, view: ExecutionView) -> None: ...
    def review(self, objective: str, progress: ExplorationProgress) -> ReviewDecision: ...
