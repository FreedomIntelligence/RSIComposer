"""Public lifecycle boundaries; no concrete loop or backend imports."""

from dataclasses import dataclass, field
from copy import deepcopy
import json
import math
from typing import Protocol, Literal

from .artifacts import ArtifactRef
from .lifecycle import ExecutionStatus

from .directive import Directive
from .feedback import Feedback
from .jobs import JobRef, JobProgress
from .model_service import ModelRef, TokenUsage
from .metrics import validated_metrics


@dataclass(frozen=True)
class OperationResult:
    """Tool receipt; evaluation data uses data.quality_metrics/evidence_refs.

    Keep measured values and their source references together in data. Research
    input views preserve exact aggregate facts and archive references; oversized
    prose and per-item tables may be explicitly excerpted, including the latest receipt.
    """
    summary: str
    artifacts: tuple[ArtifactRef, ...]
    simulated: bool
    status: ExecutionStatus = ExecutionStatus.COMPLETED
    error: str | None = None
    data: dict[str, object] = field(default_factory=dict)
    adopted_candidate_ids: tuple[str, ...] = ()

    def __post_init__(self):
        if not isinstance(self.status, ExecutionStatus):
            raise TypeError('Operation status must be a terminal ExecutionStatus')
        if (not isinstance(self.adopted_candidate_ids, tuple)
                or any(not isinstance(ident, str) or not ident.strip() for ident in self.adopted_candidate_ids)
                or len(set(self.adopted_candidate_ids)) != len(self.adopted_candidate_ids)):
            raise ValueError('adopted_candidate_ids must be a tuple of distinct nonempty IDs')
        if self.adopted_candidate_ids and self.status != ExecutionStatus.COMPLETED:
            raise ValueError('Only completed operation receipts may confirm adoption')
        if not isinstance(self.data, dict):
            raise TypeError('Operation result data must be a dict')
        for name in ('quality_metrics', 'training_metrics', 'execution_metrics'):
            if name in self.data:
                validated_metrics(self.data[name], 'data.' + name)
        json.dumps(self.data, allow_nan=False)
        object.__setattr__(self, 'data', deepcopy(self.data))
        if self.status not in (ExecutionStatus.COMPLETED, ExecutionStatus.FAILED, ExecutionStatus.CANCELLED):
            raise ValueError("An operation result must be terminal")
        if self.status == ExecutionStatus.FAILED and not self.error:
            raise ValueError("A failed operation requires an error")
        if self.status == ExecutionStatus.COMPLETED and self.error is not None:
            raise ValueError("A completed operation cannot carry an error")


@dataclass(frozen=True)
class LoopStepResult:
    kind: Literal["progressed", "dispatch", "waiting", "finished"]
    directive: Directive | None = None
    job: JobRef | None = None
    operation_result: OperationResult | None = None
    reason: str = ""
    decision_id: str | None = None

    def __post_init__(self):
        if self.decision_id is not None and (not isinstance(self.decision_id, str) or not self.decision_id.strip()):
            raise ValueError('decision_id must be nonempty or None')
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
    version: int = 0

    def __post_init__(self):
        if type(self.version) is not int or self.version < 0:
            raise ValueError('Progress version must be a nonnegative integer')
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


class RunControlHintsEndpoint(DecisionEndpoint, Protocol):
    """Optional hints; the host enforces its limits even without this endpoint."""
    def update_run_control(self, context: dict[str, object]) -> None: ...


class ResearchRestartEndpoint(DecisionEndpoint, Protocol):
    """Optional restart of a finished research session."""
    def restart_research(self, context: dict[str, object]) -> None: ...


class ResearchCycleDecisionEndpoint(RunControlHintsEndpoint, ResearchRestartEndpoint, Protocol):
    """Compatibility name for endpoints implementing both optional capabilities."""


class PendingResearchJobEndpoint(DecisionEndpoint, Protocol):
    """Optional observation of a tool job awaiting external completion handoff."""
    def pending_research_job(self) -> JobRef | None: ...


class RecoverableExplorationEndpoint(ExplorationEndpoint, Protocol):
    """Optional discovery of the command still owned by the exploration side."""
    def pending(self) -> ExplorationProgress | None: ...


class ObservableExplorationEndpoint(ExplorationEndpoint, Protocol):
    def execution_view(self) -> ExecutionView: ...


class ReviewingDecisionEndpoint(DecisionEndpoint, Protocol):
    def update_execution(self, view: ExecutionView) -> None: ...
    def review(self, objective: str, progress: ExplorationProgress) -> ReviewDecision: ...


class ReviewAuthorizingEndpoint(DecisionEndpoint, Protocol):
    """Optional last-moment check for model-proposed controls, including retries."""
    def authorize_review(self, result: ReviewDecision) -> None: ...


@dataclass(frozen=True)
class DirectiveUpdate:
    """Versioned backend control, not replacement of the immutable directive.

    The backend explicitly defines supported changes. The same key must always
    identify the same request, including expected_version, even after completion.
    """
    directive_id: str
    expected_version: int
    idempotency_key: str
    changes: dict[str, object]

    def __post_init__(self):
        import json
        if not isinstance(self.directive_id, str) or not self.directive_id:
            raise ValueError('An update requires a directive ID')
        if not isinstance(self.idempotency_key, str) or not self.idempotency_key:
            raise ValueError('An update requires an idempotency key')
        if type(self.expected_version) is not int or self.expected_version < 0:
            raise ValueError('expected_version must be a nonnegative integer')
        if not isinstance(self.changes, dict) or not self.changes:
            raise ValueError('An update requires nonempty changes')
        if any(not isinstance(key, str) for key in self.changes):
            raise ValueError('Update fields must be strings')
        json.dumps(self.changes, allow_nan=False)


@dataclass(frozen=True)
class UpdateReceipt:
    request: DirectiveUpdate
    version: int

    def __post_init__(self):
        if self.version != self.request.expected_version + 1:
            raise ValueError('An accepted update increments the version exactly once')


class ConcurrentExplorationEndpoint(ExplorationEndpoint, Protocol):
    def pending_all(self) -> tuple[ExplorationProgress, ...]: ...
    def execution_views(self) -> dict[str, ExecutionView]: ...
    def update(self, request: DirectiveUpdate) -> UpdateReceipt: ...


class ConcurrentDecisionEndpoint(DecisionEndpoint, Protocol):
    def pending_all(self) -> tuple[Directive, ...]: ...
    def update_executions(self, views: dict[str, ExecutionView]) -> None: ...
