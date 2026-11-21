"""Public lifecycle boundaries; no concrete loop or backend imports."""

from dataclasses import asdict, dataclass, field
from copy import deepcopy
import json
from typing import Protocol, Literal

from .artifacts import ArtifactRef
from .delivery import FeedbackDelivery
from .lifecycle import ExecutionStatus

from .jobs import JobRef
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
    feedback_delivery: FeedbackDelivery | None = None
    call_id: str | None = None

    def __post_init__(self):
        if self.call_id is not None and (not isinstance(self.call_id, str) or not self.call_id or len(self.call_id) > 200):
            raise ValueError('Receipt call_id must be a nonempty string of at most 200 characters')
        if self.feedback_delivery is not None:
            if not isinstance(self.feedback_delivery, FeedbackDelivery):
                raise TypeError('feedback_delivery must be FeedbackDelivery or None')
            self.feedback_delivery.__post_init__()
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

    def payload(self):
        """Canonical receipt/history fields; hash these instead of asdict(self).

        Absent optional metadata preserves the exact historical archive format.
        """
        value = asdict(self)
        if self.feedback_delivery is None:
            value.pop('feedback_delivery')
        return value


@dataclass(frozen=True)
class LoopStepResult:
    kind: Literal["progressed", "waiting", "finished"]
    job: JobRef | None = None
    operation_result: OperationResult | None = None
    reason: str = ""
    decision_id: str | None = None
    decision_steps: int = 1

    def __post_init__(self):
        if type(self.decision_steps) is not int or self.decision_steps < 0:
            raise ValueError("decision_steps must be a nonnegative integer")
        if self.decision_id is not None and (not isinstance(self.decision_id, str) or not self.decision_id.strip()):
            raise ValueError('decision_id must be nonempty or None')
        if self.kind not in ('progressed', 'waiting', 'finished'):
            raise ValueError('Unknown decision step kind')










class DecisionEndpoint(Protocol):
    def step(self, objective: str) -> LoopStepResult: ...


class RunControlHintsEndpoint(DecisionEndpoint, Protocol):
    """Optional hints; the host enforces its limits even without this endpoint."""
    def update_run_control(self, context: dict[str, object]) -> None: ...


class ResearchRestartEndpoint(DecisionEndpoint, Protocol):
    """Optional restart of a finished research session."""
    def restart_research(self, context: dict[str, object]) -> None: ...




class PendingResearchJobEndpoint(DecisionEndpoint, Protocol):
    """Optional observation of a tool job awaiting external completion handoff."""
    def pending_research_job(self) -> JobRef | None: ...
