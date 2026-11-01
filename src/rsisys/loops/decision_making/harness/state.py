from dataclasses import dataclass, field

from rsisys.experience.records import ExperienceRecord
from rsisys.loops._memory import Memory, HistoryEntry
from rsisys.loops.contracts.candidate import Candidate
from rsisys.loops.contracts.feedback import Feedback
from rsisys.loops.contracts.jobs import JobRef
from rsisys.loops.contracts.directive import Directive
from rsisys.loops.contracts.artifacts import ArtifactRef
from rsisys.loops.contracts.coordination import ExecutionView, ReviewDecision
from rsisys.loops.contracts.lifecycle import ExecutionStatus
from rsisys.loops.contracts.model_service import ModelRef

from ..action_space import OperationCall, OperationResult, OperationSpec


@dataclass(frozen=True)
class DirectiveProvenance:
    """Bindings observed before deciding; not inferred from a later training result."""
    directive_id: str
    decision_model: ModelRef | None = None
    execution_model: ModelRef | None = None


@dataclass(frozen=True)
class CandidateEvaluation:
    """Latest receipt per candidate, including failures and different protocols.

    This is a bounded-per-candidate summary, not a complete evaluation archive;
    full older receipts remain in history and Experience.
    """
    candidate_id: str
    directive_id: str
    protocol_ref: str
    input_refs: tuple[ArtifactRef, ...]
    status: ExecutionStatus
    quality_metrics: dict[str, float]
    simulated: bool
    evidence_refs: tuple[str, ...] = ()


@dataclass(frozen=True)
class DecisionContext:
    objective: str
    experience: tuple[ExperienceRecord, ...]
    operations: tuple[OperationSpec, ...]
    operation_results: tuple[OperationResult, ...]
    memories: tuple[Memory, ...] = ()
    last_feedback: Feedback | None = None
    history: tuple[HistoryEntry, ...] = ()
    directive_modes: tuple[str, ...] = ("execute",)
    candidates: tuple[Candidate, ...] = ()
    execution: ExecutionView = field(default_factory=ExecutionView)
    decision_model: ModelRef | None = None
    directive_provenance: tuple[DirectiveProvenance, ...] = ()
    candidate_evaluations: tuple[CandidateEvaluation, ...] = ()
    review_actions: tuple[str, ...] = ('continue', 'cancel')


@dataclass(frozen=True)
class _PendingOperationRecord:
    record_id: str
    call: OperationCall | None
    result: OperationResult
    elapsed_seconds: float | None
    job: JobRef | None
    resume_result: OperationResult | None = None
    task_id: str | None = None
    event: str | None = None


@dataclass(frozen=True)
class _PendingReviewRecord:
    record_id: str
    objective: str
    directive_id: str
    decision: ReviewDecision
    details: dict[str, object]


@dataclass
class _ReviewSession:
    objective: str
    directive_id: str
    context: DecisionContext
    tool_limit: int
    queries_used: int = 0
    query_refs: list[str] = field(default_factory=list)
    simulated: bool = False
    pending_record: _PendingOperationRecord | None = None


@dataclass
class DecisionState:
    issued: int = 0
    completed: int = 0
    pending_directive_id: str | None = None
    pending_directive: Directive | None = None
    feedback_record_id: str | None = None
    last_directive: Directive | None = None
    issued_directive_ids: set[str] = field(default_factory=set)
    last_feedback: Feedback | None = None
    operation_results: list[OperationResult] = field(default_factory=list)
    pending_job: JobRef | None = None
    pending_operation: OperationCall | None = None
    pending_simulated: bool = False
    pending_operation_record: _PendingOperationRecord | None = None
    finished: bool = False
    recorded_training: set[JobRef] = field(default_factory=set)
    applied_training: set[JobRef] = field(default_factory=set)
    training_record_ids: dict[JobRef, str] = field(default_factory=dict)
    measured_input_tokens: int = 0
    measured_output_tokens: int = 0
    calls_with_missing_usage: int = 0
    candidates: dict[str, Candidate] = field(default_factory=dict)
    execution: ExecutionView = field(default_factory=ExecutionView)
    directive_provenance: dict[str, DirectiveProvenance] = field(default_factory=dict)
    candidate_evaluations: dict[str, CandidateEvaluation] = field(default_factory=dict)
    pending_review_record: _PendingReviewRecord | None = None
    review_session: _ReviewSession | None = None
