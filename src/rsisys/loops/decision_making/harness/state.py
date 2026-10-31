from dataclasses import dataclass, field

from rsisys.experience.records import ExperienceRecord
from rsisys.loops._memory import Memory, HistoryEntry
from rsisys.loops.contracts.feedback import Feedback
from rsisys.loops.contracts.jobs import JobRef
from rsisys.loops.contracts.directive import Directive

from ..action_space import OperationCall, OperationResult, OperationSpec


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


@dataclass(frozen=True)
class _PendingOperationRecord:
    record_id: str
    call: OperationCall | None
    result: OperationResult
    elapsed_seconds: float | None
    job: JobRef | None
    resume_result: OperationResult | None = None


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
