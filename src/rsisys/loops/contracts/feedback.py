from dataclasses import dataclass, field

from .lifecycle import ExecutionStatus
from .artifacts import ArtifactRef
from .candidate import Candidate
from .jobs import JobRef


@dataclass(frozen=True)
class Feedback:
    directive_id: str
    status: ExecutionStatus
    summary: str
    simulated: bool
    quality_metrics: dict[str, float] = field(default_factory=dict)
    training_metrics: dict[str, float] = field(default_factory=dict)
    execution_metrics: dict[str, float] = field(default_factory=dict)
    artifact_refs: tuple[ArtifactRef, ...] = ()
    evidence_refs: tuple[str, ...] = ()
    candidates: tuple[Candidate, ...] = ()
    jobs: tuple[JobRef, ...] = ()
    error: str | None = None

    def __post_init__(self):
        if not isinstance(self.status, ExecutionStatus):
            raise TypeError('Feedback status must be a terminal ExecutionStatus')
        if self.status == ExecutionStatus.FAILED and not self.error:
            raise ValueError('Failed feedback requires an error')
        if self.status == ExecutionStatus.COMPLETED and self.error is not None:
            raise ValueError('Completed feedback cannot carry an error')
