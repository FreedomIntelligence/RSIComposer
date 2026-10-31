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
