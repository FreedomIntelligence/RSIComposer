from dataclasses import dataclass, field

from .lifecycle import ExecutionStatus
from .artifacts import ArtifactRef
from .candidate import Candidate
from .jobs import JobRef
from .metrics import InvalidMetrics, validated_metrics


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
    adopted_candidate_ids: tuple[str, ...] = ()

    def __post_init__(self):
        for name in ('quality_metrics', 'training_metrics', 'execution_metrics'):
            object.__setattr__(self, name, validated_metrics(getattr(self, name), name))
        if (not isinstance(self.adopted_candidate_ids, tuple)
                or any(not isinstance(ident, str) or not ident.strip() for ident in self.adopted_candidate_ids)
                or len(set(self.adopted_candidate_ids)) != len(self.adopted_candidate_ids)):
            raise ValueError('adopted_candidate_ids must be a tuple of distinct nonempty IDs')
        if self.adopted_candidate_ids and self.status != ExecutionStatus.COMPLETED:
            raise ValueError('Only completed feedback may confirm adoption')
        if not isinstance(self.status, ExecutionStatus):
            raise TypeError('Feedback status must be a terminal ExecutionStatus')
        if self.status == ExecutionStatus.FAILED and not self.error:
            raise ValueError('Failed feedback requires an error')
        if self.status == ExecutionStatus.COMPLETED and self.error is not None:
            raise ValueError('Completed feedback cannot carry an error')

    @classmethod
    def from_backend(cls, directive_id, status, summary, simulated, **fields):
        """Convert invalid provider measurements into a deliverable failure.

        Identity/status errors remain strict. Evidence/artifact references remain
        attached; rejected measurements are named in the error, never scored zero.
        Providers should archive their raw output before using this boundary.
        """
        errors = []
        for name in ('quality_metrics', 'training_metrics', 'execution_metrics'):
            try:
                fields[name] = validated_metrics(fields.get(name, {}), name)
            except InvalidMetrics as exc:
                errors.append(str(exc))
                fields[name] = {}
        if errors:
            if not isinstance(status, ExecutionStatus):
                raise TypeError('Feedback status must be a terminal ExecutionStatus')
            status = ExecutionStatus.FAILED
            fields['error'] = '; '.join(filter(None, (fields.get('error'), *errors)))
            fields['adopted_candidate_ids'] = ()
        return cls(directive_id, status, summary, simulated, **fields)
