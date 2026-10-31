from dataclasses import dataclass
from typing import Protocol

from rsisys.loops.contracts.candidate import Candidate
from rsisys.loops.contracts.jobs import JobRef, JobOutcome


@dataclass(frozen=True)
class EvaluationRequest:
    candidate: Candidate
    protocol_ref: str

    def __post_init__(self):
        if not self.protocol_ref:
            raise ValueError('An evaluation protocol is required')


@dataclass(frozen=True)
class EvaluationResult:
    candidate_id: str
    protocol_ref: str
    metrics: dict[str, float]
    evidence_refs: tuple[str, ...]
    simulated: bool = False


class Evaluator(Protocol):
    def submit(self, request: EvaluationRequest, *, idempotency_key: str) -> JobRef: ...
    def poll(self, job: JobRef) -> JobOutcome[EvaluationResult]: ...
    def cancel(self, job: JobRef) -> None: ...
    def result(self, job: JobRef) -> EvaluationResult: ...


class UnimplementedEvaluator:
    def submit(self, request: EvaluationRequest, *, idempotency_key: str = '') -> JobRef:
        raise NotImplementedError('Candidate evaluation is not implemented.')

    def poll(self, job: JobRef) -> JobOutcome[EvaluationResult]:
        raise NotImplementedError('Candidate evaluation is not implemented.')

    def cancel(self, job: JobRef) -> None:
        raise NotImplementedError('Candidate evaluation is not implemented.')

    def result(self, job: JobRef) -> EvaluationResult:
        raise NotImplementedError('Candidate evaluation is not implemented.')
