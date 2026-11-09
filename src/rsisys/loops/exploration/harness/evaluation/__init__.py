from __future__ import annotations

from dataclasses import asdict, dataclass, replace
import math
from typing import TYPE_CHECKING, Protocol
from uuid import uuid4

from rsisys.runtime.evidence.records import ExperienceRecord
from rsisys.runtime.evidence.retrieval import ExperienceSystem
from rsisys.loops.contracts.artifacts import ArtifactRef
from rsisys.loops.contracts.candidate import Candidate
from rsisys.loops.contracts.jobs import JobRef, JobOutcome, JobStatus
from rsisys.runtime.orchestration.jobs import JobBook, InvalidJobResult

if TYPE_CHECKING:
    from ..state import ExplorationState


@dataclass(frozen=True)
class EvaluationRequest:
    candidate: Candidate
    protocol_ref: str
    data_refs: tuple[ArtifactRef, ...] = ()

    def __post_init__(self):
        if not self.protocol_ref:
            raise ValueError('An evaluation protocol is required')
        if not isinstance(self.data_refs, tuple) or any(not isinstance(ref, ArtifactRef) for ref in self.data_refs):
            raise TypeError('Evaluation data_refs must be a tuple of ArtifactRef values')


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


class EvaluationExecution:
    """Validate and record evaluation jobs with independently retryable evidence."""

    def __init__(self, experience: ExperienceSystem, state: ExplorationState,
                 jobs: JobBook[EvaluationRequest, EvaluationResult]):
        self.experience, self.state, self.jobs = experience, state, jobs

    def submit(self, request: EvaluationRequest, *, idempotency_key: str) -> JobRef:
        if not isinstance(request, EvaluationRequest):
            raise TypeError("Expected EvaluationRequest")
        return self.jobs.submit(request, idempotency_key=idempotency_key)

    def poll(self, job: JobRef) -> JobOutcome[EvaluationResult]:
        return self._record(self.jobs.poll(job, normalize=self._validate_metrics))

    def cancel(self, job: JobRef) -> JobOutcome[EvaluationResult]:
        return self._record(self.jobs.cancel(job, normalize=self._validate_metrics))

    def _validate_metrics(self, outcome):
        request = self.jobs.request(outcome.job)
        result = outcome.result
        # Identity violations retain the strict InvalidJobResult path below.
        if (outcome.status != JobStatus.SUCCEEDED or not isinstance(result, EvaluationResult)
                or result.candidate_id != request.candidate.candidate_id
                or result.protocol_ref != request.protocol_ref):
            return outcome
        try:
            valid = isinstance(result.metrics, dict) and all(
                isinstance(key, str) and key.strip() and type(value) in (int, float)
                and math.isfinite(value) for key, value in result.metrics.items())
        except (ValueError, OverflowError):
            valid = False
        if valid:
            return outcome
        return replace(outcome, status=JobStatus.FAILED, result=None,
            simulated=outcome.simulated or result.simulated,
            error='Invalid evaluation metrics: expected nonempty names and finite numeric values')

    def _record(self, outcome: JobOutcome[EvaluationResult]) -> JobOutcome[EvaluationResult]:
        request = self.jobs.request(outcome.job)
        if outcome.status == JobStatus.SUCCEEDED:
            result = outcome.result
            if (not isinstance(result, EvaluationResult)
                    or result.candidate_id != request.candidate.candidate_id
                    or result.protocol_ref != request.protocol_ref):
                raise InvalidJobResult("Evaluation result does not match the candidate and protocol", outcome)
        if outcome.terminal and outcome.job not in self.state.recorded_evaluation:
            self.experience.record(ExperienceRecord(
                record_id=self.state.evaluation_record_ids.setdefault(outcome.job, str(uuid4())),
                kind="evaluation", scope="actor", directive_id=None,
                summary=f"Candidate evaluation {outcome.status}",
                simulated=outcome.simulated or bool(outcome.result and outcome.result.simulated),
                source_refs=outcome.result.evidence_refs if outcome.result else (),
                details={"request": asdict(request), "outcome": asdict(outcome)},
            ))
            self.state.recorded_evaluation.add(outcome.job)
        return outcome
