from dataclasses import dataclass
from typing import Protocol

from rsisys.loops.contracts.training import TrainingResult
from rsisys.loops.contracts.jobs import JobRef, JobOutcome
from rsisys.loops.contracts.model_service import ModelRef


@dataclass(frozen=True)
class DecisionTrainingRequest:
    base: ModelRef
    experience_refs: tuple[str, ...]
    recipe_ref: str

    def __post_init__(self):
        if not self.recipe_ref or not self.experience_refs:
            raise ValueError("Decision training requires a recipe and experience records")


class DecisionLearner(Protocol):
    def submit(self, request: DecisionTrainingRequest, *, idempotency_key: str) -> JobRef: ...

    def poll(self, job: JobRef) -> JobOutcome[TrainingResult]: ...

    def cancel(self, job: JobRef) -> None: ...

    def result(self, job: JobRef) -> TrainingResult: ...


class UnimplementedDecisionLearner:
    def submit(self, request: DecisionTrainingRequest, *, idempotency_key: str = "") -> JobRef:
        raise NotImplementedError("Decision-maker training is not implemented.")

    def result(self, job: JobRef) -> TrainingResult:
        raise NotImplementedError("Decision-maker training is not implemented.")

    def poll(self, job: JobRef) -> JobOutcome[TrainingResult]:
        raise NotImplementedError("Training backend is not implemented.")

    def cancel(self, job: JobRef) -> None:
        raise NotImplementedError("Training backend is not implemented.")
