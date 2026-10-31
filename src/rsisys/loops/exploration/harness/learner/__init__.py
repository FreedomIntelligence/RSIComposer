from dataclasses import dataclass
from typing import Protocol

from rsisys.loops.contracts.artifacts import ArtifactRef
from rsisys.loops.contracts.training import TrainingResult
from rsisys.loops.contracts.jobs import JobRef, JobOutcome
from rsisys.loops.contracts.model_service import ModelRef


@dataclass(frozen=True)
class ActorTrainingRequest:
    base: ModelRef
    data_refs: tuple[ArtifactRef, ...]
    recipe_ref: str
    method: str = "sft"
    trajectory_refs: tuple[str, ...] = ()

    def __post_init__(self):
        if not isinstance(self.base, ModelRef):
            raise ValueError("Training base must be a ModelRef")
        if not isinstance(self.method, str) or not self.method.strip():
            raise ValueError("Training method must be a nonempty string")
        if not isinstance(self.recipe_ref, str) or not self.recipe_ref.strip():
            raise ValueError("Training recipe must be a nonempty string")
        if not isinstance(self.data_refs, tuple) or not all(isinstance(ref, ArtifactRef) for ref in self.data_refs):
            raise ValueError("data_refs must be a tuple of ArtifactRef objects")
        if not isinstance(self.trajectory_refs, tuple) or not all(isinstance(ref, str) for ref in self.trajectory_refs):
            raise ValueError("trajectory_refs must be a tuple of strings")


class ActorLearner(Protocol):
    def submit(self, request: ActorTrainingRequest, *, idempotency_key: str) -> JobRef: ...

    def poll(self, job: JobRef) -> JobOutcome[TrainingResult]: ...

    def cancel(self, job: JobRef) -> None: ...

    def result(self, job: JobRef) -> TrainingResult: ...


class UnimplementedActorLearner:
    def submit(self, request: ActorTrainingRequest, *, idempotency_key: str = "") -> JobRef:
        raise NotImplementedError("Actor training is not implemented.")

    def result(self, job: JobRef) -> TrainingResult:
        raise NotImplementedError("Actor training is not implemented.")

    def poll(self, job: JobRef) -> JobOutcome[TrainingResult]:
        raise NotImplementedError("Training backend is not implemented.")

    def cancel(self, job: JobRef) -> None:
        raise NotImplementedError("Training backend is not implemented.")
