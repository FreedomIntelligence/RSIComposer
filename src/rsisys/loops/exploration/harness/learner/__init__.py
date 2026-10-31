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
        if self.method not in ("sft", "rl"):
            raise ValueError("Training method must be sft or rl")
        if not self.recipe_ref:
            raise ValueError("A training recipe is required")
        if self.method == "sft" and not self.data_refs:
            raise ValueError("SFT requires data artifacts")
        if self.method == "rl" and not self.trajectory_refs:
            raise ValueError("RL requires trajectory records")


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
