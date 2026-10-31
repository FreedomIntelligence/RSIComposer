"""Training outputs describe candidates; they do not authorize shared-model updates."""

from dataclasses import dataclass, field
from typing import Protocol

from .candidate import Candidate
from .model_service import ModelRef


@dataclass(frozen=True)
class TrainingResult:
    base: ModelRef
    model: ModelRef
    candidate: Candidate
    metrics: dict[str, float] = field(default_factory=dict)
    simulated: bool = False

    def __post_init__(self):
        if self.candidate.base_version != self.base.revision:
            raise ValueError("Candidate base version must match the training base")
        if self.model == self.base:
            raise ValueError("Training must produce a new model version")


class ModelActivationPending(RuntimeError):
    """Deployment is not confirmed ready; retry the same activation key later."""


class ModelActivator(Protocol):
    """Load/publish and verify a trained version before a role binds to it.

    Return result.model only when ready. Raise ModelActivationPending for ongoing
    work or an uncertain response. Repeated keys must resume the same operation.
    Other exceptions indicate activation failure. Does not mutate role policies.
    """

    def activate(self, result: TrainingResult, *, idempotency_key: str) -> ModelRef: ...
