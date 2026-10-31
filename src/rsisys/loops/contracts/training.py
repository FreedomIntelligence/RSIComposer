"""Training outputs describe candidates; they do not authorize shared-model updates."""

from dataclasses import dataclass, field

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
