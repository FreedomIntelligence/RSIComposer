"""Training outputs describe candidates; they do not authorize shared-model updates."""

from dataclasses import dataclass, field
from enum import StrEnum
from typing import Protocol

from .artifacts import ArtifactRef
from .candidate import Candidate
from .model_service import ModelRef


class ResumeMode(StrEnum):
    RESTART = "restart"
    WEIGHTS_ONLY = "weights_only"
    FULL_STATE = "full_state"


@dataclass(frozen=True)
class TrainingOrigin:
    """Experiment ancestry and training initialization are independent.

    Restart begins a fresh run from initial_model, without a run checkpoint.
    Weights-only loads checkpoint weights when supplied, otherwise initial_model;
    optimizer, scheduler, step and RNG state start fresh. Full-state restores all
    training state from checkpoint. Backends must reject unsupported semantics,
    never silently downgrade a full-state resume to a weights-only load.
    parent_system identifies an experimental parent, not a weight revision.
    """

    parent_system: str
    initial_model: ModelRef
    resume_mode: ResumeMode = ResumeMode.RESTART
    checkpoint: ArtifactRef | None = None

    def __post_init__(self):
        if not isinstance(self.parent_system, str) or not self.parent_system.strip():
            raise ValueError('parent_system must be a nonempty stable system ID')
        if not isinstance(self.initial_model, ModelRef) or any(
                not isinstance(value, str) or not value.strip()
                for value in (self.initial_model.backend_id, self.initial_model.model_id,
                              self.initial_model.revision)):
            raise ValueError('initial_model must be a ModelRef with nonempty identity fields')
        if not isinstance(self.resume_mode, ResumeMode):
            raise TypeError('resume_mode must be a ResumeMode')
        if self.checkpoint is not None:
            if not isinstance(self.checkpoint, ArtifactRef):
                raise TypeError('checkpoint must be an ArtifactRef or None')
            self.checkpoint.__post_init__()
        if self.resume_mode == ResumeMode.RESTART and self.checkpoint is not None:
            raise ValueError('restart cannot carry a checkpoint')
        if self.resume_mode == ResumeMode.FULL_STATE and self.checkpoint is None:
            raise ValueError('full_state requires a checkpoint')


@dataclass(frozen=True)
class TrainingResult:
    base: ModelRef
    model: ModelRef
    candidate: Candidate
    metrics: dict[str, float] = field(default_factory=dict)
    simulated: bool = False

    def __post_init__(self):
        if not isinstance(self.base, ModelRef) or not isinstance(self.model, ModelRef):
            raise TypeError('Training base and output model must be ModelRef')
        if not isinstance(self.candidate, Candidate):
            raise TypeError('Training output must contain a Candidate')
        if self.candidate.base_version != self.base.revision:
            raise ValueError("Candidate base version must match the training base")
        if self.model == self.base:
            raise ValueError("Training must produce a new model version")

    def require_target(self, role):
        """A single-role training request or adoption cannot apply a multi-role change."""
        if self.candidate.affected_roles != (role,):
            raise ValueError(f'Training result must target only {role}')


class ModelActivationPending(RuntimeError):
    """Deployment is not confirmed ready; retry the same activation key later."""


class ModelActivator(Protocol):
    """Load/publish and verify a trained version before a role binds to it.

    Return result.model only when ready. Raise ModelActivationPending for ongoing
    work or an uncertain response. Repeated keys must resume the same operation.
    Other exceptions indicate activation failure. Does not mutate role policies.
    """

    def activate(self, result: TrainingResult, *, idempotency_key: str) -> ModelRef: ...
