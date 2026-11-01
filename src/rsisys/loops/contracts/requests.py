"""Cross-loop intent. Concrete learners and evaluators own backend requests."""

from dataclasses import dataclass

from .candidate import Candidate
from .model_service import ModelRef
from .training import TrainingOrigin


@dataclass(frozen=True)
class PromptSpec:
    instructions: str

    def __post_init__(self):
        if not isinstance(self.instructions, str):
            raise ValueError('Prompt instructions must be a string (empty clears them)')


@dataclass(frozen=True)
class SFTSpec:
    recipe_ref: str
    base: ModelRef | None = None
    apply_result: bool = False

    def __post_init__(self):
        if (not isinstance(self.recipe_ref, str) or not self.recipe_ref.strip()
                or type(self.apply_result) is not bool):
            raise ValueError('SFT requires a recipe and a boolean apply_result')
        if self.base is not None and not isinstance(self.base, ModelRef):
            raise TypeError('SFT base must be a ModelRef')


@dataclass(frozen=True)
class RLSpec:
    recipe_ref: str
    updates: int = 1

    def __post_init__(self):
        if (not isinstance(self.recipe_ref, str) or not self.recipe_ref.strip()
                or type(self.updates) is not int or self.updates < 1):
            raise ValueError('RL requires a recipe and positive updates')


@dataclass(frozen=True)
class EvaluationSpec:
    candidate: Candidate
    protocol_ref: str

    def __post_init__(self):
        if (not isinstance(self.candidate, Candidate) or not isinstance(self.protocol_ref, str)
                or not self.protocol_ref.strip()):
            raise ValueError('Evaluation requires a candidate and protocol')


@dataclass(frozen=True)
class ApplyCandidateSpec:
    candidate_id: str

    def __post_init__(self):
        if not isinstance(self.candidate_id, str) or not self.candidate_id.strip():
            raise ValueError('Applying a candidate requires a nonempty candidate_id')


@dataclass(frozen=True)
class TrainSpec:
    """Delegate a complete training job; algorithm/data requirements belong to its backend."""

    recipe_ref: str
    method: str
    base: ModelRef | None = None
    apply_result: bool = False
    origin: TrainingOrigin | None = None

    def __post_init__(self):
        if any(not isinstance(value, str) or not value.strip() for value in (self.recipe_ref, self.method)):
            raise ValueError('Training requires nonempty recipe and method names')
        if self.base is not None and not isinstance(self.base, ModelRef):
            raise TypeError('Training base must be a ModelRef')
        if type(self.apply_result) is not bool:
            raise TypeError('apply_result must be boolean')
        if self.origin is not None:
            if not isinstance(self.origin, TrainingOrigin):
                raise TypeError('origin must be TrainingOrigin')
            if self.base is not None and self.base != self.origin.initial_model:
                raise ValueError('Training base differs from origin.initial_model')
