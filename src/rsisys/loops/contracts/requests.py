"""Cross-loop intent. Concrete learners and evaluators own backend requests."""

from dataclasses import dataclass

from .candidate import Candidate
from .model_service import ModelRef


@dataclass(frozen=True)
class SFTSpec:
    recipe_ref: str
    base: ModelRef | None = None
    apply_result: bool = False

    def __post_init__(self):
        if not self.recipe_ref or type(self.apply_result) is not bool:
            raise ValueError('SFT requires a recipe and a boolean apply_result')


@dataclass(frozen=True)
class RLSpec:
    recipe_ref: str
    updates: int = 1

    def __post_init__(self):
        if not self.recipe_ref or type(self.updates) is not int or self.updates < 1:
            raise ValueError('RL requires a recipe and positive updates')


@dataclass(frozen=True)
class EvaluationSpec:
    candidate: Candidate
    protocol_ref: str

    def __post_init__(self):
        if not isinstance(self.candidate, Candidate) or not self.protocol_ref:
            raise ValueError('Evaluation requires a candidate and protocol')
