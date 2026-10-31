from dataclasses import dataclass

from .artifacts import ArtifactRef
from .requests import SFTSpec, RLSpec, EvaluationSpec


@dataclass(frozen=True)
class Directive:
    """An immutable request from the decision-making loop to exploration."""

    directive_id: str
    objective: str
    strategy: str
    mode: str = "execute"
    input_refs: tuple[ArtifactRef, ...] = ()
    allowed_actions: tuple[str, ...] = ("echo",)
    max_steps: int = 4
    simulated: bool = False
    payload: SFTSpec | RLSpec | EvaluationSpec | None = None

    def __post_init__(self):
        if not isinstance(self.directive_id, str) or not self.directive_id:
            raise ValueError('A directive ID is required')
        if type(self.max_steps) is not int or self.max_steps < 1:
            raise ValueError("max_steps must be a positive integer")
