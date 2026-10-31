from dataclasses import dataclass

from .artifacts import ArtifactRef
from .requests import SFTSpec, RLSpec, EvaluationSpec, PromptSpec


@dataclass(frozen=True)
class Directive:
    """System instruction sent by a decision-maker or external controller.

    The exploration harness executes the requested mode. allowed_actions only
    constrains Actor/environment interactions inside execute or RL rollouts.
    """

    directive_id: str
    objective: str
    strategy: str
    mode: str = "execute"
    input_refs: tuple[ArtifactRef, ...] = ()
    allowed_actions: tuple[str, ...] = ("echo",)
    max_steps: int = 4
    simulated: bool = False
    payload: SFTSpec | RLSpec | EvaluationSpec | PromptSpec | None = None

    def __post_init__(self):
        if not isinstance(self.directive_id, str) or not self.directive_id:
            raise ValueError('A directive ID is required')
        if type(self.max_steps) is not int or self.max_steps < 1:
            raise ValueError("max_steps must be a positive integer")
