from copy import deepcopy
from dataclasses import asdict, dataclass
import json

from .artifacts import ArtifactRef


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
    payload: object | None = None

    def __post_init__(self):
        if not isinstance(self.directive_id, str) or not self.directive_id:
            raise ValueError('A directive ID is required')
        if type(self.max_steps) is not int or self.max_steps < 1:
            raise ValueError("max_steps must be a positive integer")
        # Receipts compare deep-copied directives; default histories serialize them.
        # Fail at construction rather than after a handler has performed work.
        try:
            json.dumps(asdict(self)['payload'], allow_nan=False)
            if deepcopy(self.payload) != self.payload:
                raise ValueError('Payload does not preserve value equality when copied')
        except Exception as exc:
            raise TypeError('Directive payload must serialize as JSON data and preserve value equality '
                            'when deep-copied; use plain values or value-equal dataclasses') from exc
