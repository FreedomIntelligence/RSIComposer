from copy import deepcopy
from dataclasses import asdict, dataclass
import json
import math

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
    review_after_seconds: float | None = None

    def __post_init__(self):
        if not isinstance(self.directive_id, str) or not self.directive_id:
            raise ValueError('A directive ID is required')
        if type(self.max_steps) is not int or self.max_steps < 1:
            raise ValueError("max_steps must be a positive integer")
        if self.review_after_seconds is not None and (
                type(self.review_after_seconds) not in (int, float)
                or not math.isfinite(self.review_after_seconds) or self.review_after_seconds <= 0):
            raise ValueError('review_after_seconds must be finite and positive or None')
        # Receipts compare deep-copied directives; default histories serialize them.
        # Fail at construction rather than after a handler has performed work.
        try:
            json.dumps(asdict(self)['payload'], allow_nan=False)
            if deepcopy(self.payload) != self.payload:
                raise ValueError('Payload does not preserve value equality when copied')
        except Exception as exc:
            raise TypeError('Directive payload must serialize as JSON data and preserve value equality '
                            'when deep-copied; use plain values or value-equal dataclasses') from exc
