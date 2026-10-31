from rsisys.loops.contracts.model_service import TokenUsage

from dataclasses import dataclass
from typing import Literal

from rsisys.loops.contracts.directive import Directive
from rsisys.loops.contracts.jobs import JobRef

from ..action_space import OperationCall
from rsisys.loops.contracts.coordination import LoopStepResult


@dataclass(frozen=True)
class Decision:
    kind: Literal["invoke", "explore", "wait", "finish"]
    operation: OperationCall | None = None
    directive: Directive | None = None
    job: JobRef | None = None
    reason: str = ""
    simulated: bool = False
    usage: TokenUsage | None = None

    def __post_init__(self):
        expected = {"invoke": "operation", "explore": "directive", "wait": "job", "finish": None}
        if self.kind not in expected:
            raise ValueError(f"Unknown decision kind: {self.kind}")
        present = {key for key in ("operation", "directive", "job") if getattr(self, key) is not None}
        wanted = set() if expected[self.kind] is None else {expected[self.kind]}
        if present != wanted:
            raise ValueError(f"Decision {self.kind} requires only {expected[self.kind]}")


@dataclass(frozen=True)
class DecisionRunResult:
    status: Literal["dispatch", "waiting", "finished", "step_limit"]
    steps: tuple[LoopStepResult, ...]
