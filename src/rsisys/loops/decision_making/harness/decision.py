from rsisys.loops.contracts.model_service import TokenUsage

from dataclasses import dataclass, field
from typing import Literal
from uuid import uuid4

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
    decision_id: str = field(default_factory=lambda: str(uuid4()))
    depends_on_decision_ids: tuple[str, ...] = ()
    source_decision_ids: tuple[str, ...] = ()

    def __post_init__(self):
        if not isinstance(self.decision_id, str) or not self.decision_id.strip():
            raise ValueError('decision_id must be a nonempty string')
        for name in ('depends_on_decision_ids', 'source_decision_ids'):
            refs = getattr(self, name)
            if (not isinstance(refs, tuple) or any(not isinstance(ref, str) or not ref.strip() for ref in refs)
                    or len(set(refs)) != len(refs) or self.decision_id in refs):
                raise ValueError(f'{name} must contain distinct nonempty IDs, excluding self')
        expected = {"invoke": "operation", "explore": "directive", "wait": "job", "finish": None}
        if self.kind not in expected:
            raise ValueError(f"Unknown decision kind: {self.kind}")
        present = {key for key in ("operation", "directive", "job") if getattr(self, key) is not None}
        # Waiting without a job yields one scheduling turn. A supplied job keeps
        # the existing explicit external-operation completion handoff semantics.
        wanted = set() if expected[self.kind] is None or (self.kind == 'wait' and self.job is None) else {expected[self.kind]}
        if present != wanted:
            raise ValueError(f"Decision {self.kind} requires only {expected[self.kind]}")


@dataclass(frozen=True)
class DecisionRunResult:
    status: Literal["dispatch", "waiting", "finished", "step_limit"]
    steps: tuple[LoopStepResult, ...]
