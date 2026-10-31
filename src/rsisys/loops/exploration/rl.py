"""Local rollout/update sequencing. Scheduling between loops is intentionally absent."""

from dataclasses import dataclass, field
from typing import Literal
from uuid import uuid4

from rsisys.loops.contracts.directive import Directive
from rsisys.loops.contracts.feedback import Feedback
from rsisys.loops.contracts.jobs import JobRef
from rsisys.loops.contracts.training import TrainingResult

from .learner import ActorTrainingRequest


@dataclass(frozen=True)
class RLPlan:
    rollout: Directive
    recipe_ref: str
    updates: int

    def __post_init__(self):
        if self.rollout.mode != 'execute':
            raise ValueError('An RL rollout must use execute mode')
        if type(self.updates) is not int or self.updates < 1:
            raise ValueError('updates must be a positive integer')
        if not self.recipe_ref:
            raise ValueError('An RL recipe is required')


@dataclass
class RLState:
    plan: RLPlan
    run_id: str = field(default_factory=lambda: str(uuid4()))
    phase: Literal['rollout', 'submitting', 'training', 'completed', 'failed', 'cancelled'] = 'rollout'
    updates_completed: int = 0
    job: JobRef | None = None
    request: ActorTrainingRequest | None = None
    feedback: list[Feedback] = field(default_factory=list)
    results: list[TrainingResult] = field(default_factory=list)
    error: str | None = None
    submission_attempted: bool = False
    cancel_requested: bool = False
    jobs: list[JobRef] = field(default_factory=list)
    simulated: bool = False

    @property
    def terminal(self) -> bool:
        return self.phase in ('completed', 'failed', 'cancelled')
