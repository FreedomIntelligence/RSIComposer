"""Local rollout/update sequencing. Scheduling between loops is intentionally absent."""

from __future__ import annotations

from copy import deepcopy
from dataclasses import dataclass, field, replace
from typing import TYPE_CHECKING, Literal
from uuid import uuid4

from rsisys.loops.contracts.directive import Directive
from rsisys.loops.contracts.feedback import Feedback
from rsisys.loops.contracts.jobs import JobRef, JobStatus, JobOutcome
from rsisys.loops.contracts.lifecycle import ExecutionStatus
from rsisys.loops.contracts.model_service import ModelUpdatable
from rsisys.loops.contracts.training import TrainingResult

from ..action_space import ActionSpace
from rsisys.runtime.orchestration.jobs import InvalidJobResult
from .learner import ActorTrainingRequest, LearningExecution

if TYPE_CHECKING:
    from .environment import EpisodeExecution
    from .state import ExplorationState


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


class LocalRLExecution:
    """Sequence rollout and existing learner jobs without implementing training."""

    def __init__(self, episodes: EpisodeExecution, learning: LearningExecution,
                 state: ExplorationState):
        self.episodes, self.learning, self.state = episodes, learning, state

    def start(self, plan: RLPlan) -> RLState:
        if not isinstance(plan, RLPlan):
            raise TypeError("Expected RLPlan")
        if self.state.rl is not None and not self.state.rl.terminal:
            raise RuntimeError("An RL cycle is already active")
        if self.state.active_directive_id is not None or self.state.pending_episode is not None or self.learning.jobs.busy:
            raise RuntimeError("Actor is busy")
        if self.learning.jobs.backend is None:
            raise NotImplementedError("RL requires a learner")
        if not isinstance(self.learning.policy, ModelUpdatable):
            raise TypeError("RL requires a policy that can bind updated model versions")
        self.state.rl = RLState(deepcopy(plan))
        return deepcopy(self.state.rl)

    def step(self, action_space: ActionSpace) -> RLState:
        """Advance one local phase; waiting jobs are polled once, never busy-waited."""
        state = self.state.rl
        if state is None:
            raise RuntimeError("No RL cycle has been started")
        if state.terminal:
            return deepcopy(state)
        if state.phase == "rollout":
            directive = replace(state.plan.rollout,
                                directive_id=f"{state.run_id}:{state.updates_completed}")
            try:
                feedback = self.episodes.run(directive, action_space)
            except BaseException as exc:
                if self.state.pending_episode is not None:
                    raise  # Retry evidence persistence before advancing or repeating this rollout.
                state.phase = "failed" if isinstance(exc, Exception) else "cancelled"
                state.error = f"{type(exc).__name__}: {exc}"
                if self.state.last_feedback is not None:
                    state.feedback.append(self.state.last_feedback)
                raise
            state.feedback.append(feedback)
            if state.cancel_requested:
                state.phase = "cancelled"
            elif feedback.status in (ExecutionStatus.FAILED, ExecutionStatus.CANCELLED):
                state.phase = "failed" if feedback.status == ExecutionStatus.FAILED else "cancelled"
                state.error = feedback.error
            else:
                state.request = ActorTrainingRequest(self.learning.policy.model, (), state.plan.recipe_ref,
                                                     "rl", tuple(r.record_id for r in self.episodes.memory.history(
                                                         task_id=directive.directive_id, kind="trajectory")))
                state.phase = "submitting"
        elif state.phase == "submitting":
            # Keep request and key stable if submission raises after a backend accepted it.
            state.submission_attempted = True
            state.job = self.learning.submit(state.request,
                idempotency_key=f"{state.run_id}:{state.updates_completed}")
            if state.job not in state.jobs:
                state.jobs.append(state.job)
            state.phase = "training"
        elif state.phase == "training":
            outcome = self._poll_learning(state)
            state.simulated = state.simulated or outcome.simulated
            if state.cancel_requested:
                if outcome.terminal:
                    state.phase = "cancelled"
                return deepcopy(state)
            if outcome.status == JobStatus.SUCCEEDED:
                self.learning.apply(state.job)
                state.results.append(outcome.result)
                state.updates_completed += 1
                state.phase = "completed" if state.updates_completed == state.plan.updates else "rollout"
                state.job = None
                state.request = None
                state.submission_attempted = False
            elif outcome.status in (JobStatus.FAILED, JobStatus.CANCELLED):
                state.phase = "failed" if outcome.status == JobStatus.FAILED else "cancelled"
                state.error = outcome.error
        return deepcopy(state)

    def _poll_learning(self, state: RLState) -> JobOutcome[TrainingResult]:
        try:
            return self.learning.cancel(state.job) if state.cancel_requested else self.learning.poll(state.job)
        except InvalidJobResult as exc:
            state.phase = 'failed'
            state.error = f'{type(exc).__name__}: {exc}'
            state.simulated = state.simulated or exc.simulated
            raise

    def cancel(self) -> RLState:
        state = self.state.rl
        if state is None:
            raise RuntimeError("No RL cycle has been started")
        if state.terminal:
            return deepcopy(state)
        state.cancel_requested = True
        if self.state.pending_episode is not None:
            state.feedback.append(self.episodes.flush())
        if state.phase == "submitting" and state.submission_attempted:
            # A lost submit response may hide a real job. Resolve the same key before cancelling.
            return deepcopy(state)
        if state.job is not None:
            outcome = self._poll_learning(state)
            state.simulated = state.simulated or outcome.simulated
            if not outcome.terminal:
                return deepcopy(state)
        state.phase = "cancelled"
        return deepcopy(state)
