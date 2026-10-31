"""Bounded transport between public loop endpoints; no policy or backend decisions."""

from copy import deepcopy
from dataclasses import asdict, dataclass
from typing import Literal

from rsisys.loops.contracts.coordination import DecisionEndpoint, ExplorationEndpoint
from rsisys.loops.contracts.directive import Directive
from rsisys.loops.contracts.feedback import Feedback
from rsisys.loops.contracts.jobs import JobRef


@dataclass(frozen=True)
class RunResult:
    status: Literal["completed", "finished", "waiting", "step_limit"]
    rounds_completed: int
    decision_steps: int
    events: list[dict[str, object]]
    waiting_job: JobRef | None = None
    waiting_directive_id: str | None = None


@dataclass(frozen=True)
class RunnerStep:
    kind: Literal['progressed', 'completed', 'waiting', 'finished']
    event: dict[str, object]
    decision_steps: int = 0
    job: JobRef | None = None
    directive_id: str | None = None


class Runner:
    def __init__(self, decision_making: DecisionEndpoint, exploration: ExplorationEndpoint):
        self.decision_making, self.exploration = decision_making, exploration
        self._directive: Directive | None = None
        self._feedback: Feedback | None = None

    @property
    def busy(self) -> bool:
        return self._directive is not None or self.decision_making.pending() is not None

    def _deliver(self):
        # Retain the outbox until both delivery and acknowledgement succeed.
        self.decision_making.receive(deepcopy(self._directive), deepcopy(self._feedback))
        self.exploration.acknowledge(self._directive.directive_id)
        self._directive = None
        self._feedback = None

    def _validate_progress(self, progress):
        if progress.directive != self._directive or progress.directive_id != self._directive.directive_id:
            raise ValueError('Exploration progress belongs to another directive')
        if progress.feedback is not None and progress.feedback.directive_id != progress.directive_id:
            raise ValueError('Exploration feedback belongs to another directive')

    def step(self, objective: str) -> RunnerStep:
        """At most one decision and one exploration phase/poll; never sleeps."""
        self._directive = self._directive or self.decision_making.pending()
        decision_steps = 0
        if self._directive is None:
            result = self.decision_making.step(objective)
            event = asdict(result)
            decision_steps = 1
            if result.kind != 'dispatch':
                return RunnerStep(result.kind, event, decision_steps, result.job)
            self._directive = deepcopy(result.directive)
        else:
            event = {'kind': 'dispatch', 'directive': asdict(self._directive)}
        directive_id = self._directive.directive_id
        if self._feedback is None:
            try:
                progress = self.exploration.accept(self._directive)
                self._validate_progress(progress)
                if not progress.terminal:
                    progress = self.exploration.advance(directive_id)
                    self._validate_progress(progress)
            except BaseException as exc:
                # Execution failures may already contain artifacts/evidence. Submission or
                # polling failures with no terminal feedback stay pending for a safe retry.
                try:
                    progress = self.exploration.inspect(directive_id)
                    self._validate_progress(progress)
                    if progress.terminal:
                        self._feedback = progress.feedback
                        self._deliver()
                except Exception as delivery_error:
                    exc.add_note(f'Failure feedback could not be recorded: {delivery_error}')
                raise
            if not progress.terminal:
                event['phase'] = progress.phase
                return RunnerStep('waiting', event, decision_steps, progress.job, directive_id)
            self._feedback = progress.feedback
        event['feedback'] = asdict(self._feedback)
        self._deliver()
        return RunnerStep('completed', event, decision_steps, directive_id=directive_id)

    def cancel(self):
        """Request cancellation; step/run subsequently confirms and delivers the outcome."""
        directive = self._directive or self.decision_making.pending()
        if directive is None:
            raise RuntimeError('No pending exploration directive')
        self._directive = deepcopy(directive)
        progress = self.exploration.accept(directive)
        return progress if progress.terminal else self.exploration.cancel(directive.directive_id)

    def run(self, objective: str, rounds: int, *, max_decision_steps: int | None = None) -> RunResult:
        if type(rounds) is not int or rounds < 1:
            raise ValueError('rounds must be a positive integer')
        limit = rounds * 8 if max_decision_steps is None else max_decision_steps
        if type(limit) is not int or limit < 1:
            raise ValueError('max_decision_steps must be a positive integer')
        events, completed, decision_steps = [], 0, 0
        while decision_steps < limit:
            result = self.step(objective)
            events.append(result.event)
            decision_steps += result.decision_steps
            if result.kind in ('waiting', 'finished'):
                return RunResult(result.kind, completed, decision_steps, events, result.job, result.directive_id)
            if result.kind == 'completed':
                completed += 1
                if completed == rounds:
                    return RunResult('completed', completed, decision_steps, events)
        return RunResult('step_limit', completed, decision_steps, events)
