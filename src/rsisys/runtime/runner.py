"""Bounded transport between public loop endpoints; no policy or backend decisions."""

from copy import deepcopy
from dataclasses import asdict, dataclass
from time import monotonic
from typing import Callable, Literal

from rsisys.loops.contracts.coordination import (
    DecisionEndpoint, ExplorationEndpoint, ExplorationProgress, ExecutionView, ReviewDecision,
)
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
    def __init__(self, decision_making: DecisionEndpoint, exploration: ExplorationEndpoint,
                 *, clock: Callable[[], float] = monotonic):
        self.decision_making, self.exploration = decision_making, exploration
        self._directive: Directive | None = None
        self._feedback: Feedback | None = None
        self._clock = clock
        self._review_directive_id: str | None = None
        self._review_at: float | None = None
        self._review_result: ReviewDecision | None = None
        self._review_pending = False
        self._pending_cancel = False

    def _exploration_pending(self):
        discover = getattr(self.exploration, 'pending', None)
        return discover() if discover is not None else None

    @property
    def busy(self) -> bool:
        return (self._directive is not None or self.decision_making.pending() is not None
                or self._exploration_pending() is not None)

    def _recover(self):
        pending = self.decision_making.pending()
        active = self._exploration_pending()
        directive = self._directive or pending
        if directive is not None and pending is not None and directive != pending:
            raise ValueError('Runner and decision endpoint own different directives')
        if active is not None:
            if not isinstance(active, ExplorationProgress):
                raise TypeError('pending() must return ExplorationProgress or None')
            if directive is not None and directive != active.directive:
                raise ValueError('Loop endpoints own different directives')
            if directive is None and not active.terminal:
                raise RuntimeError('Exploration has an unowned active directive')
            directive = active.directive
        self._directive = deepcopy(directive)
        if directive is not None and self._review_directive_id != directive.directive_id:
            self._review_directive_id = directive.directive_id
            delay = directive.review_after_seconds
            self._review_at = None if delay is None else self._clock() + delay
            self._review_result = None
            self._review_pending = False

    def _update_execution(self, progress=None):
        update = getattr(self.decision_making, 'update_execution', None)
        if update is None:
            return
        describe = getattr(self.exploration, 'execution_view', None)
        view = describe() if describe is not None else ExecutionView(progress=progress)
        if not isinstance(view, ExecutionView):
            raise TypeError('execution_view() must return ExecutionView')
        update(deepcopy(view))

    def _review(self, objective, progress, event):
        if self._review_at is None or self._clock() < self._review_at:
            return progress, 0
        review = getattr(self.decision_making, 'review', None)
        if review is None:
            self._review_at = None
            return progress, 0
        called = 0
        if self._review_result is None:
            self._review_pending = True
            self._update_execution(progress)
            result = review(objective, deepcopy(progress))
            if not isinstance(result, ReviewDecision):
                raise TypeError('review() must return ReviewDecision')
            result.__post_init__()
            self._review_result = deepcopy(result)
            self._review_pending = False
            called = 1
        result = self._review_result
        if result.action == 'cancel':
            # Keep the review result on an uncertain cancellation response.
            progress = self._request_cancel(progress)
        event['review'] = asdict(result)
        delay = None if result.action == 'cancel' else result.next_review_seconds
        self._review_at = None if delay is None else self._clock() + delay
        self._review_result = None
        return progress, called

    def _request_cancel(self, progress):
        self._pending_cancel = True
        progress = self.exploration.cancel(progress.directive_id)
        self._validate_progress(progress)
        self._pending_cancel = False
        return progress

    def _deliver(self):
        # Retain the outbox until both delivery and acknowledgement succeed.
        self.decision_making.receive_feedback(deepcopy(self._directive), deepcopy(self._feedback))
        self.exploration.acknowledge(self._directive.directive_id)
        self._directive = None
        self._feedback = None
        self._review_directive_id = None
        self._review_at = None
        self._review_result = None
        self._review_pending = False
        self._pending_cancel = False
        self._update_execution()

    def _validate_progress(self, progress):
        if not isinstance(progress, ExplorationProgress):
            raise TypeError('Exploration endpoint must return ExplorationProgress')
        progress.__post_init__()
        if progress.directive != self._directive or progress.directive_id != self._directive.directive_id:
            raise ValueError('Exploration progress belongs to another directive')
        if progress.feedback is not None and progress.feedback.directive_id != progress.directive_id:
            raise ValueError('Exploration feedback belongs to another directive')

    def step(self, objective: str) -> RunnerStep:
        """At most one decision and one exploration phase/poll; never sleeps."""
        self._recover()
        self._update_execution()
        decision_steps = 0
        if self._directive is None:
            result = self.decision_making.step(objective)
            event = asdict(result)
            decision_steps = 1
            if result.kind != 'dispatch':
                return RunnerStep(result.kind, event, decision_steps, result.job)
            self._directive = deepcopy(result.directive)
            self._recover()
        else:
            event = {'kind': 'dispatch', 'directive': asdict(self._directive)}
        directive_id = self._directive.directive_id
        if self._feedback is None:
            try:
                progress = self.exploration.accept(self._directive)
                self._validate_progress(progress)
                if not progress.terminal and (self._review_pending or self._review_result is not None):
                    progress, reviews = self._review(objective, progress, event)
                    decision_steps += reviews
                elif not progress.terminal and self._pending_cancel:
                    progress = self._request_cancel(progress)
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
            if not progress.terminal and decision_steps == 0:
                progress, reviews = self._review(objective, progress, event)
                decision_steps += reviews
            if not progress.terminal:
                event['phase'] = progress.phase
                event['progress'] = asdict(progress)
                return RunnerStep('waiting', event, decision_steps, progress.job, directive_id)
            self._feedback = progress.feedback
        event['feedback'] = asdict(self._feedback)
        self._deliver()
        return RunnerStep('completed', event, decision_steps, directive_id=directive_id)

    def cancel(self):
        """Request cancellation; step/run subsequently confirms and delivers the outcome."""
        self._recover()
        directive = self._directive
        if directive is None:
            raise RuntimeError('No pending exploration directive')
        self._directive = deepcopy(directive)
        self._pending_cancel = True
        progress = self.exploration.accept(directive)
        self._validate_progress(progress)
        if progress.terminal:
            self._pending_cancel = False
            return progress
        return self._request_cancel(progress)

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
