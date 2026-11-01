"""Bounded transport between public loop endpoints; no policy or backend decisions."""

from copy import deepcopy
from collections import deque
from dataclasses import asdict, dataclass
from time import monotonic
from typing import Callable, Literal

from rsisys.loops.contracts.coordination import (
    DecisionEndpoint, ExplorationEndpoint, ExplorationProgress, ExecutionView, ReviewDecision, DirectiveUpdate,
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
                 *, clock: Callable[[], float] = monotonic, parallel: bool = False):
        if type(parallel) is not bool:
            raise TypeError('parallel must be boolean')
        self.parallel = parallel
        if parallel and any(not callable(getattr(endpoint, 'pending_all', None))
                            for endpoint in (decision_making, exploration)):
            raise TypeError('Parallel endpoints must implement pending_all()')
        self._slots: dict[str, Runner] = {}
        self._rotation = deque()
        self._decision_finished = False
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
        if self.parallel:
            return bool(self._slots or self.decision_making.pending_all() or self.exploration.pending_all())
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
            authorize = getattr(self.decision_making, 'authorize_review', None)
            if callable(authorize):
                try:
                    authorize(deepcopy(result))
                except PermissionError:
                    # Stop retrying the old model intent after permission revocation.
                    # A request already received by the backend may still take effect.
                    self._review_result = None
                    self._pending_cancel = False
                    self._review_at = self._clock()
                    raise
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
        """One decision/review flow and exploration phase/poll; reviews may query tools."""
        if self.parallel:
            return self._parallel_step(objective)
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
                self._update_execution(progress)
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

    def cancel(self, directive_id: str | None = None):
        """Request cancellation; step/run subsequently confirms and delivers the outcome."""
        if self.parallel:
            self._recover_parallel()
            if directive_id is None:
                if len(self._slots) != 1:
                    raise ValueError('Specify directive_id when multiple or no experiments are pending')
                directive_id = next(iter(self._slots))
            return self._slots[directive_id].cancel(directive_id)
        self._recover()
        if directive_id is not None and (self._directive is None or self._directive.directive_id != directive_id):
            raise ValueError('Cancellation ID does not match the pending directive')
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

    def update(self, request: DirectiveUpdate):
        """Forward versioned control; unsupported backends must reject explicitly."""
        if not isinstance(request, DirectiveUpdate):
            raise TypeError('Expected DirectiveUpdate')
        update = getattr(self.exploration, 'update', None)
        if update is None:
            raise NotImplementedError('Exploration does not support live updates')
        return update(deepcopy(request))

    def _recover_parallel(self):
        pending = {d.directive_id: d for d in self.decision_making.pending_all()}
        active = {p.directive_id: p for p in self.exploration.pending_all()}
        for ident, progress in active.items():
            if ident in pending and progress.directive != pending[ident]:
                raise ValueError('Loop endpoints own different directives')
            if ident not in pending and ident not in self._slots and not progress.terminal:
                raise RuntimeError('Exploration has an unowned active directive')
            pending.setdefault(ident, progress.directive)
        for ident, directive in pending.items():
            if ident not in self._slots:
                slot = Runner(_DecisionSlot(self.decision_making, ident),
                              _ExplorationSlot(self.exploration, ident), clock=self._clock)
                slot._directive = deepcopy(directive)
                self._slots[ident] = slot
                self._rotation.append(ident)
            elif self._slots[ident]._directive != directive:
                raise ValueError('Runner and endpoints own different directives')

    def _parallel_views(self):
        describe = getattr(self.exploration, 'execution_views', None)
        views = describe() if describe else {p.directive_id: ExecutionView(progress=p)
                                            for p in self.exploration.pending_all()}
        update = getattr(self.decision_making, 'update_executions', None)
        if update is not None:
            update(deepcopy(views))
        # A multiplexed endpoint has no single current model. Named views carry
        # the bindings; do not attribute the last-polled child to a new decision.
        update_one = getattr(self.decision_making, 'update_execution', None)
        if update_one is not None:
            update_one(ExecutionView())

    def _parallel_step(self, objective):
        self._recover_parallel()
        self._parallel_views()
        decision_steps, event = 0, {}
        # Never sample another policy decision while an acknowledged outcome is
        # still in a slot's outbox. Deliver it first, retaining its exact payload.
        retrying_delivery = any(slot._feedback is not None for slot in self._slots.values())
        result = None
        ready = getattr(self.decision_making, 'can_step', None)
        if not self._decision_finished and not retrying_delivery and (ready is None or ready()):
            result = self.decision_making.step(objective)
            event['decision'] = asdict(result)
            decision_steps = 1
            self._decision_finished = result.kind == 'finished'
            self._recover_parallel()
            if result.kind == 'dispatch':
                progress = self.exploration.accept(deepcopy(result.directive))
                self._slots[result.directive.directive_id]._validate_progress(progress)
                self._slots[result.directive.directive_id]._update_execution(progress)
        if not self._rotation:
            kind = 'finished' if self._decision_finished else result.kind if result else 'waiting'
            return RunnerStep(kind, event, decision_steps, result.job if result else None)
        ident = self._rotation.popleft()
        self._rotation.append(ident)  # Rotate before I/O so a failing target cannot starve peers.
        slot = self._slots[ident]
        try:
            progress = slot.step(objective)
        finally:
            # Delivery/ack can succeed before a view-notification callback fails.
            # A cleared outbox must not leave a dead slot that resubmits or tries
            # to make a new decision through the ID-scoped adapter.
            if slot._directive is None:
                self._rotation.remove(ident)
                del self._slots[ident]
        event.update(progress.event)
        decision_steps += progress.decision_steps
        self._parallel_views()
        kind = progress.kind
        if kind == 'waiting' and result is not None and result.kind in ('dispatch', 'progressed'):
            kind = 'progressed'
        return RunnerStep(kind, event, decision_steps, progress.job, ident)

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


class _DecisionSlot:
    """An ID-scoped view so the proven single-command outbox can be reused."""
    def __init__(self, endpoint, ident):
        self.endpoint, self.ident = endpoint, ident

    def pending(self):
        return next((d for d in self.endpoint.pending_all() if d.directive_id == self.ident), None)

    def receive_feedback(self, directive, feedback):
        self.endpoint.receive_feedback(directive, feedback)

    def update_execution(self, view):
        update = getattr(self.endpoint, 'update_execution', None)
        if update is not None:
            update(view)

    def review(self, objective, progress):
        review = getattr(self.endpoint, 'review', None)
        return review(objective, progress) if review is not None else ReviewDecision()

    def authorize_review(self, result):
        authorize = getattr(self.endpoint, 'authorize_review', None)
        if authorize is not None:
            authorize(result)


class _ExplorationSlot:
    def __init__(self, endpoint, ident):
        self.endpoint, self.ident = endpoint, ident

    def __getattr__(self, name):
        return getattr(self.endpoint, name)

    def pending(self):
        return next((p for p in self.endpoint.pending_all() if p.directive_id == self.ident), None)

    def execution_view(self):
        describe = getattr(self.endpoint, 'execution_views', None)
        return describe().get(self.ident, ExecutionView()) if describe else ExecutionView(progress=self.pending())
