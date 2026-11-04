"""Bounded transport between public loop endpoints; no policy or backend decisions."""

from copy import deepcopy
from collections import deque
from dataclasses import asdict, dataclass, replace
from time import monotonic
from typing import Callable, Literal

from rsisys.loops.contracts.coordination import (
    DecisionEndpoint, ExplorationEndpoint, ExplorationProgress, ExecutionView, ReviewDecision, DirectiveUpdate,
)
from rsisys.loops.contracts.directive import Directive
from rsisys.loops.contracts.feedback import Feedback
from rsisys.loops.contracts.jobs import JobRef
from .recovery import durable_mutation
from .run_policy import (RunPolicy, ResearcherRunPolicy, TimeRunPolicy, GoalRunPolicy,
                         RunContext, RunDecision, ExperimentOutcome, MetricGoal, positive)


@dataclass(frozen=True)
class RunResult:
    status: Literal["completed", "finished", "waiting", "step_limit"]
    rounds_completed: int
    decision_steps: int
    events: list[dict[str, object]]
    waiting_job: JobRef | None = None
    waiting_directive_id: str | None = None
    stop_reason: str | None = None
    shutdown_complete: bool | None = None


@dataclass(frozen=True)
class RunnerStep:
    kind: Literal['progressed', 'completed', 'waiting', 'finished']
    event: dict[str, object]
    decision_steps: int = 0
    job: JobRef | None = None
    directive_id: str | None = None


class _RunStopping(BaseException):
    """Internal control signal; tool-error recovery must not swallow a run stop."""


class Runner:
    def __init__(self, decision_making: DecisionEndpoint, exploration: ExplorationEndpoint,
                 *, clock: Callable[[], float] = monotonic, parallel: bool = False,
                 run_policy: RunPolicy | None = None, cleanup_timeout_seconds: float = 30):
        if type(parallel) is not bool:
            raise TypeError('parallel must be boolean')
        self.run_policy = run_policy if run_policy is not None else ResearcherRunPolicy()
        if not callable(getattr(self.run_policy, 'evaluate', None)):
            raise TypeError('run_policy must implement evaluate(context)')
        positive(cleanup_timeout_seconds, 'cleanup_timeout_seconds')
        self.cleanup_timeout_seconds = cleanup_timeout_seconds
        self._run_started = None
        self._research_round = 1
        self._researcher_finished = False
        self._restart_at = None
        self._restart_context = None
        self._stop_decision = None
        self._stop_started = None
        self._terminal_step = None
        self._outcomes = {}
        self._on_outcome = self._record_outcome
        self._quiescing = False
        self._guard_actions = type(self.run_policy) is not ResearcherRunPolicy
        self._action_guard = self._guard_run
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

    def _guard_run(self, objective):
        if self._run_started is None:
            return
        if self._stop_decision is None:
            decision = self._evaluate_run(self._run_context(objective))
            if decision.action == 'stop':
                self._begin_stop(decision)
        if self._stop_decision is not None:
            raise _RunStopping('Run stopping before a new action')

    def _decision_step(self, objective):
        guarded = getattr(self.decision_making, 'step_guarded', None)
        if self._guard_actions and callable(guarded):
            return guarded(objective, lambda: self._action_guard(objective))
        return self.decision_making.step(objective)

    def _pending_records(self):
        pending = getattr(self.decision_making, 'pending_records', None)
        return pending() if callable(pending) else False

    def _record_outcome(self, directive, feedback):
        outcome = ExperimentOutcome(deepcopy(directive), deepcopy(feedback))
        previous = self._outcomes.get(directive.directive_id)
        if previous is not None and previous != outcome:
            raise ValueError('Conflicting delivered run outcome')
        self._outcomes[directive.directive_id] = outcome

    def _run_context(self, objective):
        return RunContext(objective, max(0.0, self._clock() - self._run_started),
                          self._research_round, self._researcher_finished or self._decision_finished,
                          deepcopy(tuple(self._outcomes.values())))

    def _control_context(self, context):
        policy = self.run_policy
        seconds = policy.seconds if isinstance(policy, TimeRunPolicy) else getattr(policy, 'max_seconds', None)
        mode = ('researcher' if type(policy) is ResearcherRunPolicy else
                'time' if type(policy) is TimeRunPolicy else 'goal' if type(policy) is GoalRunPolicy else 'custom')
        return {'research_round': context.research_round, 'mode': mode,
                'elapsed_seconds': context.elapsed_seconds,
                'remaining_seconds': max(0.0, seconds - context.elapsed_seconds) if seconds is not None else None,
                'researcher_finish_ends_run': type(policy) is ResearcherRunPolicy,
                'goal': asdict(policy.goal) if isinstance(policy, GoalRunPolicy) and isinstance(policy.goal, MetricGoal) else None}

    def _research_job(self):
        pending = getattr(self.decision_making, 'pending_research_job', None)
        return pending() if callable(pending) else None

    def _evaluate_run(self, context):
        decision = self.run_policy.evaluate(context)
        if not isinstance(decision, RunDecision):
            raise TypeError('RunPolicy must return RunDecision')
        decision.__post_init__()
        if decision.action == 'restart' and not context.researcher_finished:
            raise ValueError('RunPolicy can restart only after researcher finish')
        return deepcopy(decision)

    def _begin_stop(self, decision):
        if self._stop_decision is None:
            self._stop_decision, self._stop_started = decision, self._clock()
        self._quiescing = True

    def _stop_step(self, objective, original=None):
        if self.parallel:
            self._recover_parallel()
            pending = tuple(self._slots)
        else:
            self._recover()
            pending = (self._directive.directive_id,) if self._directive else ()
        research_job = self._research_job()
        if research_job is not None and self._stop_decision.shutdown == 'cancel':
            cancel_research = getattr(self.decision_making, 'cancel_research', None)
            if callable(cancel_research):
                cancel_research()
                research_job = self._research_job()  # an acknowledgement alone is not completion
        pending_records = self._pending_records()
        complete = not pending and research_job is None and not pending_records
        expired = self._clock() - self._stop_started >= self.cleanup_timeout_seconds
        stop = {'reason': self._stop_decision.reason, 'shutdown': self._stop_decision.shutdown,
                'evidence_refs': list(self._stop_decision.evidence_refs),
                'shutdown_complete': complete, 'cleanup_timed_out': bool(expired and not complete),
                'unresolved_directives': list(pending),
                'unresolved_research_job': asdict(research_job) if research_job else None,
                'unresolved_records': bool(pending_records)}
        if complete or expired:
            event = {**(original.event if original else {}), 'run_stop': stop}
            if original is not None and original.kind == 'completed':
                # Preserve the final feedback event; the next tick reports run termination.
                self._terminal_step = RunnerStep('finished', {'run_stop': stop})
                return replace(original, event=event)
            self._terminal_step = RunnerStep('finished', event, original.decision_steps if original else 0)
            return deepcopy(self._terminal_step)
        if original is not None:
            return replace(original, event={**original.event, 'run_stop': stop})
        if pending_records and not pending:
            flush = getattr(self.decision_making, 'flush_pending_records', None)
            if callable(flush):
                flush()
            if not self._pending_records():
                return self._stop_step(objective, RunnerStep('progressed', {'run_control': 'records_flushed'}))
        if not pending:
            return RunnerStep('waiting', {'run_stop': stop}, job=research_job)
        # Cancel/drain one experiment per tick. Failures keep their outbox and
        # are reported while other experiments still get a fair cleanup turn.
        ident = self._rotation[0] if self.parallel else pending[0]
        try:
            if self._stop_decision.shutdown == 'cancel':
                self.cancel(ident)
            result = self._transport_step(objective)
        except Exception as exc:
            if self.parallel and self._rotation and self._rotation[0] == ident:
                self._rotation.rotate(-1)
            raise
        return self._stop_step(objective, result)

    @durable_mutation
    def step(self, objective: str) -> RunnerStep:
        """Apply external run control around one bounded transport tick; never sleep."""
        if self._terminal_step is not None:
            return deepcopy(self._terminal_step)
        if self._run_started is None:
            self._run_started = self._clock()
        if self._stop_decision is not None:
            return self._stop_step(objective)
        context = self._run_context(objective)
        decision = self._evaluate_run(context)
        if decision.action == 'stop':
            self._begin_stop(decision)
            return self._stop_step(objective)
        if context.researcher_finished and not self.busy and self._research_job() is None:
            if decision.action != 'restart':
                return RunnerStep('waiting', {'run_control': 'researcher_finished'})
            if self._restart_at is None:
                self._restart_at = self._clock() + decision.restart_delay_seconds
            if self._clock() < self._restart_at:
                return RunnerStep('waiting', {'run_control': 'restart_wait',
                    'retry_after_seconds': self._restart_at - self._clock()})
            restart = self._control_method('restart_research')
            if not callable(restart):
                raise NotImplementedError('Continued research requires restart_research(context)')
            if self._restart_context is None:
                self._restart_context = self._control_context(replace(context, research_round=self._research_round + 1))
            restart(deepcopy(self._restart_context))
            self._research_round += 1
            self._researcher_finished = self._decision_finished = False
            self._restart_at = self._restart_context = None
            return RunnerStep('progressed', {'run_control': 'research_restarted', 'research_round': self._research_round})
        update = self._control_method('update_run_control')
        if callable(update) and type(self.run_policy) is not ResearcherRunPolicy:
            update(self._control_context(context))
        try:
            result = self._transport_step(objective)
        except _RunStopping:
            return self._stop_step(objective, RunnerStep('waiting',
                {'run_control': 'stopping_before_action'}, decision_steps=1))
        if result.kind == 'finished' or self._decision_finished:
            self._researcher_finished = True
        decision = self._evaluate_run(self._run_context(objective))
        if decision.action == 'stop':
            self._begin_stop(decision)
            return self._stop_step(objective, result)
        if result.kind == 'finished':
            if decision.action == 'restart' and self._restart_at is None:
                self._restart_at = self._clock() + decision.restart_delay_seconds
            return replace(result, kind='waiting', event={**result.event, 'run_control': 'researcher_finished'})
        return result

    def _control_method(self, name):
        capabilities = getattr(self.decision_making, 'run_control_capabilities', None)
        if callable(capabilities) and name not in capabilities():
            return None
        method = getattr(self.decision_making, name, None)
        return method if callable(method) else None

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
        if self._quiescing:
            return progress, 0
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
            guarded = getattr(self.decision_making, 'review_guarded', None)
            if self._guard_actions and callable(guarded):
                result = guarded(objective, deepcopy(progress), lambda: self._action_guard(objective))
            else:
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
        self._on_outcome(self._directive, self._feedback)
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

    def _transport_step(self, objective: str) -> RunnerStep:
        """One decision/review flow and exploration phase/poll; reviews may query tools."""
        if self.parallel:
            return self._parallel_step(objective)
        self._recover()
        self._update_execution()
        decision_steps = 0
        if self._directive is None:
            result = self._decision_step(objective)
            event = asdict(result)
            decision_steps = 1
            if result.kind != 'dispatch':
                return RunnerStep(result.kind, event, decision_steps, result.job)
            self._directive = deepcopy(result.directive)
            self._recover()
        else:
            event = {'kind': 'dispatch', 'directive': asdict(self._directive)}
        directive_id = self._directive.directive_id
        if self._run_started is not None and not self._quiescing:
            control = self._evaluate_run(self._run_context(objective))
            if control.action == 'stop':
                self._begin_stop(control)
                return RunnerStep('waiting', event, decision_steps, directive_id=directive_id)
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

    @durable_mutation
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

    @durable_mutation
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
                slot._guard_actions = self._guard_actions
                slot._action_guard = self._action_guard
                slot._on_outcome = self._record_outcome
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
        if not self._quiescing and not self._decision_finished and not retrying_delivery and (ready is None or ready()):
            result = self._decision_step(objective)
            event['decision'] = asdict(result)
            decision_steps = 1
            self._decision_finished = result.kind == 'finished'
            self._recover_parallel()
            if result.kind == 'dispatch':
                if self._run_started is not None:
                    control = self._evaluate_run(self._run_context(objective))
                    if control.action == 'stop':
                        self._begin_stop(control)
                        return RunnerStep('waiting', event, decision_steps, directive_id=result.directive.directive_id)
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
            slot._quiescing = self._quiescing
            progress = slot._transport_step(objective)
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
            if result.kind == 'finished' and 'feedback' in result.event:
                completed += 1
            if result.kind in ('waiting', 'finished'):
                return RunResult(result.kind, completed, decision_steps, events, result.job, result.directive_id,
                                 result.event.get('run_stop', {}).get('reason'),
                                 result.event.get('run_stop', {}).get('shutdown_complete'))
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

    def review_guarded(self, objective, progress, before_action):
        review = getattr(self.endpoint, 'review_guarded', None)
        return review(objective, progress, before_action) if callable(review) else self.review(objective, progress)

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
