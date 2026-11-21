"""Bounded transport between public loop endpoints; no policy or backend decisions."""

from copy import deepcopy
from dataclasses import asdict, dataclass, replace
from time import monotonic, sleep
from typing import Callable, Literal, Protocol
import warnings

from rsicomposer.loops.contracts.lifecycle import ExecutionStatus
from rsicomposer.loops.contracts.coordination import DecisionEndpoint
from rsicomposer.loops.contracts.jobs import JobRef
from rsicomposer.runtime.state.recovery import durable_mutation
from rsicomposer.runtime.observation import report_step, trigger_probe
from rsicomposer.runtime.orchestration.run_policy import (RunPolicy, ResearcherRunPolicy, TimeRunPolicy, GoalRunPolicy,
                         RunContext, RunDecision, ExperimentOutcome, MetricGoal, positive)


@dataclass(frozen=True)
class RunResult:
    status: Literal["completed", "finished", "waiting", "step_limit"]
    rounds_completed: int
    decision_steps: int
    events: list[dict[str, object]]
    waiting_job: JobRef | None = None
    stop_reason: str | None = None
    shutdown_complete: bool | None = None


@dataclass(frozen=True)
class RunnerStep:
    kind: Literal['progressed', 'completed', 'waiting', 'finished']
    event: dict[str, object]
    decision_steps: int = 0
    job: JobRef | None = None


class RunObserver(Protocol):
    """Observes a completed tick; does not authorize or replay execution."""
    def on_step(self, result: RunnerStep) -> None: ...


class _RunStopping(BaseException):
    """Internal control signal; tool-error recovery must not swallow a run stop."""


def _shutdown_error_details(error):
    details = {'error_type': type(error).__name__, 'message': str(error)}
    if isinstance(error, BaseExceptionGroup):
        details['errors'] = [_shutdown_error_details(child) for child in error.exceptions]
    return details


class Runner:
    def __init__(self, decision_making: DecisionEndpoint,
                 *, clock: Callable[[], float] = monotonic,
                 run_policy: RunPolicy | None = None, cleanup_timeout_seconds: float = 30,
                 execution_shutdown: Callable[[str], bool] | None = None,
                 finish_execution: Callable | None = None):
        self.host_budget = None
        self.run_policy = run_policy if run_policy is not None else ResearcherRunPolicy()
        if not callable(getattr(self.run_policy, 'evaluate', None)):
            raise TypeError('run_policy must implement evaluate(context)')
        positive(cleanup_timeout_seconds, 'cleanup_timeout_seconds')
        self.cleanup_timeout_seconds = cleanup_timeout_seconds
        if execution_shutdown is not None and not callable(execution_shutdown):
            raise TypeError('execution_shutdown must be callable')
        self._finish_execution = finish_execution
        self._finishing_complete = False
        self._execution_shutdown = execution_shutdown
        self._execution_shutdown_error = None
        self._run_started = None
        self._decision_step_budget = None
        self._research_round = 1
        self._researcher_finished = False
        self._restart_at = None
        self._restart_context = None
        self._stop_decision = None
        self._stop_started = None
        self._terminal_step = None
        self._outcomes = {}
        self._quiescing = False
        self._guard_actions = type(self.run_policy) is not ResearcherRunPolicy
        self._action_guard = self._guard_run
        self._decision_finished = False
        self.decision_making = decision_making
        self._clock = clock
        self._observers = []

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

    def _record_outcome(self, receipt):
        if receipt.call_id is None:
            return
        outcome = ExperimentOutcome(deepcopy(receipt))
        previous = self._outcomes.get(receipt.call_id)
        if previous is not None and previous != outcome:
            prior = previous.receipt
            if not (prior.status == ExecutionStatus.FAILED
                    and prior.data.get('effects_uncertain') is True and prior.data.get('retryable') is True):
                raise ValueError('Conflicting delivered run outcome')
        self._outcomes[receipt.call_id] = outcome

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
                'budget': self.host_budget.remaining() if self.host_budget is not None else {},
                'remaining_seconds': max(0.0, seconds - context.elapsed_seconds) if seconds is not None else None,
                'researcher_finish_ends_run': type(policy) is ResearcherRunPolicy,
                'goal': asdict(policy.goal) if isinstance(policy, GoalRunPolicy) and isinstance(policy.goal, MetricGoal) else None,
                **({'decision_steps': deepcopy(self._decision_step_budget)} if self._decision_step_budget is not None else {})}

    def _research_job(self):
        pending = getattr(self.decision_making, 'pending_research_job', None)
        return pending() if callable(pending) else None

    def _evaluate_run(self, context):
        if self.host_budget is not None:
            from rsicomposer.runtime.execution.budget import BudgetExhausted
            try:
                self.host_budget.check()
            except BudgetExhausted as exc:
                return RunDecision('stop', str(exc), shutdown='cancel')
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
        research_job = self._research_job()
        if research_job is not None and self._stop_decision.shutdown == 'cancel':
            cancel_research = getattr(self.decision_making, 'cancel_research', None)
            if callable(cancel_research):
                cancel_research()
                research_job = self._research_job()  # an acknowledgement alone is not completion
        if research_job is not None and self._stop_decision.shutdown == 'drain':
            # Finish the current policy generation boundary, then close its worker
            # without dispatching another Lab operation during host shutdown.
            drain_research = getattr(self.decision_making, 'drain_research', None)
            if callable(drain_research):
                drain_research()
                research_job = self._research_job()
        pending_records = self._pending_records()
        execution_complete = True
        execution_error = None
        if self._execution_shutdown is not None:
            try:
                execution_complete = self._execution_shutdown(self._stop_decision.shutdown)
                if type(execution_complete) is not bool:
                    raise TypeError('Execution shutdown must return bool confirming owned work has exited')
            except Exception as exc:
                execution_complete = False
                execution_error = exc
                self._execution_shutdown_error = _shutdown_error_details(exc)
        complete = not self.busy and research_job is None and not pending_records and execution_complete
        expired = self._clock() - self._stop_started >= self.cleanup_timeout_seconds
        stop = {'reason': self._stop_decision.reason, 'shutdown': self._stop_decision.shutdown,
                'evidence_refs': list(self._stop_decision.evidence_refs),
                'shutdown_complete': complete, 'cleanup_timed_out': bool(expired and not complete),
                'unresolved_operations': bool(self.busy),
                'unresolved_research_job': asdict(research_job) if research_job else None,
                'unresolved_execution': not execution_complete,
                'unresolved_records': bool(pending_records)}
        if self._execution_shutdown_error is not None:
            stop['execution_shutdown_error'] = {**self._execution_shutdown_error, 'resolved': execution_complete}
        # A pending drain adoption must not suppress hard-stop cancellation of
        # the old Controller's jobs or bypass the bounded cleanup deadline.
        if not expired and self._stop_decision.shutdown == 'cancel' and self.busy:
            drain = getattr(self.decision_making, 'drain_operations', None)
            result = drain('cancel') if callable(drain) else None
            if result is not None:
                if result.operation_result is not None:
                    self._record_outcome(result.operation_result)
                event = {**asdict(result), 'run_stop': stop}
                if original is not None:
                    event['preceding_step'] = original.event
                return RunnerStep(result.kind, event,
                    result.decision_steps + (original.decision_steps if original else 0), result.job)
        boundary = self._checkpoint_step() if not expired else None
        if boundary is not None:
            if original is not None:
                return replace(boundary, event={**boundary.event, 'preceding_step': original.event},
                               decision_steps=boundary.decision_steps + original.decision_steps)
            return boundary
        if complete or expired:
            event = {**(original.event if original else {}), 'run_stop': stop}
            if original is not None and original.kind == 'completed':
                # Preserve the final feedback event; the next tick reports run termination.
                self._terminal_step = RunnerStep('finished', {'run_stop': stop})
                return replace(original, event=event)
            self._terminal_step = RunnerStep('finished', event, original.decision_steps if original else 0)
            return deepcopy(self._terminal_step)
        if execution_error is not None:
            raise execution_error
        if original is not None:
            return replace(original, event={**original.event, 'run_stop': stop})
        if pending_records:
            flush = getattr(self.decision_making, 'flush_pending_records', None)
            if callable(flush):
                flush()
            if not self._pending_records():
                return self._stop_step(objective, RunnerStep('progressed', {'run_control': 'records_flushed'}))
        if self.busy:
            drain = getattr(self.decision_making, 'drain_operations', None)
            result = drain(self._stop_decision.shutdown) if callable(drain) else None
            if result is not None:
                if result.operation_result is not None:
                    self._record_outcome(result.operation_result)
                return RunnerStep(result.kind, {**asdict(result), 'run_stop': stop}, 0, result.job)
        return RunnerStep('waiting', {'run_stop': stop}, job=research_job)

    def add_observer(self, observer: RunObserver):
        if not callable(getattr(observer, 'on_step', None)):
            raise TypeError('RunObserver needs on_step(result)')
        self._observers.append(observer)
        def detach():
            if observer in self._observers:
                self._observers.remove(observer)
        return detach

    def step(self, objective: str) -> RunnerStep:
        result = self._step(objective)
        for observer in tuple(self._observers):
            try:
                observer.on_step(deepcopy(result))
            except Exception as error:
                warnings.warn(f'Run observation failed ({type(error).__name__})', RuntimeWarning)
        return result

    @durable_mutation
    def _step(self, objective: str) -> RunnerStep:
        """Apply external run control around one bounded transport tick; never sleep."""
        if self._terminal_step is not None:
            return deepcopy(self._terminal_step)
        if self._run_started is None:
            self._run_started = self._clock()
        if self._stop_decision is not None:
            return self._stop_step(objective)
        if self._researcher_finished and not self._finishing_complete:
            finished = self._finish_step()
            if finished is not None:
                return finished
        context = self._run_context(objective)
        decision = self._evaluate_run(context)
        if decision.action == 'stop':
            self._begin_stop(decision)
            return self._stop_step(objective)
        boundary = self._checkpoint_step()
        if boundary is not None:
            return boundary
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
        if callable(update) and (type(self.run_policy) is not ResearcherRunPolicy or self._decision_step_budget is not None or self.host_budget is not None):
            update(self._control_context(context))
        try:
            result = self._transport_step(objective)
        except _RunStopping:
            return self._stop_step(objective, RunnerStep('waiting',
                {'run_control': 'stopping_before_action'}, decision_steps=1))
        if result.kind == 'finished' or self._decision_finished:
            self._researcher_finished = True
            finished = self._finish_step()
            if finished is not None:
                return replace(finished, decision_steps=result.decision_steps)
        decision = self._evaluate_run(self._run_context(objective))
        if decision.action == 'stop':
            self._begin_stop(decision)
            return self._stop_step(objective, result)
        if result.kind == 'finished':
            if decision.action == 'restart' and self._restart_at is None:
                self._restart_at = self._clock() + decision.restart_delay_seconds
            return replace(result, kind='waiting', event={**result.event, 'run_control': 'researcher_finished'})
        return result

    def _finish_step(self):
        boundary = self._checkpoint_step()
        if boundary is not None:
            return boundary
        if self._finishing_complete:
            return None
        recovery = getattr(self, 'recovery', None)
        hook = recovery.commit_in_flight if recovery is not None else None
        event = self._finish_execution(hook) if self._finish_execution is not None else None
        if event is not None:
            return RunnerStep('progressed', event)
        self._finishing_complete = True
        return None

    def _checkpoint_step(self):
        boundary = getattr(self.decision_making, 'checkpoint_boundary', None)
        adopted = boundary() if callable(boundary) else None
        if adopted is None:
            return None
        if adopted.operation_result is not None:
            self._record_outcome(adopted.operation_result)
        return RunnerStep(adopted.kind, asdict(adopted), adopted.decision_steps, adopted.job)

    def _control_method(self, name):
        capabilities = getattr(self.decision_making, 'run_control_capabilities', None)
        if callable(capabilities) and name not in capabilities():
            return None
        method = getattr(self.decision_making, name, None)
        return method if callable(method) else None

    @property
    def busy(self) -> bool:
        pending = getattr(self.decision_making, 'in_flight_operations', None)
        return bool(pending()) if callable(pending) else False

    def _transport_step(self, objective: str) -> RunnerStep:
        """One bounded Controller/Lab scheduling tick."""
        result = self._decision_step(objective)
        if result.operation_result is not None:
            self._record_outcome(result.operation_result)
        return RunnerStep(result.kind, asdict(result), result.decision_steps, result.job)

    def run(self, objective: str, rounds: int, *, max_decision_steps: int | None = None,
            wait_for_jobs: bool = False) -> RunResult:
        if type(wait_for_jobs) is not bool:
            raise TypeError('wait_for_jobs must be boolean')
        if type(rounds) is not int or rounds < 1:
            raise ValueError('rounds must be a positive integer')
        limit = rounds * 8 if max_decision_steps is None else max_decision_steps
        if type(limit) is not int or limit < 1:
            raise ValueError('max_decision_steps must be a positive integer')
        events, completed, decision_steps = [], 0, 0
        previous_budget = self._decision_step_budget
        try:
            while decision_steps < limit:
                self._decision_step_budget = {'scope': 'this_run_invocation', 'limit': limit,
                                              'used': decision_steps, 'remaining': limit - decision_steps}
                result = self.step(objective)
                can_poll = getattr(self.decision_making, 'can_poll_job', None)
                managed_wait = (wait_for_jobs and result.kind == 'waiting' and result.job is not None
                                and callable(can_poll) and can_poll(result.job))
                if managed_wait and result.decision_steps == 0:
                    sleep(.1)
                    continue  # Idle polls do not inflate events or decision budgets.
                events.append(result.event)
                report_step(result.event)
                trigger_probe('step', result.event)
                if result.event.get('operation_result') or result.event.get('feedback'):
                    trigger_probe('feedback', result.event)
                decision_steps += result.decision_steps
                if result.kind == 'finished' and 'feedback' in result.event:
                    completed += 1
                if managed_wait:
                    sleep(.1)
                    continue
                if result.kind in ('waiting', 'finished'):
                    return RunResult(result.kind, completed, decision_steps, events, result.job,
                                     result.event.get('run_stop', {}).get('reason'),
                                     result.event.get('run_stop', {}).get('shutdown_complete'))
                if result.kind == 'completed':
                    completed += 1
                    if completed == rounds:
                        return RunResult('completed', completed, decision_steps, events)
            return RunResult('step_limit', completed, decision_steps, events)
        finally:
            self._decision_step_budget = previous_budget
            update = self._control_method('update_run_control')
            if callable(update) and self._run_started is not None:
                update(self._control_context(self._run_context(objective)))
