"""Run-owned execution state, policy delivery and call-correlated scheduling."""
from concurrent.futures import ThreadPoolExecutor, Future
from contextlib import ExitStack
from contextvars import copy_context
from copy import deepcopy
from dataclasses import asdict, dataclass, field, replace
from queue import Queue, Empty
from time import perf_counter
from types import SimpleNamespace
from uuid import uuid4

from rsicomposer.harnesses.decision.tools import execute_operation
from rsicomposer.harnesses.memory import HistoryEntry
from rsicomposer.loops.contracts.coordination import LoopStepResult, OperationResult
from rsicomposer.loops.contracts.decision import Decision, DecisionInput, _host_decision_id
from rsicomposer.loops.contracts.jobs import JobOutcome, JobRef, JobStatus, OperationProgress
from rsicomposer.loops.contracts.lifecycle import ExecutionStatus
from rsicomposer.loops.decision_making.action_space.access import authorized_space
from rsicomposer.runtime.capabilities import optional_method
from rsicomposer.runtime.evidence.records import ExperienceRecord
from rsicomposer.runtime.observation import observe_operation, report_stage_start
from .ledger import OperationLedger, _PendingOperationRecord, validate_metadata_record


@dataclass
class HostExecutionState:
    history_entries: tuple[HistoryEntry, ...] = ()
    decision_feedback_cursor: int = 0
    operation_progress: list[OperationProgress] = field(default_factory=list)
    progress_cursor: int = 0
    operation_tasks: dict[str, dict] = field(default_factory=dict)
    finish_after_operations: LoopStepResult | None = None
    operation_results: list[OperationResult] = field(default_factory=list)
    pending_job: JobRef | None = None
    pending_operation_record: _PendingOperationRecord | None = None
    finished: bool = False
    measured_input_tokens: int = 0
    measured_output_tokens: int = 0
    calls_with_missing_usage: int = 0
    run_control: dict[str, object] = field(default_factory=dict)
    research_cycles: dict[int, dict[str, object]] = field(default_factory=dict)
    decision_records: dict[str, dict] = field(default_factory=dict)
    pending_operation_decision_id: str | None = None
    pending_decision: Decision | None = None
    decision_results: dict[str, LoopStepResult] = field(default_factory=dict)

    def __post_init__(self):
        if (not isinstance(self.history_entries, tuple)
                or any(not isinstance(row, HistoryEntry) for row in self.history_entries)
                or len({row.record_id for row in self.history_entries}) != len(self.history_entries)):
            raise ValueError('Invalid canonical execution history')
        if (type(self.decision_feedback_cursor) is not int or not 0 <= self.decision_feedback_cursor <= len(self.operation_results)
                or type(self.progress_cursor) is not int or not 0 <= self.progress_cursor <= len(self.operation_progress)):
            raise ValueError('Invalid Controller delivery cursor')
        if not isinstance(self.decision_records, dict):
            raise ValueError('Decision records must be indexed by ID')
        for ident, record in self.decision_records.items():
            if (not isinstance(ident, str) or not ident.strip() or not isinstance(record, dict)
                    or record.get('decision_id') != ident
                    or record.get('status') not in ('pending', 'executing', 'completed', 'failed', 'cancelled', 'partial', 'blocked')):
                raise ValueError('Invalid decision ledger record')
            validate_metadata_record(record)
            for name in ('depends_on_decision_ids', 'source_decision_ids', 'adopted_candidate_ids', 'adopted_decision_ids',
                         'unresolved_adopted_candidate_ids'):
                refs = record.get(name, ())
                if (not isinstance(refs, tuple) or any(not isinstance(ref, str) or not ref.strip() for ref in refs)
                        or len(set(refs)) != len(refs)):
                    raise ValueError(f'Invalid ledger {name}')
            if record.get('adopted_candidate_ids') and record['status'] != 'completed':
                raise ValueError('Only completed decisions may confirm adoption')
        for record in self.decision_records.values():
            if not record.get('validation_error'):
                for name in ('depends_on_decision_ids', 'source_decision_ids'):
                    if any(ref == record['decision_id'] or ref not in self.decision_records
                           for ref in record.get(name, ())):
                        raise ValueError('Confirmed lineage requires known non-self references')
                if any(self.decision_records[ref]['status'] != 'completed'
                       for ref in record.get('depends_on_decision_ids', ())):
                    raise ValueError('Confirmed dependencies require completed decisions')
            edges = record.get('confirmed_edges', ())
            if not isinstance(edges, tuple) or any(not isinstance(edge, dict)
                    or set(edge) != {'from', 'to', 'kind', 'confirmed'} or edge['confirmed'] is not True
                    or edge['to'] != record['decision_id'] or edge['from'] not in self.decision_records
                    or edge['from'] == edge['to'] or edge['kind'] not in ('depends_on', 'source', 'adopted') for edge in edges):
                raise ValueError('Invalid confirmed decision edges')
            expected = tuple({'from': ref, 'to': record['decision_id'], 'kind': kind, 'confirmed': True}
                for kind, refs in (('depends_on', record.get('depends_on_decision_ids', ())),
                                   ('source', record.get('source_decision_ids', ())))
                for ref in refs) if not record.get('validation_error') else ()
            if edges != expected:
                raise ValueError('Confirmed edges differ from validated lineage')
        if self.pending_operation_decision_id is not None and self.pending_operation_decision_id not in self.decision_records:
            raise ValueError('Pending operation must reference a known decision')
        if self.pending_decision is not None and (not isinstance(self.pending_decision, Decision)
                or self.pending_decision.decision_id not in self.decision_records):
            raise ValueError('Pending decision must retain its registered identity')
        if not isinstance(self.decision_results, dict) or any(k not in self.decision_records
                or not isinstance(v, LoopStepResult) or v.decision_id != k for k, v in self.decision_results.items()):
            raise ValueError('Decision results must reference their ledger identity')


class HostExecution:
    def __init__(self, *, experience):
        self._memory = _HostHistory(self)
        self.experience = experience
        self.state = HostExecutionState()
        self.endpoint = None
        self.in_flight_hook = None
        self._receipt_listeners, self._cleanup_callbacks = [], []
        self.operation_scheduler = OperationScheduler(self)
        self.ledger = OperationLedger(self)

    def activate(self, endpoint, controller_version=None, *, reset_delivery=True, delivery=None):
        if endpoint.host_execution is not self:
            raise ValueError('Controller endpoint belongs to another run execution')
        changing = self.endpoint is not endpoint
        if changing and reset_delivery and delivery is not None:
            if (not isinstance(delivery, dict) or set(delivery) != {'feedback', 'progress'}
                    or type(delivery['feedback']) is not int
                    or not 0 <= delivery['feedback'] <= len(self.state.operation_results)
                    or type(delivery['progress']) is not int
                    or not 0 <= delivery['progress'] <= len(self.state.operation_progress)):
                raise ValueError('Invalid Controller version delivery boundary')
        self.endpoint = endpoint
        self.controller_version = controller_version
        endpoint.controller_version = controller_version
        bind = optional_method(endpoint.harness, 'bind_execution')
        if bind is not None:
            bind(self.execution_view)
        if changing and reset_delivery:
            if delivery is not None:
                self.state.decision_feedback_cursor = delivery['feedback']
                self.state.progress_cursor = delivery['progress']
            self.state.finished = False
            self.state.finish_after_operations = None
            self.state.pending_decision = None

    @property
    def harness(self):
        if self.endpoint is None:
            raise RuntimeError('HostExecution requires an activated Controller endpoint')
        return self.endpoint.harness

    @property
    def policy(self):
        return self.endpoint.policy

    @property
    def tools(self):
        return self.endpoint.operation_router

    @property
    def memory(self):
        return self._memory

    def current_model(self):
        return self.endpoint.current_model()

    def decision_graph(self):
        return self.ledger._decision_graph()

    def add_receipt_listener(self, listener, *, owner):
        self._receipt_listeners.append((owner, listener))

    def add_cleanup(self, callback):
        self._cleanup_callbacks.append(callback)

    def close(self):
        self.operation_scheduler.close()
        callbacks, self._cleanup_callbacks = self._cleanup_callbacks, []
        failed, errors = [], []
        for callback in reversed(callbacks):
            try:
                callback()
            except BaseException as error:
                failed.append(callback)
                errors.append(error)
        self._cleanup_callbacks.extend(reversed(failed))
        if errors:
            for error in errors[1:]:
                errors[0].add_note(f'Additional execution cleanup failure: {type(error).__name__}: {error}')
            raise errors[0]

    def observe(self, event):
        from rsicomposer.runtime.storage.codec import encode
        row = self.state.decision_records.get(event.decision_id, {})
        event = replace(event, identity={**event.identity,
            'controller_version': self.controller_version,
            'issuing_controller_version': row.get('controller_version'),
            'issuing_model': row.get('decision_model')})
        self.experience.record(ExperienceRecord(event.event_id, 'progress', 'shared', event.task_id or None,
            'Lab job ' + event.kind, False, details={'event': encode(asdict(event))}))
        self.harness.observe(deepcopy(event))

    def execution_view(self):
        """Detached facts for Harness context; no scheduling or mutable state access."""
        return deepcopy({'operation_results': tuple(self.state.operation_results),
            'progresses': tuple(self.state.operation_progress[self.state.progress_cursor:]),
            'run_control': self.state.run_control, 'pending': tuple(self.state.operation_tasks),
            'decision_graph': self.decision_graph(), 'decision_model': self.current_model()})

    def _inference_ready(self):
        """True when the Controller's own background inference finished and it should decide
        without waiting for outstanding Lab operations (optional policy inference_ready())."""
        ready = optional_method(self.policy, 'inference_ready')
        return ready is not None and bool(ready())

    def _context(self, objective, action_space):
        return SimpleNamespace(objective=objective, operations=action_space.describe(),
                               decision_model=self.current_model())

    def _call_policy(self, method, objective, action_space):
        capabilities = tuple({'kind': 'invoke', **asdict(spec)} for spec in action_space.describe())
        feedback = deepcopy(tuple(self.state.operation_results[self.state.decision_feedback_cursor:]))
        window = self.harness.window(objective, capabilities, feedback)
        integration = getattr(self.endpoint, 'integration', None)
        if integration is not None:
            window = integration.project_window(window)
        if not isinstance(window, DecisionInput):
            raise TypeError('ControllerHarness.window must return DecisionInput')
        ident = str(uuid4())
        token = _host_decision_id.set(ident)
        received = 0
        def record(usage):
            nonlocal received
            received += 1
            self._record_usage(usage)
        try:
            scope = self.endpoint.model_scope
            with ExitStack() as stack:
                if scope is not None:
                    stack.enter_context(scope(decision_id=ident))
                observer = optional_method(self.policy, 'track_usage')
                if observer is not None:
                    stack.enter_context(observer(record))
                result = method(deepcopy(window))
            if not received and isinstance(result, Decision) and not getattr(self.policy, 'asynchronous_usage', False):
                self._record_usage(result.usage)
            drain = optional_method(self.policy, 'drain_usage')
            if drain is not None:
                for usage in drain():
                    self._record_usage(usage)
            return result
        finally:
            _host_decision_id.reset(token)


    def observed_decisions(self):
        """Return detached recorded decisions for optional observation clients."""
        return deepcopy(tuple(self.state.decision_records.values()))

    @staticmethod
    def _copy_run_control(context: dict[str, object]) -> dict[str, object]:
        if not isinstance(context, dict) or any(not isinstance(key, str) for key in context):
            raise TypeError('Run control context must be a dict with string keys')
        return deepcopy(context)

    def update_run_control(self, context: dict[str, object]) -> None:
        """Replace external running hints without changing research state or objectives."""
        self.state.run_control = self._copy_run_control(context)

    def can_poll_job(self, job):
        if self.operation_scheduler.can_poll_job(job):
            return True
        can_poll = optional_method(self.policy, 'can_poll_job')
        return self.state.pending_job == job and can_poll is not None and can_poll(job) is True

    def _complete_policy_wait(self, outcome):
        job, ident = self.state.pending_job, self.state.pending_operation_decision_id
        if not isinstance(outcome, JobOutcome) or outcome.job != job:
            raise TypeError('Controller polling requires a matching JobOutcome')
        outcome.__post_init__()
        if not outcome.terminal:
            return LoopStepResult('waiting', job=job, decision_id=ident, decision_steps=0)
        row = self.state.decision_records[ident]
        row['status'] = ('completed' if outcome.status == JobStatus.SUCCEEDED else
                         'cancelled' if outcome.status == JobStatus.CANCELLED else 'failed')
        if outcome.error:
            row['error'] = outcome.error
        result = LoopStepResult('progressed', reason=outcome.error or 'Controller worker ready',
                                decision_id=ident, decision_steps=0)
        self.state.decision_results[ident] = deepcopy(result)
        self.state.pending_job = None
        self.state.pending_operation_decision_id = None
        return result

    def poll_research(self):
        """Poll a policy-owned wait without fabricating a Lab operation receipt."""
        job = self.state.pending_job
        if job is None:
            return None
        can_poll = optional_method(self.policy, 'can_poll_job')
        poll = optional_method(self.policy, 'poll_job')
        if can_poll is None or poll is None or can_poll(job) is not True:
            return LoopStepResult('waiting', job=job,
                                  decision_id=self.state.pending_operation_decision_id, decision_steps=0)
        return self._complete_policy_wait(poll(job))

    def drain_research(self):
        if self.state.pending_job is None:
            drain = optional_method(self.policy, 'drain_research')
            if drain is not None:
                drain()
                usages = optional_method(self.policy, 'drain_usage')
                if usages is not None:
                    for usage in usages():
                        self._record_usage(usage)
            return None
        job = self.state.pending_job
        can_poll = optional_method(self.policy, 'can_poll_job')
        poll = optional_method(self.policy, 'poll_job')
        if job is None or can_poll is None or poll is None or can_poll(job) is not True:
            return None
        outcome = poll(job)
        if not isinstance(outcome, JobOutcome) or outcome.job != job:
            raise TypeError('Controller polling requires a matching JobOutcome')
        if outcome.terminal:
            close = optional_method(self.policy, 'close')
            if close is not None:
                close()
        return self._complete_policy_wait(outcome)

    def cancel_research(self):
        native_cancel = optional_method(self.policy, 'cancel_research')
        if self.state.pending_job is None and native_cancel is not None:
            complete = native_cancel()
            drain = optional_method(self.policy, 'drain_usage')
            if drain is not None:
                for usage in drain():
                    self._record_usage(usage)
            return complete
        job = self.state.pending_job
        can_poll = optional_method(self.policy, 'can_poll_job')
        cancel = optional_method(self.policy, 'cancel_job')
        if job is None or can_poll is None or cancel is None or can_poll(job) is not True:
            return False
        outcome = cancel(job)
        self._complete_policy_wait(outcome)
        return self.state.pending_job is None

    def pending_research_job(self) -> JobRef | None:
        """Report work awaiting external resume, without polling or cancelling it."""
        if self.state.pending_job is not None:
            return self.state.pending_job
        read = optional_method(self.policy, 'pending_research_job')
        return read() if read is not None else None

    def pending_records(self) -> bool:
        """Include unfinished reviews even when their tool receipts are recorded."""
        return self.state.pending_operation_record is not None

    def flush_pending_records(self) -> None:
        """Retry existing receipts without executing the operation again."""
        if self.state.pending_operation_record is not None:
            call_id = self.state.pending_operation_record.result.call_id
            self.ledger._flush_operation_record()
            self.state.operation_tasks.pop(call_id, None)

    def restart_research(self, context: dict[str, object]) -> None:
        """Reopen a finished, drained cycle; retries of accepted rounds are no-ops."""
        context = self._copy_run_control(context)
        research_round = context.get('research_round')
        if type(research_round) is not int or research_round < 1:
            raise ValueError('research_round must be a positive integer')
        previous = self.state.research_cycles.get(research_round)
        if previous is not None:
            if previous != context:
                raise ValueError('Research round context differs from its original request')
            return
        if not self.state.finished:
            raise RuntimeError('Research can restart only from a finished state')
        if self.pending_research_job() is not None:
            raise RuntimeError('Cannot restart research with unresolved Controller inference')
        if (self.state.operation_tasks or self.state.pending_job is not None
                or self.state.pending_operation_record is not None):
            raise RuntimeError('Cannot restart research with pending jobs or operations')
        run_control = deepcopy(context)
        self.state.research_cycles[research_round] = context
        self.state.run_control = run_control
        self.state.finished = False
        self.state.finish_after_operations = None

    def reopen_in_flight(self, calls):
        """Requeue only after the host has resolved every interrupted call safely."""
        self.operation_scheduler.reopen(calls)

    def drain_operations(self, action_space, mode):
        """Host stop control drains receipts without sampling another policy."""
        return self.operation_scheduler.drain_plan(action_space,
            {ident: mode for ident in self.state.operation_tasks})

    def in_flight_operations(self):
        return self.operation_scheduler.in_flight()

    def drain_operation_plan(self, action_space, plan):
        return self.operation_scheduler.drain_plan(action_space, plan)

    def step(self, objective: str, action_space) -> LoopStepResult:
        action_space = authorized_space(action_space, 'decision')
        if self.state.pending_operation_record is not None:
            ident = self.state.pending_operation_record.decision_id
            call_id = self.state.pending_operation_record.result.call_id
            result = self.ledger._flush_operation_record()
            self.state.operation_tasks.pop(call_id, None)
            return LoopStepResult("progressed", operation_result=result, decision_id=ident, decision_steps=0)
        if self.state.finished:
            return LoopStepResult("finished")
        if self.state.finish_after_operations is not None:
            if self.state.operation_tasks:
                return self.operation_scheduler.advance(action_space)
            self.state.finished = True
            return deepcopy(self.state.finish_after_operations)
        if self.state.pending_job is not None:
            return self.poll_research()
        if (self.state.operation_tasks and self.state.pending_decision is None
                and self.state.decision_feedback_cursor == len(self.state.operation_results)
                and not any(task.get('checkpoint') is not None for task in self.state.operation_tasks.values())
                and not self._inference_ready()):
            return self.operation_scheduler.advance(action_space, interrupt=self._inference_ready)
        ready = optional_method(self.policy, 'can_decide')
        if self.state.pending_decision is None and ready is not None and ready() is False:
            drain = optional_method(self.policy, 'drain_usage')
            if drain is not None:
                for usage in drain():
                    self._record_usage(usage)
            return LoopStepResult('waiting', reason='Controller inference is generating', decision_steps=0)
        context = self._context(objective, action_space)
        saved_pending = self.state.pending_decision
        sample = optional_method(self.policy, 'decide')
        if sample is None:
            if not callable(self.policy):
                raise TypeError('Controller policy requires decide(input) or a callable')
            sample = self.policy
        decision = saved_pending or self._call_policy(sample, objective, action_space)
        if not isinstance(decision, Decision):
            raise TypeError('Controller policy must return Decision')
        self.state.decision_feedback_cursor = len(self.state.operation_results)
        self.state.progress_cursor = len(self.state.operation_progress)
        metadata = None
        if saved_pending is None:
            read_metadata = optional_method(self.harness, 'freeze_decision_metadata')
            metadata = read_metadata(self.policy, decision.decision_id) if read_metadata is not None else {}
        row = self.ledger._begin_decision(decision, context, requested=metadata)
        ident = decision.decision_id
        previous = self.state.decision_results.get(ident)
        if previous is not None:
            self.state.pending_decision = None
            return deepcopy(previous)
        try:
            if decision.kind == "invoke":
                self.operation_scheduler.submit(decision, action_space)
                return replace(self.operation_scheduler.advance(action_space, interrupt=self._inference_ready),
                               decision_steps=1)
            elif decision.kind == 'control':
                self.operation_scheduler.control(decision.control)
                row['status'] = 'completed'
                result = LoopStepResult('progressed', decision_id=ident, reason='Job checkpoint ' + decision.control.action)
            elif decision.kind == "wait":
                if self.state.operation_tasks and decision.job is None:
                    row['status'] = 'completed'
                    self.state.pending_decision = None
                    return replace(self.operation_scheduler.advance(action_space, interrupt=self._inference_ready),
                                   decision_steps=1)
                self.state.pending_job = decision.job
                self.state.pending_operation_decision_id = ident if decision.job else None
                row['status'] = 'pending' if decision.job else 'completed'
                result = LoopStepResult("waiting", job=decision.job, decision_id=ident)
            else:
                row['status'] = 'completed'
                result = LoopStepResult("finished", reason=decision.reason, decision_id=ident)
                if self.state.operation_tasks:
                    self.state.finish_after_operations = replace(result, decision_steps=0)
                    self.state.pending_decision = None
                    return replace(self.operation_scheduler.advance(action_space), decision_steps=1)
                self.state.finished = True
        except Exception as error:
            if self.state.pending_operation_record is None:
                row.update(status='failed', error=f'{type(error).__name__}: {error}')
                self.state.pending_decision = None
                self.state.pending_operation_decision_id = None
            raise
        self.state.decision_results[ident] = deepcopy(result)
        self.state.pending_decision = None
        return result

    def _record_usage(self, usage) -> None:
        if usage is None or usage.input_tokens is None or usage.output_tokens is None:
            self.state.calls_with_missing_usage += 1
        if usage is not None:
            self.state.measured_input_tokens += usage.input_tokens or 0
            self.state.measured_output_tokens += usage.output_tokens or 0


class _HostHistory:
    """Canonical history belongs to a run; Harnesses receive detached projections."""
    def __init__(self, execution):
        self.execution = execution

    def append(self, entry):
        from rsicomposer.harnesses.memory import HistoryEntry
        from rsicomposer.loops.contracts.feedback_context import EvidenceEvent
        if not isinstance(entry, HistoryEntry):
            raise TypeError('Host history requires HistoryEntry')
        entries = self.execution.state.history_entries
        known = next((row for row in entries if row.record_id == entry.record_id), None)
        if known is not None and known != entry:
            raise ValueError('Duplicate host history identity has different contents')
        if known is None:
            self.execution.state.history_entries += (deepcopy(entry),)
        self.execution.harness.observe(EvidenceEvent('history:' + entry.record_id,
            'controller', 'history', task_id=entry.task_id or '', output={'history_entry': asdict(entry)},
            identity={'controller_version': self.execution.controller_version}))

    def history(self, *, task_id=None, kind=None, limit=None):
        rows = tuple(row for row in self.execution.state.history_entries
                     if (task_id is None or row.task_id == task_id) and (kind is None or row.kind == kind))
        return deepcopy(rows[-limit:] if limit is not None else rows)


class OperationScheduler:
    def __init__(self, execution):
        self.host = execution
        self._executors = {}
        self._futures = {}
        self._arrivals = Queue()

    @property
    def tasks(self):
        return self.host.state.operation_tasks

    def in_flight(self):
        return [dict(decision_id=task['decision_id'], operation=deepcopy(task['call']),
                     status=task['status']) for task in self.tasks.values()]

    def capacity(self, name):
        lab = getattr(self.host.tools, 'lab', None)
        capacity = optional_method(lab, 'operation_capacity')
        return capacity(name) if capacity is not None else 1

    @staticmethod
    def alias(call):
        if call.name.startswith('host.'):
            return '<host>'
        return call.name.rsplit('.', 1)[0] if '.' in call.name else '<host>'

    def submit(self, decision, action_space):
        # Admit the complete batch before starting any effects.
        seen = {call['call_id'] for row in self.host.state.decision_records.values()
                if row['decision_id'] != decision.decision_id
                for call in row['proposal'].get('operations', ())}
        lab = getattr(self.host.tools, 'lab', None)
        can_retry = optional_method(lab, 'can_retry')
        retries = {}
        for call in decision.operations:
            if call.call_id in self.tasks:
                raise ValueError('Operation call_id already has an active execution')
            if call.call_id in seen:
                previous = [issued for row in self.host.state.decision_records.values()
                            if row['decision_id'] != decision.decision_id
                            for issued in row['proposal'].get('operations', ())
                            if issued['call_id'] == call.call_id]
                same_request = all(issued['name'] == call.name and issued['arguments'] == call.arguments
                                   for issued in previous)
                if not same_request or can_retry is None or not can_retry(call):
                    raise ValueError('Operation call_id was already issued by another decision')
                sequence = optional_method(lab, 'receipt_sequence')
                retries[call.call_id] = sequence(call) if sequence is not None else None
        for call in decision.operations:
            try:
                action_space.validate(call)
            except PermissionError as denied:
                row = self.host.state.decision_records[decision.decision_id]
                row['authorization_rejected'] = True
                self.host.state.pending_operation_decision_id = decision.decision_id
                self.host.ledger._record_operation(call, OperationResult('Decision operation failed', (),
                    decision.simulated, ExecutionStatus.FAILED, f'PermissionError: {denied}', call_id=call.call_id))
                raise
            except (TypeError, ValueError):
                pass  # The shared executor returns the exact validation failure.
        for call in decision.operations:
            self.tasks[call.call_id] = {'decision_id': decision.decision_id, 'call': deepcopy(call),
                'simulated': decision.simulated, 'status': 'queued', 'job': None,
                'last_event': 0, 'checkpoint': None, 'retry': call.call_id in retries,
                'retry_sequence': retries.get(call.call_id)}
        self.host.state.pending_decision = None
        self.host.state.decision_records[decision.decision_id]['status'] = 'executing'
        self._launch(action_space)

    def _checkpoint(self):
        if self.host.in_flight_hook is not None:
            self.host.in_flight_hook(self.in_flight())

    def _launch(self, action_space):
        active = {}
        for task in self.tasks.values():
            if task['status'] == 'running' or (task['status'] == 'waiting' and task['checkpoint'] is None):
                alias = self.alias(task['call'])
                active[alias] = active.get(alias, 0) + 1
        ready = []
        for ident, task in self.tasks.items():
            alias = self.alias(task['call'])
            if task['status'] == 'queued' and active.get(alias, 0) < self.capacity(task['call'].name):
                active[alias] = active.get(alias, 0) + 1
                task['status'] = 'running'
                ready.append((ident, task))
        if not ready:
            return
        # Persist ALL running and queued calls before launching the first worker.
        self._checkpoint()
        tools = self.host.tools
        for ident, task in ready:
            row = self.host.state.decision_records[task['decision_id']]
            report_stage_start(task['decision_id'], task['call'].name, task['call'].arguments, {
                **{key: row.get(key) for key in ('reason', 'depends_on_decision_ids', 'source_decision_ids')},
                **deepcopy(row.get('context_metadata', {}))}, call_id=ident)
            # ContextVars preserve model/accounting and artifact scopes in the Lab worker.
            context = copy_context()
            def run(task=deepcopy(task)):
                token = _host_decision_id.set(task['decision_id'])
                try:
                    with observe_operation(task['call'].call_id):
                        return execute_operation(tools, action_space, task['call'],
                                                 simulated=task['simulated'], clock=perf_counter, retry=task['retry'],
                                                 retry_sequence=task['retry_sequence'])
                finally:
                    _host_decision_id.reset(token)
            alias = self.alias(task['call'])
            if alias == '<host>':
                future = Future()
                try:
                    future.set_result(context.run(run))
                except BaseException as error:
                    future.set_exception(error)
            else:
                if alias not in self._executors:
                    self._executors[alias] = ThreadPoolExecutor(max_workers=self.capacity(task['call'].name),
                                                               thread_name_prefix='rsicomposer-' + alias)
                future = self._executors[alias].submit(context.run, run)
            self._futures[ident] = future
            future.add_done_callback(lambda future, ident=ident: self._arrivals.put(ident))

    def _poll_jobs(self, call_ids=None):
        from rsicomposer.loops.contracts.feedback_context import EvidenceEvent
        lab = getattr(self.host.tools, 'lab', None)
        poll = optional_method(lab, 'poll_job')
        if poll is None:
            return None
        progressed = False
        for ident, task in list(self.tasks.items()):
            if call_ids is not None and ident not in call_ids:
                continue
            if task['status'] != 'waiting':
                continue
            result, events = poll(task['call'], task['job'], decision_id=task['decision_id'], after=task['last_event'])
            for event in events:
                self.host.observe(EvidenceEvent(f'{ident}:progress:{event.sequence}', 'lab', event.kind,
                    task_id=ident, status='running', output=asdict(event), request=asdict(task['call']),
                    references=(asdict(event.checkpoint),) if event.checkpoint is not None else (),
                    decision_id=task['decision_id'], identity={'call_id': ident}))
                self.host.state.operation_progress.append(deepcopy(event))
                task['last_event'] = event.sequence
                if event.kind == 'checkpoint':
                    task['checkpoint'] = event.sequence
                progressed = True
            if result is not None:
                return self.complete(ident, result)
        return LoopStepResult('progressed', reason='Lab job progress', decision_steps=0) if progressed else None

    def can_poll_job(self, job):
        check = optional_method(getattr(self.host.tools, 'lab', None), 'can_poll_job')
        return check is not None and any(task['job'] == job and check(task['call'], job)
                                         for task in self.tasks.values())

    def advance(self, action_space, call_ids=None, *, interrupt=None):
        """Run queued operations until one completes, or until interrupt() reports Controller work."""
        while True:
            progress = self._poll_jobs(call_ids)
            if progress is not None:
                return progress
            self._launch(action_space)
            futures = {key: value for key, value in self._futures.items() if call_ids is None or key in call_ids}
            if not futures:
                waiting = next((task for key, task in self.tasks.items()
                                if task['job'] is not None and (call_ids is None or key in call_ids)), None)
                if waiting is not None:
                    return LoopStepResult('waiting', job=waiting['job'], decision_id=waiting['decision_id'], decision_steps=0)
                raise RuntimeError('Operation queue cannot make progress')
            if call_ids is not None:
                ident = next((key for key, future in futures.items() if future.done()), None)
                if ident is None:
                    from concurrent.futures import wait, FIRST_COMPLETED
                    wait(tuple(futures.values()), timeout=.05, return_when=FIRST_COMPLETED)
                    continue
            else:
                try:
                    ident = self._arrivals.get(timeout=.05)
                except Empty:
                    if interrupt is not None and interrupt():
                        # The Controller decides next; the running operations keep going.
                        return LoopStepResult('progressed', reason='Controller inference finished', decision_steps=0)
                    continue
                if ident not in self._futures:
                    continue  # A selective drain already consumed this arrival.
            execution = self._futures.pop(ident).result()
            task = self.tasks[ident]
            if isinstance(execution.outcome, JobRef):
                task.update(status='waiting', job=execution.outcome)
                self._checkpoint()
                continue
            return self.complete(ident, execution.observation(), elapsed=execution.elapsed_seconds,
                                 error=execution.error)

    def drain_plan(self, action_space, plan):
        """Apply only the named swap actions, preserving handover operations."""
        from rsicomposer.loops.contracts.jobs import JobControl
        for ident, mode in plan.items():
            task = self.tasks.get(ident)
            if task is None:
                continue
            if mode not in ('drain', 'cancel'):
                raise ValueError('Selective drain accepts drain or cancel')
            if mode == 'cancel' and task['status'] == 'queued':
                return self.complete(ident, OperationResult('Cancelled for Controller replacement', (),
                    task['simulated'], ExecutionStatus.CANCELLED, call_id=ident))
            if task['checkpoint'] is not None:
                self.control(JobControl(ident, task['checkpoint'], 'stop' if mode == 'cancel' else 'continue'))
            elif mode == 'cancel' and task['job'] is not None:
                cancel = optional_method(getattr(self.host.tools, 'lab', None), 'cancel_job')
                if cancel is None:
                    raise TypeError('Swap cancellation requires a Lab cancel_job contract or a job checkpoint')
                cancel(task['call'], task['job'], decision_id=task['decision_id'])
        return self.advance(action_space, set(plan)) if any(key in self.tasks for key in plan) else None

    def control(self, control):
        task = self.tasks.get(control.call_id)
        if task is None or task['job'] is None or task['checkpoint'] != control.sequence:
            raise ValueError('Job control must match a pending checkpoint event')
        apply = optional_method(getattr(self.host.tools, 'lab', None), 'control_job')
        if apply is None:
            raise TypeError('Lab group has no job control capability')
        # The control key (call_id, sequence) is durable before applying it.
        self._checkpoint()
        apply(task['call'], task['job'], control, decision_id=task['decision_id'])
        task['checkpoint'] = None

    def complete(self, ident, result, *, elapsed=None, error=None):
        task = self.tasks[ident]
        read_elapsed = optional_method(getattr(self.host.tools, 'lab', None), 'receipt_elapsed')
        if read_elapsed is not None:
            saved_elapsed = read_elapsed(task['call'])
            if saved_elapsed is not None:
                elapsed = saved_elapsed
        result = replace(result, call_id=ident, simulated=result.simulated or task['simulated'])
        try:
            self.host.ledger._adoption(task['decision_id'], result)
        except (TypeError, ValueError) as invalid:
            error = invalid
            result = OperationResult('Invalid executor adoption receipt', (), result.simulated,
                                     ExecutionStatus.FAILED, str(invalid), call_id=ident)
        self.host.state.pending_operation_decision_id = task['decision_id']
        self.host.ledger._record_operation(task['call'], result, elapsed_seconds=elapsed)
        self.tasks.pop(ident, None)
        if error is not None:
            raise error
        return LoopStepResult('progressed', operation_result=result, decision_id=task['decision_id'], decision_steps=0)

    def reopen(self, calls):
        if self._futures:
            raise RuntimeError('Cannot reopen operations while workers are active')
        for call in calls:
            task = self.tasks.get(call.call_id)
            if task is None or task['call'] != call or task['call'].call_id != call.call_id:
                raise ValueError('Interrupted call is absent from the restored operation queue')
        lab = getattr(self.host.tools, 'lab', None)
        stored = optional_method(lab, 'stored_receipt')
        for call in calls:
            saved = stored(call) if stored is not None else None
            if isinstance(saved, JobRef) and lab.reattach_job(call, saved) is None:
                self.tasks[call.call_id].update(status='waiting', job=saved)
            else:
                self.tasks[call.call_id].update(status='queued', job=None, checkpoint=None)
            self.host.state.decision_records[self.tasks[call.call_id]['decision_id']]['status'] = 'executing'

    def close(self):
        for executor in self._executors.values():
            executor.shutdown(wait=True, cancel_futures=True)
        self._executors.clear()
