"""Route namespaced operations and own registered Lab lifecycles."""
from contextlib import nullcontext, contextmanager
from copy import deepcopy
from dataclasses import asdict, replace
import json
from threading import RLock
from time import perf_counter
from pathlib import Path

from rsicomposer.loops.contracts.coordination import OperationResult
from rsicomposer.loops.contracts.decision import OperationCall, OperationSpec
from rsicomposer.runtime.capabilities import optional_method
from rsicomposer.runtime.execution.errors import WorkerLostError
from rsicomposer.runtime.observation import suppress_observations
from rsicomposer.runtime.orchestration.profiles import ProfileBinding
from rsicomposer.loops.contracts.lifecycle import ExecutionStatus
from rsicomposer.loops.contracts.jobs import JobRef, JobOutcome, JobStatus, OperationProgress
from rsicomposer.runtime.storage.files import atomic_write, json_bytes


_RECEIPT_SCHEMA = 24


class LabGroup:
    """Expose multiple Labs as alias.operation through the existing Lab port."""
    def __init__(self, *, receipts=None, artifacts=None, experience=None, artifact_access=None):
        """receipts: a run-owned directory; completed receipts are kept by call_id.

        A call whose receipt is stored returns it without running the Lab again, so
        a resumed run never repeats an operation that finished before an interruption.
        """
        self._lock = RLock()
        self._capacities = {}
        self._state_locks = {}
        self._before_operation = {}
        self._reserved_once = {}
        self._active_calls = set()
        self._invocations = set()
        self._receipt_sequences = {}
        self._applied_sequences = {}
        self._labs, self._operations, self._specs = {}, {}, []
        self._owned = []
        self._bindings = {}
        self._model_scopes = {}
        self._artifacts, self._experience = artifacts, experience
        self._artifact_access = artifact_access
        self._final_calls = None
        self._final_index = 0
        self._applied_receipts = set()
        self._receipts = Path(receipts) if receipts is not None else None
        if self._receipts is not None:
            self._receipts.mkdir(parents=True, exist_ok=True)

    def add(self, alias, lab, *, model_scope=None, max_concurrent=1):
        if type(max_concurrent) is not int or max_concurrent < 1:
            raise ValueError("Lab max_concurrent must be a positive integer")
        # Own a newly constructed provider before validating its catalog.
        close = optional_method(lab, 'close')
        if close is not None:
            self._owned.append(close)
        if not callable(getattr(lab, 'describe', None)) or not callable(getattr(lab, 'execute', None)):
            raise TypeError(f'Lab {alias} must implement describe/execute')
        if any(existing is lab for existing in self._labs.values()):
            if close is not None:
                self._owned.pop()
            raise ValueError('Lab aliases must own distinct instances')
        if alias in self._labs:
            raise ValueError(f'Duplicate Lab alias: {alias}')
        catalog = tuple(lab.describe())
        if any(not isinstance(spec, OperationSpec) for spec in catalog):
            raise TypeError('Lab catalogs must contain OperationSpec entries')
        self._bindings[alias] = ProfileBinding(alias, lab, catalog)
        if optional_method(lab, 'state') is not None and getattr(lab, 'concurrent_state_safe', False) is not True:
            self._state_locks[alias] = RLock()
        self._capacities[alias] = 1 if alias in self._state_locks else max_concurrent
        self._labs[alias] = lab
        self._model_scopes[alias] = model_scope
        for spec in catalog:
            if not isinstance(spec, OperationSpec) or not isinstance(spec.name, str) or not spec.name:
                raise TypeError('Lab catalogs must contain named OperationSpec entries')
            name = f'{alias}.{spec.name}'
            if name in self._operations:
                raise ValueError(f'Duplicate Lab operation: {name}')
            self._operations[name] = (lab, spec.name)
            self._specs.append(replace(spec, name=name))

    @property
    def labs(self):
        return dict(self._labs)

    @property
    def aliases(self):
        return tuple(self._labs)

    def operation_capacity(self, name):
        return self._capacities.get(name.rsplit('.', 1)[0], 1)

    def describe(self):
        return deepcopy(tuple(self._specs))

    def profile_identity(self):
        return {alias: binding.identity() for alias, binding in self._bindings.items()
                if binding.profiles or binding.metric is not None or binding.model_roles or binding.data_bindings}

    def primary_metric(self, alias):
        """The declaration of this Lab instance; callers must not compare across aliases."""
        return self._bindings[alias].metric

    def validate_controller(self, controller):
        required = optional_method(controller, 'required_profiles')
        if required is None:
            return
        from collections.abc import Mapping
        names = required()
        if not isinstance(names, Mapping):
            raise TypeError('required_profiles() must map Lab aliases to profile IDs')
        for alias, identity in names.items():
            binding = self._bindings.get(alias)
            if binding is None or identity not in {p.identity for p in binding.profiles}:
                raise ValueError(f'Controller requires Lab {alias} profile {identity}')

    def execute(self, call):
        if self._final_calls is not None and any(
                final.name.rsplit('.', 1)[0] == call.name.rsplit('.', 1)[0] for final in self._final_calls):
            return OperationResult('Lab search is sealed', (), False, ExecutionStatus.FAILED,
                                   'Search cannot continue after host final evaluation has started', call_id=call.call_id)
        return self._execute(call, finishing=False)

    def can_retry(self, call):
        """Only an explicit idempotent retry of a recorded uncertain failure is valid."""
        with self._lock:
            if call.call_id in self._active_calls:
                return False
            saved = self._saved(call)
            spec = next((spec for spec in self._specs if spec.name == call.name), None)
            outcome = saved['outcome'] if saved is not None else None
            return (spec is not None and spec.idempotent and isinstance(outcome, OperationResult)
                    and outcome.status == ExecutionStatus.FAILED
                    and outcome.data.get('effects_uncertain') is True
                    and outcome.data.get('retryable') is True)

    def retry(self, call, *, after_sequence=None):
        """Retry a failed infrastructure attempt; ordinary execute always replays it."""
        if self._final_calls is not None:
            raise ValueError('Cannot retry after host final evaluation has started')
        return self._execute(call, finishing=False, retry=True, after_sequence=after_sequence)

    def _execute(self, call, *, finishing, retry=False, after_sequence=None):
        alias = call.name.rsplit('.', 1)[0]
        with self._lock:
            if call.call_id in self._invocations:
                raise ValueError('Operation call_id already has an active execution')
            self._invocations.add(call.call_id)
        try:
            with self._state_boundary(alias):
                return self._execute_call(call, finishing=finishing, retry=retry, after_sequence=after_sequence)
        finally:
            with self._lock:
                self._invocations.remove(call.call_id)

    @contextmanager
    def _state_boundary(self, alias, *, wait=True):
        lock = self._state_locks.get(alias)
        if lock is None:
            yield True
            return
        if not lock.acquire(blocking=wait):
            yield False
            return
        try:
            with self._lock:
                self._before_operation[alias] = deepcopy(self._labs[alias].state())
            try:
                yield True
            finally:
                with self._lock:
                    self._before_operation.pop(alias, None)
        finally:
            lock.release()

    def capture_lab_states(self):
        with self._lock:
            states = {}
            for alias, lab in self._labs.items():
                state = self._before_operation[alias] if alias in self._before_operation else lab.state()
                if not isinstance(state, dict):
                    raise TypeError(f'Lab {alias} state() must return a dict')
                states[alias] = deepcopy(state)
            return states

    def _execute_call(self, call, *, finishing, retry, after_sequence):
        started = perf_counter()
        if call.name not in self._operations:
            raise ValueError(f'Unknown Lab operation: {call.name}')
        lab, name = self._operations[call.name]
        alias = call.name.rsplit('.', 1)[0]
        binding = self._bindings[alias]
        local_call = OperationCall(name, deepcopy(call.arguments), call_id=call.call_id)
        with self._lock:
            saved = self._saved(call)
            if retry and after_sequence is not None:
                if (type(after_sequence) is not int or after_sequence < 1 or saved is None
                        or after_sequence > saved['sequence']):
                    raise ValueError('Infrastructure retry source receipt sequence does not match')
                if saved['sequence'] > after_sequence:
                    retry = False  # This explicit retry already published its terminal receipt.
            if retry and not self.can_retry(call):
                raise ValueError('Only idempotent recorded uncertain infrastructure failures may be retried')
            # Authorization also applies when replaying an existing receipt.
            try:
                binding.check(local_call, finishing=finishing, history=False)
            except ValueError as error:
                return OperationResult('Lab profile rejected the operation', (), False,
                                       ExecutionStatus.FAILED, str(error), call_id=call.call_id)
            if saved is not None:
                sequence = saved['sequence']
                if sequence > self._applied_sequences.get(alias, 0):
                    if saved['lab_state'] is not None:
                        restore = optional_method(lab, 'restore')
                        if restore is None or restore(deepcopy(saved['lab_state'])) is not None:
                            raise TypeError('A journaled Lab state needs restore(state) returning None')
                    binding.restore(saved['profile_state'])
                    self._applied_sequences[alias] = sequence
                self._receipt_sequences[alias] = max(sequence, self._receipt_sequences.get(alias, 0))
                self._applied_receipts.add(call.call_id)
                if alias in self._before_operation:
                    self._before_operation[alias] = deepcopy(lab.state())
                if not retry:
                    return saved['outcome']
            keys = tuple((alias, binding._once_key(rule, local_call)) for rule in binding.rules
                         if rule.kind == 'once_per' and (not rule.operation or rule.operation == name))
            try:
                binding.check(local_call, finishing=finishing)
                if any(key in self._reserved_once for key in keys):
                    raise ValueError('Operation conflicts with an in-flight once_per request')
            except ValueError as error:
                return OperationResult('Lab profile rejected the operation', (), False,
                                       ExecutionStatus.FAILED, str(error), call_id=call.call_id)
            for key in keys:
                self._reserved_once[key] = call.call_id
            self._active_calls.add(call.call_id)
        try:
            scope = self._model_scopes[alias]
            from rsicomposer.loops.contracts.decision import _host_decision_id
            access = self._artifact_access
            role = binding.role(local_call)
            with (scope(decision_id=_host_decision_id.get(), call_id=call.call_id) if scope else nullcontext()), (
                    access.scope('lab:' + alias, split=local_call.arguments.get('split'), role=role, finishing=finishing)
                    if access is not None else nullcontext()), (
                    suppress_observations() if role is not None and not role.per_item_results else nullcontext()):
                try:
                    outcome = lab.execute(local_call)
                except WorkerLostError as error:
                    outcome = self._worker_lost(call, error)
                if access is not None and role is not None and not role.per_item_results:
                    access.restrict_result(outcome, owner='lab:' + alias, split=local_call.arguments.get('split'))
            with self._lock:
                if isinstance(outcome, OperationResult):
                    outcome = replace(outcome, call_id=call.call_id)
                outcome = binding.validate_result(local_call, outcome)
                binding.observe(local_call, outcome)
                outcome = binding.aggregate(local_call, outcome, self._artifacts)
                self._store(call, outcome, perf_counter() - started)
                return outcome
        finally:
            with self._lock:
                self._active_calls.discard(call.call_id)
                for key in keys:
                    self._reserved_once.pop(key, None)

    def _worker_lost(self, call, error):
        spec = next(spec for spec in self._specs if spec.name == call.name)
        return OperationResult('Worker lost; operation effects uncertain', (), False,
            ExecutionStatus.FAILED, f'{type(error).__name__}: {error}', call_id=call.call_id,
            data={'effects_uncertain': True, 'retryable': spec.idempotent})

    def _store(self, call, outcome, elapsed):
        if self._receipts is None or not isinstance(outcome, (OperationResult, JobRef)):
            return
        from rsicomposer.runtime.storage.codec import encode
        alias = call.name.rsplit('.', 1)[0]
        lab, _ = self._operations[call.name]
        binding = self._bindings[alias]
        capture = optional_method(lab, 'state')
        sequence = self._receipt_sequences.get(alias, 0) + 1
        atomic_write(self._receipt_path(call), json_bytes(encode(
            {'schema': _RECEIPT_SCHEMA, 'name': call.name, 'arguments': call.arguments, 'outcome': outcome,
             'lab_state': capture() if capture is not None else None,
             'profile_state': binding.state(), 'sequence': sequence, 'elapsed_seconds': elapsed})))
        self._receipt_sequences[alias] = self._applied_sequences[alias] = sequence
        self._applied_receipts.add(call.call_id)
        self._before_operation.pop(alias, None)

    @contextmanager
    def _job_scope(self, call, decision_id):
        alias = call.name.rsplit('.', 1)[0]
        local = replace(call, name=self._operations[call.name][1])
        scope = self._model_scopes[alias]
        access = self._artifact_access
        with (scope(decision_id=decision_id, call_id=call.call_id) if scope else nullcontext()), (
                access.scope('lab:' + alias, split=call.arguments.get('split'),
                             role=self._bindings[alias].role(local), finishing=False)
                if access is not None else nullcontext()):
            yield self._labs[alias], local, self._bindings[alias]

    def can_poll_job(self, call, job):
        lab, _ = self._operations[call.name]
        return optional_method(lab, 'poll_job') is not None

    def poll_job(self, call, job, *, decision_id, after):
        with self._state_boundary(call.name.rsplit('.', 1)[0], wait=False) as available:
            if not available:
                return None, ()
            return self._poll_job(call, job, decision_id=decision_id, after=after)

    def _poll_job(self, call, job, *, decision_id, after):
        with self._job_scope(call, decision_id) as (lab, local, binding):
            try:
                result, progress = self._job_observation(lab, call, job, after)
            except WorkerLostError as error:
                result, progress = self._worker_lost(call, error), ()
            if result is None:
                return None, progress
            return self._record_job_result(call, local, binding, result), progress

    def _record_job_result(self, call, local, binding, result):
        with self._lock:
            result = binding.validate_result(local, replace(result, call_id=call.call_id))
            binding.observe(local, result)
            result = binding.aggregate(local, result, self._artifacts)
            self._store(call, result, self.receipt_elapsed(call))
        return result

    @staticmethod
    def _job_observation(lab, call, job, after):
        poll = optional_method(lab, 'poll_job')
        if poll is None:
            return None, ()  # External explicit resume remains available.
        outcome = poll(job)
        if not isinstance(outcome, JobOutcome) or outcome.job != job:
            raise TypeError('Lab poll_job must return the matching JobOutcome')
        events = optional_method(lab, 'job_events')
        progress = events(job, after=after) if events is not None else ()
        if (not isinstance(progress, tuple) or any(not isinstance(row, OperationProgress)
                or row.job != job or row.call_id != call.call_id for row in progress)
                or [row.sequence for row in progress] != sorted({row.sequence for row in progress})
                or any(row.sequence <= after for row in progress)):
            raise ValueError('Lab job events must be increasing, call-correlated OperationProgress observations')
        if not outcome.terminal:
            return None, progress
        result = outcome.result
        if result is None:
            result = OperationResult('Lab job ' + outcome.status.value, (), outcome.simulated,
                ExecutionStatus.FAILED if outcome.status == JobStatus.FAILED else ExecutionStatus.CANCELLED,
                outcome.error if outcome.status == JobStatus.FAILED else None)
        if not isinstance(result, OperationResult):
            raise TypeError('Lab job completion must contain OperationResult')
        return result, progress

    def control_job(self, call, job, control, *, decision_id):
        with self._state_boundary(call.name.rsplit('.', 1)[0]), self._job_scope(call, decision_id) as (lab, _, _):
            apply = optional_method(lab, 'control_job')
            if apply is None or apply(job, control) is not None:
                raise TypeError('Checkpoint jobs require idempotent control_job(job, control) returning None')

    def cancel_job(self, call, job, *, decision_id):
        """Request cancellation; terminal delivery still uses the normal poll journal."""
        with self._state_boundary(call.name.rsplit('.', 1)[0]), self._job_scope(call, decision_id) as (lab, _, _):
            cancel = optional_method(lab, 'cancel_job')
            if cancel is None:
                raise TypeError('Lab does not implement cancel_job(job)')
            outcome = cancel(job)
            if not isinstance(outcome, JobOutcome) or outcome.job != job:
                raise TypeError('Lab cancellation must return the matching JobOutcome')

    def reattach_job(self, call, job):
        lab, name = self._operations[call.name]
        reattach = optional_method(lab, 'reattach_job')
        if reattach is None:
            raise TypeError('Journaled jobs require reattach_job(job) returning None')
        try:
            value = reattach(job)
        except WorkerLostError as error:
            binding = self._bindings[call.name.rsplit('.', 1)[0]]
            return self._record_job_result(call, replace(call, name=name), binding, self._worker_lost(call, error))
        if value is not None:
            raise TypeError('Journaled jobs require reattach_job(job) returning None')
        return None

    def profile_state(self):
        return {'bindings': {alias: binding.state() for alias, binding in self._bindings.items()},
                'final_calls': deepcopy(self._final_calls), 'final_index': self._final_index}

    def restore_profile_state(self, state):
        if (not isinstance(state, dict) or set(state) != {'bindings', 'final_calls', 'final_index'}
                or set(state['bindings']) != set(self._bindings)):
            raise ValueError('Invalid Lab profile state')
        for alias, binding in self._bindings.items():
            binding.restore(state['bindings'][alias])
        calls, index = state['final_calls'], state['final_index']
        if (calls is not None and (not isinstance(calls, tuple)
                or any(not isinstance(call, OperationCall) for call in calls))):
            raise ValueError('Invalid host final evaluation plan')
        if type(index) is not int or not 0 <= index <= len(calls or ()):
            raise ValueError('Invalid final evaluation cursor')
        self._final_calls, self._final_index = deepcopy(calls), index

    def finish_step(self, before_execute=None):
        """One host-only sealed evaluation after Controller finish, or None.

        Labs declare final_evaluations() using their recorded selection. The
        complete plan and its call IDs are checkpointed before the first effect.
        Nothing from this phase is delivered back to Controller or Memory.
        """
        if self._final_calls is None:
            calls = []
            for alias, lab in self._labs.items():
                final = optional_method(lab, 'final_evaluations')
                plan = final() if final is not None else ()
                if not isinstance(plan, tuple) or any(not isinstance(call, OperationCall) for call in plan):
                    raise TypeError('final_evaluations() must return an OperationCall tuple')
                self._bindings[alias].check_finish(plan)
                for call in plan:
                    if call.name not in ('evaluate', 'execute'):
                        raise ValueError('Final evaluations must be evaluate or execute measurements')
                    name = f'{alias}.{call.name}'
                    if name not in self._operations:
                        raise ValueError(f'Unknown final evaluation: {name}')
                    self._bindings[alias].check(call, finishing=True)
                    calls.append(replace(call, name=name))
            self._final_calls = tuple(calls)
        if self._final_index == len(self._final_calls):
            return None
        call = self._final_calls[self._final_index]
        if before_execute is not None:
            before_execute([{'owner': 'host_finish', 'decision_id': call.call_id, 'operation': deepcopy(call), 'status': 'running'}])
        result = self._execute(call, finishing=True)
        if not isinstance(result, OperationResult):
            raise TypeError('Host finishing evaluations must return terminal receipts')
        if self._experience is not None:
            from rsicomposer.runtime.evidence.records import ExperienceRecord
            # Stable IDs make retrying evidence delivery safe after receipt replay.
            self._experience.record(ExperienceRecord('sealed-' + call.call_id, 'evaluation', 'shared', None,
                'Host final evaluation', result.simulated,
                source_refs=tuple(ref.artifact_id for ref in result.artifacts), details={
                    'phase': 'host_finish', 'operation': asdict(call), 'receipt': result.payload()}))
        self._final_index += 1
        return {'phase': 'host_finish', 'operation': asdict(call), 'receipt': result.payload()}

    def receipt_state(self):
        """Receipt effects already included in the current Lab states."""
        with self._lock:
            return {'applied': sorted(self._applied_receipts), 'sequences': dict(self._applied_sequences)}

    def restore_receipt_state(self, state):
        if (not isinstance(state, dict) or set(state) != {'applied', 'sequences'}
                or not isinstance(state['applied'], list) or not isinstance(state['sequences'], dict)
                or any(not isinstance(key, str) for key in state['applied'])
                or any(alias not in self._labs or type(n) is not int or n < 1
                       for alias, n in state['sequences'].items())):
            raise ValueError('Invalid applied Lab receipt state')
        self._applied_receipts = set(state['applied'])
        self._applied_sequences = dict(state['sequences'])
        self._receipt_sequences = dict(state['sequences'])

    def _receipt_path(self, call):
        return self._receipts / (call.call_id + '.json')

    def _saved(self, call):
        if self._receipts is None or not self._receipt_path(call).exists():
            return None
        from rsicomposer.runtime.storage.codec import decode
        saved = decode(json.loads(self._receipt_path(call).read_text()))
        if not isinstance(saved, dict) or saved.get('schema') != _RECEIPT_SCHEMA:
            raise ValueError('Lab receipt uses an unsupported RSIComposer protocol; start a new run')
        if set(saved) != {'schema', 'name', 'arguments', 'outcome', 'lab_state', 'profile_state', 'sequence', 'elapsed_seconds'}:
            raise ValueError('Invalid Lab receipt fields')
        if saved['name'] != call.name or saved['arguments'] != call.arguments:
            raise ValueError(f'call_id {call.call_id} was already used for a different operation')
        return saved

    def receipt_sequence(self, call):
        saved = self._saved(call)
        return saved['sequence'] if saved is not None else None

    def receipt_elapsed(self, call):
        saved = self._saved(call)
        return saved['elapsed_seconds'] if saved is not None else None

    def stored_receipt(self, call):
        """The completed receipt for call.call_id, or None. Rejects a reused call_id."""
        saved = self._saved(call)
        return saved['outcome'] if saved is not None else None

    def restore_journaled_calls(self, calls):
        """Apply saved states in completion order before any interrupted work restarts."""
        saved = [(call.name.rsplit('.', 1)[0], receipt['sequence'], call)
                 for call in calls if (receipt := self._saved(call)) is not None]
        for _, _, call in sorted(saved, key=lambda item: item[:2]):
            self._execute(call, finishing=False)

    def in_flight_resolution(self, call):
        """How an interrupted call can continue: 'receipt', 'idempotent', or None (needs review)."""
        saved = self.stored_receipt(call)
        if isinstance(saved, JobRef):
            lab, _ = self._operations[call.name]
            return 'job' if optional_method(lab, 'reattach_job') is not None else None
        if saved is not None:
            return 'receipt'
        spec = next((s for s in self._specs if s.name == call.name), None)
        return 'idempotent' if spec is not None and spec.idempotent else None

    def runtime_context(self):
        values = {}
        for alias, lab in self._labs.items():
            read = optional_method(lab, 'runtime_context')
            if read is not None:
                value = read()
                if not isinstance(value, str):
                    raise TypeError('Lab runtime_context must return text')
                values[alias] = value
        return json.dumps(values, ensure_ascii=False) if values else ''

    def shutdown(self, mode):
        complete, errors = True, []
        for lab in self._labs.values():
            shutdown = optional_method(lab, 'shutdown')
            if shutdown is not None:
                try:
                    stopped = shutdown(mode)
                    if type(stopped) is not bool:
                        raise TypeError('Lab shutdown must confirm completion with a bool')
                    complete = stopped and complete
                except Exception as exc:
                    errors.append(exc)
        if errors:
            raise ExceptionGroup('Lab shutdown failed', errors)
        return complete

    def close(self):
        failed, errors = [], []
        for close in reversed(self._owned):
            try:
                close()
            except Exception as exc:
                failed.append(close)
                errors.append(exc)
        self._owned = list(reversed(failed))
        if errors:
            raise ExceptionGroup('Lab cleanup failed', errors)
