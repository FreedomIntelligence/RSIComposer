"""Route namespaced operations and own registered Lab lifecycles."""
from contextlib import nullcontext, contextmanager
from copy import deepcopy
from dataclasses import asdict, replace
from errno import ENAMETOOLONG
from hashlib import sha256
import json
import os
from stat import S_ISREG
from threading import RLock
from time import perf_counter
from pathlib import Path
import warnings

from rsicomposer.loops.contracts.coordination import OperationResult
from rsicomposer.loops.contracts.decision import OperationCall, OperationSpec
from rsicomposer.loops.contracts.execution import execute_lab_call
from rsicomposer.runtime.capabilities import optional_method
from rsicomposer.runtime.execution.errors import WorkerLostError
from rsicomposer.runtime.observation import suppress_observations
from rsicomposer.runtime.orchestration.profiles import ProfileBinding
from rsicomposer.loops.contracts.lifecycle import ExecutionStatus
from rsicomposer.loops.contracts.jobs import JobRef, JobOutcome, JobStatus, OperationProgress
from rsicomposer.runtime.storage.files import atomic_write, json_bytes


_LEGACY_RECEIPT_SCHEMA = 24
_RECEIPT_SCHEMA = 26
_RECEIPT_FIELDS = {'schema', 'name', 'arguments', 'outcome', 'lab_state', 'profile_state',
                   'sequence', 'elapsed_seconds'}


def _open_receipt(path, flags):
    return os.open(path, flags | os.O_NOFOLLOW | os.O_NONBLOCK)


def _read_receipt(path):
    with open(path, encoding='utf-8', opener=_open_receipt) as stream:
        if not S_ISREG(os.fstat(stream.fileno()).st_mode):
            raise ValueError('Lab receipt must be a regular file')
        return stream.read()


def _bind_receipt(call, result):
    if not isinstance(result, OperationResult):
        raise TypeError('Lab job completion must contain OperationResult')
    if result.call_id is not None and result.call_id != call.call_id:
        raise ValueError('Operation receipt call_id does not match the issued call')
    return replace(result, call_id=call.call_id)


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
        self._memory_receipts = {} if receipts is None else None
        if self._receipts is not None:
            self._receipts.mkdir(parents=True, exist_ok=True)

    def add(self, alias, lab, *, model_scope=None, max_concurrent=1, declare_roles=None):
        if any(existing is lab for existing in self._labs.values()):
            raise ValueError('Lab aliases must own distinct instances')
        # Own the new provider before calling any of its declaration ports.
        close = optional_method(lab, 'close')
        if close is not None:
            self._owned.append(close)
        if type(max_concurrent) is not int or max_concurrent < 1:
            raise ValueError('Lab max_concurrent must be a positive integer')
        if not isinstance(alias, str) or not alias or '.' in alias:
            raise ValueError('Lab aliases must be nonempty and cannot contain dots')
        if not callable(getattr(lab, 'describe', None)) or not callable(getattr(lab, 'execute', None)):
            raise TypeError(f'Lab {alias} must implement describe/execute')
        if alias in self._labs:
            raise ValueError(f'Duplicate Lab alias: {alias}')
        if declare_roles is not None:
            roles = optional_method(lab, 'resource_roles')
            if roles is not None:
                declare_roles(roles())
        catalog = tuple(lab.describe())
        for spec in catalog:
            if not isinstance(spec, OperationSpec) or not isinstance(spec.name, str) or not spec.name:
                raise TypeError('Lab catalogs must contain named OperationSpec entries')
            if '.' in spec.name:
                raise ValueError('Lab local operation names cannot contain dots: ' + spec.name)
        if len({spec.name for spec in catalog}) != len(catalog):
            raise ValueError('Lab operation names must be distinct')
        self._bindings[alias] = ProfileBinding(alias, lab, catalog)
        if optional_method(lab, 'state') is not None and getattr(lab, 'concurrent_state_safe', False) is not True:
            self._state_locks[alias] = RLock()
            if max_concurrent > 1:
                # The Lab holds mutable state and never promised concurrent safety,
                # so calls are serialized. Say so: a silent drop to one turns a
                # configured parallelism setting into a mystery slowdown.
                warnings.warn(
                    f'Lab {alias!r} keeps state and does not set concurrent_state_safe = True; '
                    f'running it with max_concurrent=1 instead of {max_concurrent}',
                    RuntimeWarning, stacklevel=3)
        self._capacities[alias] = 1 if alias in self._state_locks else max_concurrent
        self._labs[alias] = lab
        self._model_scopes[alias] = model_scope
        for spec in catalog:
            name = f'{alias}.{spec.name}'
            self._operations[name] = (lab, spec.name)
            self._specs.append(replace(spec, name=name))

    @property
    def labs(self):
        return dict(self._labs)

    @property
    def aliases(self):
        return tuple(self._labs)

    def catalog_metrics(self):
        """The detached per-alias metric declarations, including unscored Labs."""
        return deepcopy({alias: binding.metric for alias, binding in self._bindings.items()})

    def candidate_identity_fields(self):
        """Qualified paths from the installed bindings, not a second provider query."""
        return {f'{alias}.{name}': fields for alias, binding in self._bindings.items()
                for name, fields in binding.candidate_fields.items()}

    def operation_capacity(self, name):
        return self._capacities.get(name.rsplit('.', 1)[0], 1)

    def describe(self):
        return deepcopy(tuple(self._specs))

    def profile_identity(self):
        return {alias: binding.identity() for alias, binding in self._bindings.items()
                if (binding.profiles or binding.metric is not None or binding.model_roles or binding.data_bindings
                    or binding.roles or binding.rules or binding.evaluations or binding.candidate_fields)}

    def is_evaluation(self, name):
        """Check the installed Lab catalog, not an operation receipt's claims."""
        alias, _, local = name.rpartition('.')
        return alias in self._bindings and local in self._bindings[alias].evaluations

    def primary_metric(self, alias):
        """The declaration of this Lab instance; callers must not compare across aliases."""
        return self._bindings[alias].metric

    def validate_bindings(self, requirements):
        """Check method-specific requirements inside each mounted Lab, without exposing it to the Policy."""
        for alias, requirement in requirements.items():
            validate = optional_method(self._labs.get(alias), 'validate_binding')
            if validate is None:
                raise ValueError(f'Lab {alias!r} cannot validate the required Controller binding')
            if validate(requirement) is not None:
                raise TypeError('Lab binding validators must return None')

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

    @contextmanager
    def _invocation(self, call):
        with self._lock:
            available = call.call_id not in self._invocations
            if available:
                self._invocations.add(call.call_id)
        try:
            yield available
        finally:
            if available:
                with self._lock:
                    self._invocations.remove(call.call_id)

    def _execute(self, call, *, finishing, retry=False, after_sequence=None):
        with self._invocation(call) as available:
            if not available:
                raise ValueError('Operation call_id already has an active execution')
            with self._state_boundary(call.name.rsplit('.', 1)[0]):
                return self._execute_call(call, finishing=finishing, retry=retry, after_sequence=after_sequence)

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
        local_call = replace(call, name=name)
        with self._lock:
            if not finishing and self._final_calls is not None and any(
                    final.name.rpartition('.')[0] == alias or final.call_id == call.call_id
                    for final in self._final_calls):
                return OperationResult('Lab search is sealed', (), False, ExecutionStatus.FAILED,
                    'Search cannot continue after host final evaluation has started', call_id=call.call_id)
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
                self._apply_receipt(saved)
                if not retry:
                    return saved['outcome']
            try:
                binding.check(local_call, finishing=finishing)
                keys = tuple((alias, key) for key in binding.once_keys(local_call))
                if any(key in self._reserved_once for key in keys):
                    raise ValueError('Operation conflicts with an in-flight once_per request')
            except ValueError as error:
                rejected = OperationResult('Lab profile rejected the operation', (), False,
                                           ExecutionStatus.FAILED, str(error), call_id=call.call_id)
                _, rejected = binding.settle_result(local_call, rejected, self._artifacts)
                self._store(call, rejected, perf_counter() - started, finishing=finishing)
                return rejected
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
                    # A provider can normalize its own arguments, not the Host's admitted identity.
                    outcome = execute_lab_call(lab, replace(local_call, arguments=deepcopy(local_call.arguments)))
                except WorkerLostError as error:
                    outcome = self._worker_lost(call, error)
                if access is not None and role is not None and not role.per_item_results:
                    access.restrict_result(outcome, owner='lab:' + alias, split=local_call.arguments.get('split'))
            with self._lock:
                if isinstance(outcome, OperationResult):
                    outcome = _bind_receipt(local_call, outcome)
                return self._commit_outcome(call, local_call, binding, outcome,
                                            perf_counter() - started, finishing=finishing)
        finally:
            with self._lock:
                self._active_calls.discard(call.call_id)
                for key in keys:
                    self._reserved_once.pop(key, None)

    def _apply_receipt(self, row):
        """Apply a committed state once, under the caller's alias boundary and group lock."""
        alias = row['name'].rpartition('.')[0]
        sequence = row['sequence']
        if sequence > self._applied_sequences.get(alias, 0):
            if row['lab_state'] is not None:
                restore = optional_method(self._labs[alias], 'restore')
                if restore is None or restore(deepcopy(row['lab_state'])) is not None:
                    raise TypeError('A journaled Lab state needs restore(state) returning None')
            self._bindings[alias].restore(row['profile_state'])
            self._applied_sequences[alias] = sequence
            if alias in self._before_operation:
                self._before_operation[alias] = deepcopy(self._labs[alias].state())
        self._receipt_sequences[alias] = max(sequence, self._receipt_sequences.get(alias, 0))
        self._applied_receipts.add(row['call_id'])

    def _commit_outcome(self, call, local, binding, outcome, elapsed, **metadata):
        # Publish the profile transition (including JobRef reservations) and
        # its receipt together. A failed terminal write must not release a key.
        outcome, result = binding.settle_result(local, outcome, self._artifacts)
        previous = binding.state()
        try:
            binding.observe(local, outcome)
            self._store(call, result, elapsed, **metadata)
        except BaseException:
            binding.restore(previous)
            raise
        return result

    def _worker_lost(self, call, error):
        spec = next(spec for spec in self._specs if spec.name == call.name)
        return OperationResult('Worker lost; operation effects uncertain', (), False,
            ExecutionStatus.FAILED, f'{type(error).__name__}: {error}', call_id=call.call_id,
            data={'effects_uncertain': True, 'retryable': spec.idempotent})

    def _write_receipt(self, row):
        if self._receipts is None:
            self._memory_receipts[row['call_id']] = deepcopy(row)
            return
        from rsicomposer.runtime.storage.codec import encode
        atomic_write(self._receipt_path(row['call_id']), json_bytes(encode(row)))

    def _store(self, call, outcome, elapsed, *, finishing=False, job=None, progress=()):
        if not isinstance(outcome, (OperationResult, JobRef)):
            return
        alias = call.name.rsplit('.', 1)[0]
        lab, _ = self._operations[call.name]
        binding = self._bindings[alias]
        capture = optional_method(lab, 'state')
        sequence = self._receipt_sequences.get(alias, 0) + 1
        row = {'schema': _RECEIPT_SCHEMA, 'call_id': call.call_id, 'name': call.name,
            'arguments': call.arguments, 'outcome': outcome, 'lab_state': capture() if capture is not None else None,
            'profile_state': binding.state(), 'sequence': sequence, 'elapsed_seconds': elapsed,
            'finishing': finishing, 'job': outcome if isinstance(outcome, JobRef) else job, 'progress': progress}
        self._write_receipt(self._validate_receipt(row, call.call_id))
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
        operation = self._operations.get(call.name)
        if operation is None:
            return False
        if optional_method(operation[0], 'poll_job') is not None:
            return True
        # A lost worker may have produced a terminal receipt on reattachment,
        # even for a provider without a poll port. Only the journal is needed.
        row = self._saved(call)
        return row is not None and row['job'] == job and isinstance(row['outcome'], OperationResult)

    def can_cancel_job(self, call, job):
        operation = self._operations.get(call.name)
        return operation is not None and optional_method(operation[0], 'cancel_job') is not None

    def _job_receipt(self, call, job):
        row = self._saved(call)
        if not isinstance(job, JobRef) or row is None or row['job'] != job:
            raise ValueError('Job does not match the journaled Lab submission')
        return row

    def poll_job(self, call, job, *, decision_id, after):
        if type(after) is not int or after < 0:
            raise ValueError('Job progress cursor must be a nonnegative integer')
        with self._invocation(call) as available:
            if not available:
                return None, ()
            row = self._job_receipt(call, job)
            alias = call.name.rpartition('.')[0]
            missing = tuple(event for event in row['progress'] if event.sequence > after)
            terminal = row['outcome'] if isinstance(row['outcome'], OperationResult) else None
            if ((terminal is not None or missing) and call.call_id in self._applied_receipts
                    and row['sequence'] <= self._applied_sequences.get(alias, 0)):
                return terminal, missing  # Normal delivery retry: no provider methods, even state().
            with self._state_boundary(alias, wait=False) as available:
                if not available:
                    return None, ()
                with self._lock:
                    self._apply_receipt(row)
                if terminal is not None or missing:
                    return terminal, missing
                with self._job_scope(call, decision_id) as (lab, local, binding):
                    try:
                        result, events = self._job_observation(lab, call, job, after)
                    except WorkerLostError as error:
                        result, events = self._worker_lost(call, error), ()
                    if result is not None or events:
                        # Journal observations before delivery. On a repeated poll
                        # the Host's cursor, not the worker's lifetime, decides
                        # which events still need to be observed.
                        progress = (*row['progress'], *events)
                        with self._lock:
                            if result is None:
                                self._store(call, job, row['elapsed_seconds'], progress=progress)
                            else:
                                result = self._commit_outcome(call, local, binding, _bind_receipt(local, result),
                                    row['elapsed_seconds'], job=job, progress=progress)
                    return result, events

    @staticmethod
    def _job_observation(lab, call, job, after):
        poll = optional_method(lab, 'poll_job')
        if poll is None:
            return None, ()  # External explicit resume remains available.
        outcome = poll(job)
        if not isinstance(outcome, JobOutcome) or outcome.job != job:
            raise TypeError('Lab poll_job must return the matching JobOutcome')
        outcome.__post_init__()
        result = outcome.result
        if result is not None:
            result = _bind_receipt(call, result)
        events = optional_method(lab, 'job_events')
        progress = events(job, after=after) if events is not None else ()
        LabGroup._validate_progress(progress, call.call_id, job, after=after)
        if not outcome.terminal:
            return None, progress
        if result is None:
            result = OperationResult('Lab job ' + outcome.status.value, (), outcome.simulated,
                ExecutionStatus.FAILED if outcome.status == JobStatus.FAILED else ExecutionStatus.CANCELLED,
                outcome.error if outcome.status == JobStatus.FAILED else None)
        return result, progress

    @staticmethod
    def _validate_progress(progress, call_id, job, *, after):
        if (not isinstance(progress, tuple) or any(not isinstance(row, OperationProgress)
                or row.job != job or row.call_id != call_id for row in progress)
                or [row.sequence for row in progress] != sorted({row.sequence for row in progress})
                or any(row.sequence <= after for row in progress)):
            raise ValueError('Lab job events must be increasing, call-correlated OperationProgress observations')

    def control_job(self, call, job, control, *, decision_id):
        with self._state_boundary(call.name.rsplit('.', 1)[0]), self._job_scope(call, decision_id) as (lab, _, _):
            if isinstance(self._job_receipt(call, job)['outcome'], OperationResult):
                return
            apply = optional_method(lab, 'control_job')
            if apply is None or apply(job, control) is not None:
                raise TypeError('Checkpoint jobs require idempotent control_job(job, control) returning None')

    def cancel_job(self, call, job, *, decision_id):
        """Request cancellation; terminal delivery still uses the normal poll journal."""
        with self._state_boundary(call.name.rsplit('.', 1)[0]), self._job_scope(call, decision_id) as (lab, _, _):
            if isinstance(self._job_receipt(call, job)['outcome'], OperationResult):
                return
            cancel = optional_method(lab, 'cancel_job')
            if cancel is None:
                raise TypeError('Lab does not implement cancel_job(job)')
            outcome = cancel(job)
            if not isinstance(outcome, JobOutcome) or outcome.job != job:
                raise TypeError('Lab cancellation must return the matching JobOutcome')
            outcome.__post_init__()
            if outcome.result is not None:
                _bind_receipt(call, outcome.result)

    def reattach_job(self, call, job):
        with self._invocation(call) as available:
            if not available:
                raise ValueError('Operation call_id already has an active execution')
            with self._state_boundary(call.name.rpartition('.')[0]):
                return self._reattach_job(call, job)

    def _reattach_job(self, call, job):
        row = self._job_receipt(call, job)
        with self._lock:
            self._apply_receipt(row)
        if isinstance(row['outcome'], OperationResult):
            return row['outcome']
        lab, name = self._operations[call.name]
        reattach = optional_method(lab, 'reattach_job')
        if reattach is None:
            raise TypeError('Journaled jobs require reattach_job(job) returning None')
        try:
            value = reattach(job)
        except WorkerLostError as error:
            binding = self._bindings[call.name.rsplit('.', 1)[0]]
            with self._lock:
                return self._commit_outcome(call, replace(call, name=name), binding, self._worker_lost(call, error),
                    row['elapsed_seconds'], job=job, progress=row['progress'])
        if value is not None:
            raise TypeError('Journaled jobs require reattach_job(job) returning None')
        return None

    def profile_state(self):
        with self._lock:
            return {'bindings': {alias: binding.state() for alias, binding in self._bindings.items()},
                    'final_calls': deepcopy(self._final_calls), 'final_index': self._final_index}

    def restore_profile_state(self, state):
        if (not isinstance(state, dict) or set(state) != {'bindings', 'final_calls', 'final_index'}
                or not isinstance(state['bindings'], dict) or set(state['bindings']) != set(self._bindings)):
            raise ValueError('Invalid Lab profile state')
        calls, index = state['final_calls'], state['final_index']
        if (calls is not None and (not isinstance(calls, tuple)
                or any(not isinstance(call, OperationCall) for call in calls))):
            raise ValueError('Invalid host final evaluation plan')
        if type(index) is not int or not 0 <= index <= len(calls or ()):
            raise ValueError('Invalid final evaluation cursor')
        with self._lock:
            before = {alias: binding.state() for alias, binding in self._bindings.items()}
            try:
                for alias, binding in self._bindings.items():
                    binding.restore(state['bindings'][alias])
                if calls is not None:
                    self._check_final_plan(calls, restored_index=index)
            except Exception:
                for alias, binding in self._bindings.items():
                    binding.restore(before[alias])
                raise
            self._final_calls, self._final_index = deepcopy(calls), index

    def _check_final_plan(self, calls, *, restored_index=None):
        ids = {call.call_id for call in calls}
        if len(ids) != len(calls):
            raise ValueError('Host final evaluation plan contains duplicate call IDs')
        if ids & self._invocations:
            raise ValueError('A final evaluation call_id already has an active execution')
        plans = {alias: [] for alias in self._bindings}
        for position, call in enumerate(calls):
            if call.name not in self._operations:
                raise ValueError(f'Unknown final evaluation: {call.name}')
            alias, _, name = call.name.rpartition('.')
            plans[alias].append(replace(call, name=name))
            row = self._saved(call)  # Also rejects earlier calls with different arguments or aliases.
            if row is not None and (restored_index is None or not row['finishing']):
                raise ValueError('Final evaluation call_id was already recorded: ' + call.call_id)
            if row is None and (call.call_id in self._applied_receipts or position < (restored_index or 0)):
                raise ValueError('Final evaluation call_id has a missing applied receipt: ' + call.call_id)
        for alias, plan in plans.items():
            binding = self._bindings[alias]
            binding.check_finish(plan)
            for local in plan:
                binding.check(local, finishing=True, history=restored_index is None)

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
                calls.extend(replace(call, name=f'{alias}.{call.name}') for call in plan)
            with self._lock:
                self._check_final_plan(calls)
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

    def _receipt_path(self, call_id):
        digest = sha256(call_id.encode('utf-8')).hexdigest()
        # Historical filenames ended in .json; keep new writes in a disjoint namespace.
        return self._receipts / f'sha256-{digest}.receipt'

    def _legacy_receipt_path(self, call_id):
        name = call_id + '.json'
        # Only probe a bounded, literal child filename, never a legacy path expression.
        if any(ch in name for ch in '/\\:\x00') or len(os.fsencode(name)) > 255:
            return None
        path = self._receipts / name
        try:
            path.lstat()
        except FileNotFoundError:
            return None
        except OSError as error:
            if error.errno == ENAMETOOLONG:
                return None
            raise
        # Old JobRefs carry no call_id. Require exact spelling even on filesystems
        # where lookup can match another case or Unicode normalization.
        if not any(entry.name == name for entry in self._receipts.iterdir()):
            raise ValueError('Legacy Lab receipt filename does not match the call_id exactly')
        return path

    def _saved(self, call):
        saved = self._saved_by_id(call.call_id)
        # Python treats True == 1 == 1.0 even inside composite targets. A stored
        # receipt belongs to the exact admitted JSON call, not Python equality.
        if saved is not None and (saved['name'] != call.name or
                json.dumps(saved['arguments'], sort_keys=True) != json.dumps(call.arguments, sort_keys=True)):
            raise ValueError(f'call_id {call.call_id} was already used for a different operation')
        return saved

    def _saved_by_id(self, call_id):
        if self._receipts is None:
            return deepcopy(self._memory_receipts.get(call_id))
        legacy = False
        try:
            raw = _read_receipt(self._receipt_path(call_id))
        except FileNotFoundError:
            path = self._legacy_receipt_path(call_id)
            if path is None:
                return None
            raw = _read_receipt(path)
            legacy = True
        from rsicomposer.runtime.storage.codec import decode
        return self._validate_receipt(decode(json.loads(raw)), call_id, legacy=legacy)

    def _validate_receipt(self, saved, call_id, *, legacy=False):
        schema = _LEGACY_RECEIPT_SCHEMA if legacy else _RECEIPT_SCHEMA
        if not isinstance(saved, dict) or saved.get('schema') != schema:
            raise ValueError('Lab receipt uses an unsupported RSIComposer protocol; start a new run')
        if set(saved) != (_RECEIPT_FIELDS if legacy else _RECEIPT_FIELDS | {'call_id', 'job', 'progress', 'finishing'}):
            raise ValueError('Invalid Lab receipt fields')
        if not legacy and saved['call_id'] != call_id:
            raise ValueError('Lab receipt call_id does not match')
        outcome = saved['outcome']
        if legacy:
            saved = {**saved, 'schema': _RECEIPT_SCHEMA, 'call_id': call_id, 'finishing': False,
                     'job': outcome if isinstance(outcome, JobRef) else None, 'progress': ()}
        if (not isinstance(saved['name'], str) or saved['name'] not in self._operations
                or not isinstance(saved['arguments'], dict) or type(saved['finishing']) is not bool):
            raise ValueError('Invalid Lab receipt request')
        if isinstance(outcome, OperationResult):
            if outcome.call_id != call_id:
                raise ValueError('Terminal Lab receipt call_id does not match')
        elif not isinstance(outcome, JobRef) or saved['job'] != outcome:
            raise ValueError('Invalid Lab receipt outcome')
        job = saved['job']
        if job is not None and (not isinstance(job, JobRef) or not isinstance(job.job_id, str)
                or not job.job_id.strip() or not isinstance(job.backend, str) or not job.backend.strip()
                or saved['finishing']):
            raise ValueError('Invalid Lab receipt job')
        if saved['progress'] and job is None:
            raise ValueError('Progress requires a Lab job receipt')
        self._validate_progress(saved['progress'], call_id, job, after=0)
        if type(saved['sequence']) is not int or saved['sequence'] < 1:
            raise ValueError('Invalid Lab receipt sequence')
        return saved

    def _recovery_receipt_evidence(self, call):
        """Read-only description of a stored receipt, not authorization to replay it."""
        if self._receipts is None:
            return {'receipt': 'unavailable'}
        try:
            saved = self._saved(call)
        except (OSError, ValueError, KeyError, TypeError) as error:
            return {'receipt': 'unreadable_or_invalid', 'receipt_error_type': type(error).__name__}
        if saved is None:
            return {'receipt': 'missing'}
        outcome = saved['outcome']
        return {'receipt': 'terminal' if isinstance(outcome, OperationResult) else 'job',
                'receipt_sequence': saved['sequence']}

    def receipt_sequence(self, call):
        saved = self._saved(call)
        return saved['sequence'] if saved is not None else None

    def receipt_job(self, call):
        """The original managed JobRef, retained even after terminal settlement."""
        saved = self._saved(call)
        return saved['job'] if saved is not None else None

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

    def in_flight_resolution(self, call, *, job=None):
        """Find a replay/reattach proof, without changing any Lab state."""
        row = self._saved(call)
        if job is not None and (row is None or row['job'] != job):
            return None
        saved = row['outcome'] if row is not None else None
        if isinstance(saved, JobRef):
            lab, _ = self._operations[call.name]
            return 'job' if optional_method(lab, 'reattach_job') is not None else None
        if saved is not None:
            return 'receipt'
        if job is not None:
            return None  # A known submission is not safe to launch again.
        spec = next((s for s in self._specs if s.name == call.name), None)
        return 'idempotent' if spec is not None and spec.idempotent else None

    def _portable_receipt(self, row, sequences):
        if row is None or row['name'] not in self._operations:
            raise ValueError('Portable Lab journal is missing an applied operation')
        if not isinstance(row['outcome'], OperationResult):
            raise ValueError('Portable Lab journal requires terminal receipts')
        alias = row['name'].rpartition('.')[0]
        if row['sequence'] > sequences.get(alias, 0):
            raise ValueError('Portable Lab receipt is newer than the captured state')
        return row

    def export_receipt_files(self, directory, inventory):
        """Stream exactly the applied journal, not a plugin's workspace or filename layout."""
        from rsicomposer.runtime.storage.codec import encode
        with self._lock, (directory / 'receipts.jsonl').open('xb') as stream:
            if inventory != self.receipt_state():
                raise RuntimeError('Lab receipt inventory changed during snapshot capture')
            for call_id in inventory['applied']:
                row = self._portable_receipt(self._saved_by_id(call_id), self._applied_sequences)
                stream.write(json_bytes(encode(row)) + b'\n')

    def restore_receipt_files(self, directory, inventory):
        """Reconstruct retry and replay evidence without executing or restoring a Lab."""
        from rsicomposer.runtime.storage.codec import decode
        if not isinstance(inventory, dict) or inventory.keys() != {'applied', 'sequences'}:
            raise ValueError('Invalid portable Lab receipt inventory')
        applied, sequences = inventory['applied'], inventory['sequences']
        if (not isinstance(applied, list) or any(not isinstance(i, str) or not i.strip() for i in applied)
                or applied != sorted(set(applied)) or not isinstance(sequences, dict)
                or any(alias not in self._labs or type(n) is not int or n < 1 for alias, n in sequences.items())):
            raise ValueError('Invalid portable Lab receipt inventory')

        def rows(stream):
            for call_id in applied:
                line = stream.readline()
                if not line:
                    raise ValueError('Portable Lab journal is missing an applied receipt')
                row = self._validate_receipt(decode(json.loads(line)), call_id)
                yield call_id, self._portable_receipt(row, sequences)
            if stream.read(1):
                raise ValueError('Portable Lab journal contains an unlisted receipt')

        # A component may use an external journal. Validate all rows and all
        # pre-existing receipts before installing even the first missing row.
        with self._lock, (directory / 'receipts.jsonl').open('rb') as stream:
            missing, latest = set(), {}
            for call_id, row in rows(stream):
                alias = row['name'].rpartition('.')[0]
                latest[alias] = max(row['sequence'], latest.get(alias, 0))
                prior = self._saved_by_id(call_id)
                if prior is None:
                    missing.add(call_id)
                elif prior != row:
                    raise ValueError('Cannot overwrite a conflicting Lab receipt on restore')
            if latest != sequences:
                raise ValueError('Portable Lab receipt sequences do not match the captured inventory')
            if missing:
                stream.seek(0)
                for call_id, row in rows(stream):
                    if call_id in missing:
                        self._write_receipt(row)

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
