"""Single-writer checkpoint and call-correlated durable mutation boundaries."""

import json
import sys
from copy import deepcopy
from dataclasses import asdict
from functools import wraps
from hashlib import sha256
from pathlib import Path
from threading import RLock
from time import time

from rsicomposer.runtime.orchestration.run_policy import (
    GoalRunPolicy,
    MetricGoal,
    ResearcherRunPolicy,
    TimeRunPolicy,
)
from rsicomposer.runtime.storage.files import atomic_write, json_bytes


_CHECKPOINT_SCHEMA = 34


class RecoveryRequired(RuntimeError):
    """An interrupted boundary needs evidence-backed reconciliation before execution."""


from rsicomposer.runtime.state.contracts import select_contract


def _intervention_summary(incident):
    """Public diagnostics only; the journal and repair callback retain the full incident."""
    if not isinstance(incident, dict):
        return {'reason': 'unconfirmed_mutation'}
    reason = incident.get('reason')
    if incident.get('operation') == 'reconcile':
        reason = 'interrupted_reconciliation'
    elif reason not in ('interrupted_mutation', 'interrupted_operation'):
        reason = 'unconfirmed_mutation'
    summary = {'reason': reason}
    for key in ('generation', 'checkpoint_generation'):
        value = incident.get(key)
        if type(value) is int and value >= 0:
            summary[key] = value
    rows = incident.get('checkpoint_in_flight_evidence')
    if isinstance(rows, list):
        summary['calls'] = []
        for row in rows:
            if not isinstance(row, dict):
                continue
            call = {}
            call_id = row.get('call_id')
            if type(call_id) is str:
                # IDs can be supplied by callers and need not be UUIDs. Keep a
                # stable correlation reference without publishing their text.
                call['call_id_sha256'] = sha256(call_id.encode('utf-8', errors='surrogatepass')).hexdigest()
            status = row.get('checkpoint_status')
            if status in ('queued', 'running', 'waiting'):
                call['checkpoint_status'] = status
            receipt = row.get('receipt')
            if receipt in ('terminal', 'job', 'missing', 'unavailable', 'unreadable_or_invalid'):
                call['receipt'] = receipt
            sequence = row.get('receipt_sequence')
            if type(sequence) is int and sequence > 0:
                call['receipt_sequence'] = sequence
            if call:
                summary['calls'].append(call)
    return summary


def durable_mutation(method):
    @wraps(method)
    def wrapped(self, *args, **kwargs):
        recovery = getattr(self, 'recovery', None)
        if recovery is None:
            return method(self, *args, **kwargs)
        return recovery.mutate("step" if method.__name__ == "_step" else method.__name__, (args, kwargs), lambda: method(self, *args, **kwargs))
    return wrapped


# Only data fields: callbacks, endpoints and clocks are rebound by the assembly.
_RUN_FIELDS = ('_research_round', '_researcher_finished', '_restart_context', '_stop_decision',
               '_terminal_step', '_outcomes', '_quiescing', '_decision_finished',
               '_execution_shutdown_error', '_execution_shutdown_complete', '_finishing_complete')
_RUN_TIMERS = ('_run_started', '_restart_at', '_stop_started')


def capture_runner(runner):
    now = runner._clock()
    return {**{key: deepcopy(getattr(runner, key)) for key in _RUN_FIELDS},
            'timers': {key: None if getattr(runner, key) is None else getattr(runner, key) - now
                       for key in _RUN_TIMERS},
            'cleanup_timeout_seconds': runner.cleanup_timeout_seconds}


def restore_runner(runner, state, downtime):
    state = dict(state)
    # Older captures did not record whether execution shutdown had settled.
    state.setdefault('_execution_shutdown_complete', False)
    if type(state['_execution_shutdown_complete']) is not bool:
        raise ValueError('Invalid execution shutdown state')
    if set(state) != {*_RUN_FIELDS, 'timers', 'cleanup_timeout_seconds'} or set(state['timers']) != set(_RUN_TIMERS):
        raise ValueError('Unsupported runner recovery fields')
    for key in _RUN_FIELDS:
        setattr(runner, key, deepcopy(state[key]))
    now = runner._clock()
    for key, offset in state['timers'].items():
        setattr(runner, key, None if offset is None else now + offset - downtime)
    runner.cleanup_timeout_seconds = state['cleanup_timeout_seconds']


def policy_state(policy, *, external=False):
    if type(policy) in (ResearcherRunPolicy, TimeRunPolicy):
        return {'type': type(policy).__name__, 'fields': asdict(policy)}
    if type(policy) is GoalRunPolicy and type(policy.goal) is MetricGoal:
        return {'type': 'GoalRunPolicy', 'fields': asdict(policy)}
    if external:
        return {'type': 'external', 'identity': f'{type(policy).__module__}.{type(policy).__qualname__}'}
    raise TypeError('Custom RunPolicy/goal requires an explicit recovery_contract')


def restore_policy(state, supplied=None):
    kind = state['type']
    fields = dict(state.get('fields', {}))
    if kind == 'external':
        if supplied is None or policy_state(supplied, external=True) != state:
            raise TypeError('Supply the original custom RunPolicy and recovery_contract')
        return supplied
    if supplied is not None and policy_state(supplied) != state:
        raise ValueError('Cannot replace a recovered run policy/budget')
    if kind == 'GoalRunPolicy':
        fields['goal'] = MetricGoal(**fields['goal'])
    constructors = {'ResearcherRunPolicy': ResearcherRunPolicy, 'TimeRunPolicy': TimeRunPolicy,
                    'GoalRunPolicy': GoalRunPolicy}
    if kind not in constructors:
        raise ValueError('Unknown recovered run policy')
    return constructors[kind](**fields)


def _config_identity(config):
    value = json.loads(json_bytes(config.to_dict()))
    value['data_dir'] = str(config.data_dir.resolve())
    return value


class DurableRecovery:
    """Two atomically replaced files; checkpoint generation is the commit record.

    A journal ahead of the checkpoint is uncertain. Never roll back SQLite/artifacts
    or re-execute an operation whose effects are uncertain. All mutations use
    the owning Runner thread.
    """
    def __init__(self, app, contract=None, *, wall_clock=time):
        self.app = app
        self.contract = select_contract(contract, has_state_adapters=bool(app.assembly.state_adapters))
        self.root = app.config.data_dir / 'recovery'
        self.wall_clock = wall_clock
        self.generation = 0
        self.intervention = None
        self.downtime = 0.0
        self._active = False
        self._lock = RLock()

    def _checkpoint(self, generation, intervention=None, in_flight=None):
        from rsicomposer.runtime.state.provenance import code_revision
        from rsicomposer.runtime.storage.codec import encode
        if in_flight is None:
            execution = self.app.runner.decision_making
            read = getattr(execution, 'in_flight_operations', None)
            in_flight = read() if callable(read) else []
        payload = {'schema': _CHECKPOINT_SCHEMA, 'generation': generation, 'contract_id': self.contract.contract_id,
                   'source': code_revision(), 'python': list(sys.version_info[:2]),
                   'run_id': self.app.config.run_id, 'objective': self.app.config.objective,
                   'config': _config_identity(self.app.config), 'saved_at': self.wall_clock(),
                   'run_bootstrap_sha256': sha256(self.app._run_record).hexdigest(),
                   'policy': policy_state(self.app.runner.run_policy, external=True),
                   'action_permissions': self.app._action_permissions.to_dict(),
                   'runner': capture_runner(self.app.runner), 'components': self.contract.capture(self.app),
                   'intervention': intervention, 'in_flight': in_flight}
        encoded = encode(payload)
        atomic_write(self.root / 'checkpoint.json', json_bytes(
            {'payload': encoded, 'sha256': sha256(json_bytes(encoded)).hexdigest()}))
        self.generation = generation

    def initialize(self):
        # There are no effects to reconcile yet. The first checkpoint is the
        # commit record, so its required journal must already be durable.
        atomic_write(self.root / 'journal.json', json_bytes({'generation': 0, 'status': 'committed'}))
        self._checkpoint(0)

    @staticmethod
    def read(root):
        from rsicomposer.runtime.storage.codec import decode
        document = json.loads((Path(root) / 'recovery' / 'checkpoint.json').read_text())
        if sha256(json_bytes(document['payload'])).hexdigest() != document['sha256']:
            raise ValueError('Recovery checkpoint digest mismatch')
        state = decode(document['payload'])
        if state['schema'] != _CHECKPOINT_SCHEMA:
            raise ValueError('Unsupported recovery checkpoint schema; resume with the original framework revision '
                             'or start a new run')
        return state

    @staticmethod
    def validate_workspace(state, config):
        from rsicomposer.runtime.state.provenance import code_revision
        if state['source'] != code_revision() or state['python'] != list(sys.version_info[:2]):
            raise ValueError('Recovery requires the same framework source and Python major/minor')
        if state['config'] != _config_identity(config):
            raise ValueError('Recovery configuration/run identity mismatch')

    def load(self, state, supplied_policy=None):
        self.validate_workspace(state, self.app.config)
        if state['contract_id'] != self.contract.contract_id:
            raise TypeError('Supply the original recovery contract')
        journal = json.loads((self.root / 'journal.json').read_text())
        self.generation = state['generation']
        if journal['generation'] not in (self.generation, self.generation + 1):
            raise ValueError('Recovery journal/checkpoint generation mismatch')
        if journal.get('status') not in ('started', 'committed'):
            raise ValueError('Invalid recovery journal status')
        self.intervention = state['intervention']
        in_flight = state['in_flight']
        if journal['generation'] > self.generation:
            if journal['status'] != 'started':
                raise ValueError('Missing committed recovery checkpoint')
            # The new mutation has no committed prefix. Even a terminal receipt
            # matching a prior call cannot prove that nothing else happened (e.g.
            # Controller adoption, progress observations, or feedback delivery).
            self.intervention = {**journal, 'reason': 'interrupted_mutation',
                                 'checkpoint_generation': self.generation,
                                 'checkpoint_in_flight_evidence': self._in_flight_evidence(in_flight)}
        self.downtime = max(0.0, self.wall_clock() - state['saved_at'])
        self.app.runner.run_policy = restore_policy(state['policy'], supplied_policy)
        self.app.runner._guard_actions = True
        restore_runner(self.app.runner, state['runner'], self.downtime)
        # The very first step may die before any checkpoint records its start.
        # Its write-ahead marker still fixes the budget's start across restart.
        incident = self.intervention
        while self.app.runner._run_started is None and incident:
            if incident.get('operation') == 'step' and 'started_at' in incident:
                elapsed = max(0.0, self.wall_clock() - incident['started_at'])
                self.app.runner._run_started = self.app.runner._clock() - elapsed
                break
            incident = incident.get('previous')
        if self.contract.restore(self.app, state['components']) is not None:
            raise TypeError('Recovery restore must return None')
        if in_flight and self.intervention is None:
            self._resume_in_flight(in_flight)

    def _in_flight_evidence(self, in_flight):
        from rsicomposer.runtime.orchestration.lab_group import LabGroup
        lab = self.app.assembly.lab_execution
        evidence = []
        for item in in_flight:
            call = item['operation']
            # Inspect the built-in receipt files only; do not invoke custom Lab
            # hooks while the effect boundary is uncertain.
            receipt = lab._recovery_receipt_evidence(call) if type(lab) is LabGroup else {'receipt': 'unavailable'}
            evidence.append({**receipt, 'decision_id': item['decision_id'], 'operation': call.name,
                             'call_id': call.call_id, 'checkpoint_status': item['status']})
        return evidence

    def _resume_in_flight(self, in_flight):
        """An operation was running when the process stopped. Continue only when that is safe.

        Safe means the Lab journal holds the call's receipt (the operation finished),
        or the Lab declared the operation idempotent for a repeated call_id. Otherwise
        the run stays blocked until reconcile(), exactly as for any interrupted step.
        """
        from rsicomposer.loops.contracts.jobs import JobRef
        from rsicomposer.runtime.capabilities import optional_method
        from rsicomposer.runtime.orchestration.lab_group import LabGroup
        if not isinstance(in_flight, list):
            raise ValueError('In-flight operations must be a list')
        lab, host = self.app.assembly.lab_execution, self.app.assembly.host_execution
        resolve = optional_method(lab, 'in_flight_resolution')
        reopen = optional_method(host, 'reopen_in_flight')
        read = optional_method(host, 'in_flight_operations')
        restored = {row['operation'].call_id: row for row in read()} if read is not None else None
        resolutions, unsafe = [], []
        for item in in_flight:
            call = item['operation']
            host_finish = item.get('owner') == 'host_finish'
            current = restored.get(call.call_id) if restored is not None and not host_finish else None
            job = item.get('job')
            resolution = None
            if restored is not None and not host_finish and (current is None
                    or any(item[key] != current[key] for key in ('decision_id', 'operation', 'status'))
                    or ('job' in item and job != current.get('job'))):
                pass  # The saved intent has no matching execution owner.
            elif item['status'] == 'queued':
                resolution = 'queued'
            elif resolve is not None and (host_finish or reopen is not None):
                if item['status'] == 'waiting' and type(lab) is LabGroup:
                    # Older checkpoints omitted job from the in-flight summary;
                    # the restored Host queue retains that same full JobRef.
                    job = current.get('job') if current is not None else job
                    if isinstance(job, JobRef):
                        resolution = resolve(call, job=job)
                else:
                    resolution = resolve(call)
            resolutions.append(resolution)
            if resolution is None:
                unsafe.append({'decision_id': item['decision_id'], 'operation': call.name, 'call_id': call.call_id})
        # Resolve the complete set before changing a single task or executing a receipt.
        if unsafe:
            self.intervention = {'reason': 'interrupted_operation', 'operations': unsafe,
                                 'checkpoint_generation': self.generation,
                                 'checkpoint_in_flight_evidence': self._in_flight_evidence(in_flight)}
            return
        calls = tuple(item['operation'] for item in in_flight if item.get('owner') != 'host_finish')
        if calls:
            apply = optional_method(self.app.assembly.lab_execution, 'restore_journaled_calls')
            if apply is not None:
                apply(calls)
            reopen(calls)
        atomic_write(self.root / f'automatic-resume-{self.generation}.json', json_bytes({
            'operations': [{'decision_id': item['decision_id'], 'operation': item['operation'].name,
                            'call_id': item['operation'].call_id, 'resolution': resolution}
                           for item, resolution in zip(in_flight, resolutions)], 'at': self.wall_clock()}))

    def commit_in_flight(self, in_flight):
        """Checkpoint the current step just before its Lab operation starts."""
        with self._lock:
            if not self._active:
                raise RuntimeError('In-flight checkpoints belong to an active durable step')
            self._checkpoint(self._pending_generation, in_flight=in_flight)

    def mutate(self, operation, arguments, callback):
        from rsicomposer.runtime.storage.codec import encode
        with self._lock:
            if self.app._closed:
                raise RuntimeError('Application is closed')
            if self._active:
                return callback()  # e.g. a step's internal cancel belongs to the same boundary
            if self.intervention is not None:
                summary = json.dumps(_intervention_summary(self.intervention))
                raise RecoveryRequired('Interrupted mutation requires reconciliation: ' + summary)
            # Reject uncontracted component swaps before allowing another external effect.
            json_bytes(encode(self.contract.capture(self.app)))
            policy_state(self.app.runner.run_policy, external=True)
            generation = self.generation + 1
            intent = {'generation': generation, 'status': 'started', 'operation': operation,
                      'arguments': encode(arguments), 'started_at': self.wall_clock()}
            self.intervention = intent  # also fail closed if writing the intent fails
            atomic_write(self.root / 'journal.json', json_bytes(intent))
            self._active = True
            self._pending_generation = generation
            try:
                result = callback()
                self._checkpoint(generation)
                self.intervention = None
                # checkpoint is the commit; no second post-effect write is necessary
                return result
            except BaseException as error:
                self.intervention = {**intent, 'reason': 'interrupted_mutation',
                                     'error_type': type(error).__name__}
                try:
                    # Preserve in-memory outbox/receipts if the process is still alive.
                    self._checkpoint(generation, self.intervention)
                except BaseException as checkpoint_error:
                    error.add_note(f'Interrupted state could not be checkpointed: {checkpoint_error}')
                raise
            finally:
                self._active = False

    def reconcile(self, repair, *, evidence: str):
        """Operator/adapter verifies effects and repairs state; never automatic replay.

        repair(app, intervention) must attach existing jobs/results or mark failure,
        and return None. The evidence should identify the checks/receipts used.
        If reconciliation is interrupted, execution remains blocked.
        """
        with self._lock:
            if self.app._closed or self._active:
                raise RuntimeError('Cannot reconcile a closed or executing application')
            if self.intervention is None:
                raise ValueError('No interrupted boundary to reconcile')
            if not isinstance(evidence, str) or not evidence.strip() or not callable(repair):
                raise ValueError('Reconciliation needs a repair callback and nonempty evidence')
            incident = deepcopy(self.intervention)
            # Preserve an audit record before allowing any repair; never expose clear/retry.
            atomic_write(self.root / f'reconciliation-{self.generation + 1}.json',
                         json_bytes({'incident': incident, 'evidence': evidence, 'at': self.wall_clock()}))
            intent = {'generation': self.generation + 1, 'status': 'started',
                      'operation': 'reconcile', 'previous': incident, 'evidence': evidence}
            self.intervention = intent
            atomic_write(self.root / 'journal.json', json_bytes(intent))
            result = repair(self.app, incident)
            if result is not None:
                raise TypeError('Reconciliation callback must return None')
            self._checkpoint(self.generation + 1)
            self.intervention = None
