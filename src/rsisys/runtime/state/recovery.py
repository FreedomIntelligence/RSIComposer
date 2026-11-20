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

from rsisys.runtime.orchestration.run_policy import (
    GoalRunPolicy,
    MetricGoal,
    ResearcherRunPolicy,
    TimeRunPolicy,
)
from rsisys.runtime.storage.files import atomic_write, json_bytes


_CHECKPOINT_SCHEMA = 33


class RecoveryRequired(RuntimeError):
    """An interrupted boundary needs evidence-backed reconciliation before execution."""


from rsisys.runtime.state.contracts import select_contract


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
               '_execution_shutdown_error', '_finishing_complete')
_RUN_TIMERS = ('_run_started', '_restart_at', '_stop_started')


def capture_runner(runner):
    now = runner._clock()
    return {**{key: deepcopy(getattr(runner, key)) for key in _RUN_FIELDS},
            'timers': {key: None if getattr(runner, key) is None else getattr(runner, key) - now
                       for key in _RUN_TIMERS},
            'cleanup_timeout_seconds': runner.cleanup_timeout_seconds}


def restore_runner(runner, state, downtime):
    state = dict(state)
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
    or replay an interrupted operation. All mutations use the owning Runner thread.
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
        from rsisys.runtime.state.provenance import code_revision
        from rsisys.runtime.storage.codec import encode
        if in_flight is None:
            execution = self.app.runner.decision_making
            read = getattr(execution, 'in_flight_operations', None)
            in_flight = read() if callable(read) else []
        payload = {'schema': _CHECKPOINT_SCHEMA, 'generation': generation, 'contract_id': self.contract.contract_id,
                   'source': code_revision(), 'python': list(sys.version_info[:2]),
                   'run_id': self.app.config.run_id, 'objective': self.app.config.objective,
                   'config': _config_identity(self.app.config), 'saved_at': self.wall_clock(),
                   'policy': policy_state(self.app.runner.run_policy, external=True),
                   'runner': capture_runner(self.app.runner), 'components': self.contract.capture(self.app),
                   'intervention': intervention, 'in_flight': in_flight}
        encoded = encode(payload)
        atomic_write(self.root / 'checkpoint.json', json_bytes(
            {'payload': encoded, 'sha256': sha256(json_bytes(encoded)).hexdigest()}))
        self.generation = generation

    def initialize(self):
        self._checkpoint(0)
        atomic_write(self.root / 'journal.json', json_bytes({'generation': 0, 'status': 'committed'}))

    @staticmethod
    def read(root):
        from rsisys.runtime.storage.codec import decode
        document = json.loads((Path(root) / 'recovery' / 'checkpoint.json').read_text())
        if sha256(json_bytes(document['payload'])).hexdigest() != document['sha256']:
            raise ValueError('Recovery checkpoint digest mismatch')
        state = decode(document['payload'])
        if state['schema'] != _CHECKPOINT_SCHEMA:
            raise ValueError('Recovery checkpoint uses an earlier RSISys protocol; start a new run')
        return state

    def load(self, supplied_policy=None):
        from rsisys.runtime.state.provenance import code_revision
        state = self.read(self.app.config.data_dir)
        if state['source'] != code_revision() or state['python'] != list(sys.version_info[:2]):
            raise ValueError('Recovery requires the same framework source and Python major/minor')
        if state['contract_id'] != self.contract.contract_id:
            raise TypeError('Supply the original recovery contract')
        if state['config'] != _config_identity(self.app.config):
            raise ValueError('Recovery configuration/run identity mismatch')
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
            self.intervention = {**journal, 'reason': 'interrupted_mutation'}
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

    def _resume_in_flight(self, in_flight):
        """An operation was running when the process stopped. Continue only when that is safe.

        Safe means the Lab journal holds the call's receipt (the operation finished),
        or the Lab declared the operation idempotent for a repeated call_id. Otherwise
        the run stays blocked until reconcile(), exactly as for any interrupted step.
        """
        from rsisys.runtime.capabilities import optional_method
        if not isinstance(in_flight, list):
            raise ValueError('In-flight operations must be a list')
        resolve = optional_method(self.app.assembly.lab_execution, 'in_flight_resolution')
        reopen = optional_method(self.app.assembly.host_execution, 'reopen_in_flight')
        resolutions, unsafe = [], []
        for item in in_flight:
            call = item['operation']
            host_finish = item.get('owner') == 'host_finish'
            resolution = ('queued' if item['status'] == 'queued' else
                          resolve(call) if resolve is not None and (host_finish or reopen is not None) else None)
            resolutions.append(resolution)
            if resolution is None:
                unsafe.append({'decision_id': item['decision_id'], 'operation': call.name, 'call_id': call.call_id})
        # Resolve the complete set before changing a single task or executing a receipt.
        if unsafe:
            self.intervention = {'reason': 'interrupted_operation', 'operations': unsafe}
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
        from rsisys.runtime.storage.codec import encode
        with self._lock:
            if self.app._closed:
                raise RuntimeError('Application is closed')
            if self._active:
                return callback()  # e.g. a step's internal cancel belongs to the same boundary
            if self.intervention is not None:
                raise RecoveryRequired('Interrupted mutation requires reconciliation: ' + str(self.intervention))
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
