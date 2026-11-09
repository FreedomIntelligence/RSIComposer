"""Single-writer checkpoint plus one durable mutation boundary, never effect replay."""

from copy import deepcopy
from dataclasses import asdict
from functools import wraps
from hashlib import sha256
import json
from pathlib import Path
import sys
from threading import RLock
from time import monotonic, time
from typing import Protocol

from rsisys.runtime.storage.files import atomic_write, json_bytes
from rsisys.runtime.orchestration.run_policy import ResearcherRunPolicy, TimeRunPolicy, GoalRunPolicy, MetricGoal


class RecoveryRequired(RuntimeError):
    """An interrupted boundary needs evidence-backed reconciliation before execution."""


class RecoveryContract(Protocol):
    """Explicit assembly state; capture is observational, restore never submits work.

    Include all custom policy/backend/tool state, durable job identities, and receipts.
    Reject state that cannot be reattached safely. Never serialize credentials/callables.
    The caller supplies the same factories and contract again when resuming.
    """
    contract_id: str

    def capture(self, application) -> object: ...
    def restore(self, application, state: object) -> None: ...


def durable_mutation(method):
    @wraps(method)
    def wrapped(self, *args, **kwargs):
        recovery = getattr(self, 'recovery', None)
        if recovery is None:
            return method(self, *args, **kwargs)
        return recovery.mutate(method.__name__, (args, kwargs), lambda: method(self, *args, **kwargs))
    return wrapped


# Only data fields: callbacks, endpoints and clocks are rebound by the assembly.
_RUN_FIELDS = ('_research_round', '_researcher_finished', '_restart_context', '_stop_decision',
               '_terminal_step', '_outcomes', '_quiescing', '_decision_finished', '_directive',
               '_feedback', '_review_directive_id', '_review_result', '_review_pending', '_pending_cancel',
               '_execution_shutdown_error')
_RUN_TIMERS = ('_run_started', '_restart_at', '_stop_started', '_review_at')


def capture_runner(runner):
    if runner.parallel:
        raise TypeError('Parallel runner recovery requires a separate implementation')
    now = runner._clock()
    return {**{key: deepcopy(getattr(runner, key)) for key in _RUN_FIELDS},
            'timers': {key: None if getattr(runner, key) is None else getattr(runner, key) - now
                       for key in _RUN_TIMERS},
            'cleanup_timeout_seconds': runner.cleanup_timeout_seconds}


def restore_runner(runner, state, downtime):
    state = dict(state)
    state.setdefault('_execution_shutdown_error', None)  # additive field in older checkpoints
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


def decision_extension(harness):
    from rsisys.loops.decision_making.harness.engine import DefaultDecisionHarness
    if type(harness) is DefaultDecisionHarness:
        return False
    return (isinstance(harness, DefaultDecisionHarness)
            and isinstance(getattr(harness, 'recovery_id', None), str) and bool(harness.recovery_id)
            and callable(getattr(harness, 'export_recovery_state', None))
            and callable(getattr(harness, 'restore_recovery_state', None)))


def _capture_book(book):
    from rsisys.runtime.orchestration.jobs import JobBook, MockJobQueue
    if type(book) is not JobBook:
        raise TypeError('Custom job books need an explicit recovery contract')
    queue = None
    if book.backend is not None:
        q = book.backend.jobs
        if type(q) is not MockJobQueue:
            raise TypeError('Only built-in mock jobs can be reconstructed locally')
        queue = {'backend': q.backend, 'pending_polls': q.pending_polls, 'keys': q._keys,
                 'jobs': {key: (value.request, value.remaining_polls, value.outcome)
                          for key, value in q._jobs.items()}}
    return {'requests': book._requests, 'keys': book._keys,
            'handles': {job: (handle.outcome, handle.cancel_requested) for job, handle in book._handles.items()},
            'queue': queue}


def _restore_book(book, state):
    from rsisys.runtime.orchestration.jobs import JobHandle, _MockJob
    queue = state['queue']
    if queue is not None:
        q = book.backend.jobs
        # Recreate only the known mock's deferred closure in an isolated scratch queue.
        # submit here cannot execute a factory, launch a process, or contact a service.
        scratch = type(book.backend)(book.backend.artifacts)
        for key, (request, polls, outcome) in queue['jobs'].items():
            temporary = scratch.submit(request, idempotency_key=key)
            factory = scratch.jobs._jobs[temporary.job_id].factory
            q._jobs[key] = _MockJob(request, factory, polls, outcome)
        q.backend, q.pending_polls, q._keys = queue['backend'], queue['pending_polls'], queue['keys']
    book._requests, book._keys = state['requests'], state['keys']
    book._handles = {}
    for job, (outcome, cancelled) in state['handles'].items():
        handle = JobHandle(job, book.backend.poll, book.backend.cancel)
        handle.outcome, handle.cancel_requested = outcome, cancelled
        book._handles[job] = handle


class BuiltinRecovery:
    contract_id = 'rsisys-local/v1'

    def capture(self, app):
        from rsisys.snapshot.local import validate_local_application, research_configuration
        dm, actor, _, permissions = validate_local_application(app, ongoing=True)
        if app.sandbox.active_count or actor.state.active_directive_id is not None:
            raise RecoveryRequired('An active environment/process needs reconciliation; it cannot be restored')
        extension = ({'id': dm.recovery_id, 'state': dm.export_recovery_state()}
                     if decision_extension(dm) else None)
        return {'decision': dm.state, 'actor': actor.state,
                'decision_extension': extension,
                'research_agenda': dm.research_agenda.export(),
                'context_compactor': research_configuration(dm),
                'models': (dm.policy.model, actor.policy.model),
                'limits': (dm.experience_limit, dm.memory_limit, dm.review_tool_limit, actor.memory_limit),
                'permissions': permissions,
                'commands': {'active': actor.commands.active, 'receipts': actor.commands.receipts},
                'command_clock': monotonic(),
                'books': [_capture_book(book) for book in (dm.training, actor.training, actor.evaluation)]}

    def restore(self, app, state):
        from rsisys.snapshot.local import validate_local_application, restore_research
        from rsisys.loops.decision_making.action_space.access import ActionPermissions
        dm, actor, _, _ = validate_local_application(app, ongoing=True)
        extension = state['decision_extension']
        if (extension is None) != (not decision_extension(dm)) or (
                extension is not None and extension['id'] != dm.recovery_id):
            raise TypeError('Decision recovery_id differs; supply the original decision_factory')
        dm.state, actor.state = state['decision'], state['actor']
        restore_research(dm, state['research_agenda'], state['context_compactor'])
        dm.policy.model, actor.policy.model = state['models']
        dm.experience_limit, dm.memory_limit, dm.review_tool_limit, actor.memory_limit = state['limits']
        app.runner.decision_making.action_space.permissions = ActionPermissions.from_dict(state['permissions'])
        actor.commands.active = state['commands']['active']
        actor.commands.receipts = state['commands']['receipts']
        for receipt in actor.commands.receipts.values():
            if receipt.started_at is not None:
                receipt.started_at = monotonic() + receipt.started_at - state['command_clock'] - app.recovery.downtime
        for book, saved in zip((dm.training, actor.training, actor.evaluation), state['books']):
            _restore_book(book, saved)
        if extension is not None:
            dm.restore_recovery_state(extension['state'])


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
        from rsisys.runtime.state.component_state import AssemblyRecovery
        self.contract = (contract if contract is not None else
                         AssemblyRecovery() if app.assembly.state_adapters else BuiltinRecovery())
        if (not isinstance(getattr(self.contract, 'contract_id', None), str) or not self.contract.contract_id
                or not callable(getattr(self.contract, 'capture', None))
                or not callable(getattr(self.contract, 'restore', None))):
            raise TypeError('Recovery contract needs contract_id, capture(app), restore(app, state)')
        self.root = app.config.data_dir / 'recovery'
        self.wall_clock = wall_clock
        self.generation = 0
        self.intervention = None
        self.downtime = 0.0
        self._active = False
        self._lock = RLock()

    def _checkpoint(self, generation, intervention=None):
        from rsisys.snapshot.codec import encode
        from rsisys.snapshot.local import code_revision
        payload = {'schema': 1, 'generation': generation, 'contract_id': self.contract.contract_id,
                   'source': code_revision(), 'python': list(sys.version_info[:2]),
                   'run_id': self.app.config.run_id, 'objective': self.app.config.objective,
                   'config': _config_identity(self.app.config), 'saved_at': self.wall_clock(),
                   'policy': policy_state(self.app.runner.run_policy, external=type(self.contract) is not BuiltinRecovery),
                   'runner': capture_runner(self.app.runner), 'components': self.contract.capture(self.app),
                   'intervention': intervention}
        encoded = encode(payload)
        atomic_write(self.root / 'checkpoint.json', json_bytes(
            {'payload': encoded, 'sha256': sha256(json_bytes(encoded)).hexdigest()}))
        self.generation = generation

    def initialize(self):
        self._checkpoint(0)
        atomic_write(self.root / 'journal.json', json_bytes({'generation': 0, 'status': 'committed'}))

    @staticmethod
    def read(root):
        from rsisys.snapshot.codec import decode
        document = json.loads((Path(root) / 'recovery' / 'checkpoint.json').read_text())
        if sha256(json_bytes(document['payload'])).hexdigest() != document['sha256']:
            raise ValueError('Recovery checkpoint digest mismatch')
        state = decode(document['payload'])
        if state['schema'] != 1:
            raise ValueError('Unsupported recovery schema')
        return state

    def load(self, supplied_policy=None):
        from rsisys.snapshot.local import code_revision
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
        if journal['generation'] > self.generation:
            if journal['status'] != 'started':
                raise ValueError('Missing committed recovery checkpoint')
            self.intervention = {**journal, 'reason': 'interrupted_mutation'}
        self.downtime = max(0.0, self.wall_clock() - state['saved_at'])
        self.app.runner.run_policy = restore_policy(state['policy'], supplied_policy)
        self.app.runner._guard_actions = type(self.app.runner.run_policy) is not ResearcherRunPolicy
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
        self.contract.restore(self.app, state['components'])

    def mutate(self, operation, arguments, callback):
        from rsisys.snapshot.codec import encode
        with self._lock:
            if self.app._closed:
                raise RuntimeError('Application is closed')
            if self._active:
                return callback()  # e.g. a step's internal cancel belongs to the same boundary
            if self.intervention is not None:
                raise RecoveryRequired('Interrupted mutation requires reconciliation: ' + str(self.intervention))
            # Reject uncontracted component swaps before allowing another external effect.
            json_bytes(encode(self.contract.capture(self.app)))
            policy_state(self.app.runner.run_policy, external=type(self.contract) is not BuiltinRecovery)
            generation = self.generation + 1
            intent = {'generation': generation, 'status': 'started', 'operation': operation,
                      'arguments': encode(arguments), 'started_at': self.wall_clock()}
            self.intervention = intent  # also fail closed if writing the intent fails
            atomic_write(self.root / 'journal.json', json_bytes(intent))
            self._active = True
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
