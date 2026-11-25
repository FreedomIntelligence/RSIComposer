"""CPU-only hard-exit tests of real Application, LabGroup, Host and SQLite boundaries."""
from concurrent.futures import ThreadPoolExecutor
from contextlib import chdir
from hashlib import sha256
import json
import os
from pathlib import Path
import sqlite3
import subprocess
import sys
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch

from rsicomposer.algorithms.ttrl import verl
from rsicomposer.application import Application, ApplicationConfig
from rsicomposer.assembly import ModuleAssembly
from rsicomposer.harnesses.defaults import build_harness_decision
from rsicomposer.loops.contracts.coordination import OperationResult
from rsicomposer.loops.contracts.decision import Decision, OperationCall, OperationSpec
from rsicomposer.loops.contracts.harness import NullHarness
from rsicomposer.loops.contracts.jobs import JobOutcome, JobRef, JobStatus
from rsicomposer.loops.contracts.lifecycle import ExecutionStatus
from rsicomposer.loops.contracts.model_service import BackendBinding, ModelRef
from rsicomposer.loops.decision_making.action_space.access import (
    ActionPermissions, SensitivePolicy, authorized_space,
)
from rsicomposer.method_state import method_state_factory
from rsicomposer.methods import ControllerDefinition
from rsicomposer.registry import Registry
from rsicomposer.research import MemoryAssembly, ResearchAssembly
from rsicomposer.runtime.execution.errors import WorkerLostError
from rsicomposer.runtime.orchestration.lab_group import LabGroup
from rsicomposer.runtime.state.recovery import DurableRecovery, RecoveryRequired
from rsicomposer.runtime.storage.codec import decode, encode
from rsicomposer.runtime.storage.files import json_bytes
from rsicomposer.snapshot.local import LocalSnapshotCapture, restore_snapshot


_EXIT_AT_BOUNDARY = 83
_OBJECTIVE = 'Verify durable receipt recovery'
_PRIVATE_STEP = 'synthetic-secret-argument from the interrupted step'
_PRIVATE_REPAIR = 'synthetic-secret-evidence for reconciliation'


def _log(root, name, value):
    with (root / name).open('a') as stream:
        stream.write(value + '\n')


def _lines(root, name):
    path = root / name
    return path.read_text().splitlines() if path.exists() else []


def _cut():
    os._exit(_EXIT_AT_BOUNDARY)


class _Harness(NullHarness):
    def __init__(self, root, fault):
        self.root, self.fault = root, fault

    def observe(self, event):
        if event.source != 'lab' or event.kind != 'receipt':
            return
        if self.fault in ('before_observer', 'poll_tick'):
            _cut()
        if self.fault == 'caught':
            raise RuntimeError('injected observer failure')
        _log(self.root, 'observer', event.event_id)
        if self.fault == 'after_observer':
            _cut()


class _Policy:
    def __init__(self, root, calls):
        self.root, self.calls = root, calls
        self.issued = False

    def decide(self, context):
        if self.issued:
            for result in context.feedbacks:
                _log(self.root, 'policy_feedback', result.call_id)
            return Decision('finish')
        self.issued = True
        _log(self.root, 'sample', 'issued')
        return Decision('invoke', operations=self.calls)

    def state(self):
        return {'issued': self.issued}

    def restore(self, state):
        self.issued = state['issued']


class _Lab:
    def __init__(self, root, *, idempotent, managed, fault):
        self.root, self.idempotent, self.managed, self.fault = root, idempotent, managed, fault
        self.job = None

    def describe(self):
        return (OperationSpec('act', 'A durable side effect', {
            'type': 'object', 'properties': {'value': {'type': 'integer'}}, 'required': ['value'],
        }, idempotent=self.idempotent),)

    def execute(self, call):
        _log(self.root, 'invocations', call.call_id)
        effect = self.root / ('effect-' + call.call_id)
        if not self.idempotent or not effect.exists():
            effect.touch(exist_ok=False)
            _log(self.root, 'effects', call.call_id)
        if self.fault == 'after_effect':
            _cut()
        if self.managed:
            self.job = JobRef('job:' + call.call_id, 'fake')
            return self.job
        return OperationResult('finished', (), False)

    def state(self):
        return {'job': self.job}

    def restore(self, state):
        self.job = state['job']

    def reattach_job(self, job):
        if job != self.job:
            raise ValueError('Job differs from captured state')
        _log(self.root, 'reattach', job.job_id)
        if (self.root / 'job-lost').exists():
            raise WorkerLostError('The fixture worker is gone')

    def poll_job(self, job):
        if job != self.job:
            raise ValueError('Job differs from captured state')
        _log(self.root, 'polls', job.job_id)
        if (self.root / 'job-ready').exists():
            return JobOutcome(job, JobStatus.SUCCEEDED, OperationResult('finished', (), False))
        return JobOutcome(job, JobStatus.RUNNING)


def _call(alias, value):
    return OperationCall(f'{alias}.act', {'value': value}, call_id=f'call-{alias}')


def _modules(*, managed=False, idempotent=False, batch=False, fault=None, on_build=None, calls=None):
    if calls is None:
        calls = (_call('a', 1), _call('b', 2)) if batch else (_call('a', 1),)

    def lab_factory(context):
        if on_build is not None:
            on_build(context.action_permissions)
        group = LabGroup(receipts=context.data_dir / 'lab_receipts')
        for call in calls:
            group.add(call.name.split('.')[0], _Lab(context.data_dir, idempotent=idempotent,
                                                     managed=managed, fault=fault))
        return group

    def decision_factory(context, lab):
        endpoint = build_harness_decision(context, lab, policy=_Policy(context.data_dir, calls),
                                          harness=_Harness(context.data_dir, fault))
        context.host_execution.activate(endpoint)
        return endpoint

    return ModuleAssembly(decision_factory=decision_factory, lab_factory=lab_factory,
                          state_factory=method_state_factory)


def _crash(root, fault, *, idempotent):
    config = ApplicationConfig(_OBJECTIVE, root, backends={}, roles={})
    with Application(config, durable=True, modules=_modules(
            managed=fault in ('poll_tick', 'waiting'), idempotent=idempotent, batch=fault == 'batch', fault=fault)) as app:
        execution = app.assembly.host_execution
        if fault == 'intent':
            with patch.object(execution, 'step', side_effect=lambda *_: _cut()):
                app.runner.step(_OBJECTIVE)
        elif fault == 'admitted':
            def checkpoint_then_exit(in_flight):
                app.recovery.commit_in_flight(in_flight)
                _cut()
            execution.in_flight_hook = checkpoint_then_exit
            app.runner.step(_OBJECTIVE)
        elif fault == 'batch':
            submit = ThreadPoolExecutor.submit
            launched = []

            def submit_then_exit(executor, *args, **kwargs):
                if launched:
                    _cut()
                result = submit(executor, *args, **kwargs)
                result.result(timeout=5)  # Ensure A's receipt is durable before attempting B.
                launched.append(result)
                return result

            with patch.object(ThreadPoolExecutor, 'submit', submit_then_exit):
                app.runner.step(_OBJECTIVE)
        elif fault in ('poll_tick', 'waiting'):
            with patch.object(ThreadPoolExecutor, 'submit', _completed_submit):
                step = app.runner.step(_OBJECTIVE)
            if step.job != JobRef('job:call-a', 'fake'):
                raise AssertionError('Fixture never committed a waiting JobRef')
            if fault == 'waiting':
                _cut()
            (root / 'job-ready').touch()
            app.runner.step(_PRIVATE_STEP)
        elif fault == 'caught':
            try:
                _deliver(app)
            except RuntimeError as error:
                if str(error) != 'injected observer failure':
                    raise
                return
        else:
            _deliver(app)
    raise AssertionError(f'Fixture did not hit the {fault} boundary')


_submit = ThreadPoolExecutor.submit


def _completed_submit(executor, *args, **kwargs):
    """Have the real worker finish before the scheduler probes its future."""
    future = _submit(executor, *args, **kwargs)
    future.result(timeout=5)
    return future


def _deliver(app):
    with patch.object(ThreadPoolExecutor, 'submit', _completed_submit):
        step = app.runner.step(_OBJECTIVE)
    result = step.event.get('operation_result')
    if result is None:
        raise AssertionError('Fixture did not deliver an operation result')
    return result


def _sqlite_ids(root):
    connection = sqlite3.connect(root / 'experience.sqlite3')
    try:
        return [row[0] for row in connection.execute(
            "SELECT record_id FROM experience WHERE kind='runtime_feedback' ORDER BY seq")]
    finally:
        connection.close()


class DurableRecoveryTests(unittest.TestCase):
    def crash(self, fault, *, idempotent=False):
        directory = tempfile.TemporaryDirectory()
        self.addCleanup(directory.cleanup)
        root = Path(directory.name).resolve()
        child = subprocess.run([sys.executable, str(Path(__file__).resolve()), '--crash', str(root),
                                fault, str(int(idempotent))], capture_output=True, text=True, timeout=20)
        self.assertEqual(child.returncode, 0 if fault == 'caught' else _EXIT_AT_BOUNDARY,
                         child.stdout + child.stderr)
        return root

    def assert_blocked(self, app, reason):
        incident = app.recovery.intervention
        self.assertIsNotNone(incident)
        self.assertEqual(incident['reason'], reason)
        with self.assertRaises(RecoveryRequired):
            app.runner.step(_OBJECTIVE)
        return incident

    def test_intent_without_in_flight_checkpoint_is_blocked(self):
        root = self.crash('intent', idempotent=True)
        with Application.resume(root, modules=_modules(idempotent=True)) as app:
            incident = self.assert_blocked(app, 'interrupted_mutation')
            self.assertEqual(incident['generation'], incident['checkpoint_generation'] + 1)
            self.assertEqual(incident['checkpoint_in_flight_evidence'], [])
        self.assertEqual(_lines(root, 'sample'), [])
        self.assertEqual(_lines(root, 'invocations'), [])
        self.assertEqual(_sqlite_ids(root), [])

    def test_pre_submit_and_post_effect_require_idempotence(self):
        for fault in ('admitted', 'after_effect'):
            for idempotent in (False, True):
                with self.subTest(fault=fault, idempotent=idempotent):
                    root = self.crash(fault, idempotent=idempotent)
                    prior = ['call-a'] if fault == 'after_effect' else []
                    self.assertEqual(_lines(root, 'invocations'), prior)
                    self.assertEqual(_lines(root, 'effects'), prior)
                    with Application.resume(root, modules=_modules(idempotent=idempotent)) as app:
                        if idempotent:
                            self.assertIsNone(app.recovery.intervention)
                            self.assertEqual(_deliver(app)['call_id'], 'call-a')
                        else:
                            incident = self.assert_blocked(app, 'interrupted_operation')
                            [row] = incident['checkpoint_in_flight_evidence']
                            self.assertEqual((row['call_id'], row['checkpoint_status'], row['receipt']),
                                             ('call-a', 'running', 'missing'))
                    self.assertEqual(_lines(root, 'invocations'), prior + ['call-a'] if idempotent else prior)
                    self.assertEqual(_lines(root, 'effects'), ['call-a'] if idempotent else prior)
                    self.assertEqual(len(_sqlite_ids(root)), int(idempotent))
                    self.assertEqual(_lines(root, 'observer'), _sqlite_ids(root))
                    self.assertEqual(_lines(root, 'sample'), ['issued'])

    def test_receipt_replay_deduplicates_sqlite_not_the_external_observer(self):
        for fault in ('before_observer', 'after_observer'):
            with self.subTest(fault=fault):
                root = self.crash(fault)
                [record_id] = _sqlite_ids(root)
                before = [record_id] if fault == 'after_observer' else []
                self.assertEqual(_lines(root, 'observer'), before)
                with Application.resume(root, modules=_modules()) as app:
                    self.assertIsNone(app.recovery.intervention)
                    self.assertEqual(_deliver(app)['call_id'], 'call-a')
                    app.runner.step(_OBJECTIVE)
                self.assertEqual(_lines(root, 'invocations'), ['call-a'])
                self.assertEqual(_lines(root, 'effects'), ['call-a'])
                self.assertEqual(_sqlite_ids(root), [record_id])
                self.assertEqual(_lines(root, 'observer'), before + [record_id])
                self.assertEqual(_lines(root, 'policy_feedback'), ['call-a'])

    def test_caught_exception_is_durably_blocked(self):
        root = self.crash('caught')
        saved = DurableRecovery.read(root)
        self.assertEqual(saved['intervention']['error_type'], 'RuntimeError')
        with Application.resume(root, modules=_modules()) as app:
            self.assert_blocked(app, 'interrupted_mutation')
        self.assertEqual(_lines(root, 'invocations'), ['call-a'])
        self.assertEqual(_lines(root, 'observer'), [])
        self.assertEqual(len(_sqlite_ids(root)), 1)

    def test_newer_poll_tick_stays_blocked_despite_terminal_receipt(self):
        root = self.crash('poll_tick')
        [record_id] = _sqlite_ids(root)
        prior_polls = _lines(root, 'polls')
        self.assertGreaterEqual(len(prior_polls), 2)
        modules = _modules(managed=True)
        with Application.resume(root, modules=modules) as app:
            incident = self.assert_blocked(app, 'interrupted_mutation')
            self.assertEqual(incident['generation'], incident['checkpoint_generation'] + 1)
            [row] = incident['checkpoint_in_flight_evidence']
            self.assertEqual((row['call_id'], row['checkpoint_status'], row['receipt'], row['receipt_sequence']),
                             ('call-a', 'waiting', 'terminal', 2))
        # A different request or result is not evidence even if the filename matches.
        [file] = (root / 'lab_receipts').glob('*.receipt')
        receipt = decode(json.loads(file.read_text()))
        self.assertEqual(receipt['outcome'].call_id, 'call-a')
        for field, other in (('arguments', {'value': 2}),
                             ('outcome', OperationResult('Wrong call', (), False, call_id='call-b'))):
            with self.subTest(field=field):
                file.write_text(json.dumps(encode({**receipt, field: other})))
                with Application.resume(root, modules=modules) as app:
                    incident = self.assert_blocked(app, 'interrupted_mutation')
                    [row] = incident['checkpoint_in_flight_evidence']
                    self.assertEqual(row['receipt'], 'unreadable_or_invalid')
        self.assertEqual(_lines(root, 'invocations'), ['call-a'])
        self.assertEqual(_lines(root, 'reattach'), [])
        self.assertEqual(_lines(root, 'polls'), prior_polls)
        self.assertEqual(_lines(root, 'observer'), [])
        self.assertEqual(_sqlite_ids(root), [record_id])

    def test_public_recovery_error_redacts_private_journal_and_repair_details(self):
        root = self.crash('poll_tick')
        modules = _modules(managed=True)
        journal_path = root / 'recovery' / 'journal.json'
        original_journal = journal_path.read_text()
        intent = json.loads(original_journal)
        self.assertEqual(decode(intent['arguments']), ((_PRIVATE_STEP,), {}))
        generation = intent['generation']
        call = {'call_id_sha256': sha256(b'call-a').hexdigest(), 'checkpoint_status': 'waiting',
                'receipt': 'terminal', 'receipt_sequence': 2}

        def diagnostic(app):
            with self.assertRaises(RecoveryRequired) as caught:
                app.runner.step(_OBJECTIVE)
            message = str(caught.exception)
            for private_text in (_PRIVATE_STEP, _PRIVATE_REPAIR, _OBJECTIVE, 'call-a'):
                self.assertNotIn(private_text, message)
            prefix = 'Interrupted mutation requires reconciliation: '
            self.assertTrue(message.startswith(prefix))
            return json.loads(message[len(prefix):])

        with Application.resume(root, modules=modules) as app:
            self.assertEqual(diagnostic(app), {'reason': 'interrupted_mutation', 'generation': generation,
                                              'checkpoint_generation': generation - 1, 'calls': [call]})
            self.assertEqual(journal_path.read_text(), original_journal)
            original_incident = app.recovery.intervention

            def interrupted_repair(owner, incident):
                self.assertIs(owner, app)
                self.assertEqual(incident, original_incident)
                self.assertEqual(decode(incident['arguments']), ((_PRIVATE_STEP,), {}))
                self.assertEqual(incident['checkpoint_in_flight_evidence'][0]['call_id'], 'call-a')
                raise RuntimeError('fixture repair did not finish')

            with self.assertRaisesRegex(RuntimeError, 'fixture repair did not finish'):
                app.recovery.reconcile(interrupted_repair, evidence=_PRIVATE_REPAIR)
            self.assertEqual(diagnostic(app), {'reason': 'interrupted_reconciliation', 'generation': generation})
            repair_intent = json.loads(journal_path.read_text())
            self.assertEqual(repair_intent['previous'], original_incident)
            self.assertEqual(repair_intent['evidence'], _PRIVATE_REPAIR)
            audit = json.loads((journal_path.parent / f'reconciliation-{generation}.json').read_text())
            self.assertEqual(audit['incident'], original_incident)
            self.assertEqual(audit['evidence'], _PRIVATE_REPAIR)

        with Application.resume(root, modules=modules) as app:
            self.assertEqual(diagnostic(app), {'reason': 'interrupted_reconciliation', 'generation': generation,
                                              'checkpoint_generation': generation - 1, 'calls': [call]})
        self.assertEqual(_lines(root, 'invocations'), ['call-a'])
        self.assertEqual(_lines(root, 'observer'), [])
        self.assertEqual(len(_sqlite_ids(root)), 1)

    def test_mixed_batch_does_not_deliver_partial_recovery(self):
        root = self.crash('batch')
        with Application.resume(root, modules=_modules(batch=True)) as app:
            incident = self.assert_blocked(app, 'interrupted_operation')
            self.assertEqual([row['call_id'] for row in incident['operations']], ['call-b'])
            self.assertEqual({row['call_id']: row['receipt'] for row in incident['checkpoint_in_flight_evidence']},
                             {'call-a': 'terminal', 'call-b': 'missing'})
        self.assertEqual(_lines(root, 'invocations'), ['call-a'])
        self.assertEqual(_lines(root, 'observer'), [])
        self.assertEqual(_sqlite_ids(root), [])

    def test_matching_job_reattaches_without_resubmission_or_polling_on_open(self):
        for lost in (False, True):
            with self.subTest(lost=lost):
                root = self.crash('waiting')
                prior_polls = _lines(root, 'polls')
                (root / ('job-lost' if lost else 'job-ready')).touch()
                with Application.resume(root, modules=_modules(managed=True)) as app:
                    self.assertIsNone(app.recovery.intervention)
                    self.assertEqual(_lines(root, 'reattach'), ['job:call-a'])
                    self.assertEqual(_lines(root, 'polls'), prior_polls)
                    outcome = _deliver(app)
                    self.assertEqual(outcome['call_id'], 'call-a')
                    if lost:
                        self.assertEqual(outcome['status'], ExecutionStatus.FAILED)
                        self.assertEqual(outcome['data'], {'effects_uncertain': True, 'retryable': False})
                    else:
                        self.assertEqual(outcome['status'], ExecutionStatus.COMPLETED)
                self.assertEqual(_lines(root, 'invocations'), ['call-a'])
                self.assertEqual(_lines(root, 'polls'), prior_polls + ([] if lost else ['job:call-a']))
                self.assertEqual(len(_sqlite_ids(root)), 1)
                self.assertEqual(_lines(root, 'observer'), _sqlite_ids(root))


class RecoveryPermissionsTests(unittest.TestCase):
    def setUp(self):
        self.directory = Path(self.enterContext(tempfile.TemporaryDirectory())).resolve()
        self.root = self.directory / 'workspace'
        self.permissions = ActionPermissions(['a.act'], SensitivePolicy('fixture', '1', (
            {'name': 'a.act', 'argument': 'value', 'equals': 1},
        )))
        self.config = ApplicationConfig(_OBJECTIVE, self.root, backends={}, roles={})

    def assert_recovered_policy(self, app, built_with):
        self.assertEqual(built_with, [self.permissions])
        space = authorized_space(app.runner.decision_making.action_space)
        self.assertEqual([spec.name for spec in space.describe()], ['a.act'])
        self.assertEqual(space.validate(_call('a', 2)).arguments, {'value': 2})
        for call in (_call('a', 1), _call('b', 2)):
            with self.subTest(call=call), self.assertRaises(PermissionError):
                space.validate(call)
        self.assertEqual(_lines(app.config.data_dir, 'invocations'), [])

    def test_durable_permissions_with_relative_workspace(self):
        with chdir(self.directory):
            config = ApplicationConfig(_OBJECTIVE, Path('workspace'), backends={}, roles={})
            with Application(config, durable=True, modules=_modules(batch=True),
                             action_permissions=self.permissions) as app:
                self.assertEqual(app.config.data_dir, self.root)
        built_with = []
        with Application.resume(self.root, modules=_modules(batch=True, on_build=built_with.append)) as app:
            self.assert_recovered_policy(app, built_with)

    def test_portable_snapshot_rebuilds_with_original_permissions(self):
        with Application(self.config, modules=_modules(batch=True), action_permissions=self.permissions) as app:
            capture = LocalSnapshotCapture(app)
            snapshot = capture.store.directory(capture.capture('permission fixture').snapshot_id)
        built_with = []
        target = self.directory / 'restored'
        with restore_snapshot(snapshot, target, modules=_modules(batch=True, on_build=built_with.append)) as app:
            self.assertEqual(app.config.data_dir, target)
            self.assert_recovered_policy(app, built_with)

    def test_restored_permissions_guard_actual_dispatch(self):
        for restore in ('durable', 'portable'):
            for allowed in (True, False):
                with self.subTest(restore=restore, allowed=allowed):
                    source = self.directory / f'{restore}-{allowed}'
                    config = ApplicationConfig(_OBJECTIVE, source, {}, {})
                    modules = _modules(calls=(_call('a', 2 if allowed else 1),))
                    with Application(config, modules=modules, action_permissions=self.permissions,
                                     durable=restore == 'durable') as app:
                        if restore == 'portable':
                            capture = LocalSnapshotCapture(app)
                            snapshot = capture.store.directory(capture.capture('permission fixture').snapshot_id)
                    if restore == 'durable':
                        reopened = Application.resume(source, modules=modules)
                    else:
                        reopened = restore_snapshot(snapshot, self.directory / f'target-{allowed}', modules=modules)
                    with reopened as app:
                        if allowed:
                            app.run(max_decision_steps=1, wait_for_jobs=True)
                        else:
                            with self.assertRaises(PermissionError):
                                app.run(max_decision_steps=1, wait_for_jobs=True)
                        self.assertEqual(_lines(app.config.data_dir, 'invocations'), ['call-a'] if allowed else [])

    def test_legacy_checkpoint_without_permissions_is_closed_before_assembly(self):
        with Application(self.config, durable=True, modules=_modules()):
            pass
        # Emit the exact older envelope: absence of the field did not identify the original policy.
        checkpoint = DurableRecovery.read(self.root)
        del checkpoint['action_permissions']
        payload = encode(checkpoint)
        (self.root / 'recovery' / 'checkpoint.json').write_bytes(json_bytes({
            'payload': payload, 'sha256': sha256(json_bytes(payload)).hexdigest(),
        }))
        config_path = self.root / 'config.json'
        config = json.loads(config_path.read_bytes())
        config_path.write_bytes(json_bytes({**config, 'data_dir': 'workspace'}))
        built_with = []
        with Application.resume(self.root, modules=_modules(on_build=built_with.append)) as app:
            self.assertEqual(built_with, [ActionPermissions([])])
            self.assertEqual(authorized_space(app.runner.decision_making.action_space).describe(), ())

    def test_legacy_snapshot_without_permissions_is_closed_before_assembly(self):
        def legacy_encode(state):
            return encode({name: value for name, value in state.items() if name != 'action_permissions'})

        with Application(self.config, modules=_modules()) as app:
            capture = LocalSnapshotCapture(app)
            with patch('rsicomposer.snapshot.components.encode', legacy_encode):
                snapshot = capture.store.directory(capture.capture('legacy permission fixture').snapshot_id)
        built_with = []
        with restore_snapshot(snapshot, self.directory / 'restored',
                              modules=_modules(on_build=built_with.append)) as app:
            self.assertEqual(built_with, [ActionPermissions([])])
            self.assertEqual(authorized_space(app.runner.decision_making.action_space).describe(), ())

    def test_resume_reconstructs_lazy_registered_memory(self):
        calls = []

        def memory(options):
            calls.append(options)
            return MemoryAssembly(ResearchAssembly())

        registry = Registry()
        registry.register_controller('recovery_memory', ControllerDefinition(
            lambda ctx, _: _Policy(ctx.data_dir, ()), memory=memory), version='1')
        registry.register_lab('recovery_memory', lambda ctx, _: _Lab(
            ctx.data_dir, idempotent=False, managed=False, fault=None), version='1')
        selection = {'controller': {'name': 'recovery_memory'}, 'labs': {'a': {'name': 'recovery_memory'}}}
        config = ApplicationConfig(_OBJECTIVE, self.root, {'offline': BackendBinding('offline', 'mock')},
                                   {'decision_maker': ModelRef('offline', 'fixture', 'v1')})
        modules = registry.build(selection)
        self.assertEqual(calls, [])
        with Application(config, durable=True, modules=modules, action_permissions=self.permissions):
            self.assertEqual(calls, [{}])
        self.assertTrue((self.root / 'research_assembly.json').is_file())
        modules = registry.build(selection)
        self.assertEqual(calls, [{}])
        with Application.resume(self.root, modules=modules) as app:
            self.assertIsNotNone(app.runner.decision_making.integration)
            self.assertEqual(calls, [{}, {}])

    def test_malformed_saved_permissions_are_rejected_before_construction(self):
        with Application(self.config, durable=True, modules=_modules(),
                         action_permissions=self.permissions) as app:
            capture = LocalSnapshotCapture(app)

            def invalid_encode(state):
                return encode({**state, 'action_permissions': {}})

            with patch('rsicomposer.snapshot.components.encode', invalid_encode):
                snapshot = capture.store.directory(capture.capture('invalid permission fixture').snapshot_id)
        checkpoint = DurableRecovery.read(self.root)
        payload = encode({**checkpoint, 'action_permissions': {}})
        (self.root / 'recovery' / 'checkpoint.json').write_bytes(json_bytes({
            'payload': payload, 'sha256': sha256(json_bytes(payload)).hexdigest(),
        }))
        built_with = []
        modules = _modules(on_build=built_with.append)
        with self.assertRaisesRegex(ValueError, 'Invalid saved action permissions'):
            Application.resume(self.root, modules=modules)
        target = self.directory / 'restored'
        with self.assertRaisesRegex(ValueError, 'Invalid saved action permissions'):
            restore_snapshot(snapshot, target, modules=modules)
        self.assertEqual(built_with, [])
        self.assertFalse(target.exists())

    def test_custom_permissions_do_not_build_durable_or_publish_portable_state(self):
        policy = ActionPermissions(check=lambda request: False)
        built_with = []
        with self.assertRaisesRegex(TypeError, 'custom permission callbacks are unsupported'):
            Application(self.config, durable=True, action_permissions=policy,
                        modules=_modules(on_build=built_with.append))
        self.assertFalse(self.root.exists())
        self.assertEqual(built_with, [])
        with Application(self.config, action_permissions=policy, modules=_modules()) as app:
            self.assertEqual(authorized_space(app.runner.decision_making.action_space).describe(), ())
            capture = LocalSnapshotCapture(app)
            with self.assertRaisesRegex(TypeError, 'custom permission callbacks are unsupported'):
                capture.capture('no custom callback serialization')
            self.assertFalse(capture.store.root.exists())


class NativeRecoveryTests(unittest.TestCase):
    def setUp(self):
        directory = tempfile.TemporaryDirectory()
        self.addCleanup(directory.cleanup)
        self.root = Path(directory.name)
        self.config = {'upstream_root': str(self.root)}
        self.engine = verl.VerlTTRLEngine(self.config)
        self.job = JobRef(str(self.root), self.engine.name)
        self.descriptor = dict(call_id='native-call', pid=123, start_ticks='original', submitted_at=1.0)
        self.write('job', self.descriptor)

    def write(self, file, data):
        (self.root / f'{file}.json').write_text(json.dumps(data))

    def test_running_identity_and_terminal_status(self):
        with patch.object(verl, 'process_identity', return_value='original'):
            self.assertIsNone(self.engine.reattach_fit(self.job))
            self.assertEqual(self.engine.poll_fit(self.job).status, JobStatus.RUNNING)
        with patch.object(verl, 'process_identity', side_effect=AssertionError('terminal must not probe the PID')):
            self.write('status', {'status': 'succeeded'})
            self.write('response', {'weights_changed': True, 'steps': 1})
            self.assertIsNone(self.engine.reattach_fit(self.job))
            self.assertEqual(self.engine.poll_fit(self.job).result, {'weights_changed': True, 'steps': 1})
            self.write('status', {'status': 'failed', 'error': 'training failed'})
            self.assertIsNone(self.engine.reattach_fit(self.job))
            outcome = self.engine.poll_fit(self.job)
            self.assertEqual((outcome.status, outcome.error), (JobStatus.FAILED, 'training failed'))

    def test_lost_identity_requires_no_terminal_publication(self):
        for identity in (None, 'replacement'):
            with self.subTest(identity=identity), patch.object(verl, 'process_identity', return_value=identity), \
                 patch.object(verl.os, 'kill', side_effect=ProcessLookupError):
                for method in (self.engine.reattach_fit, self.engine.poll_fit):
                    with self.assertRaises(WorkerLostError):
                        method(self.job)

        def publish_during_probe(_):
            self.write('status', {'status': 'failed', 'error': 'terminal failure'})
            return 'replacement'

        with patch.object(verl, 'process_identity', side_effect=publish_during_probe):
            self.assertEqual(self.engine.poll_fit(self.job).error, 'terminal failure')

    def test_missing_invalid_and_unreadable_inputs_are_not_loss(self):
        (self.root / 'job.json').unlink()
        with self.assertRaises(FileNotFoundError):
            self.engine.reattach_fit(self.job)
        for descriptor in ({}, {**self.descriptor, 'pid': -1}, {**self.descriptor, 'start_ticks': False}):
            with self.subTest(descriptor=descriptor):
                self.write('job', descriptor)
                with self.assertRaises(ValueError):
                    self.engine.reattach_fit(self.job)
        self.write('job', self.descriptor)
        with self.assertRaisesRegex(ValueError, 'backend'):
            self.engine.reattach_fit(JobRef(self.job.job_id, 'wrong-backend'))
        with patch.object(verl, 'process_identity', side_effect=PermissionError('identity inaccessible')):
            with self.assertRaises(PermissionError):
                self.engine.reattach_fit(self.job)
        with (patch.object(verl.sys, 'platform', 'darwin'),
              patch.object(verl, 'process_identity', return_value=None),
              patch.object(verl.os, 'kill', return_value=None)):
            with self.assertRaisesRegex(RuntimeError, 'Cannot verify') as caught:
                self.engine.reattach_fit(self.job)
            self.assertNotIsInstance(caught.exception, WorkerLostError)
        self.write('job', {**self.descriptor, 'start_ticks': None})
        with patch.object(verl, 'process_identity', return_value='some-process'):
            with self.assertRaisesRegex(RuntimeError, 'no captured process identity') as caught:
                self.engine.reattach_fit(self.job)
            self.assertNotIsInstance(caught.exception, WorkerLostError)
        self.write('job', self.descriptor)
        for status in ({'status': 'running'}, {'status': 'failed'}, {'status': 'failed', 'error': 42},
                       {'status': 'succeeded', 'error': 'contradictory'}):
            with self.subTest(status=status):
                self.write('status', status)
                with self.assertRaisesRegex(ValueError, 'terminal status'):
                    self.engine.reattach_fit(self.job)
        self.write('status', {'status': 'succeeded'})
        with self.assertRaises(FileNotFoundError):
            self.engine.reattach_fit(self.job)

    def test_native_worker_loss_becomes_correlated_failure_receipt(self):
        cfg = SimpleNamespace(data_dir=self.root, artifacts=None)
        verifier = SimpleNamespace(name='fake', parse=lambda _: None, is_correct=lambda *_: False)
        with patch.object(self.engine, 'bind_execution'):
            lab = verl.TTRLVerlLab(cfg, self.config, engine=self.engine, verifier=verifier)
        lab.native_jobs[self.job.job_id] = {'job': self.job, 'call_id': 'native-call'}
        group = LabGroup(receipts=self.root / 'receipts')
        group.add('ttrl', lab)
        call = OperationCall('ttrl.fit', {}, call_id='native-call')
        for reattach in (True, False):
            with self.subTest(reattach=reattach), patch.object(verl, 'process_identity', return_value='replacement'):
                # Start from a submission receipt in either branch; do not launch a worker.
                group._store(call, self.job, 0.0)
                if reattach:
                    result = group.reattach_job(call, self.job)
                else:
                    result, progress = group.poll_job(call, self.job, decision_id='decision', after=0)
                    self.assertEqual(progress, ())
                self.assertEqual((result.call_id, result.status), ('native-call', ExecutionStatus.FAILED))
                self.assertEqual(result.data, {'effects_uncertain': True, 'retryable': False})
                self.assertEqual(group.stored_receipt(call), result)
                self.assertFalse(group.can_retry(call))


if __name__ == '__main__':
    if len(sys.argv) == 5 and sys.argv[1] == '--crash':
        _crash(Path(sys.argv[2]), sys.argv[3], idempotent=sys.argv[4] == '1')
    else:
        unittest.main()
