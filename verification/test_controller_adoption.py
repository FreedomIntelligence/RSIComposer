"""Keyless Controller adoption regressions using real Application and Host ports."""
from dataclasses import replace
import json
from pathlib import Path
from tempfile import TemporaryDirectory
from threading import Event
import unittest
from unittest.mock import patch

from rsicomposer.api import (
    BackendBinding, ControllerPort, LocalSnapshotCapture, MemoryAssembly, ResearchAssembly, RunDecision, TimeRunPolicy,
)
from rsicomposer.application import Application, ApplicationConfig
from rsicomposer.loops.contracts.candidate import Candidate
from rsicomposer.loops.contracts.model_service import ModelRef
from rsicomposer.loops.contracts.training import ModelActivationPending, TrainingResult
from rsicomposer.harnesses.defaults import build_harness_decision
from rsicomposer.loops.contracts.controller_version import ControllerVersion
from rsicomposer.loops.contracts.coordination import LoopStepResult, OperationResult
from rsicomposer.loops.contracts.decision import Decision, OperationCall, OperationSpec
from rsicomposer.loops.contracts.harness import NullHarness
from rsicomposer.loops.contracts.jobs import JobOutcome, JobProgress, JobRef, JobStatus, OperationProgress
from rsicomposer.methods import ControllerDefinition, MethodDefinition
from rsicomposer.runtime.state.controller_versions import save_controller_version
from rsicomposer.runtime.state.recovery import RecoveryRequired


class Policy:
    def __init__(self, *, invoke=False, fail_restore=False):
        self.invoke, self.fail_restore = invoke, fail_restore
        self.decisions = self.closes = 0

    def decide(self, window):
        self.decisions += 1
        if self.invoke and self.decisions == 1:
            return Decision('invoke', operations=(OperationCall('work.start', {}, call_id='original-call'),))
        return Decision('finish', reason='done')

    def plan_swap(self, calls):
        return {call.call_id: 'handover' for call in calls}

    def state(self):
        return {'decisions': self.decisions}

    def restore(self, state):
        if self.fail_restore:
            raise RuntimeError('candidate restore failed')
        self.decisions = state['decisions']

    def close(self):
        self.closes += 1


class Harness(NullHarness):
    def __init__(self, *, on_bind=None, fail_close=0):
        self.on_bind, self.fail_close = on_bind, fail_close
        self.binds = self.closes = 0
        self.events = []

    def bind_execution(self, read):
        self.binds += 1
        self.read = read
        self.bind_view = read()
        if self.on_bind is not None:
            self.on_bind()

    def observe(self, event):
        self.events.append(event)

    def close(self):
        self.closes += 1
        if self.closes <= self.fail_close:
            raise RuntimeError('harness close failed')


class Lab:
    def __init__(self):
        self.job = JobRef('example-job', 'fixture')
        self.calls, self.cancels, self.done = [], 0, False

    def describe(self):
        return (OperationSpec('start', 'Start the test job', {'type': 'object'}),)

    def execute(self, call):
        self.calls.append(call.call_id)
        return self.job

    def poll_job(self, job):
        if self.done:
            return JobOutcome(job, JobStatus.SUCCEEDED, OperationResult('finished job', (), False))
        return JobOutcome(job, JobStatus.RUNNING)

    def cancel_job(self, job):
        self.cancels += 1
        return JobOutcome(job, JobStatus.CANCELLED)

    def state(self):
        return {'calls': list(self.calls), 'cancels': self.cancels, 'done': self.done}

    def restore(self, state):
        self.calls, self.cancels, self.done = state['calls'], state['cancels'], state['done']


def make_version(*, policy_state=None):
    return ControllerVersion('replacement-v1', {'name': 'replacement', 'config': {}}, policy_state=policy_state)


def replacement_checkpoint(app, host):
    saved = host.capture()
    saved['archives'] = [{'version': saved['current_version'], 'state': saved['active']}]
    saved['current_version'] = version = make_version()
    saved['adopted'] = {version.version_id: save_controller_version(app.artifacts, version)}
    return saved


def adopted_events(app):
    return [event for row in app.experience.retrieve(kinds=('controller_event',))
            if (event := row.details['event'])['kind'] == 'controller_adopted']


class ControllerAdoptionTests(unittest.TestCase):
    def application(self, definition, *, old_policy=None, old_harness=None, old_memory=None, durable=False,
                    lab_type=Lab, model=None, validator=None):
        old_policy = old_policy if old_policy is not None else Policy()
        old_harness = old_harness if old_harness is not None else Harness()
        baseline = ControllerDefinition(lambda context, options: old_policy, harness=lambda context, options: old_harness,
                                        memory=old_memory)
        modules = MethodDefinition.from_factories(controller=baseline, labs={'work': lambda context, options: lab_type()},
                                                  version='1').factory({})
        modules = replace(modules, controller_catalog={**modules.controller_catalog, 'replacement': definition},
                          controller_validators={'replacement': validator} if validator else {})
        root = Path(self.enterContext(TemporaryDirectory()))
        config = ApplicationConfig('test adoption', root,
            {model.backend_id: BackendBinding(model.backend_id, 'mock')} if model else {},
            {'decision_maker': model} if model else {})
        app = self.enterContext(Application(config, modules=modules, durable=durable))
        return app, app.assembly.controller_host

    def queue(self, app, host, version=None):
        version = version if version is not None else make_version()
        ref = save_controller_version(app.artifacts, version)
        self.assertEqual(host.request_version(ref).data['status'], 'pending')
        return version

    def test_manual_list_state_round_trips_before_and_after_retirement(self):
        class ListPort:
            def __init__(self, instance):
                self.instance = instance
                self.endpoint, self.context = instance.endpoint, instance.context

            def capture(self):
                return [self.instance.capture()]

            def restore(self, state):
                [saved] = state
                self.instance.restore(saved)
                restored.append(self.instance.policy.decisions)

            def close(self):
                self.instance.close()

        initial_decisions, restored = iter((17, 0, 0)), []

        def baseline(ctx, opts):
            policy = Policy()
            policy.decisions = next(initial_decisions)
            return policy

        base = MethodDefinition.from_factories(controller=baseline, labs={'work': lambda *_: Lab()},
                                               version='1').factory({})

        def manual(ctx, lab):
            port = base.controller_factory(ctx, lab)
            return ControllerPort(ListPort(port.instance), port.version)

        modules = replace(base, controller_factory=manual, controller_catalog={**base.controller_catalog,
            'replacement': ControllerDefinition(lambda *_: Policy())})
        root = Path(self.enterContext(TemporaryDirectory()))
        with Application(ApplicationConfig('manual archive', root, {}, {}), modules=modules, durable=True) as app:
            initial = app.assembly.controller_host.current_version
        with Application.resume(root, modules=modules) as app:
            host = app.assembly.controller_host
            self.assertEqual(restored, [17])  # Restored the list payload rather than a fresh Policy.
            self.assertEqual(host.active.instance.policy.decisions, 17)
            initial_ref = save_controller_version(app.artifacts, initial)
            self.assertEqual(host.request_version(initial_ref).data['status'], 'active')
            archive = host.active.capture()
            version = self.queue(app, host)
            self.assertIn('Controller adopted:', app.runner.step(app.config.objective).event['reason'])
            conflict = replace(version, version_id=initial.version_id)
            with self.assertRaisesRegex(ValueError, 'cannot be reused'):
                host.request_version(save_controller_version(app.artifacts, conflict))
        with Application.resume(root, modules=modules) as app:
            host = app.assembly.controller_host
            self.assertEqual(host.current_version, version)
            self.assertEqual(host.archives, [{'version': initial, 'state': archive}])
            with self.assertRaisesRegex(ValueError, 'cannot be reused'):
                host.version(version_id=initial.version_id)
            for obsolete in (initial, conflict):
                with self.assertRaisesRegex(ValueError, 'cannot be reused'):
                    host.request_version(save_controller_version(app.artifacts, obsolete))
            self.assertIsNone(host.pending)
            self.assertEqual(app.run().status, 'finished')

    def test_durable_candidate_validation_subclass_rejects_without_blocking(self):
        def validate(options):
            json.loads(options['json'])

        app, host = self.application(ControllerDefinition(lambda *_: Policy()), durable=True, validator=validate)
        old = host.active
        version = replace(make_version(), config={'name': 'replacement', 'config': {'json': 'invalid'}})
        self.queue(app, host, version)
        self.assertIn('candidate construction failed', app.runner.step(app.config.objective).event['reason'])
        self.assertIs(host.active, old)
        self.assertIsNone(host.pending)
        self.assertIsNone(app.recovery.intervention)
        rejected, = [row.details['event']['output'] for row in app.experience.retrieve(kinds=('controller_event',))
                     if row.details['event']['kind'] == 'controller_adoption_rejected']
        self.assertEqual(rejected['error_type'], 'JSONDecodeError')
        self.assertEqual(app.run().status, 'finished')

    def test_candidate_infrastructure_and_control_faults_do_not_commit_rejection(self):
        from rsicomposer.runtime.execution.errors import WorkerLostError
        from rsicomposer.runtime.models.accounting import HostAdmissionStop

        for error in (OSError('storage unavailable'), WorkerLostError('worker lost'),
                      RecoveryRequired('reconciliation needed'), HostAdmissionStop('closed')):
            with self.subTest(error=type(error).__name__):
                class Fault(Policy):
                    def restore(self, state):
                        raise error

                policy = Fault()
                app, host = self.application(ControllerDefinition(lambda *_: policy), durable=True)
                old = host.active
                version = self.queue(app, host, make_version(policy_state={'decisions': 7}))
                with self.assertRaises(type(error)) as caught:
                    app.runner.step(app.config.objective)
                self.assertIs(caught.exception, error)
                self.assertEqual(policy.closes, 1)
                self.assertIs(host.active, old)
                self.assertEqual(host.pending['version'], version)
                self.assertEqual(app.recovery.intervention['reason'], 'interrupted_mutation')

    def test_pending_model_activation_blocks_snapshot_and_clean_shutdown(self):
        calls = []

        class Activator:
            def activate(self, result, *, idempotency_key):
                calls.append((result.model, idempotency_key))
                raise ModelActivationPending('fixture deployment still in flight')

        app, host = self.application(ControllerDefinition(lambda *_: Policy()), durable=True)
        host.activator = Activator()
        old = host.active
        base = ModelRef('fixture', 'original', 'v1')
        model = ModelRef('fixture', 'candidate', 'v2')
        version = replace(make_version(), model=model)
        ref = save_controller_version(app.artifacts, version)
        training = TrainingResult(base, model, Candidate('candidate', base.revision, {}, ('decision_maker',)))
        host.request_version(ref, training_result=training)
        self.assertFalse(app.runner.decision_making.pending_stop_checkpoint())
        result = app.runner.step(app.config.objective)
        self.assertEqual(result.event['reason'], 'Controller model activation pending')
        self.assertTrue(app.runner.decision_making.pending_stop_checkpoint())
        self.assertTrue(host.capture()['pending']['activation_pending'])
        with self.assertRaisesRegex(RuntimeError, 'quiescent|activation'):
            LocalSnapshotCapture(app).capture('pending')

        now = 0.
        app.runner._clock = lambda: now
        app.runner.cleanup_timeout_seconds = 1.
        app.runner._begin_stop(RunDecision('stop', 'fixture stop', 'cancel'))
        waiting = app.runner.step(app.config.objective)
        self.assertEqual(waiting.kind, 'waiting')
        self.assertTrue(waiting.event['run_stop']['unresolved_checkpoint'])
        now = 1.
        stopped = app.runner.step(app.config.objective)
        self.assertFalse(stopped.event['run_stop']['shutdown_complete'])
        self.assertTrue(stopped.event['run_stop']['cleanup_timed_out'])
        self.assertEqual(calls, [(model, 'controller-version:' + version.version_id)])
        self.assertIs(host.active, old)

    def test_bind_failure_preserves_active_delivery_and_can_retry(self):
        seen = []

        def failing_bind():
            self.assertIs(host.active, old)
            self.assertIs(execution.endpoint, old.endpoint)
            self.assertEqual(execution.controller_version, old_version)
            self.assertEqual(host.pending, pending)
            raise RuntimeError('candidate bind failed')

        def harness_factory(context, options):
            harness = Harness(on_bind=failing_bind if not seen else None)
            seen.append(harness)
            return harness

        app, host = self.application(ControllerDefinition(lambda context, options: Policy(), harness=harness_factory))
        execution = app.assembly.host_execution
        old, old_version = host.active, host.version_id
        version = self.queue(app, host)
        pending = host.pending.copy()
        execution.state.finished = True
        finish = LoopStepResult('finished', reason='old finished', decision_steps=0)
        execution.state.finish_after_operations = finish
        execution.state.operation_results.append(OperationResult('old receipt', (), False, call_id='old-call'))
        progress = OperationProgress('old-call', JobRef('old-job', 'fixture'), 1, 'progress', JobProgress())
        execution.state.operation_progress.append(progress)
        execution.state.decision_feedback_cursor = execution.state.progress_cursor = 1

        with self.assertRaisesRegex(RuntimeError, 'candidate bind failed'):
            app.runner.step(app.config.objective)
        self.assertIs(host.active, old)
        self.assertIs(execution.endpoint, old.endpoint)
        self.assertEqual((host.version_id, execution.controller_version), (old_version, old_version))
        self.assertEqual(host.pending, pending)
        self.assertEqual(host.archives, [])
        self.assertEqual(host._adopted, {})
        self.assertEqual(seen[0].bind_view['progresses'], (progress,))
        self.assertEqual(old.endpoint.harness.read()['progresses'], ())
        self.assertTrue(execution.state.finished)
        self.assertIs(execution.state.finish_after_operations, finish)
        self.assertEqual((execution.state.decision_feedback_cursor, execution.state.progress_cursor), (1, 1))
        self.assertEqual((old.policy.closes, old.endpoint.harness.closes, seen[0].closes), (0, 0, 1))
        self.assertEqual(adopted_events(app), [])

        result = host.advance()
        self.assertEqual((result.kind, result.decision_steps), ('progressed', 0))
        self.assertIs(host.active.endpoint, execution.endpoint)
        self.assertIs(host.active.endpoint.harness, seen[1])
        self.assertEqual((host.version_id, execution.controller_version), (version.version_id, version.version_id))
        self.assertIsNone(host.pending)
        self.assertFalse(execution.state.finished)
        self.assertIsNone(execution.state.finish_after_operations)
        self.assertEqual((execution.state.decision_feedback_cursor, execution.state.progress_cursor), (0, 0))
        execution.state.progress_cursor = 1
        self.assertEqual(seen[1].read()['progresses'], ())  # The bound reader is now live.
        self.assertEqual((old.policy.closes, old.endpoint.harness.closes), (1, 1))
        events = adopted_events(app)
        self.assertEqual([event['output']['adoption_id'] for event in events], [pending['adoption_id']])
        self.assertIsNone(host.advance())
        self.assertEqual(adopted_events(app), events)

    def test_policy_restore_failure_rejects_candidate_before_binding(self):
        for durable in (False, True):
            with self.subTest(durable=durable):
                candidate, harness = Policy(fail_restore=True), Harness()
                app, host = self.application(ControllerDefinition(lambda *_: candidate, harness=lambda *_: harness),
                                             durable=durable)
                old = host.active
                version = self.queue(app, host, make_version(policy_state={'decisions': 3}))
                if durable:
                    result = app.runner.step(app.config.objective)
                    self.assertIn('state could not be restored', result.event['reason'])
                    self.assertIsNone(host.pending)
                    self.assertIsNone(app.recovery.intervention)
                else:
                    with self.assertRaisesRegex(RuntimeError, 'candidate restore failed'):
                        host.advance()
                    self.assertEqual(host.pending['version'], version)
                self.assertIs(host.active, old)
                self.assertIs(app.assembly.host_execution.endpoint, old.endpoint)
                self.assertEqual((candidate.closes, harness.closes, harness.binds), (1, 1, 0))
                self.assertEqual(old.policy.closes, 0)
                self.assertEqual(adopted_events(app), [])

    def test_durable_candidate_state_is_checked_before_publication(self):
        class MissingState(Policy):
            state = restore = None

        class OpaqueState(Policy):
            def state(self):
                return {'opaque': object()}

        class NonFiniteState(Policy):
            def state(self):
                return {'score': float('nan')}

        for policy, error_type in ((MissingState, 'TypeError'), (OpaqueState, 'TypeError'),
                                   (NonFiniteState, 'ValueError')):
            with self.subTest(policy=policy.__name__):
                candidate, harness = policy(), Harness()
                app, host = self.application(ControllerDefinition(lambda *_: candidate, harness=lambda *_: harness),
                                             durable=True)
                old, version = host.active, self.queue(app, host)
                result = app.runner.step(app.config.objective)
                self.assertIn('state cannot be checkpointed', result.event['reason'])
                self.assertEqual((result.decision_steps, harness.binds, harness.closes, candidate.closes), (0, 1, 1, 1))
                self.assertIs(host.active, old)
                self.assertIs(app.assembly.host_execution.endpoint, old.endpoint)
                self.assertEqual((old.policy.closes, host.pending, host.archives, host._retired), (0, None, [], []))
                self.assertIsNone(app.recovery.intervention)
                self.assertEqual(adopted_events(app), [])
                rejected, = [row.details['event']['output'] for row in app.experience.retrieve(kinds=('controller_event',))
                             if row.details['event']['kind'] == 'controller_adoption_rejected']
                self.assertEqual((rejected['controller_version'], rejected['error_type']), (version.version_id, error_type))
                self.assertEqual(app.run(wait_for_jobs=True).status, 'finished')

    def test_binding_is_checked_on_the_restored_policy_for_adoption_seed_and_resume(self):
        class Ruler(Policy):
            unit = 'cm'
            def required_lab_bindings(self): return {'work': {'unit': self.unit}}
            def state(self): return {'unit': self.unit}
            def restore(self, state): self.unit = state['unit']

        class Centimeters(Lab):
            def validate_binding(self, requirement):
                if requirement != {'unit': 'cm'}:
                    raise ValueError('Only centimeters are measured')

        app, host = self.application(ControllerDefinition(lambda *_: Ruler()), lab_type=Centimeters, durable=True)
        old = host.active
        self.queue(app, host, make_version(policy_state={'unit': 'mm'}))
        self.assertIn('incompatible with its Labs', app.runner.step(app.config.objective).event['reason'])
        self.assertIs(host.active, old)
        self.assertIsNone(host.pending)
        self.assertEqual(adopted_events(app), [])
        self.queue(app, host, make_version(policy_state={'unit': 'cm'}))
        self.assertIn('Controller adopted:', app.runner.step(app.config.objective).event['reason'])

        config, modules = app.config, app.modules
        # Save an instance whose dynamic requirements no longer match its Lab.
        host.active.policy.unit = 'mm'
        app.runner.step(config.objective)
        app.close()
        with self.assertRaisesRegex(ValueError, 'Only centimeters'):
            Application.resume(config.data_dir, modules=modules)
        seed_config = replace(config, data_dir=Path(self.enterContext(TemporaryDirectory())))
        with self.assertRaisesRegex(ValueError, 'Only centimeters'):
            Application(seed_config, modules=modules, _initial_controller_version=make_version(policy_state={'unit': 'mm'}))

    def test_resume_checks_fixed_endpoint_after_lab_and_policy_restore(self):
        from rsicomposer.api import restore_snapshot
        from rsicomposer.assembly import ModuleAssembly
        from rsicomposer.method_state import method_state_factory

        class Fixed(Policy):
            def required_operations(self):
                return ('work.other',) if self.decisions else ('work.start',)

        root = Path(self.enterContext(TemporaryDirectory()))
        lab = MethodDefinition.from_factories(controller=lambda *_: Fixed(), labs={'work': lambda *_: Lab()},
                                             version='1').factory({}).lab_factory
        modules = ModuleAssembly(lab_factory=lab, state_factory=method_state_factory,
            decision_factory=lambda ctx, group: build_harness_decision(ctx, group, policy=Fixed(), harness=Harness()))
        with Application(ApplicationConfig('fixed endpoint', root, {}, {}), modules=modules, durable=True) as app:
            app.runner.step(app.config.objective)
            capture = LocalSnapshotCapture(app)
            snapshot = capture.store.directory(capture.capture('fixed Controller').snapshot_id)
        with self.assertRaisesRegex(ValueError, 'missing Lab operations: work.other'):
            Application.resume(root, modules=modules)
        with self.assertRaisesRegex(ValueError, 'missing Lab operations: work.other'):
            restore_snapshot(snapshot, root / 'restored', modules=modules)

    def test_custom_adapter_restoring_lab_last_cannot_invalidate_hosted_controller(self):
        from rsicomposer.method_state import method_state_factory

        class Ruler(Policy):
            def required_lab_bindings(self): return {'work': {'unit': 'cm'}}

        class Scale(Lab):
            unit = 'cm'
            def validate_binding(self, requirement):
                if requirement['unit'] != self.unit:
                    raise ValueError('Restored Lab has mismatched units')
            def state(self): return {'unit': self.unit}
            def restore(self, state): self.unit = state['unit']

        def adapters(ctx, decision):
            bound = method_state_factory(ctx, decision)
            return {key: bound[key] for key in ('execution', 'decision', 'lab_execution')}

        root = Path(self.enterContext(TemporaryDirectory()))
        standard = MethodDefinition.from_factories(controller=lambda *_: Ruler(), labs={'work': lambda *_: Scale()},
                                                  version='1').factory({})
        modules = replace(standard, state_factory=adapters)
        with Application(ApplicationConfig('restore both sides', root, {}, {}), modules=modules, durable=True) as app:
            app.assembly.lab_execution.labs['work'].unit = 'mm'
            app.runner.step(app.config.objective)
        with self.assertRaisesRegex(ValueError, 'Restored Lab has mismatched units'):
            with Application.resume(root, modules=modules):
                self.fail('A late Lab restore invalidated the already checked Controller')

    def test_non_durable_controller_without_state_runs_without_advertising_adoption(self):
        class NoArchive(Policy):
            state = restore = None

        for initial in (True, False):
            with self.subTest(initial=initial):
                app, host = self.application(ControllerDefinition(lambda *_: NoArchive()),
                                             old_policy=NoArchive() if initial else Policy())
                if not initial:
                    self.queue(app, host)
                    self.assertIn('Controller adopted:', host.advance().reason)
                space = host.active.endpoint.action_space
                visible = {spec.name for spec in space.describe()}
                self.assertIn('work.start', visible)
                self.assertNotIn('host.adopt_controller', visible)
                ref = save_controller_version(app.artifacts, make_version())
                with self.assertRaises(PermissionError):
                    space.validate(OperationCall('host.adopt_controller', {'version': ref}))
                with self.assertRaisesRegex(ValueError, 'archivable'):
                    host.request_version(ref)
                self.assertEqual(app.run(wait_for_jobs=True).status, 'finished')

    def test_non_durable_context_only_memory_runs_without_advertising_adoption(self):
        from rsicomposer.harnesses.decision.feedback.context_memory import CallbackContextMemory
        memory = lambda _: MemoryAssembly(ResearchAssembly(context_memory=CallbackContextMemory(lambda query: '')))
        app, host = self.application(ControllerDefinition(lambda *_: Policy()), old_memory=memory,
                                     model=ModelRef('mock', 'fixture', 'v1'))
        self.assertNotIn('host.adopt_controller', {spec.name for spec in host.active.endpoint.action_space.describe()})
        with self.assertRaisesRegex(ValueError, 'archivable'):
            host.request_version(save_controller_version(app.artifacts, make_version()))
        self.assertEqual(app.run(wait_for_jobs=True).status, 'finished')

    def test_durable_adoption_does_not_require_portable_snapshot(self):
        class DurableOnly(Policy):
            def validate_snapshot(self):
                raise RuntimeError('not portable')

        app, host = self.application(ControllerDefinition(lambda *_: DurableOnly()), durable=True)
        version = self.queue(app, host)
        self.assertEqual(app.runner.step(app.config.objective).event['reason'], 'Controller adopted: ' + version.version_id)
        self.assertIsNone(app.recovery.intervention)
        with self.assertRaisesRegex(RuntimeError, 'not portable'):
            LocalSnapshotCapture(app).capture('unsupported')

    def test_durable_restore_rejects_unencodable_reconstructed_state(self):
        class Opaque(Policy):
            def state(self):
                return {'opaque': object()}

        candidate, harness = Opaque(), Harness()
        app, host = self.application(ControllerDefinition(lambda *_: candidate, harness=lambda *_: harness), durable=True)
        old = host.active
        saved = replacement_checkpoint(app, host)
        with self.assertRaisesRegex(TypeError, 'Unsupported checkpoint state type'):
            host.restore(saved)
        self.assertIs(host.active, old)
        self.assertIs(app.assembly.host_execution.endpoint, old.endpoint)
        self.assertEqual((harness.binds, harness.closes, candidate.closes, old.policy.closes), (1, 1, 1, 0))
        self.assertEqual(adopted_events(app), [])

    def test_feedback_listeners_release_on_rejected_build_retirement_and_failed_close(self):
        class Memory:
            def context(self, query): return ''
            def export(self): return {}
            def restore(self, state): pass

        class InstalledMemory(MemoryAssembly):
            def install(self, endpoint, **kwargs):
                service = super().install(endpoint, **kwargs)
                if fail_build.pop(0):
                    raise ValueError('failed after listener installation')
                return service

        fail_build = [False, True, False]
        definition = ControllerDefinition(lambda *_: Policy(), harness=lambda *_: Harness(),
            memory=lambda _: InstalledMemory(ResearchAssembly(context_memory=Memory())))
        app, host = self.application(definition, model=ModelRef('mock', 'fixture', 'v1'), durable=True)
        listeners = app.assembly.host_execution._receipt_listeners
        self.queue(app, host)
        host.advance()
        old = host.active
        self.assertEqual([owner for owner, _ in listeners], [old.endpoint.harness])
        self.queue(app, host, replace(make_version(), version_id='again'))
        result = app.runner.step(app.config.objective)
        self.assertIn('candidate construction failed', result.event['reason'])
        self.assertIsNone(host.pending)
        self.assertIsNone(app.recovery.intervention)
        self.assertEqual([owner for owner, _ in listeners], [old.endpoint.harness])
        self.queue(app, host, replace(make_version(), version_id='again'))
        with patch.object(old.endpoint.integration.learning, 'close', side_effect=RuntimeError('cleanup failure')):
            with self.assertRaisesRegex(ExceptionGroup, 'Retired Controller cleanup failed'):
                host.advance()
        self.assertEqual([owner for owner, _ in listeners], [host.active.endpoint.harness])
        host.stop_checkpoint()
        app.close()
        self.assertEqual(listeners, [])

    def test_adopted_versions_retire_local_scope_but_share_live_run_services(self):
        from rsicomposer.assembly import AssemblyContext

        contexts, retired = [], []

        def build(ctx, options):
            contexts.append(ctx)
            ctx.add_cleanup(lambda: retired.append(ctx.data_dir))
            return Policy()

        app, host = self.application(ControllerDefinition(build), model=ModelRef('mock', 'fixture', 'v1'))
        first = self.queue(app, host)
        host.advance()
        first_owner = host.active.context
        [previous] = contexts
        self.assertNotIsInstance(first_owner, AssemblyContext)
        event = previous.record('interpretation', {'message': 'local scope'})
        self.assertEqual(event.identity['controller_version'], first.version_id)
        artifact = previous.artifacts.put(b'shared evidence', kind='note')

        second = self.queue(app, host, replace(first, version_id='successor'))
        host.advance()
        current = contexts[1]
        self.assertNotEqual(previous.data_dir, current.data_dir)
        self.assertEqual(retired, [previous.data_dir])
        with self.assertRaisesRegex(RuntimeError, 'closed'):
            previous.record('stale')
        with self.assertRaisesRegex(RuntimeError, 'closed'):
            previous.add_cleanup(lambda: None)
        self.assertIs(previous.models, current.models)
        self.assertIs(previous.executor, current.executor)
        self.assertEqual(current.models.direct_attribution()['component'], 'controller')
        self.assertEqual(current.artifacts.read(artifact), b'shared evidence')
        self.assertIn(event.event_id, [row.event_id for row in current.evidence_view().events()])
        self.assertEqual(current.record('continuation').identity['controller_version'], second.version_id)
        app.close()
        self.assertEqual(retired, [previous.data_dir, current.data_dir])

    def test_separate_factories_transfer_cleanup_once_and_leave_borrowed_resources_live(self):
        policies, harnesses, order = [], [], []
        run_resource, first_resource = Policy(), Policy()

        class TrackedPolicy(Policy):
            def close(self):
                order.append(('policy', self))
                super().close()

        def policy(ctx, options):
            item = TrackedPolicy()
            policies.append(item)
            ctx.add_cleanup(item.close)
            ctx.add_owned_cleanup(run_resource.close)
            # Only the first factory acquires this resource. The second factory
            # cannot claim it even on a failed build before returning its Harness.
            ctx.add_owned_cleanup(first_resource.close)
            return item

        def harness(ctx, options):
            owning_policy = policies[-1]
            ctx.add_cleanup(lambda: order.append(('dependency', owning_policy)))
            ctx.add_owned_cleanup(owning_policy.close)
            if options.get('fail'):
                raise ValueError('failure before the candidate Harness was returned')
            item = Harness()
            harnesses.append(item)
            ctx.add_owned_cleanup(item.close)
            return item

        app, host = self.application(ControllerDefinition(policy, harness=harness))
        app.assembly.add_owned_cleanup(run_resource.close)
        version = self.queue(app, host)
        host.advance()
        self.queue(app, host, replace(version, version_id='failed',
            config={'name': 'replacement', 'config': {'fail': True}}))
        with self.assertRaisesRegex(ValueError, 'before the candidate Harness'):
            host.advance()
        self.assertEqual([p.closes for p in policies], [0, 1])
        self.assertEqual((run_resource.closes, first_resource.closes), (0, 0))
        app.close()
        self.assertEqual([p.closes for p in policies], [1, 1])
        self.assertEqual([h.closes for h in harnesses], [1])
        self.assertEqual((run_resource.closes, first_resource.closes), (1, 1))
        self.assertEqual(order, [(kind, policy) for policy in reversed(policies) for kind in ('dependency', 'policy')])

    def test_fresh_adapter_cannot_claim_borrowed_policy_on_rejection(self):
        from rsicomposer.harnesses.decision.public_policy import ResearchPolicyAdapter

        active = Policy()

        def factory(ctx, options):
            adapter = ResearchPolicyAdapter(active, None)
            ctx.add_owned_cleanup(adapter.close)
            return adapter

        app, host = self.application(ControllerDefinition(factory), old_policy=active)
        self.queue(app, host)
        with self.assertRaisesRegex(TypeError, 'independently owned policy'):
            host.advance()
        self.assertEqual(active.closes, 0)
        app.close()
        self.assertEqual(active.closes, 1)

    def test_shared_policy_is_rejected_without_closing_active_policy(self):
        old_policy = Policy()

        def borrowed_policy(context, options):
            context.add_owned_cleanup(old_policy.close)
            return old_policy

        app, host = self.application(ControllerDefinition(borrowed_policy), old_policy=old_policy)
        self.queue(app, host)
        with self.assertRaisesRegex(TypeError, 'independently owned policy'):
            host.advance()
        self.assertIs(host.active.policy, old_policy)
        self.assertEqual(old_policy.closes, 0)
        self.assertEqual(adopted_events(app), [])
        app.close()
        self.assertEqual(old_policy.closes, 1)

    def test_shared_harness_is_not_closed_if_factory_returns_or_raises(self):
        for rejects_in_factory in (False, True):
            with self.subTest(rejects_in_factory=rejects_in_factory):
                old_harness, new_policy = Harness(), Policy()

                def borrowed_harness(context, options):
                    context.add_owned_cleanup(old_harness.close)
                    if rejects_in_factory:
                        raise ValueError('factory rejected before returning an endpoint')
                    return old_harness

                definition = ControllerDefinition(lambda context, options: new_policy, harness=borrowed_harness)
                app, host = self.application(definition, old_harness=old_harness)
                self.queue(app, host)
                kind = ValueError if rejects_in_factory else TypeError
                message = 'factory rejected' if rejects_in_factory else 'independently owned harness'
                with self.assertRaisesRegex(kind, message):
                    host.advance()
                self.assertIs(host.active.endpoint.harness, old_harness)
                self.assertEqual((old_harness.binds, old_harness.closes, new_policy.closes), (1, 0, 1))
                self.assertEqual(adopted_events(app), [])
                app.close()
                self.assertEqual(old_harness.closes, 1)

    def test_explicit_endpoint_cannot_reuse_policy_with_new_harness(self):
        old_policy, new_harness = Policy(), Harness()

        def borrowed_policy_endpoint(context, execution, options):
            return build_harness_decision(context, execution, policy=old_policy, harness=new_harness)

        app, host = self.application(ControllerDefinition(borrowed_policy_endpoint, endpoint=True), old_policy=old_policy)
        self.queue(app, host)
        with self.assertRaisesRegex(TypeError, 'independently owned policy'):
            host.advance()
        self.assertEqual((old_policy.closes, new_harness.closes, new_harness.binds), (0, 1, 0))
        self.assertEqual(adopted_events(app), [])

    def test_endpoint_owns_independent_parts_once_across_retirement_and_close(self):
        from rsicomposer.harnesses.decision.public_policy import ResearchPolicyAdapter
        from rsicomposer.harnesses.defaults import build_default_decision

        class Both(Harness):
            def decide(self, window):
                return Decision('finish')

        for mode in ('independent', 'shared', 'native-adapter'):
            with self.subTest(mode=mode):
                pairs, native_closes = [], []

                def factory(context, lab, cfg):
                    if mode == 'native-adapter':
                        policy = Policy()
                        policy.model = None
                        loop = build_default_decision(context, lab, policy=policy)
                        harness = loop.harness
                        loop.policy = ResearchPolicyAdapter(policy, harness)
                        harness.add_cleanup(lambda: native_closes.append(harness))
                        harness.add_cleanup(policy.close)  # Existing native endpoint ownership.
                        context.add_owned_cleanup(loop.policy.close)
                    else:
                        harness = Both() if mode == 'shared' else Harness()
                        policy = harness if mode == 'shared' else Policy()
                        loop = build_harness_decision(context, lab, policy=policy, harness=harness)
                    context.add_owned_cleanup(harness.close)
                    pairs.append((policy, harness))
                    return loop

                app, host = self.application(ControllerDefinition(factory, endpoint=True))
                for ident in ('one', 'two'):
                    self.queue(app, host, replace(make_version(), version_id=ident))
                    host.advance()
                app.close()
                self.assertEqual([p.closes for p, _ in pairs], [1, 1])
                if mode == 'native-adapter':
                    self.assertEqual(native_closes, [h for _, h in pairs])
                else:
                    self.assertEqual([h.closes for _, h in pairs], [1, 1])

    def test_endpoint_close_failure_retains_policy_until_harness_cleanup_succeeds(self):
        policy, harness = Policy(), Harness(fail_close=1)

        def factory(ctx, lab, cfg):
            ctx.add_owned_cleanup(policy.close)
            ctx.add_owned_cleanup(harness.close)
            return build_harness_decision(ctx, lab, policy=policy, harness=harness)

        app, host = self.application(ControllerDefinition(factory, endpoint=True))
        self.queue(app, host)
        host.advance()
        with self.assertRaisesRegex(RuntimeError, 'harness close failed'):
            app.close()
        self.assertEqual((harness.closes, policy.closes), (1, 0))
        app.close()
        app.close()
        self.assertEqual((harness.closes, policy.closes), (2, 1))

    def test_endpoint_borrowing_harness_closes_only_its_fresh_policy(self):
        new_policy, old_harness = Policy(), Harness()

        def factory(context, lab, cfg):
            context.add_owned_cleanup(old_harness.close)
            return build_harness_decision(context, lab, policy=new_policy, harness=old_harness)

        app, host = self.application(ControllerDefinition(factory, endpoint=True), old_harness=old_harness)
        self.queue(app, host)
        with self.assertRaisesRegex(TypeError, 'independently owned harness'):
            host.advance()
        self.assertEqual((new_policy.closes, old_harness.closes), (1, 0))
        app.close()
        self.assertEqual((new_policy.closes, old_harness.closes), (1, 1))

    def test_failed_memory_install_preserves_owned_close_for_application_close(self):
        candidate, harness = Policy(), Harness(fail_close=1)

        class BrokenMemory:
            def install(self, endpoint, *, owner, path):
                raise RuntimeError('memory install failed')

        definition = ControllerDefinition(lambda context, options: candidate, memory=lambda options: BrokenMemory(),
                                          harness=lambda context, options: harness)
        app, host = self.application(definition)
        old = host.active
        self.queue(app, host)
        with self.assertRaisesRegex(RuntimeError, 'memory install failed') as failure:
            host.advance()
        self.assertTrue(any('harness close failed' in note for note in failure.exception.__notes__))
        self.assertEqual((harness.closes, candidate.closes), (1, 0))
        self.assertIs(host.active, old)
        self.assertEqual(adopted_events(app), [])
        with self.assertRaisesRegex(RuntimeError, 'cleanup must finish'):
            host.validate_snapshot()
        app.close()
        self.assertEqual((harness.closes, candidate.closes, old.policy.closes), (2, 1, 1))

    def test_retired_close_retry_never_rolls_back_adoption(self):
        for durable in (False, True):
            with self.subTest(durable=durable):
                old_harness, new_harness = Harness(fail_close=1), Harness()
                definition = ControllerDefinition(lambda context, options: Policy(), harness=lambda context, options: new_harness)
                app, host = self.application(definition, old_harness=old_harness, durable=durable)
                old = host.active
                version = self.queue(app, host)
                with self.assertRaisesRegex(ExceptionGroup, 'Retired Controller cleanup failed'):
                    app.runner.step(app.config.objective)
                self.assertEqual(host.version_id, version.version_id)
                self.assertIs(host.active.endpoint, app.assembly.host_execution.endpoint)
                self.assertIs(host.active.endpoint.harness, new_harness)
                self.assertIsNone(host.pending)
                self.assertEqual((old_harness.closes, old.policy.closes, new_harness.closes), (1, 0, 0))
                events = adopted_events(app)
                self.assertEqual(len(events), 1)
                self.assertTrue(app.runner.decision_making.pending_stop_checkpoint())
                if durable:
                    with self.assertRaisesRegex(RecoveryRequired, 'requires reconciliation'):
                        app.runner.step(app.config.objective)
                    self.assertEqual(old_harness.closes, 1)
                    app.recovery.reconcile(lambda live, incident: live.runner.decision_making.stop_checkpoint_boundary(),
                                           evidence='Adoption event confirmed; retry only the retained old owner')
                else:
                    self.assertIsNone(host.advance())
                self.assertEqual((old_harness.closes, old.policy.closes, new_harness.binds), (2, 1, 1))
                self.assertFalse(app.runner.decision_making.pending_stop_checkpoint())
                self.assertEqual(adopted_events(app), events)

    def test_restore_replacement_bind_failure_does_not_publish_it(self):
        def failing_bind():
            self.assertIs(host.active, old)
            self.assertIs(execution.endpoint, old.endpoint)
            raise RuntimeError('recovered bind failed')

        replacement = Harness(on_bind=failing_bind)
        definition = ControllerDefinition(lambda context, options: Policy(), harness=lambda context, options: replacement)
        app, host = self.application(definition)
        execution = app.assembly.host_execution
        old, old_version = host.active, host.version_id
        saved = replacement_checkpoint(app, host)
        execution.state.finished = True
        with self.assertRaisesRegex(RuntimeError, 'recovered bind failed'):
            host.restore(saved)
        self.assertIs(host.active, old)
        self.assertIs(execution.endpoint, old.endpoint)
        self.assertEqual((host.version_id, execution.controller_version), (old_version, old_version))
        self.assertTrue(execution.state.finished)
        self.assertEqual((old.endpoint.harness.closes, replacement.closes), (0, 1))
        self.assertEqual(adopted_events(app), [])

    def test_guard_rejects_adoption_before_build_and_after_binding(self):
        class StopDuringAdoption(BaseException):
            pass

        created = []

        def factory(context, options):
            harness = Harness()
            created.append(harness)
            return harness

        app, host = self.application(ControllerDefinition(lambda context, options: Policy(), harness=factory))
        endpoint = app.runner.decision_making
        old = host.active
        self.queue(app, host)
        pending = host.pending.copy()

        def stop_before_build():
            raise StopDuringAdoption('before build')

        with self.assertRaisesRegex(StopDuringAdoption, 'before build'):
            endpoint.checkpoint_boundary_guarded(stop_before_build)
        self.assertEqual(created, [])

        def stop_after_binding():
            if created and created[0].binds:
                raise StopDuringAdoption('after binding')

        with self.assertRaisesRegex(StopDuringAdoption, 'after binding'):
            endpoint.checkpoint_boundary_guarded(stop_after_binding)
        self.assertEqual([(harness.binds, harness.closes) for harness in created], [(1, 1)])
        self.assertIs(host.active, old)
        self.assertIs(app.assembly.host_execution.endpoint, old.endpoint)
        self.assertEqual(host.pending, pending)
        self.assertEqual(adopted_events(app), [])

        self.assertEqual(endpoint.checkpoint_boundary().decision_steps, 0)  # Legacy path still works.
        self.assertIs(host.active.endpoint.harness, created[1])

    def test_stop_checkpoint_publishes_only_already_committed_adoption(self):
        created = []

        def factory(context, options):
            policy = Policy()
            created.append(policy)
            return policy

        app, host = self.application(ControllerDefinition(factory))
        endpoint = app.runner.decision_making
        self.queue(app, host)
        self.assertFalse(endpoint.pending_stop_checkpoint())
        record = app.assembly.record

        def fail_publish(kind, payload):
            if kind == 'controller_adopted':
                raise RuntimeError('adoption event unavailable')
            return record(kind, payload)

        app.assembly.record = fail_publish
        try:
            with self.assertRaisesRegex(RuntimeError, 'event unavailable'):
                host.advance()
        finally:
            app.assembly.record = record
        active = host.active
        version = replace(make_version(), version_id='replacement-v2')
        self.queue(app, host, version)
        self.assertEqual(adopted_events(app), [])

        self.assertTrue(endpoint.pending_stop_checkpoint())
        self.assertIsNone(endpoint.stop_checkpoint_boundary())
        self.assertFalse(endpoint.pending_stop_checkpoint())
        self.assertIs(host.active, active)
        self.assertEqual(host.pending['version'], version)
        self.assertEqual(created, [active.policy])
        self.assertEqual([event['output']['controller_version'] for event in adopted_events(app)],
                         ['replacement-v1'])

    def test_swap_flush_delivers_existing_receipt_before_adopting(self):
        class OnceFailingHarness(Harness):
            fail = True

            def observe(self, event):
                if self.fail and event.identity.get('call_id') == 'original-call':
                    self.fail = False
                    raise RuntimeError('receipt observer unavailable')
                super().observe(event)

        app, host = self.application(ControllerDefinition(lambda context, options: Policy()),
                                     old_policy=Policy(invoke=True), old_harness=OnceFailingHarness())
        execution = app.assembly.host_execution
        lab = app.assembly.lab_execution.labs['work']
        step = lambda: app.runner.step(app.config.objective)
        first = step()
        if first.job is None:
            arrived = Event()
            execution.operation_scheduler._futures['original-call'].add_done_callback(lambda _: arrived.set())
            self.assertTrue(arrived.wait(2), 'fixture operation did not arrive')
            self.assertEqual(step().job, lab.job)
        lab.done = True
        with self.assertRaisesRegex(RuntimeError, 'receipt observer unavailable'):
            step()
        self.assertTrue(execution.pending_records())
        original = host.active
        version = self.queue(app, host)

        flushed = step()
        self.assertEqual(flushed.decision_steps, 0)
        self.assertEqual(flushed.event['operation_result']['call_id'], 'original-call')
        self.assertEqual(app.runner._outcomes['original-call'].receipt.call_id, 'original-call')
        self.assertIs(host.active, original)
        self.assertFalse(execution.pending_records())
        self.assertEqual(step().event['reason'], 'Controller adopted: ' + version.version_id)
        self.assertEqual(lab.calls, ['original-call'])

    def test_mixed_swap_settles_capacity_holder_but_defers_queued_handover(self):
        class MixedPolicy(Policy):
            def decide(self, window):
                self.decisions += 1
                return Decision('invoke', operations=tuple(OperationCall('work.start', {}, call_id=call_id)
                    for call_id in ('active-handover', 'queued-handover', 'drain')))

            def plan_swap(self, calls):
                return {call.call_id: 'drain' if call.call_id == 'drain' else 'handover' for call in calls}

        class MixedLab(Lab):
            def execute(self, call):
                if call.call_id == 'active-handover':
                    if not release.wait(2):
                        raise TimeoutError('fixture worker was not released')
                    return super().execute(call)
                self.calls.append(call.call_id)
                return OperationResult('finished toy work', (), False)

        release = Event()
        old_policy, old_harness, new_harness = MixedPolicy(), Harness(), Harness()
        definition = ControllerDefinition(lambda context, options: Policy(), harness=lambda context, options: new_harness)
        app, host = self.application(definition, old_policy=old_policy, old_harness=old_harness, lab_type=MixedLab)
        self.addCleanup(release.set)
        old_version = host.version_id
        lab = app.assembly.lab_execution.labs['work']
        scheduler = app.assembly.host_execution.operation_scheduler

        def wait_for_arrival(call_id):
            arrived = Event()
            scheduler._futures[call_id].add_done_callback(lambda future: arrived.set())
            self.assertTrue(arrived.wait(2), 'fixture operation did not arrive')

        self.assertEqual(app.runner.step(app.config.objective).kind, 'waiting')
        version = self.queue(app, host)
        pending = host.advance()
        self.assertEqual((pending.kind, pending.job, pending.decision_steps), ('waiting', None, 0))
        self.assertEqual(host.version_id, old_version)
        release.set()
        wait_for_arrival('active-handover')
        waiting = host.advance()
        self.assertEqual((waiting.kind, waiting.job, waiting.decision_steps), ('waiting', lab.job, 0))
        self.assertEqual(lab.calls, ['active-handover'])

        lab.done = True
        self.assertEqual(host.advance().operation_result.call_id, 'active-handover')
        drained = host.advance()
        if drained.operation_result is None:
            wait_for_arrival('drain')
            drained = host.advance()
        self.assertEqual((drained.operation_result.call_id, drained.decision_steps), ('drain', 0))
        self.assertEqual(lab.calls, ['active-handover', 'drain'])
        self.assertEqual(host.version_id, old_version)
        self.assertEqual(host.advance().decision_steps, 0)
        self.assertEqual(host.version_id, version.version_id)
        self.assertEqual(new_harness.bind_view['pending'], ('queued-handover',))
        self.assertEqual((old_policy.decisions, host.active.policy.decisions), (1, 0))

        step = host.active.endpoint.step
        handed = step(app.config.objective)
        if handed.operation_result is None:
            wait_for_arrival('queued-handover')
            handed = step(app.config.objective)
        self.assertEqual(handed.operation_result.call_id, 'queued-handover')
        self.assertEqual(lab.calls, ['active-handover', 'drain', 'queued-handover'])
        self.assertEqual((lab.cancels, old_policy.decisions), (0, 1))
        self.assertFalse(any(event.identity.get('call_id') == 'queued-handover' for event in old_harness.events))
        receipt, = [event for event in new_harness.events if event.identity.get('call_id') == 'queued-handover']
        self.assertEqual(receipt.identity['issuing_controller_version'], old_version)
        self.assertEqual(receipt.identity['controller_version'], version.version_id)

    def test_handover_keeps_job_and_attributes_receipt_to_both_versions(self):
        old_policy, old_harness, candidate_harness = Policy(invoke=True), Harness(), Harness()
        definition = ControllerDefinition(lambda context, options: Policy(), harness=lambda context, options: candidate_harness)
        app, host = self.application(definition, old_policy=old_policy, old_harness=old_harness)
        old_version = host.version_id
        lab = app.assembly.lab_execution.labs['work']
        execution = app.assembly.host_execution
        state = execution.state
        app.runner.step(app.config.objective)
        if state.operation_tasks['original-call']['job'] is None:
            execution.operation_scheduler._futures['original-call'].result(timeout=2)
            app.runner.step(app.config.objective)
        self.assertEqual(state.operation_tasks['original-call']['job'], lab.job)
        version = self.queue(app, host)
        self.assertEqual(host.pending['plan'], {'original-call': 'handover'})
        self.assertEqual(host.advance().decision_steps, 0)
        self.assertEqual(state.operation_tasks['original-call']['job'], lab.job)
        self.assertEqual(candidate_harness.bind_view['pending'], ('original-call',))
        lab.done = True
        result = host.active.endpoint.step(app.config.objective)
        self.assertEqual(result.operation_result.call_id, 'original-call')
        self.assertEqual(lab.calls, ['original-call'])
        self.assertEqual(lab.cancels, 0)
        self.assertFalse(state.operation_tasks)
        self.assertFalse(any(event.identity.get('call_id') == 'original-call' for event in old_harness.events))
        receipts = [event for event in candidate_harness.events if event.identity.get('call_id') == 'original-call']
        self.assertEqual(len(receipts), 1)
        self.assertEqual(receipts[0].identity['issuing_controller_version'], old_version)
        self.assertEqual(receipts[0].identity['controller_version'], version.version_id)


if __name__ == '__main__':
    unittest.main()
