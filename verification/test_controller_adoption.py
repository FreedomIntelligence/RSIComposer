"""Keyless Controller adoption regressions using real Application and Host ports."""
from dataclasses import replace
from pathlib import Path
from tempfile import TemporaryDirectory
from threading import Event
import unittest

from rsicomposer.application import Application, ApplicationConfig
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


def adopted_events(app):
    return [event for row in app.experience.retrieve(kinds=('controller_event',))
            if (event := row.details['event'])['kind'] == 'controller_adopted']


class ControllerAdoptionTests(unittest.TestCase):
    def application(self, definition, *, old_policy=None, old_harness=None, durable=False, lab_type=Lab):
        old_policy = old_policy if old_policy is not None else Policy()
        old_harness = old_harness if old_harness is not None else Harness()
        baseline = ControllerDefinition(lambda context, options: old_policy, harness=lambda context, options: old_harness)
        modules = MethodDefinition.from_factories(controller=baseline, labs={'work': lambda context, options: lab_type()},
                                                  version='1').factory({})
        modules = replace(modules, controller_catalog={**modules.controller_catalog, 'replacement': definition})
        root = Path(self.enterContext(TemporaryDirectory()))
        app = self.enterContext(Application(ApplicationConfig('test adoption', root, {}, {}),
                                            modules=modules, durable=durable))
        return app, app.assembly.controller_host

    def queue(self, app, host, version=None):
        version = version if version is not None else make_version()
        ref = save_controller_version(app.artifacts, version)
        self.assertEqual(host.request_version(ref).data['status'], 'pending')
        return version

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
        candidate, harness = Policy(fail_restore=True), Harness()
        app, host = self.application(ControllerDefinition(lambda context, options: candidate,
                                                           harness=lambda context, options: harness))
        old = host.active
        version = self.queue(app, host, make_version(policy_state={'decisions': 3}))
        with self.assertRaisesRegex(RuntimeError, 'candidate restore failed'):
            host.advance()
        self.assertIs(host.active, old)
        self.assertIs(app.assembly.host_execution.endpoint, old.endpoint)
        self.assertEqual(host.pending['version'], version)
        self.assertEqual((candidate.closes, harness.closes, harness.binds), (1, 1, 0))
        self.assertEqual(old.policy.closes, 0)
        self.assertEqual(adopted_events(app), [])

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
        saved = host.capture()
        saved['current_version'] = make_version()
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
