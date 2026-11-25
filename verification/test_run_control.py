"""CPU-only run control regressions through Application and real Host/Lab routing."""

from concurrent.futures import ThreadPoolExecutor
from dataclasses import replace
from pathlib import Path
from tempfile import TemporaryDirectory
from threading import Event
import unittest
from unittest.mock import patch

from rsicomposer.application import Application, ApplicationConfig
from rsicomposer.loops.contracts.coordination import LoopStepResult, OperationResult
from rsicomposer.loops.contracts.decision import Decision, OperationCall, OperationSpec
from rsicomposer.loops.contracts.jobs import JobOutcome, JobProgress, JobRef, JobStatus, OperationProgress
from rsicomposer.loops.contracts.lifecycle import ExecutionStatus
from rsicomposer.loops.contracts.profiles import SEALED
from rsicomposer.loops.decision_making.loop import DecisionMakingLoop
from rsicomposer.registry import Registry
from rsicomposer.runtime.orchestration.run_policy import ResearcherRunPolicy, TimeRunPolicy
from rsicomposer.runtime.orchestration.runner import Runner
from rsicomposer.runtime.state.controller_versions import save_controller_version
from rsicomposer.runtime.state.recovery import RecoveryRequired


class Clock:
    now = 0.

    def __call__(self):
        return self.now


class Policy:
    def __init__(self, calls=(), on_decide=None):
        self.calls = calls
        self.on_decide = on_decide
        self.turns = 0

    def decide(self, window):
        self.turns += 1
        if self.on_decide is not None:
            self.on_decide()
        if self.turns == 1 and self.calls:
            return Decision('invoke', operations=self.calls)
        return Decision('finish', reason='done')

    def plan_swap(self, calls):
        return {call.call_id: 'cancel' for call in calls}

    def state(self):
        return {'turns': self.turns}

    def restore(self, state):
        self.turns = state['turns']


class Lab:
    def __init__(self, *, blocking=False, backend=False, sealed=False):
        self.release = Event()
        self.started = Event()
        self.blocking, self.backend, self.sealed = blocking, backend, sealed
        self.calls, self.shutdowns, self.cancels = [], [], []
        self.job = JobRef('backend-work', 'fixture')
        self.settled = False

    def describe(self):
        return (OperationSpec('evaluate', 'Fixture operation', {
            'type': 'object', 'properties': {'split': {'type': 'string'}},
        }),)

    def split_roles(self):
        return {'heldout': SEALED} if self.sealed else {}

    def final_evaluations(self):
        return tuple(OperationCall('evaluate', {'split': 'heldout'}, call_id=call_id)
                     for call_id in ('final-one', 'final-two')) if self.sealed else ()

    def execute(self, call):
        self.calls.append(call.call_id)
        self.started.set()
        if self.blocking and not self.release.wait(5):
            raise TimeoutError('fixture worker was not released')
        if self.backend:
            return self.job
        return OperationResult('Lab result', (), True, data={'quality_metrics': {'score': 1.}})

    def poll_job(self, job):
        if not self.settled:
            return JobOutcome(job, JobStatus.RUNNING)
        if self.cancels:
            return JobOutcome(job, JobStatus.CANCELLED)
        return JobOutcome(job, JobStatus.SUCCEEDED, OperationResult('Finished normally', (), True))

    def cancel_job(self, job):
        self.cancels.append(job)
        return self.poll_job(job)

    def shutdown(self, mode):
        self.shutdowns.append(mode)
        return not self.backend or self.settled


class ImmediateExecutor(ThreadPoolExecutor):
    """Make the real worker's receipt visible before the first scheduling probe."""

    def submit(self, fn, *args, **kwargs):
        future = super().submit(fn, *args, **kwargs)
        future.result(timeout=2)
        return future


class LostSubmitReply(ImmediateExecutor):
    def submit(self, fn, *args, **kwargs):
        super().submit(fn, *args, **kwargs)
        raise RuntimeError('submit did not return its Future')


def operation(call_id='first'):
    return OperationCall('fixture.evaluate', {}, call_id=call_id)


class RunControlTest(unittest.TestCase):
    def setup_application(self, lab, policy, run_policy=None, *, timeout=1., capacity=1, durable=False):
        temporary = TemporaryDirectory()
        self.addCleanup(temporary.cleanup)
        self.candidates = []
        registry = Registry(builtins=False)
        registry.register_controller('original', lambda context, config: policy, version='1')

        def candidate(context, config):
            result = Policy()
            self.candidates.append(result)
            return result

        registry.register_controller('candidate', candidate, version='1')
        registry.register_lab('fixture', lambda context, config: lab, version='1')
        modules = registry.build({'controller': {'name': 'original'},
                                  'labs': {'fixture': {'name': 'fixture', 'config': {'max_concurrent': capacity}}}})
        self.app = Application(ApplicationConfig('fixture objective', Path(temporary.name), {}, {}),
                               modules=modules, run_policy=run_policy, cleanup_timeout_seconds=timeout,
                               durable=durable)
        self.addCleanup(self.app.close)
        self.addCleanup(lab.release.set)
        self.clock = Clock()
        self.app.runner._clock = self.clock
        return self.app

    def step(self):
        return self.app.runner.step(self.app.config.objective)

    def release_and_await_arrival(self, lab, call_id='first'):
        future = self.app.assembly.host_execution.operation_scheduler._futures[call_id]
        arrived = Event()
        future.add_done_callback(lambda _: arrived.set())
        lab.release.set()
        self.assertTrue(arrived.wait(2), 'fixture Future completion did not arrive')
        future.result()

    def queue_swap(self):
        host = self.app.assembly.controller_host
        version = host.version(version_id='candidate-version')
        version = replace(version, config={'name': 'candidate', 'config': {}})
        ref = save_controller_version(self.app.artifacts, version)
        self.assertEqual(host.request_version(ref).data['status'], 'pending')
        return host

    def assert_no_adoption(self, host, original):
        self.assertIs(host.active, original)
        self.assertIsNotNone(host.pending)
        self.assertEqual(self.candidates, [])
        self.assertFalse(any(record.details['event']['kind'] == 'controller_adopted'
                             for record in self.app.experience.retrieve(kinds=('controller_event',))))

    def test_managed_controller_jobs_follow_the_same_shutdown_contract(self):
        class ManagedPolicy:
            def __init__(self):
                self.job = JobRef('controller-job', 'fixture')
                self.ready = False
                self.turns = 0
                self.polled = []
                self.cancelled = []

            def decide(self, window):
                self.turns += 1
                return Decision('finish') if self.turns > 1 else Decision('wait', job=self.job)

            def can_poll_job(self, job):
                return job == self.job

            def poll_job(self, job):
                self.polled.append(job)
                if not self.ready:
                    return JobOutcome(job, JobStatus.RUNNING)
                if self.cancelled:
                    return JobOutcome(job, JobStatus.CANCELLED)
                return JobOutcome(job, JobStatus.SUCCEEDED, result='ready')

            def cancel_job(self, job):
                self.cancelled.append(job)
                # This hook accepts the request; only polling knows when work exits.
                return JobOutcome(job, JobStatus.RUNNING)

        with self.subTest(mode='ordinary'):
            lab, policy = Lab(), ManagedPolicy()
            app = self.setup_application(lab, policy)
            try:
                self.assertEqual(self.step().job, policy.job)
                policy.ready = True
                ready = self.step()
                self.assertEqual((ready.kind, ready.decision_steps), ('progressed', 0))
                self.assertIsNone(ready.event.get('operation_result'))
                self.assertEqual(self.step().kind, 'finished')
                self.assertEqual(policy.turns, 2)
                self.assertEqual(lab.calls, [])
            finally:
                app.close()

        for mode, terminal in (('drain', True), ('cancel', True), ('cancel', False)):
            with self.subTest(mode=mode, terminal=terminal):
                lab, policy = Lab(), ManagedPolicy()
                app = self.setup_application(lab, policy, TimeRunPolicy(1, shutdown=mode))
                try:
                    initial = self.step()
                    self.assertEqual((initial.kind, initial.job, initial.decision_steps), ('waiting', policy.job, 1))
                    self.clock.now = 2.
                    waiting = self.step()
                    self.assertEqual((waiting.kind, waiting.job, waiting.decision_steps), ('waiting', policy.job, 0))
                    self.assertFalse(waiting.event['run_stop']['shutdown_complete'])
                    self.assertEqual(waiting.event['run_stop']['unresolved_research_job'],
                                     {'job_id': policy.job.job_id, 'backend': policy.job.backend})
                    self.assertEqual(policy.polled, [policy.job] if mode == 'drain' else [])
                    self.assertEqual(policy.cancelled, [policy.job] if mode == 'cancel' else [])
                    if mode == 'cancel':
                        self.clock.now = 2.5
                        still_running = self.step()
                        self.assertEqual((still_running.kind, still_running.job), ('waiting', policy.job))
                        self.assertEqual(policy.cancelled, [policy.job])
                        self.assertEqual(policy.polled, [policy.job])
                    policy.ready = terminal
                    if not terminal:
                        self.clock.now = 3.
                    stopped = self.step()
                    self.assertEqual((stopped.kind, stopped.decision_steps), ('finished', 0))
                    stop = stopped.event['run_stop']
                    self.assertEqual(stop['shutdown_complete'], terminal)
                    self.assertEqual(stop['cleanup_timed_out'], not terminal)
                    self.assertEqual(stop['unresolved_research_job'], None if terminal else
                                     {'job_id': policy.job.job_id, 'backend': policy.job.backend})
                    if mode == 'cancel':
                        self.assertEqual(policy.cancelled, [policy.job])
                        self.assertEqual(policy.polled, [policy.job] * (2 if terminal else 1))
                    self.assertEqual(policy.turns, 1)
                    self.assertEqual(lab.calls, [])
                finally:
                    app.close()

    def test_resume_polls_acknowledged_controller_cancellation_without_resending(self):
        job = JobRef('durable-controller-job', 'fixture')
        calls = []
        ready = False

        class ManagedPolicy(Policy):
            def decide(self, window):
                self.turns += 1
                return Decision('wait', job=job)

            def can_poll_job(self, ref):
                return ref == job

            def cancel_job(self, ref):
                calls.append(('cancel', ref))
                return JobOutcome(ref, JobStatus.RUNNING)

            def poll_job(self, ref):
                calls.append(('poll', ref))
                return JobOutcome(ref, JobStatus.CANCELLED if ready else JobStatus.RUNNING)

        class DurableLab(Lab):
            def state(self):
                return {}

            def restore(self, state):
                if state != {}:
                    raise ValueError('Unexpected Lab state')

        lab = DurableLab()
        registry = Registry(builtins=False)
        registry.register_controller('fixture', lambda context, config: ManagedPolicy(), version='1')
        registry.register_lab('fixture', lambda context, config: lab, version='1')
        modules = registry.build({'controller': {'name': 'fixture'}, 'labs': {'fixture': {'name': 'fixture'}}})
        with TemporaryDirectory() as temp:
            config = ApplicationConfig('fixture objective', Path(temp), {}, {})
            clock = Clock()
            with Application(config, modules=modules, durable=True, cleanup_timeout_seconds=60,
                             run_policy=TimeRunPolicy(1)) as app:
                app.runner._clock = clock
                self.assertEqual(app.runner.step(config.objective).job, job)
                clock.now = 2.
                waiting = app.runner.step(config.objective)
                self.assertEqual(waiting.job, job)
                self.assertFalse(waiting.event['run_stop']['shutdown_complete'])
                self.assertEqual(calls, [('cancel', job)])

            ready = True
            with Application.resume(config.data_dir, modules=modules) as resumed:
                result = resumed.runner.step(config.objective)
                self.assertTrue(result.event['run_stop']['shutdown_complete'])
                self.assertEqual(result.decision_steps, 0)
                self.assertIsNone(result.event['run_stop']['unresolved_research_job'])
                self.assertEqual(calls, [('cancel', job), ('poll', job)])
                self.assertEqual(lab.calls, [])

    def test_local_future_yields_then_stops_and_delivers_real_receipt(self):
        lab = Lab(blocking=True)
        policy = Policy((operation(),))
        app = self.setup_application(lab, policy, TimeRunPolicy(1.))
        started = self.step()
        self.assertEqual((started.kind, started.decision_steps, started.job), ('waiting', 1, None))
        self.assertTrue(lab.started.wait(2))
        self.clock.now = 2.
        stopping = self.step()
        self.assertEqual((stopping.kind, stopping.decision_steps), ('waiting', 0))
        self.assertEqual(stopping.event['run_stop']['reason'], 'budget_exhausted')
        self.assertFalse(stopping.event['run_stop']['shutdown_complete'])
        self.assertTrue(stopping.event['run_stop']['unresolved_operations'])
        self.assertEqual(app.assembly.host_execution.state.operation_results, [])
        self.release_and_await_arrival(lab)
        settled = self.step()
        self.assertEqual((settled.kind, settled.decision_steps), ('progressed', 0))
        self.assertEqual(settled.event['operation_result']['status'], ExecutionStatus.COMPLETED)
        final = self.step()
        self.assertEqual(final.kind, 'finished')
        self.assertTrue(final.event['run_stop']['shutdown_complete'])
        self.assertEqual(policy.turns, 1)
        self.assertEqual(lab.calls, ['first'])

    def test_timeout_keeps_running_owner_and_never_starts_queued_call(self):
        lab = Lab(blocking=True)
        policy = Policy((operation(), operation('queued')))
        app = self.setup_application(lab, policy, TimeRunPolicy(1., shutdown='drain'))
        self.assertEqual(self.step().kind, 'waiting')
        self.assertTrue(lab.started.wait(2))
        self.clock.now = 2.
        stopped_queue = self.step()
        self.assertEqual(stopped_queue.event['operation_result']['call_id'], 'queued')
        self.assertEqual(stopped_queue.event['operation_result']['status'], ExecutionStatus.CANCELLED)
        self.clock.now = 3.
        final = self.step()
        self.assertEqual(final.kind, 'finished')
        stop = final.event['run_stop']
        self.assertTrue(stop['cleanup_timed_out'])
        self.assertTrue(stop['unresolved_operations'])
        self.assertFalse(stop['shutdown_complete'])
        self.assertEqual(lab.shutdowns, ['drain'])
        self.assertEqual(lab.calls, ['first'])
        self.assertEqual(policy.turns, 1)
        self.assertEqual([row['operation'].call_id for row in app.assembly.host_execution.in_flight_operations()], ['first'])
        self.assertIsNotNone(app.store)

    def test_run_waits_for_local_future_at_decision_cap_without_sampling(self):
        lab = Lab(blocking=True)
        policy = Policy((operation(),))
        app = self.setup_application(lab, policy)
        delays = []

        def release_on_wait(delay):
            delays.append(delay)
            self.assertTrue(lab.started.wait(2))
            self.release_and_await_arrival(lab)

        with patch('rsicomposer.runtime.orchestration.runner.sleep', side_effect=release_on_wait):
            result = app.run(wait_for_jobs=True, max_decision_steps=1)
        self.assertEqual((result.status, result.decision_steps, policy.turns), ('step_limit', 1, 1))
        self.assertEqual(delays, [.1])
        self.assertEqual([event['operation_result']['call_id'] for event in result.events
                          if event.get('operation_result')], ['first'])
        self.assertFalse(app.runner.busy)

    def test_cap_settles_entire_batch_without_applying_pending_swap(self):
        lab = Lab(blocking=True)
        policy = Policy((operation(), operation('second')))
        app = self.setup_application(lab, policy)
        host = app.assembly.controller_host
        original = host.active
        waited = []

        def release_on_wait():
            futures = app.assembly.host_execution.operation_scheduler._futures
            call_id = next(iter(futures))
            waited.append(call_id)
            self.release_and_await_arrival(lab, call_id)
            if call_id == 'first':
                self.queue_swap()

        with patch.object(app.runner, '_wait_for_progress', side_effect=release_on_wait):
            result = app.run(wait_for_jobs=True, max_decision_steps=1)
        self.assertEqual((result.status, result.decision_steps, policy.turns), ('step_limit', 1, 1))
        self.assertEqual(waited[0], 'first')
        self.assertEqual(lab.calls, ['first', 'second'])
        self.assertEqual([event['operation_result']['call_id'] for event in result.events
                          if event.get('operation_result')], ['first', 'second'])
        self.assertFalse(app.runner.busy)
        self.assert_no_adoption(host, original)

    def test_fast_first_receipt_settles_batch_only_with_wait_enabled(self):
        for wait in (False, True):
            with self.subTest(wait_for_jobs=wait):
                lab = Lab()
                policy = Policy((operation(), operation('second'), operation('third')))
                app = self.setup_application(lab, policy, capacity=2)
                with (patch('rsicomposer.host.execution.ThreadPoolExecutor', ImmediateExecutor),
                      patch.object(app.runner, '_wait_for_progress') as pause):
                    result = app.run(wait_for_jobs=wait, max_decision_steps=1)
                expected = ['first', 'second', 'third'] if wait else ['first']
                pause.assert_not_called()
                self.assertEqual((result.status, result.decision_steps, policy.turns), ('step_limit', 1, 1))
                self.assertEqual(lab.calls, ['first', 'second', 'third'] if wait else ['first', 'second'])
                self.assertEqual([event['operation_result']['call_id'] for event in result.events
                                  if event.get('operation_result')], expected)
                in_flight = app.assembly.host_execution.in_flight_operations()
                self.assertEqual({row['operation'].call_id: row['status'] for row in in_flight},
                                 {} if wait else {'second': 'running', 'third': 'queued'})

    def test_cap_settlement_is_committed_as_a_durable_tick(self):
        class DurableLab(Lab):
            def state(self):
                return {'calls': list(self.calls)}

            def restore(self, state):
                self.calls = state['calls']

        lab = DurableLab()
        policy = Policy((operation(), operation('second')))
        app = self.setup_application(lab, policy, durable=True)
        with patch('rsicomposer.host.execution.ThreadPoolExecutor', ImmediateExecutor):
            result = app.run(wait_for_jobs=True, max_decision_steps=1)
        self.assertEqual((result.status, result.decision_steps, policy.turns), ('step_limit', 1, 1))
        checkpoint = app.recovery.read(app.config.data_dir)
        self.assertEqual(checkpoint['generation'], 2)
        self.assertEqual(checkpoint['in_flight'], [])
        self.assertIsNone(checkpoint['intervention'])
        self.assertEqual(lab.calls, ['first', 'second'])

    def test_cap_keeps_polling_after_progress_before_next_receipt(self):
        class ProgressLab(Lab):
            def execute(self, call):
                if call.call_id == 'first':
                    return super().execute(call)
                self.calls.append(call.call_id)
                return OperationResult('Lab result', (), True)

            def job_events(self, job, *, after):
                return (OperationProgress('first', job, 1, 'progress', JobProgress(step=1)),) if after == 0 else ()

        lab = ProgressLab(backend=True)
        policy = Policy((operation(), operation('second')))
        app = self.setup_application(lab, policy)
        with (patch('rsicomposer.host.execution.ThreadPoolExecutor', ImmediateExecutor),
              patch.object(app.runner, '_wait_for_progress', side_effect=lambda: setattr(lab, 'settled', True)) as pause):
            result = app.run(wait_for_jobs=True, max_decision_steps=1)
        self.assertEqual((result.status, result.decision_steps, policy.turns), ('step_limit', 1, 1))
        pause.assert_called_once_with()
        self.assertEqual([event.get('reason') for event in result.events].count('Lab job progress'), 1)
        self.assertEqual([event['operation_result']['call_id'] for event in result.events
                          if event.get('operation_result')], ['first', 'second'])
        self.assertFalse(app.runner.busy)

    def test_cap_returns_external_or_policy_held_job_without_poll_spin(self):
        class OpaqueLab(Lab):
            poll_job = None

        class CheckpointLab(Lab):
            def job_events(self, job, *, after):
                return (OperationProgress('second', job, 1, 'checkpoint', JobProgress(step=1),
                                          self.artifact),) if after == 0 else ()

        for lab_type in (OpaqueLab, CheckpointLab):
            with self.subTest(lab_type=lab_type.__name__):
                lab = lab_type(backend=True)
                real_execute = lab.execute

                def execute(call):
                    if call.call_id == 'first':
                        lab.calls.append(call.call_id)
                        return OperationResult('Lab result', (), True)
                    return real_execute(call)

                lab.execute = execute
                policy = Policy((operation(), operation('second')))
                app = self.setup_application(lab, policy)
                lab.artifact = app.artifacts.put(b'checkpoint fixture', kind='checkpoint')
                with (patch('rsicomposer.host.execution.ThreadPoolExecutor', ImmediateExecutor),
                      patch.object(app.runner, '_wait_for_progress', side_effect=AssertionError('unmanaged wait'))):
                    result = app.run(wait_for_jobs=True, max_decision_steps=1)
                self.assertEqual((result.status, result.decision_steps, result.waiting_job, policy.turns),
                                 ('waiting', 1, lab.job, 1))
                self.assertEqual(lab.calls, ['first', 'second'])
                self.assertTrue(app.runner.busy)
                self.assertEqual(lab.cancels, [])

    def test_cap_does_not_step_custom_owner_without_settlement_capability(self):
        class CustomOwner:
            calls = 0

            def step(self, objective, action_space):
                self.calls += 1
                return LoopStepResult('progressed')

            def in_flight_operations(self):
                return ('opaque operation',)

        owner = CustomOwner()
        endpoint = DecisionMakingLoop(None, None, object(), host_execution=owner)
        result = Runner(endpoint).run('objective', 1, max_decision_steps=1, wait_for_jobs=True)
        self.assertEqual((result.status, result.decision_steps, owner.calls), ('step_limit', 1, 1))

    def test_held_job_does_not_hide_independent_pollable_job_at_cap(self):
        class MixedLab(Lab):
            def execute(self, call):
                self.calls.append(call.call_id)
                return JobRef(call.call_id, 'fixture')

            def poll_job(self, job):
                if job.job_id == 'second' and self.settled:
                    return JobOutcome(job, JobStatus.SUCCEEDED, OperationResult('Second settled', (), True))
                return JobOutcome(job, JobStatus.RUNNING)

            def job_events(self, job, *, after):
                if job.job_id == 'first' and after == 0:
                    return (OperationProgress('first', job, 1, 'checkpoint', JobProgress(step=1), self.artifact),)
                return ()

        lab = MixedLab()
        policy = Policy((operation(), operation('second')))
        app = self.setup_application(lab, policy, capacity=2)
        lab.artifact = app.artifacts.put(b'checkpoint fixture', kind='checkpoint')

        def progress_on_wait():
            self.assertFalse(lab.settled, 'ordinary polling must not spin on the held job')
            lab.settled = True

        with (patch('rsicomposer.host.execution.ThreadPoolExecutor', ImmediateExecutor),
              patch.object(app.runner, '_wait_for_progress', side_effect=progress_on_wait) as pause):
            result = app.run(wait_for_jobs=True, max_decision_steps=1)
        self.assertEqual((result.status, result.decision_steps, result.waiting_job, policy.turns),
                         ('waiting', 1, JobRef('first', 'fixture'), 1))
        self.assertEqual([event['operation_result']['call_id'] for event in result.events
                          if event.get('operation_result')], ['second'])
        self.assertEqual([row['operation'].call_id for row in app.assembly.host_execution.in_flight_operations()],
                         ['first'])
        pause.assert_called_once_with()

    def test_hard_expiry_after_cap_receipt_stops_before_queued_launch(self):
        lab = Lab()
        policy = Policy((operation(), operation('second')))
        app = self.setup_application(lab, policy, TimeRunPolicy(1.))

        class AdvanceClockAfterFirstReceipt:
            def on_step(observer, result):
                if (result.event.get('operation_result') or {}).get('call_id') == 'first':
                    self.clock.now = 2.

        app.runner.add_observer(AdvanceClockAfterFirstReceipt())
        with patch('rsicomposer.host.execution.ThreadPoolExecutor', ImmediateExecutor):
            result = app.run(wait_for_jobs=True, max_decision_steps=1)
        self.assertEqual((result.status, result.stop_reason, result.decision_steps, policy.turns),
                         ('finished', 'budget_exhausted', 1, 1))
        self.assertEqual(lab.calls, ['first'])
        self.assertEqual([(receipt['call_id'], receipt['status']) for event in result.events
                          if (receipt := event.get('operation_result'))],
                         [('first', ExecutionStatus.COMPLETED), ('second', ExecutionStatus.CANCELLED)])
        self.assertTrue(result.shutdown_complete)
        self.assertFalse(app.runner.busy)

    def test_run_manages_local_submission_and_later_pollable_job(self):
        lab = Lab(blocking=True, backend=True)
        policy = Policy((operation(),))
        app = self.setup_application(lab, policy)
        delays = []

        def progress_on_wait(delay):
            delays.append(delay)
            if len(delays) == 1:
                self.release_and_await_arrival(lab)
            elif len(delays) == 2:
                lab.settled = True
            else:
                self.fail('managed JobRef did not settle on the next poll')

        with patch('rsicomposer.runtime.orchestration.runner.sleep', side_effect=progress_on_wait):
            result = app.run(wait_for_jobs=True)
        self.assertEqual((result.status, result.decision_steps, policy.turns), ('finished', 2, 2))
        self.assertEqual(delays, [.1, .1])
        self.assertEqual([event['operation_result']['call_id'] for event in result.events
                          if event.get('operation_result')], ['first'])

    def test_unmanaged_no_job_wait_still_returns_to_caller(self):
        class Endpoint:
            def step(self, objective):
                return LoopStepResult('waiting', reason='Waiting for caller input')

        with patch('rsicomposer.runtime.orchestration.runner.sleep') as sleep:
            result = Runner(Endpoint()).run('objective', 1, wait_for_jobs=True)
        self.assertEqual((result.status, result.decision_steps, result.waiting_job), ('waiting', 1, None))
        sleep.assert_not_called()

    def test_blocked_future_timeout_returns_from_managed_run(self):
        lab = Lab(blocking=True)
        policy = Policy((operation(),))
        app = self.setup_application(lab, policy, TimeRunPolicy(.2), timeout=.2)
        with patch('rsicomposer.runtime.orchestration.runner.sleep', side_effect=self.advance_clock) as sleeps:
            result = app.run(wait_for_jobs=True, max_decision_steps=1)
        self.assertEqual((result.status, result.stop_reason, result.shutdown_complete, result.decision_steps),
                         ('finished', 'budget_exhausted', False, 1))
        self.assertEqual(sleeps.call_count, 4)
        self.assertEqual(policy.turns, 1)
        self.assertEqual(len(result.events), 2)  # admission + terminal; idle probes are not events
        self.assertTrue(result.events[-1]['run_stop']['cleanup_timed_out'])

    def advance_clock(self, delay):
        self.clock.now += delay

    def test_synchronous_shutdown_overrun_does_not_retry_after_deadline(self):
        lab = Lab()
        calls = []

        def shutdown(mode):
            calls.append(mode)
            self.clock.now = 4.  # An already entered callback can overrun the deadline.
            return False

        lab.shutdown = shutdown
        self.setup_application(lab, Policy(), TimeRunPolicy(1.))
        self.assertEqual(self.step().kind, 'waiting')
        self.clock.now = 2.
        result = self.step()
        self.assertEqual(result.kind, 'finished')
        self.assertEqual(calls, ['cancel'])
        self.assertTrue(result.event['run_stop']['unresolved_execution'])
        self.assertTrue(result.event['run_stop']['cleanup_timed_out'])
        self.assertFalse(result.event['run_stop']['shutdown_complete'])

    def test_hard_time_checks_before_each_sealed_call(self):
        lab = Lab(sealed=True)
        self.setup_application(lab, Policy(), TimeRunPolicy(1.))
        first = self.step()
        self.assertEqual((first.event['phase'], first.decision_steps), ('host_finish', 1))
        self.assertEqual(first.event['receipt']['summary'], 'Sealed evaluation receipt')
        self.clock.now = 2.
        final = self.step()
        self.assertEqual(final.kind, 'finished')
        self.assertEqual(final.event['run_stop']['reason'], 'budget_exhausted')
        self.assertEqual(lab.calls, ['final-one'])

    def test_normal_researcher_finish_still_runs_all_sealed_calls(self):
        lab = Lab(sealed=True)
        app = self.setup_application(lab, Policy(), ResearcherRunPolicy())
        result = app.run(wait_for_jobs=True)
        self.assertEqual((result.status, result.stop_reason, result.decision_steps),
                         ('finished', 'researcher_finished', 1))
        self.assertEqual(lab.calls, ['final-one', 'final-two'])

    def test_deadline_crossed_inside_policy_blocks_new_operation(self):
        lab = Lab()
        policy = Policy((operation(),), on_decide=lambda: self.advance_clock(2.))
        self.setup_application(lab, policy, TimeRunPolicy(1.))
        final = self.step()
        self.assertEqual((final.kind, final.decision_steps), ('finished', 1))
        self.assertEqual(final.event['run_stop']['reason'], 'budget_exhausted')
        self.assertEqual(lab.calls, [])
        self.assertEqual(policy.turns, 1)

    def test_hard_stop_before_pending_swap_construction(self):
        lab = Lab(sealed=True)
        self.setup_application(lab, Policy(), TimeRunPolicy(1.))
        self.assertEqual(self.step().event['phase'], 'host_finish')
        host = self.queue_swap()
        original = host.active
        self.clock.now = 2.
        final = self.step()
        self.assertEqual(final.kind, 'finished')
        self.assertTrue(final.event['run_stop']['shutdown_complete'])
        self.assert_no_adoption(host, original)

    def test_deadline_crossed_during_swap_drain_blocks_queued_dispatch(self):
        class DrainPolicy(Policy):
            def plan_swap(self, calls):
                return {call.call_id: 'drain' for call in calls}

        lab = Lab()
        policy = DrainPolicy((operation(), operation('second')))
        self.setup_application(lab, policy, TimeRunPolicy(1.))
        with patch('rsicomposer.host.execution.ThreadPoolExecutor', ImmediateExecutor):
            first = self.step()
        self.assertEqual(first.event['operation_result']['call_id'], 'first')
        host = self.queue_swap()
        original = host.active

        def cross_deadline(*args, call_id):
            self.assertEqual(call_id, 'second')
            self.clock.now = 2.

        with patch('rsicomposer.host.execution.report_stage_start', side_effect=cross_deadline) as report:
            stopped = self.step()
        report.assert_called_once()
        self.assertEqual(stopped.decision_steps, 0)
        self.assertEqual(stopped.event['run_stop']['reason'], 'budget_exhausted')
        receipt = stopped.event['operation_result']
        self.assertEqual((receipt['call_id'], receipt['status']), ('second', ExecutionStatus.CANCELLED))
        self.assertEqual(lab.calls, ['first'])
        self.assertEqual(policy.turns, 1)
        self.assert_no_adoption(host, original)

    def test_launch_preparation_failure_requeues_only_unstarted_calls(self):
        lab = Lab()
        policy = Policy(tuple(operation(call_id) for call_id in ('first', 'second', 'third')))
        app = self.setup_application(lab, policy, capacity=3)
        execution = app.assembly.host_execution

        def fail_second(*args, call_id):
            if call_id == 'second':
                raise RuntimeError('stage report unavailable')

        with (patch('rsicomposer.host.execution.ThreadPoolExecutor', ImmediateExecutor),
              patch('rsicomposer.host.execution.report_stage_start', side_effect=fail_second),
              self.assertRaisesRegex(RuntimeError, 'stage report unavailable')):
            self.step()

        self.assertEqual({row['operation'].call_id: row['status'] for row in execution.in_flight_operations()},
                         {'first': 'running', 'second': 'queued', 'third': 'queued'})
        self.assertEqual(next(iter(execution.state.decision_records.values()))['status'], 'executing')
        self.assertEqual(lab.calls, ['first'])
        settle = app.runner.decision_making.settle_admitted_operations
        self.assertEqual([settle(lambda: None).operation_result.call_id for _ in range(3)],
                         ['first', 'second', 'third'])
        self.assertEqual(lab.calls, ['first', 'second', 'third'])
        self.assertEqual(policy.turns, 1)

    def test_unknown_submission_waits_without_retry_and_bounds_hard_stop(self):
        lab = Lab()
        policy = Policy((operation(), operation('second')))
        app = self.setup_application(lab, policy, TimeRunPolicy(1.), timeout=.2)
        with (patch('rsicomposer.host.execution.ThreadPoolExecutor', LostSubmitReply),
              self.assertRaisesRegex(RuntimeError, 'submit did not return its Future')):
            self.step()
        with patch('rsicomposer.runtime.orchestration.runner.sleep', side_effect=AssertionError('unmanaged wait')):
            pending = app.run(wait_for_jobs=True)
        self.assertEqual((pending.status, pending.decision_steps), ('waiting', 0))
        self.assertIn('dispatch outcome is unknown', pending.events[-1]['reason'])

        self.clock.now = 2.
        stopped_queue = self.step()
        receipt = stopped_queue.event['operation_result']
        self.assertEqual((receipt['call_id'], receipt['status']), ('second', ExecutionStatus.CANCELLED))
        with patch('rsicomposer.runtime.orchestration.runner.sleep', side_effect=self.advance_clock):
            stopped = app.run(wait_for_jobs=True)
        self.assertEqual((stopped.status, stopped.stop_reason, stopped.shutdown_complete),
                         ('finished', 'budget_exhausted', False))
        self.assertIn('dispatch outcome is unknown', stopped.events[-1]['reason'])
        self.assertTrue(stopped.events[-1]['run_stop']['cleanup_timed_out'])
        self.assertTrue(stopped.events[-1]['run_stop']['unresolved_operations'])
        self.assertEqual(lab.calls, ['first'])
        self.assertEqual(policy.turns, 1)

    def test_raised_submit_retains_durable_launch_uncertainty(self):
        class DurableLab(Lab):
            def state(self):
                return {'calls': list(self.calls)}

            def restore(self, state):
                self.calls = state['calls']

        lab = DurableLab()
        policy = Policy((operation(), operation('second')))
        app = self.setup_application(lab, policy, durable=True)
        with (patch('rsicomposer.host.execution.ThreadPoolExecutor', LostSubmitReply),
              self.assertRaisesRegex(RuntimeError, 'submit did not return its Future')):
            self.step()

        checkpoint = app.recovery.read(app.config.data_dir)
        self.assertEqual({row['operation'].call_id: row['status'] for row in checkpoint['in_flight']},
                         {'first': 'running', 'second': 'queued'})
        self.assertEqual(checkpoint['intervention']['reason'], 'interrupted_mutation')
        with self.assertRaises(RecoveryRequired):
            self.step()
        self.assertEqual(lab.calls, ['first'])
        self.assertEqual(policy.turns, 1)

    def test_deadline_crossed_during_swap_preparation_keeps_original(self):
        lab = Lab()
        self.setup_application(lab, Policy(), TimeRunPolicy(1.))
        self.assertEqual(self.step().kind, 'waiting')
        host = self.queue_swap()
        original = host.active
        build = host.build_controller
        built = []

        def cross_deadline(version):
            candidate = build(version)
            built.append(candidate)
            self.clock.now = 2.
            return candidate

        host.build_controller = cross_deadline
        result = self.step()
        self.assertEqual((result.kind, result.decision_steps), ('finished', 0))
        self.assertEqual(result.event['run_stop']['reason'], 'budget_exhausted')
        self.assertIs(host.active, original)
        self.assertIsNotNone(host.pending)
        self.assertEqual(len(built), 1)
        self.assertTrue(built[0].context._closed)
        self.assertFalse(any(record.details['event']['kind'] == 'controller_adopted'
                             for record in self.app.experience.retrieve(kinds=('controller_event',))))

    def test_expiry_does_not_report_unclosed_retired_owner_as_quiescent(self):
        lab = Lab()
        self.setup_application(lab, Policy(), TimeRunPolicy(1.))
        self.assertEqual(self.step().kind, 'waiting')
        host = self.queue_swap()
        original = host.active
        close = original.close
        attempts = []

        def fail_once():
            attempts.append(True)
            if len(attempts) == 1:
                raise RuntimeError('retired resource still held')
            close()

        original.close = fail_once
        with self.assertRaises(ExceptionGroup):
            host.advance()
        self.assertIsNot(host.active, original)
        self.assertTrue(self.app.runner.decision_making.pending_stop_checkpoint())

        def overrun_but_confirm_lab_shutdown(mode):
            self.clock.now = 4.
            return True

        lab.shutdown = overrun_but_confirm_lab_shutdown
        self.clock.now = 2.
        result = self.step()
        self.assertEqual(result.kind, 'finished')
        stop = result.event['run_stop']
        self.assertFalse(stop['unresolved_execution'])
        self.assertTrue(stop['unresolved_checkpoint'])
        self.assertTrue(stop['cleanup_timed_out'])
        self.assertFalse(stop['shutdown_complete'])
        self.assertEqual(len(attempts), 1)
        self.app.close()
        self.assertEqual(len(attempts), 2)

    def test_run_drain_overrides_swap_cancel_for_live_job(self):
        lab = Lab(backend=True, blocking=True)
        policy = Policy((operation(),))
        self.setup_application(lab, policy, TimeRunPolicy(1., shutdown='drain'))
        self.assertEqual(self.step().kind, 'waiting')
        self.assertTrue(lab.started.wait(2))
        self.release_and_await_arrival(lab)
        self.assertEqual(self.step().job, lab.job)
        host = self.queue_swap()
        original = host.active
        self.assertEqual(host.pending['plan'], {'first': 'cancel'})
        self.clock.now = 2.
        waiting = self.step()
        self.assertEqual((waiting.kind, waiting.job), ('waiting', lab.job))
        self.assertFalse(waiting.event['run_stop']['shutdown_complete'])
        self.assertEqual(lab.cancels, [])
        lab.settled = True
        self.assertEqual(self.step().event['operation_result']['status'], ExecutionStatus.COMPLETED)
        self.assertTrue(self.step().event['run_stop']['shutdown_complete'])
        self.assertEqual(lab.cancels, [])
        self.assertEqual(lab.shutdowns, ['drain', 'drain'])
        self.assert_no_adoption(host, original)

    def test_backend_cancel_acknowledgement_is_not_settlement(self):
        lab = Lab(backend=True, blocking=True)
        self.setup_application(lab, Policy((operation(),)), TimeRunPolicy(1.))
        self.assertEqual(self.step().kind, 'waiting')
        self.release_and_await_arrival(lab)
        self.assertEqual(self.step().job, lab.job)
        self.clock.now = 2.
        pending = self.step()
        self.assertEqual((pending.kind, pending.job), ('waiting', lab.job))
        self.assertEqual(lab.cancels, [lab.job])
        self.assertFalse(pending.event['run_stop']['shutdown_complete'])
        lab.settled = True
        self.assertEqual(self.step().event['operation_result']['status'], ExecutionStatus.CANCELLED)
        self.assertTrue(self.step().event['run_stop']['shutdown_complete'])


if __name__ == '__main__':
    unittest.main()
