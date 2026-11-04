"""Execution boundaries with owned CPU processes, real policy requests and injected faults."""
from dataclasses import replace
import errno
import fcntl
import json
import os
from pathlib import Path
import sys
from tempfile import TemporaryDirectory
import unittest
from unittest.mock import Mock, patch

from rsisys.adapters.local_lab import LocalProcessLab
from rsisys.application import Application, ApplicationConfig
from rsisys.config import DemoConfig, build_demo
from rsisys.loops.contracts.decision import OperationCall
from rsisys.loops.contracts.execution import LabInfrastructureError, LabToolError
from rsisys.loops.contracts.jobs import JobStatus
from rsisys.loops.contracts.model_service import GenerationResponse
from rsisys.runtime.artifacts import InMemoryArtifactStore
from rsisys.runtime.local_process import LocalProcessBackend
from rsisys.runtime.run_policy import TimeRunPolicy
from rsisys.runtime.sandbox import ExecRequest, SandboxSpec


ROOT = Path(__file__).resolve().parents[1]
SLEEP = [sys.executable, '-c', 'import time; time.sleep(30)']


class CountingLab(LocalProcessLab):
    close_attempts = 0

    def close(self):
        self.close_attempts += 1
        return super().close()


class ExecutionContractFixesTests(unittest.TestCase):
    def setUp(self):
        directory = TemporaryDirectory()
        self.addCleanup(directory.cleanup)
        self.root = Path(directory.name)

    def lab(self, name='lab'):
        backend = LocalProcessBackend(self.root / name, max_output_bytes=4096)
        lab = CountingLab(backend, InMemoryArtifactStore(), timeout_seconds=10)
        self.addCleanup(lab.close)
        self.addCleanup(backend.close_all)
        return lab

    def owner(self, lab, kind='demo', **options):
        dependencies = []

        def factory(context):
            # Registered before the Lab: this dependency must outlive its close.
            context.add_cleanup(lambda: dependencies.append('released'))
            return lab

        if kind == 'application':
            config = ApplicationConfig.load(ROOT / 'configs/local.json')
            config = replace(config, data_dir=self.root / 'application-workspace', execution='mock',
                             roles={'decision_maker': config.roles['decision_maker']})
            owner = Application(config, lab_factory=factory, **options)
            runner = owner.runner
        else:
            config = DemoConfig.load(ROOT / 'configs/demo.json')
            config = replace(config, roles={'decision_maker': config.roles['decision_maker']})
            owner, _ = build_demo(config, lab_factory=factory, **options)
            runner = owner
        self.addCleanup(owner.close)
        return owner, runner, dependencies

    @staticmethod
    def wire(runner, *calls):
        requests = []
        calls = iter(calls)

        def generate(request):
            requests.append(request)
            name, arguments = next(calls)
            return GenerationResponse(json.dumps({'kind': 'invoke', 'operation': name,
                'arguments': arguments}), request.model, simulated=False)

        runner.decision_making.harness.policy.backend.generate = generate
        return requests

    @staticmethod
    def start(lab, key='owned', argv=None, **options):
        result = lab.execute(OperationCall('lab_start', {'key': key, 'argv': argv or SLEEP, **options}))
        ident = result.data['job_id']
        ref = lab._jobs[ident][0]
        return ident, ref, lab.backend._processes[ref]

    @staticmethod
    def call(lab, name, **args):
        return lab.execute(OperationCall('lab_' + name, args))

    def test_both_owners_retry_owned_close_and_retain_dependencies_and_lock(self):
        for kind in ('application', 'demo'):
            with self.subTest(kind=kind):
                lab = self.lab(kind)
                owner, runner, dependencies = self.owner(lab, kind)
                events = []
                owner.assembly.add_cleanup(lambda: events.append(len(owner.assembly.experience.retrieve())))
                _, _, entry = self.start(lab)
                with patch('rsisys.runtime.local_process._kill_process_group',
                           side_effect=PermissionError('temporary owned kill failure')):
                    with self.assertRaisesRegex(PermissionError, 'owned kill failure'):
                        owner.close()
                self.assertEqual(events, [0])
                self.assertEqual(dependencies, [])
                self.assertTrue(owner.assembly.cleanup_pending)
                self.assertEqual(lab.backend.active_count, 1)
                os.kill(entry.process.pid, 0)
                self.assertEqual(tuple(owner.assembly.experience.retrieve()), ())
                with self.assertRaisesRegex(RuntimeError, 'closed'):
                    owner.assembly.add_cleanup(lambda: None)
                with self.assertRaisesRegex(RuntimeError, 'closed'):
                    owner.__enter__()
                with self.assertRaisesRegex(RuntimeError, 'closed'):
                    runner.step('no new decision or execution')
                with self.assertRaisesRegex(RuntimeError, 'shutting down'):
                    self.call(lab, 'start', key='late', argv=SLEEP)
                if kind == 'application':
                    self.assertIsNotNone(owner.store)
                    self.assertFalse(owner._lock.closed)
                    with (owner.config.data_dir / '.lock').open('a+b') as lock:
                        with self.assertRaises(BlockingIOError):
                            fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
                    with self.assertRaisesRegex(RuntimeError, 'closed'):
                        owner.run()
                owner.close()
                self.assertEqual(lab.close_attempts, 2)
                self.assertFalse(owner.assembly.cleanup_pending)
                self.assertEqual(events, [0])
                self.assertEqual(dependencies, ['released'])
                self.assertEqual(lab.backend.active_count, 0)
                with self.assertRaises(ProcessLookupError):
                    os.kill(entry.process.pid, 0)
                if kind == 'application':
                    self.assertIsNone(owner.store)
                    self.assertTrue(owner._lock.closed)
                owner.close()
                self.assertEqual(lab.close_attempts, 2)

    def test_uncertain_ordinary_callback_is_not_replayed_when_owned_close_retries(self):
        lab = self.lab()
        owner, _, dependencies = self.owner(lab)
        events = []

        def fail_once():
            events.append('ordinary')
            raise OSError('ordinary callback has uncertain effects')

        owner.assembly.add_cleanup(fail_once)
        self.start(lab)
        with patch('rsisys.runtime.local_process._kill_process_group', side_effect=PermissionError('owned kill')):
            with self.assertRaises(ExceptionGroup) as caught:
                owner.close()
        self.assertEqual([type(e) for e in caught.exception.exceptions], [OSError, PermissionError])
        self.assertEqual(dependencies, [])
        owner.close()
        self.assertEqual(events, ['ordinary'])
        self.assertEqual(dependencies, ['released'])

    def test_poll_and_cancel_errors_do_not_skip_other_owned_cancellations(self):
        lab = self.lab()
        _, first, _ = self.start(lab, 'first')
        _, second, entry = self.start(lab, 'second')
        other = lab.backend.create(SandboxSpec('local', 10))
        other_ref = lab.backend.exec(other, ExecRequest(tuple(SLEEP), 10))
        poll, cancel = lab.backend.poll, lab.backend.cancel

        def broken_poll(ref):
            if ref == first:
                raise OSError('first poll failed')
            return poll(ref)

        def broken_cancel(ref):
            if ref == first:
                raise PermissionError('first cancel failed')
            cancel(ref)

        with patch.object(lab.backend, 'poll', side_effect=broken_poll), \
                patch.object(lab.backend, 'cancel', side_effect=broken_cancel) as cancellation:
            with self.assertRaises(ExceptionGroup) as caught:
                lab.shutdown('cancel')
        self.assertEqual([str(e) for e in caught.exception.exceptions], ['first poll failed', 'first cancel failed'])
        self.assertEqual([args.args[0] for args in cancellation.call_args_list], [first, second])
        self.assertTrue(entry.cancelled)
        self.assertFalse(lab.backend._processes[other_ref].cancelled)
        self.assertEqual(lab.backend.poll(other_ref).status, JobStatus.RUNNING)
        lab.close()
        self.assertEqual(lab.backend.active_count, 1)

    def test_terminal_artifact_error_does_not_skip_other_terminal_evidence(self):
        lab = self.lab()
        first, first_ref, _ = self.start(lab, 'first', [sys.executable, '-c', 'print("first")'])
        second, second_ref, _ = self.start(lab, 'second', [sys.executable, '-c', 'print("second")'])
        lab.backend.wait(first_ref)
        lab.backend.wait(second_ref)
        put = lab.artifacts.put

        def fail_first(payload, **kwargs):
            if json.loads(payload)['job_id'] == first:
                raise OSError('first artifact unavailable')
            return put(payload, **kwargs)

        with patch.object(lab.artifacts, 'put', side_effect=fail_first):
            with self.assertRaisesRegex(OSError, 'first artifact unavailable'):
                lab.shutdown('cancel')
        self.assertNotIn(first, lab._terminal)
        self.assertIn(second, lab._terminal)
        self.assertTrue(lab.shutdown('cancel'))
        self.assertIn(first, lab._terminal)

    def test_drain_poll_error_does_not_cancel_running_programs(self):
        lab = self.lab()
        self.start(lab)
        with patch.object(lab.backend, 'poll', side_effect=OSError('poll failed')), \
                patch.object(lab.backend, 'cancel', wraps=lab.backend.cancel) as cancel:
            with self.assertRaisesRegex(OSError, 'poll failed'):
                lab.shutdown('drain')
        cancel.assert_not_called()
        self.assertFalse(lab._accepting)

    def stopping_runner(self, lab):
        owner, runner, _ = self.owner(lab, run_policy=TimeRunPolicy(1), cleanup_timeout_seconds=1)
        now = [0.0]
        runner._clock = lambda: now[0]
        self.call(lab, 'write_file', path='ready', content='ready')
        requests = self.wire(runner, ('lab_read_file', {'path': 'ready'}))
        self.assertEqual(runner.step('objective').kind, 'progressed')
        now[0] = 2
        return owner, runner, now, requests

    def test_shutdown_error_cannot_bypass_deadline_and_is_reported_honestly(self):
        lab = self.lab()
        _, _, first = self.start(lab, 'first')
        _, _, second = self.start(lab, 'second')
        owner, runner, now, requests = self.stopping_runner(lab)
        with patch.object(lab.backend, 'poll', side_effect=OSError('poll infrastructure failed')):
            with self.assertRaises(ExceptionGroup):
                runner.step('objective')
            self.assertTrue(first.cancelled and second.cancelled)
            self.assertIsNone(runner._terminal_step)
            now[0] = 10
            result = runner.step('objective')
        self.assertEqual(result.kind, 'finished')
        stop = result.event['run_stop']
        self.assertFalse(stop['shutdown_complete'])
        self.assertTrue(stop['cleanup_timed_out'])
        self.assertTrue(stop['unresolved_execution'])
        self.assertFalse(stop['execution_shutdown_error']['resolved'])
        self.assertEqual(stop['execution_shutdown_error']['error_type'], 'ExceptionGroup')
        self.assertEqual([error['message'] for error in stop['execution_shutdown_error']['errors']],
                         ['poll infrastructure failed', 'poll infrastructure failed'])
        self.assertEqual(runner.step('objective'), result)
        self.assertEqual(len(requests), 1)
        owner.close()
        self.assertEqual(lab.backend.active_count, 0)

    def test_shutdown_boolean_contract_remains_strict_even_at_timeout(self):
        lab = self.lab()
        _, runner, now, _ = self.stopping_runner(lab)
        runner._execution_shutdown = Mock(return_value=1)
        with self.assertRaisesRegex(TypeError, 'must return bool'):
            runner.step('objective')
        now[0] = 4
        stop = runner.step('objective').event['run_stop']
        self.assertTrue(stop['unresolved_execution'])
        self.assertEqual(stop['execution_shutdown_error']['error_type'], 'TypeError')

    def test_recovered_shutdown_keeps_the_error_evidence_without_false_unresolved_work(self):
        lab = self.lab()
        _, runner, _, _ = self.stopping_runner(lab)
        runner._execution_shutdown = Mock(side_effect=[OSError('temporary shutdown fault'), True])
        with self.assertRaisesRegex(OSError, 'temporary shutdown fault'):
            runner.step('objective')
        stop = runner.step('objective').event['run_stop']
        self.assertTrue(stop['shutdown_complete'])
        self.assertFalse(stop['unresolved_execution'])
        self.assertTrue(stop['execution_shutdown_error']['resolved'])
        self.assertEqual(stop['execution_shutdown_error']['message'], 'temporary shutdown fault')

    def test_supervision_failure_reaches_host_with_original_archived_evidence(self):
        lab = self.lab()
        with patch('rsisys.runtime.local_process._wait_for_process_exit',
                   side_effect=OSError('owned exit observer failed')):
            ident, ref, entry = self.start(lab)
            self.assertTrue(entry.done.wait(3))
        self.assertEqual(lab.backend.poll(ref).failure_origin, 'infrastructure')
        owner, runner, _ = self.owner(lab)
        self.wire(runner, ('lab_result', {'job_id': ident}), ('lab_poll', {'job_id': ident}),
                  ('lab_result', {'job_id': ident}))
        artifacts = []
        events = []
        for _ in range(3):
            with self.assertRaises(LabInfrastructureError) as caught:
                runner.step('objective')
            result = caught.exception.result
            self.assertEqual(result.data['failure_origin'], 'infrastructure')
            self.assertIn('owned exit observer failed', result.data['stderr'])
            history = runner.decision_making.harness.memory.history(kind='operation')[-1]
            self.assertEqual(history.details['result'], json.loads(json.dumps(result.payload())))
            self.assertEqual(history.source_refs, tuple(ref.artifact_id for ref in result.artifacts))
            archived = json.loads(lab.artifacts.get(result.artifacts[0]))
            self.assertEqual(archived['stderr'], result.data['stderr'])
            self.assertEqual(archived['failure_origin'], 'infrastructure')
            artifacts.append(result.artifacts)
            events.append(result.feedback_delivery)
        self.assertEqual(artifacts[0], artifacts[1])
        self.assertEqual(artifacts[0], artifacts[2])
        self.assertIsNotNone(events[0])
        self.assertIsNone(events[1])
        self.assertEqual(events[0], events[2])
        with self.assertRaises(ProcessLookupError):
            os.kill(entry.process.pid, 0)
        # Host teardown checks exit and preserves evidence; it must not loop on
        # an already reported, fully reaped supervision failure.
        self.assertTrue(lab.shutdown('cancel'))
        owner.close()
        self.assertEqual(lab.backend.active_count, 0)

    def test_program_cannot_impersonate_infrastructure_failure_by_printing_stderr(self):
        lab = self.lab()
        text = 'Process supervision failed (OSError): owned exit observer failed'
        ident, ref, _ = self.start(lab, argv=[sys.executable, '-c',
            f'import sys; print({text!r}, file=sys.stderr); sys.exit(1)'])
        self.assertEqual(lab.backend.wait(ref).failure_origin, 'program')
        _, runner, _ = self.owner(lab)
        self.wire(runner, ('lab_result', {'job_id': ident}))
        result = runner.step('objective')
        self.assertEqual(result.kind, 'progressed')
        self.assertEqual(result.event['operation_result']['status'], 'failed')
        self.assertEqual(result.event['operation_result']['data']['failure_origin'], 'program')

    def test_timeout_cancel_and_success_provenance_are_distinct(self):
        lab = self.lab()
        _, ref, _ = self.start(lab, 'success', [sys.executable, '-c', 'print("done")'])
        success = lab.backend.wait(ref)
        _, ref, _ = self.start(lab, 'timeout', timeout_seconds=.02)
        timeout = lab.backend.wait(ref)
        _, ref, _ = self.start(lab, 'cancel')
        lab.backend.cancel(ref)
        cancelled = lab.backend.wait(ref)
        self.assertEqual((success.status, success.failure_origin), (JobStatus.SUCCEEDED, None))
        self.assertEqual((timeout.status, timeout.failure_origin), (JobStatus.FAILED, 'timeout'))
        self.assertEqual((cancelled.status, cancelled.failure_origin), (JobStatus.CANCELLED, None))

    def test_path_type_errors_reach_next_decision_and_corrected_action_succeeds(self):
        for case in ('read_parent_file', 'write_parent_file', 'write_directory'):
            with self.subTest(case=case):
                lab = self.lab(case)
                self.call(lab, 'write_file', path='ordinary', content='source')
                (lab.workspace / 'directory').mkdir()
                if case == 'read_parent_file':
                    bad = ('lab_read_file', {'path': 'ordinary/child'})
                    fixed = ('lab_read_file', {'path': 'ordinary'})
                else:
                    bad = ('lab_write_file', {'path': 'directory' if case == 'write_directory' else 'ordinary/child',
                                              'content': 'new content'})
                    fixed = ('lab_write_file', {'path': 'new_directory/file', 'content': 'new content'})
                owner, runner, _ = self.owner(lab)
                requests = self.wire(runner, bad, fixed)
                first = runner.step('objective').event['operation_result']
                self.assertEqual(first['status'], 'failed')
                self.assertIn('LabToolError', first['error'])
                second = runner.step('objective').event['operation_result']
                self.assertEqual(second['status'], 'completed')
                context = json.loads(requests[1].prompt)
                self.assertEqual(context['operation_results'][-1]['error'], first['error'])
                self.assertFalse(list(lab.workspace.glob('.lab-write-*')))
                owner.close()

    def test_symlink_root_permission_and_storage_errors_remain_host_failures(self):
        for case in ('symlink', 'root_is_file', 'root_missing', 'permission', 'no_space'):
            with self.subTest(case=case):
                lab = self.lab(case)
                self.call(lab, 'write_file', path='ordinary', content='original')
                owner, runner, _ = self.owner(lab)
                path, restore_root = 'target', None
                if case == 'symlink':
                    (lab.workspace / 'link').symlink_to(lab.workspace / 'ordinary')
                    path = 'link/target'
                elif case in ('root_is_file', 'root_missing'):
                    root = lab.workspace
                    saved = root.with_name(root.name + '-saved')
                    root.rename(saved)
                    if case == 'root_is_file':
                        root.write_text('corrupt workspace root')
                    restore_root = (root, saved)
                self.wire(runner, ('lab_write_file', {'path': path, 'content': 'rejected'}))
                fault = (PermissionError(errno.EACCES, 'fixture permission') if case == 'permission' else
                         OSError(errno.ENOSPC, 'fixture storage full') if case == 'no_space' else None)
                try:
                    if fault is None:
                        with self.assertRaises(OSError) as caught:
                            runner.step('objective')
                    else:
                        with patch('rsisys.adapters.local_lab.os.replace', side_effect=fault):
                            with self.assertRaises(OSError) as caught:
                                runner.step('objective')
                    self.assertNotIsInstance(caught.exception, LabToolError)
                    if fault is not None:
                        self.assertIs(caught.exception, fault)
                finally:
                    if restore_root is not None:
                        root, saved = restore_root
                        root.unlink(missing_ok=True)
                        saved.rename(root)
                owner.close()

    def test_reading_or_exporting_a_directory_is_a_correctable_type_error(self):
        lab = self.lab()
        (lab.workspace / 'directory').mkdir()
        for name in ('read_file', 'export'):
            with self.subTest(name=name), self.assertRaisesRegex(LabToolError, 'regular files'):
                self.call(lab, name, path='directory')


if __name__ == '__main__':
    unittest.main()
