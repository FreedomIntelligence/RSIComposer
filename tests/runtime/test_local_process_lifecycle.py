"""Owned CPU process regressions for streaming and retryable cleanup."""
import os
from pathlib import Path
import shutil
import sys
from tempfile import TemporaryDirectory
import threading
import time
import unittest
from unittest.mock import patch

from rsisys.loops.contracts.jobs import JobStatus
from rsisys.runtime.local_process import LocalProcessBackend
from rsisys.runtime.sandbox import ExecRequest, SandboxSpec


class LocalProcessLifecycleTests(unittest.TestCase):
    def setUp(self):
        self.tmp = TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.backend = LocalProcessBackend(Path(self.tmp.name) / 'work')
        self.addCleanup(self.backend.close_all)
        self.session = self.backend.create(SandboxSpec('local', 5))

    def test_flushed_short_stdout_and_stderr_are_visible_before_exit(self):
        code = ('from pathlib import Path; import sys,time; '
                'print("READY", flush=True); print("NOTE", file=sys.stderr, flush=True); '
                '\nwhile not Path("release").exists(): time.sleep(.01)')
        ref = self.backend.exec(self.session, ExecRequest((sys.executable, '-c', code), 5))
        deadline = time.monotonic() + 3
        while True:
            state = self.backend.poll(ref)
            if ('READY' in state.stdout and 'NOTE' in state.stderr) or time.monotonic() >= deadline:
                break
            time.sleep(.01)
        self.assertEqual(state.status, JobStatus.RUNNING)
        self.assertEqual(state.stdout, 'READY\n')
        self.assertEqual(state.stderr, 'NOTE\n')
        (self.backend.workspace(self.session) / 'release').touch()
        self.assertEqual(self.backend.wait(ref).status, JobStatus.SUCCEEDED)

    def test_workspace_deletion_failure_retains_ownership_for_retry(self):
        root = self.backend.workspace(self.session)
        with patch('rsisys.runtime.local_process.shutil.rmtree', side_effect=PermissionError('fixture locked')):
            with self.assertRaisesRegex(PermissionError, 'fixture locked'):
                self.backend.close(self.session)
        self.assertEqual(self.backend.active_count, 1)
        self.assertEqual(self.backend.workspace(self.session), root)
        self.assertTrue(root.exists())
        self.backend.close(self.session)
        self.assertEqual(self.backend.active_count, 0)
        self.assertFalse(root.exists())

    def test_already_removed_workspace_can_be_closed(self):
        shutil.rmtree(self.backend.workspace(self.session))
        self.backend.close(self.session)
        self.assertEqual(self.backend.active_count, 0)

    def test_observer_failure_reports_failed_outcome_after_owned_exit(self):
        with patch('rsisys.runtime.local_process._wait_for_process_exit', side_effect=RuntimeError('fixture observer')):
            ref = self.backend.exec(self.session, ExecRequest((sys.executable, '-c', 'import time; time.sleep(30)'), 5))
            state = self.backend.wait(ref)
        self.assertEqual(state.status, JobStatus.FAILED)
        self.assertEqual(state.failure_origin, 'infrastructure')
        self.assertIn('RuntimeError', state.stderr)
        self.assertIn('fixture observer', state.stderr)
        self.assertIsNotNone(state.exit_code)
        entry = self.backend._processes[ref]
        with self.assertRaises(ProcessLookupError):
            os.kill(entry.process.pid, 0)
        self.backend.close(self.session)
        self.assertEqual(self.backend.active_count, 0)

    def test_failed_observer_and_failed_kill_keep_live_process_owned_until_retry(self):
        with patch('rsisys.runtime.local_process._wait_for_process_exit', side_effect=RuntimeError('fixture observer')), \
                patch('rsisys.runtime.local_process._kill_process_group', side_effect=PermissionError('fixture kill')):
            ref = self.backend.exec(self.session, ExecRequest((sys.executable, '-c', 'import time; time.sleep(30)'), 5))
            entry = self.backend._processes[ref]
            self.assertTrue(entry.done.wait(3))
            self.assertIsNone(entry.state)
            os.kill(entry.process.pid, 0)
            with self.assertRaisesRegex(RuntimeError, 'cleanup is incomplete'):
                self.backend.poll(ref)
            with self.assertRaisesRegex(PermissionError, 'fixture kill'):
                self.backend.close(self.session)
            self.assertTrue(self.backend.workspace(self.session).exists())
            self.assertEqual(self.backend.active_count, 1)
            os.kill(entry.process.pid, 0)
        self.backend.close(self.session)
        state = self.backend.poll(ref)
        self.assertEqual(state.status, JobStatus.FAILED)
        self.assertEqual(state.failure_origin, 'infrastructure')
        self.assertIn('fixture observer', state.stderr)
        self.assertIn('fixture kill', state.stderr)
        self.assertIsNotNone(state.exit_code)
        self.assertEqual(self.backend.active_count, 0)
        with self.assertRaises(ProcessLookupError):
            os.kill(entry.process.pid, 0)

    def test_watcher_start_failure_does_not_orphan_the_created_process(self):
        with patch('rsisys.runtime.local_process.threading.Thread.start', side_effect=RuntimeError('fixture start')):
            with self.assertRaisesRegex(RuntimeError, 'fixture start'):
                self.backend.exec(self.session, ExecRequest((sys.executable, '-c', 'import time; time.sleep(30)'), 5))
        self.assertEqual(len(self.backend._processes), 1)
        entry = next(iter(self.backend._processes.values()))
        self.assertTrue(entry.done.is_set())
        self.assertEqual(entry.state.status, JobStatus.FAILED)
        self.assertEqual(entry.state.failure_origin, 'infrastructure')
        self.assertIsNotNone(entry.process.returncode)
        self.assertTrue(entry.process.stdout.closed)
        self.assertTrue(entry.process.stderr.closed)

    def test_reader_start_failure_is_failed_and_still_reaps_owned_process(self):
        start = threading.Thread.start
        count = 0
        def fail_second_reader(thread):
            nonlocal count
            count += 1
            if count == 3:  # watcher and stdout reader started, stderr reader failed
                raise RuntimeError('fixture reader start')
            return start(thread)
        with patch('rsisys.runtime.local_process.threading.Thread.start', new=fail_second_reader):
            ref = self.backend.exec(self.session, ExecRequest((sys.executable, '-c', 'import time; time.sleep(30)'), 5))
            state = self.backend.wait(ref)
        self.assertEqual(state.status, JobStatus.FAILED)
        self.assertEqual(state.failure_origin, 'infrastructure')
        self.assertIn('fixture reader start', state.stderr)
        entry = self.backend._processes[ref]
        self.assertIsNotNone(entry.process.returncode)
        self.assertTrue(entry.process.stdout.closed)
        self.assertTrue(entry.process.stderr.closed)
