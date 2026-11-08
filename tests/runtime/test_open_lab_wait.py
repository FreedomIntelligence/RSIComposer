"""Optional owned-process waiting, with immediate queries and native receipts."""
from pathlib import Path
import sys
from tempfile import TemporaryDirectory
import threading
import unittest

from rsisys.adapters.local_lab import LocalProcessLab
from rsisys.loops.contracts.decision import OperationCall
from rsisys.loops.contracts.execution import LabToolError
from rsisys.runtime.artifacts import InMemoryArtifactStore
from rsisys.runtime.local_process import LocalProcessBackend


class OpenLabWaitTests(unittest.TestCase):
    def setUp(self):
        directory = TemporaryDirectory()
        self.addCleanup(directory.cleanup)
        self.backend = LocalProcessBackend(Path(directory.name) / 'executor')
        self.lab = LocalProcessLab(self.backend, InMemoryArtifactStore(), timeout_seconds=5,
                                   max_poll_wait_seconds=2)
        self.addCleanup(self.backend.close_all)
        self.addCleanup(self.lab.close)

    def call(self, name, **arguments):
        return self.lab.execute(OperationCall('lab_' + name, arguments))

    def start(self, code):
        return self.call('start', key='once', argv=[sys.executable, '-c', code]).data['job_id']

    def test_immediate_query_remains_available_and_does_not_collect_or_cancel(self):
        ident = self.start('import time; time.sleep(30)')
        for arguments in ({}, {'wait_seconds': 0}):
            result = self.call('poll', job_id=ident, **arguments)
            self.assertEqual(result.data['state'], 'running')
            self.assertNotIn('poll_wait', result.data)
            self.assertIsNone(result.feedback_delivery)
        self.assertEqual(len(self.lab._jobs), 1)
        self.assertEqual(self.backend.poll(self.lab._jobs[ident][0]).status.value, 'running')

    def test_one_wait_observes_real_exit_without_resubmission_or_changing_collection(self):
        ident = self.start('import time; time.sleep(.1); print("native result")')
        result = self.call('poll', job_id=ident, wait_seconds=2)
        self.assertEqual(result.data['state'], 'succeeded')
        self.assertEqual(result.data['stdout'], 'native result\n')
        self.assertEqual(result.data['poll_wait']['reason'], 'terminal')
        self.assertIsNone(result.feedback_delivery)
        collected = self.call('result', job_id=ident)
        self.assertNotIn('poll_wait', collected.data)
        self.assertEqual({k: v for k, v in result.data.items() if k != 'poll_wait'}, collected.data)
        self.assertEqual(self.call('result', job_id=ident), collected)
        self.assertEqual(result.artifacts, collected.artifacts)
        self.assertEqual(len(self.lab._jobs), 1)

    def test_wait_limit_returns_running_process_and_queries_can_continue(self):
        ident = self.start('import time; time.sleep(30)')
        result = self.call('poll', job_id=ident, wait_seconds=.03)
        self.assertEqual(result.data['state'], 'running')
        self.assertEqual(result.data['poll_wait']['reason'], 'timeout')
        self.assertEqual(self.call('poll', job_id=ident).data['state'], 'running')
        self.assertEqual(len(self.lab._jobs), 1)

    def test_host_interrupt_returns_current_state_without_taking_ownership_of_shutdown(self):
        ident = self.start('import time; time.sleep(30)')
        interrupt = threading.Event()
        self.lab.wait_interrupted = interrupt.is_set
        timer = threading.Timer(.05, interrupt.set)
        timer.start()
        self.addCleanup(timer.join)
        result = self.call('poll', job_id=ident, wait_seconds=2)
        self.assertEqual(result.data['poll_wait']['reason'], 'host_interrupted')
        self.assertEqual(result.data['state'], 'running')
        self.assertIsNone(result.feedback_delivery)
        self.assertEqual(self.call('poll', job_id=ident).data['state'], 'running')

    def test_program_failure_and_cancellation_wake_wait_and_keep_native_evidence(self):
        ident = self.start('import time; time.sleep(.05); raise RuntimeError("native failure")')
        result = self.call('poll', job_id=ident, wait_seconds=2)
        self.assertEqual(result.data['poll_wait']['reason'], 'terminal')
        self.assertEqual(result.data['state'], 'failed')
        self.assertIn('native failure', result.data['stderr'])
        other = self.call('start', key='cancel', argv=[sys.executable, '-c', 'import time; time.sleep(30)']).data['job_id']
        timer = threading.Timer(.05, lambda: self.call('cancel', job_id=other))
        timer.start()
        self.addCleanup(timer.join)
        cancelled = self.call('poll', job_id=other, wait_seconds=2)
        self.assertEqual(cancelled.data['state'], 'cancelled')
        self.assertEqual(cancelled.data['poll_wait']['reason'], 'terminal')

    def test_invalid_wait_is_recoverable_and_does_not_affect_the_job(self):
        ident = self.start('import time; time.sleep(30)')
        for value in (-1, 3, 10**1000, True, '1', None, float('nan'), float('inf')):
            with self.subTest(value=value), self.assertRaises(LabToolError):
                self.call('poll', job_id=ident, wait_seconds=value)
        self.assertEqual(self.call('poll', job_id=ident).data['state'], 'running')


if __name__ == '__main__':
    unittest.main()
