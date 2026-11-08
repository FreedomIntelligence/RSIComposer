"""Executed decision operations survive uncertain history/experience writes."""

from copy import deepcopy
from dataclasses import replace
from pathlib import Path
import tempfile
import unittest
from unittest.mock import Mock, patch

from rsisys.experience.retrieval import BasicExperienceSystem
from rsisys.experience.sqlite import SQLiteExperienceStore
from rsisys.experience.storage import InMemoryExperienceStore
from rsisys.loops.contracts.jobs import JobRef
from rsisys.loops.contracts.lifecycle import ExecutionStatus
from rsisys.loops.decision_making.harness.decision import Decision
from rsisys.loops.decision_making.action_space import OperationCall, OperationResult
from rsisys.loops.decision_making.loop import DecisionMakingLoop
from rsisys.loops.decision_making.harness.engine import DefaultDecisionHarness
from rsisys.loops.decision_making.harness.memory import DecisionMemory


class OperationRetryTests(unittest.TestCase):
    def make_loop(self, *, asynchronous=False, wait=False, simulated=False,
                  store=None, memory=None):
        call = OperationCall('prepare', {'nested': ['original']})
        result = OperationResult('prepared result', (), False)
        job = JobRef('pending-operation', 'test')
        decision = (Decision('wait', job=job, simulated=simulated) if wait else
                    Decision('invoke', operation=call, simulated=simulated))
        policy = Mock(decide=Mock(side_effect=[decision, Decision('finish')]))
        space = Mock(describe=Mock(return_value=()),
                     execute=Mock(return_value=job if asynchronous else result))
        store = store if store is not None else InMemoryExperienceStore()
        actions = Mock(describe=Mock(return_value=()), validate=Mock(side_effect=lambda call: call),
                       directive_modes=Mock(return_value=('execute',)), validate_directive=Mock())
        loop = DecisionMakingLoop(DefaultDecisionHarness(policy, space, BasicExperienceSystem(store), memory=memory), actions)
        return loop, store, call, result, job

    def inject_failure(self, loop, target, after_commit, *, attempts=1):
        """Capture every payload and fail only writes of operation records."""
        history_payloads, experience_payloads = [], []
        append, record = loop.harness.memory.append, loop.harness.experience.record
        remaining = attempts

        def write(value, kind, commit, captured):
            nonlocal remaining
            captured.append(deepcopy(value))
            if kind == target and remaining:
                remaining -= 1
                if after_commit:
                    commit(value)
                raise OSError('record response lost' if after_commit else 'record unavailable')
            commit(value)

        def append_history(value):
            if value.kind == 'operation':
                write(value, 'memory', append, history_payloads)
            else:
                append(value)

        loop.harness.memory.append = append_history
        loop.harness.experience.record = lambda value: write(value, 'experience', record, experience_payloads)
        return history_payloads, experience_payloads

    def assert_recorded_once(self, loop, store, result):
        history = loop.harness.memory.history(kind='operation')
        self.assertEqual(len(history), 1)
        self.assertEqual(len(store.records()), 1)
        self.assertEqual(history[0].record_id, store.records()[0].record_id)
        self.assertEqual(loop.harness.state.operation_results, [result])
        self.assertIsNone(loop.harness.state.pending_operation_record)
        self.assertIsNone(loop.harness.state.pending_job)
        self.assertIsNone(loop.harness.state.pending_operation)
        self.assertFalse(loop.harness.state.pending_simulated)

    def test_sync_retry_preserves_identity_result_elapsed_and_does_not_reexecute(self):
        for target in ('memory', 'experience'):
            for after_commit in (False, True):
                with self.subTest(target=target, after_commit=after_commit):
                    loop, store, call, result, job = self.make_loop(simulated=True)
                    histories, experiences = self.inject_failure(loop, target, after_commit, attempts=2)
                    with patch('rsisys.loops.decision_making.harness.engine.perf_counter', side_effect=[10.0, 10.75]):
                        with self.assertRaises(OSError):
                            loop.step('task')
                    pending = deepcopy(loop.harness.state.pending_operation_record)
                    self.assertEqual(pending.elapsed_seconds, 0.75)
                    self.assertEqual(pending.result, replace(result, simulated=True))
                    self.assertEqual(loop.harness.state.operation_results, [])
                    # Neither retry may decide, execute, generate an ID, or recompute timing.
                    with patch('rsisys.loops.decision_making.harness.engine.perf_counter', side_effect=AssertionError), \
                         patch('rsisys.loops.decision_making.harness.engine.uuid4', side_effect=AssertionError):
                        with self.assertRaises(OSError):
                            loop.step('task')
                        self.assertEqual(loop.harness.state.pending_operation_record, pending)
                        retry = loop.step('task')
                    self.assertEqual(retry.kind, 'progressed')
                    self.assertEqual(retry.operation_result, pending.result)
                    self.assert_recorded_once(loop, store, pending.result)
                    self.assertTrue(all(value == histories[0] for value in histories))
                    self.assertTrue(all(value == experiences[0] for value in experiences))
                    self.assertEqual(store.records()[0].details['elapsed_seconds'], 0.75)
                    loop.harness.tools.execute.assert_called_once()
                    loop.harness.policy.decide.assert_called_once()
                    self.assertEqual(loop.step('task').kind, 'finished')
                    self.assertEqual(loop.harness.policy.decide.call_args.args[0].operation_results, (pending.result,))

    def test_async_resume_retries_the_same_completion_for_invoke_and_wait(self):
        for wait in (False, True):
            for target in ('memory', 'experience'):
                for after_commit in (False, True):
                    with self.subTest(wait=wait, target=target, after_commit=after_commit):
                        loop, store, call, result, job = self.make_loop(
                            asynchronous=True, wait=wait, simulated=True)
                        histories, experiences = self.inject_failure(loop, target, after_commit)
                        self.assertEqual(loop.step('task').job, job)
                        with self.assertRaises(OSError):
                            loop.harness.resume(job, result)
                        pending = deepcopy(loop.harness.state.pending_operation_record)
                        self.assertEqual(pending.job, job)
                        self.assertEqual(pending.resume_result, result)
                        self.assertEqual(loop.harness.state.pending_job, job)
                        self.assertIsNone(pending.elapsed_seconds)
                        self.assertEqual(loop.harness.state.operation_results, [])
                        with patch('rsisys.loops.decision_making.harness.engine.uuid4', side_effect=AssertionError):
                            loop.harness.resume(job, deepcopy(result))
                        self.assert_recorded_once(loop, store, replace(result, simulated=True))
                        self.assertTrue(all(value == histories[0] for value in histories))
                        self.assertTrue(all(value == experiences[0] for value in experiences))
                        self.assertEqual(loop.harness.tools.execute.call_count, 0 if wait else 1)
                        loop.harness.policy.decide.assert_called_once()
                        # Keep the existing contract after successful completion.
                        with self.assertRaises(ValueError):
                            loop.harness.resume(job, result)

    def test_changed_resume_job_or_raw_result_is_rejected_without_writes(self):
        loop, store, call, result, job = self.make_loop(asynchronous=True, simulated=True)
        histories, experiences = self.inject_failure(loop, 'experience', False)
        loop.step('task')
        with self.assertRaises(OSError):
            loop.harness.resume(job, result)
        pending = deepcopy(loop.harness.state.pending_operation_record)
        for wrong_job, wrong_result in (
            (JobRef('other', 'test'), result),
            (job, replace(result, summary='different result')),
            # Normalization makes both results simulated; the original payload must still match.
            (job, replace(result, simulated=True)),
            (job, replace(result, status=ExecutionStatus.CANCELLED)),
        ):
            with self.subTest(job=wrong_job, result=wrong_result), self.assertRaises(ValueError):
                loop.harness.resume(wrong_job, wrong_result)
            self.assertEqual(loop.harness.state.pending_operation_record, pending)
            self.assertEqual((len(histories), len(experiences)), (1, 1))
        loop.harness.resume(job, result)
        self.assert_recorded_once(loop, store, replace(result, simulated=True))

    def test_step_flushes_async_completion_before_waiting_or_deciding(self):
        loop, store, call, result, job = self.make_loop(asynchronous=True)
        self.inject_failure(loop, 'experience', True)
        loop.step('task')
        with self.assertRaises(OSError):
            loop.harness.resume(job, result)
        retry = loop.step('task')
        self.assertEqual((retry.kind, retry.operation_result), ('progressed', result))
        self.assert_recorded_once(loop, store, result)
        loop.harness.tools.execute.assert_called_once()
        loop.harness.policy.decide.assert_called_once()
        self.assertEqual(loop.step('task').kind, 'finished')

    def test_failed_operation_is_recorded_on_retry_without_reexecution(self):
        loop, store, call, result, job = self.make_loop()
        loop.harness.tools.execute.side_effect = RuntimeError('operation failed')
        self.inject_failure(loop, 'experience', True)
        with patch('rsisys.loops.decision_making.harness.engine.perf_counter', side_effect=[5.0, 7.0]):
            with self.assertRaisesRegex(RuntimeError, 'operation failed') as raised:
                loop.step('task')
        self.assertTrue(any('Could not record operation failure' in note for note in raised.exception.__notes__))
        pending = deepcopy(loop.harness.state.pending_operation_record)
        self.assertEqual(pending.result.status, ExecutionStatus.FAILED)
        self.assertEqual(pending.elapsed_seconds, 2.0)
        retry = loop.step('task')
        self.assertEqual(retry.operation_result, pending.result)
        self.assert_recorded_once(loop, store, pending.result)
        loop.harness.tools.execute.assert_called_once()
        self.assertEqual(loop.step('task').kind, 'finished')
        self.assertEqual(loop.harness.policy.decide.call_args.args[0].operation_results, (pending.result,))

    def test_pending_call_is_detached_from_mutable_arguments(self):
        loop, store, call, result, job = self.make_loop()
        self.inject_failure(loop, 'experience', False)
        with self.assertRaises(OSError):
            loop.step('task')
        call.arguments['nested'].append('caller mutation')
        self.assertEqual(loop.harness.state.pending_operation_record.call.arguments, {'nested': ['original']})
        loop.step('task')
        self.assert_recorded_once(loop, store, result)
        self.assertEqual(loop.harness.memory.history(kind='operation')[0].details['operation']['arguments'],
                         {'nested': ['original']})

    def test_async_call_is_detached_at_submission(self):
        loop, store, call, result, job = self.make_loop(asynchronous=True)
        loop.step('task')
        call.arguments['nested'].append('caller mutation')
        loop.harness.resume(job, result)
        self.assertEqual(loop.harness.memory.history(kind='operation')[0].details['operation']['arguments'],
                         {'nested': ['original']})

    def test_normal_sync_completion_clears_recording_state(self):
        loop, store, call, result, job = self.make_loop()
        self.assertEqual(loop.step('task').operation_result, result)
        self.assert_recorded_once(loop, store, result)

    def test_committed_sqlite_response_loss_retries_without_duplicate_records(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            store = SQLiteExperienceStore(root / 'experience.sqlite3')
            self.addCleanup(store.close)
            memory = DecisionMemory(root / 'memory.json')
            loop, _, call, result, job = self.make_loop(store=store, memory=memory)
            self.inject_failure(loop, 'experience', True)
            with self.assertRaises(OSError):
                loop.step('task')
            loop.step('task')
            self.assert_recorded_once(loop, store, result)
            reopened = DecisionMemory(memory.path)
            self.assertEqual(len(reopened.history(kind='operation')), 1)
            self.assertEqual(reopened.history(kind='operation')[0].record_id, store.records()[0].record_id)


if __name__ == '__main__':
    unittest.main()
