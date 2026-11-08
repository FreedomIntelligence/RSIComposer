"""Reproduced audit failures: SQLite, journal recovery, and tool evidence."""
from dataclasses import asdict, replace
import json
from pathlib import Path
from tempfile import TemporaryDirectory
import unittest

from rsisys.experience.sqlite import SQLiteExperienceStore
from rsisys.experience.retrieval import BasicExperienceSystem
from rsisys.loops._memory import HistoryEntry
from rsisys.loops.contracts.candidate import Candidate
from rsisys.loops.contracts.coordination import OperationResult
from rsisys.loops.contracts.directive import Directive
from rsisys.loops.contracts.feedback import Feedback
from rsisys.loops.contracts.jobs import JobStatus
from rsisys.loops.contracts.lifecycle import ExecutionStatus
from rsisys.loops.contracts.model_service import ModelRef
from rsisys.loops.decision_making.harness.context import BasicContextCompactor
from rsisys.loops.decision_making.harness.feedback_learning import FeedbackLearning, SerializedLearningBackend
from rsisys.loops.decision_making.harness.research_state import research_working_view
from rsisys.loops.decision_making.harness.state import DecisionContext
from rsisys.loops.exploration.harness.evaluation import EvaluationExecution, EvaluationRequest, EvaluationResult
from rsisys.loops.exploration.harness.state import ExplorationState
from rsisys.runtime.jobs import JobBook
from tests.runtime.test_exploration_execution import Backend
from tests.runtime import test_exploration_recovery as recovery
from tests.feedback import test_feedback_learning_integration as learning_fixture


class AuditRegressions(unittest.TestCase):
    def test_invalid_metrics_fail_once_and_sqlite_poll_cancel_remain_idempotent(self):
        for metrics in ({'accuracy': float('nan')}, {'accuracy': float('inf')},
                        {'accuracy': float('-inf')}, {'accuracy': True}, {'accuracy': '0'},
                        {'accuracy': 10**1000}, {'': 0}, {1: 0}, []):
            for cancel_first in (False, True):
                with self.subTest(metrics=repr(metrics)[:50], cancel_first=cancel_first), TemporaryDirectory() as tmp:
                    store = SQLiteExperienceStore(Path(tmp) / 'experience.sqlite3')
                    try:
                        backend = Backend(lambda r: EvaluationResult('candidate', 'protocol', metrics, (), True))
                        # A late cancel can race with a backend's completed result.
                        backend.cancel.side_effect = lambda job: None
                        jobs = JobBook(backend)
                        execution = EvaluationExecution(BasicExperienceSystem(store), ExplorationState(), jobs)
                        request = EvaluationRequest(Candidate('candidate', 'v1', {}, ('actor',)), 'protocol')
                        job = execution.submit(request, idempotency_key='one')
                        first = (execution.cancel if cancel_first else execution.poll)(job)
                        self.assertEqual(first.status, JobStatus.FAILED)
                        self.assertIn('finite numeric', first.error)
                        self.assertIsNone(first.result)
                        self.assertTrue(first.simulated)
                        self.assertEqual(jobs._handles[job].outcome, first)
                        self.assertEqual(execution.poll(job), first)
                        self.assertEqual(execution.cancel(job), first)
                        self.assertEqual(execution.submit(request, idempotency_key='one'), job)
                        self.assertFalse(jobs.busy)
                        backend.poll.assert_called_once()
                        records = store.records()
                        self.assertEqual(len(records), 1)
                        self.assertEqual(records[0].details['outcome']['status'], 'failed')
                    finally:
                        store.close()

    def test_finite_metrics_including_zero_and_negative_are_preserved(self):
        with TemporaryDirectory() as tmp:
            store = SQLiteExperienceStore(Path(tmp) / 'experience.sqlite3')
            try:
                metrics = {'accuracy': 0.0, 'reward': -2, 'count': 3}
                backend = Backend(lambda r: EvaluationResult('candidate', 'protocol', metrics, ('evidence',)))
                execution = EvaluationExecution(BasicExperienceSystem(store), ExplorationState(), JobBook(backend))
                job = execution.submit(EvaluationRequest(Candidate('candidate', 'v1', {}, ('actor',)),
                                                         'protocol'), idempotency_key='one')
                result = execution.poll(job)
                self.assertEqual(result.status, JobStatus.SUCCEEDED)
                self.assertEqual(result.result.metrics, metrics)
                self.assertEqual(store.records()[0].source_refs, ('evidence',))
            finally:
                store.close()

    def test_invalid_evaluation_finishes_command_and_acknowledges(self):
        harness, store = recovery.ExplorationRecoveryTests.build(self)
        harness.evaluation.backend.result = EvaluationResult('first', 'protocol', {'accuracy': float('nan')}, ())
        directive = recovery.ExplorationRecoveryTests.directive(self, 'evaluate')
        harness.accept(directive)
        harness.advance(directive.directive_id, recovery.ACTIONS)
        terminal = harness.advance(directive.directive_id, recovery.ACTIONS)
        self.assertEqual(terminal.feedback.status, ExecutionStatus.FAILED)
        self.assertEqual(terminal.feedback.quality_metrics, {})
        self.assertTrue(terminal.feedback.evidence_refs)
        self.assertEqual(harness.advance(directive.directive_id, recovery.ACTIONS), terminal)
        harness.acknowledge(directive.directive_id)
        self.assertFalse(harness.command_busy)

    def test_reopened_queue_allocates_last_call_to_first_capture(self):
        with TemporaryDirectory() as tmp:
            path = Path(tmp) / 'learning.json'
            backend = learning_fixture.Backend()
            shared = SerializedLearningBackend(backend)
            original = FeedbackLearning(shared, path=path, max_calls=1,
                                        executor=learning_fixture.ImmediateExecutor())
            ids = [original.capture(Directive(ident, 'objective', 'strategy'),
                Feedback(ident, ExecutionStatus.COMPLETED, 'observed', True),
                ModelRef('fixture', 'decision', 'v1')) for ident in ('0', '1')]
            # Simulate process loss before dispatch, not a graceful close/cancel.
            loaded = json.loads(path.read_text())
            self.assertNotEqual(list(loaded['jobs']), ids)
            resumed = FeedbackLearning(shared, path=path, max_calls=1,
                                       executor=learning_fixture.ImmediateExecutor())
            self.assertEqual([r['job_id'] for r in resumed.read()['items']], ids)
            for _ in range(3):
                resumed.poll()
            self.assertEqual([json.loads(r.prompt)['receipt']['directive']['directive_id']
                              for r in backend.requests], ['0'])
            self.assertEqual(resumed.state['jobs'][ids[0]]['status'], 'completed')
            self.assertEqual(resumed.state['jobs'][ids[1]]['status'], 'budget_exhausted')
            self.assertEqual(resumed.state['attempted'], 1)
            resumed.close()

    def test_operation_data_counterevidence_and_sources_survive_compaction(self):
        result = OperationResult('Measured counterexample', (), False,
            data={'quality_metrics': {'accuracy': 0.0}, 'evidence_refs': ['evaluation:negative']})
        evidence = HistoryEntry('counterexample', 'operation', None, result.summary, False,
                                details={'result': asdict(result)})
        history = (evidence, *(HistoryEntry(str(i), 'operation', None, 'x' * 800, False) for i in range(25)))
        later = OperationResult('Unrelated latest tool', (), False)
        view = research_working_view(DecisionContext('objective', (), (), (result, later), history=history), [], [])
        self.assertIn(evidence.record_id, view.research_state['protected_history_ids'])
        compactor = BasicContextCompactor(trigger_tokens=14000, target_tokens=7000, summary_tokens=200)
        projected = compactor.compact(view, summarize=lambda *args: 'Navigation only')
        self.assertEqual(compactor.compactions, 1)
        self.assertIn(evidence, projected.history)
        self.assertIn(result, projected.operation_results)
        self.assertEqual(projected.operation_results[0].data['evidence_refs'], ['evaluation:negative'])
        self.assertLess(len(projected.history), len(history))
