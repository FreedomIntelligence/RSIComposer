"""Audit regressions; synthetic meters and local CPU integration fixtures only."""
from dataclasses import asdict, replace
import unittest
from unittest.mock import Mock, patch

from rsisys.loops._memory import HistoryEntry
from rsisys.loops.decision_making.harness.context import BasicContextCompactor
from rsisys.loops.decision_making.harness.state import DecisionContext
from rsisys.runtime.json_tools import encoded
import test_medical_experiment as medical_fixture


def measure(context):
    return len(encoded(asdict(context)))


def records(prefix):
    return tuple(HistoryEntry(f'{prefix}{i}', 'operation', None, f'Observation {i}', False,
        details={'data': 'x' * 1800}) for i in range(12))


class RollingSummaryAuditTests(unittest.TestCase):
    def test_prior_summary_is_replaceable_not_an_irreducible_fact(self):
        compactor = BasicContextCompactor(trigger_tokens=22000, target_tokens=8500, summary_tokens=1800)
        raw = replace(DecisionContext('Investigate', (), (), ()), history=records('first-'))
        first = compactor.compact(raw, measure=measure, summarize=lambda *args: 'x' * 1700)
        later = replace(raw, objective='x' * 2000, history=(*raw.history, *records('later-')))
        writer = Mock(return_value='Retain the original archive sources and uncertainty.')
        second = compactor.compact(later, measure=measure, summarize=writer)
        self.assertEqual(writer.call_args.args[1], first.context_summary)
        self.assertEqual(second.history[-2:], later.history[-2:])
        self.assertLessEqual(measure(second), 8500)
        self.assertEqual(compactor.compactions, 2)
        restored = BasicContextCompactor(trigger_tokens=22000, target_tokens=8500, summary_tokens=1800)
        restored.restore_state(compactor.export_state())
        self.assertEqual(restored.compact(later, measure=measure), second)
        self.assertEqual(len(later.history), 24)


class FinalResultAuditTests(unittest.TestCase):
    setUp = medical_fixture.MedicalExperimentIntegrationTests.setUp
    execute = medical_fixture.MedicalExperimentIntegrationTests.execute
    read = medical_fixture.MedicalExperimentIntegrationTests.read

    def test_terminal_monitor_failure_still_preserves_authoritative_result(self):
        from examples.medical.experiment import JournalMonitor
        original = JournalMonitor.stage
        def fail_terminal(monitor, stage, *, status=None):
            if status is not None and status != 'running':
                raise OSError('fixture terminal monitor storage failure')
            return original(monitor, stage, **({'status': status} if status is not None else {}))
        with patch.object(JournalMonitor, 'stage', fail_terminal):
            result = self.execute([medical_fixture.FINISH])
        self.assertEqual(self.read('result.json'), result)
        self.assertEqual(result['status'], 'failed')
        self.assertEqual(result['reason'], 'cleanup_or_persistence_failed')
        self.assertTrue(result['cleanup']['confirmed'])
        self.assertTrue(any('terminal monitor storage failure' in text for text in result['cleanup_errors']))
        self.assertGreater(self.serving.release_count, 0)
        self.assertEqual(result['execution_outcome']['status'], 'completed')


class HarnessCleanupAuditTests(unittest.TestCase):
    def test_failed_callback_does_not_skip_other_releases_and_can_be_retried(self):
        from pathlib import Path
        from rsisys.config import DemoConfig, build_demo
        runner, _ = build_demo(DemoConfig.load(Path('configs/demo.json')))
        harness = runner.decision_making.harness
        events = []
        retries = [True]
        def close_worker():
            events.append('worker')
            if retries and retries.pop():
                raise TimeoutError('fixture unfinished worker')
        harness.add_cleanup(lambda: events.append('first'))
        harness.add_cleanup(close_worker)
        harness.add_cleanup(lambda: events.append('last'))
        with self.assertRaisesRegex(TimeoutError, 'unfinished worker'):
            harness.close()
        self.assertEqual(events, ['last', 'worker', 'first'])
        harness.close()
        harness.close()
        self.assertEqual(events, ['last', 'worker', 'first', 'worker'])


class MonitorRevisionAuditTests(unittest.TestCase):
    def test_bounded_history_keeps_current_revisions_for_each_visible_attempt(self):
        from rsimonitor.server import attempt_interpretations, MAX_ROWS
        first = {'attempt_id': 'a', 'revision': 1, 'title': 'First idea'}
        revisions = [first, *[{'attempt_id': 'b', 'revision': i, 'title': 'Continuing idea'}
                             for i in range(1, MAX_ROWS + 6)]]
        raw = {'schema_version': 1, 'authority': 'model_interpretation',
               'items': [first, revisions[-1]], 'revisions': revisions}
        view = attempt_interpretations(raw)
        identities = {(row['attempt_id'], row['revision']) for row in view['revisions']}
        self.assertIn(('a', 1), identities)
        self.assertIn(('b', MAX_ROWS + 5), identities)
        self.assertEqual(len(view['revisions']), MAX_ROWS)
        self.assertEqual(len(raw['revisions']), MAX_ROWS + 6)


class DecisionTrainingAuditTests(unittest.TestCase):
    def test_invalid_training_metrics_become_a_stable_failed_job_before_publication(self):
        from types import SimpleNamespace
        from rsisys.experience.retrieval import BasicExperienceSystem
        from rsisys.experience.storage import InMemoryExperienceStore
        from rsisys.loops.contracts.candidate import Candidate
        from rsisys.loops.contracts.jobs import JobRef, JobOutcome, JobStatus
        from rsisys.loops.contracts.model_service import ModelRef
        from rsisys.loops.contracts.training import TrainingResult
        from rsisys.loops.decision_making.harness.engine import DefaultDecisionHarness
        from rsisys.loops.decision_making.harness.learner import DecisionTrainingRequest
        for metrics in ({'loss': float('nan')}, {'loss': True}, {'': 1}, {'loss': '0'}):
            for cancel in (False, True):
                with self.subTest(metrics=metrics, cancel=cancel):
                    base = ModelRef('fixture', 'meta-agent', 'base')
                    job = JobRef('training', 'fixture')
                    result = TrainingResult(base, replace(base, revision='trained'),
                        Candidate('candidate', base.revision, {}, ('decision_maker',)), metrics, True)
                    backend = SimpleNamespace(submit=Mock(return_value=job), cancel=Mock(),
                        poll=Mock(return_value=JobOutcome(job, JobStatus.SUCCEEDED, result)))
                    experience = BasicExperienceSystem(InMemoryExperienceStore())
                    harness = DefaultDecisionHarness(SimpleNamespace(model=base), object(), experience, learner=backend)
                    harness.submit_learning(DecisionTrainingRequest(base, (), 'fixture'), idempotency_key='one')
                    outcome = (harness.cancel_learning if cancel else harness.poll_learning)(job)
                    self.assertEqual(outcome.status, JobStatus.FAILED)
                    self.assertIsNone(outcome.result)
                    self.assertTrue(outcome.simulated)
                    self.assertEqual(harness.poll_learning(job), outcome)
                    self.assertEqual(harness.cancel_learning(job), outcome)
                    self.assertFalse(harness.state.applied_training)
                    self.assertEqual(len(experience.retrieve()), 1)
                    backend.poll.assert_called_once()


if __name__ == '__main__':
    unittest.main()
