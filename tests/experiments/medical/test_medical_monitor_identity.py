"""Observed progress must share the backend job identity, including terminal failures."""
from pathlib import Path
from tempfile import TemporaryDirectory
from types import SimpleNamespace as NS
from unittest import TestCase
from unittest.mock import patch
from experiments.medical.experiment import evaluator_factory
from experiments.medical.evaluation import MedicalEvaluator
from rsisys.loops.contracts.jobs import JobRef, JobOutcome, JobStatus


class EvaluationMonitorIdentity(TestCase):
    def test_progress_and_failure_share_job_and_baseline_is_explicit(self):
        events = []
        monitor = NS(event=lambda event, **fields: events.append(fields))
        with TemporaryDirectory() as directory:
            cls = evaluator_factory(monitor, Path(directory))
            evaluator = object.__new__(cls)
            evaluator._partial, evaluator._outputs = {}, {}
            evaluator._task = NS(development=[NS(data=NS(read=lambda: [1]))])
            system = NS(version='candidate')
            job = JobRef('real-job', 'medical-development')
            evaluator._jobs = {job: (NS(candidate=NS(candidate_id='candidate')), None)}

            def measure(_self, _system):
                _self._partial['candidate'] = [{'correct': True}]
                return None

            def poll(_self, _job):
                _self._measure(system)
                return JobOutcome(_job, JobStatus.FAILED, error='interpretation failed')

            with patch.object(MedicalEvaluator, '_measure', measure), patch.object(MedicalEvaluator, 'poll', poll):
                evaluator.poll(job)
                self.assertEqual([r['job_id'] for r in events], ['real-job', 'real-job'])
                self.assertEqual(events[-1]['status'], 'failed')
                self.assertEqual(events[-1]['error'], 'interpretation failed')
                self.assertIsNone(evaluator._monitor_job)
                evaluator._partial.clear()
                evaluator._measure(system)
                self.assertEqual(events[-1]['role'], 'baseline')
                self.assertEqual(events[-1]['job_id'], 'baseline:candidate')
