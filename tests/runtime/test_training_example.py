"""Exercise the standalone CPU backend with real child processes."""
from tests import ROOT as REPO_ROOT
from dataclasses import replace
import importlib.util
from pathlib import Path
import sys
import time
import unittest

from rsisys.loops.contracts.jobs import JobStatus
from rsisys.loops.exploration.harness.learner import ActorTrainingRequest
from rsisys.runtime.artifacts import InMemoryArtifactStore

spec = importlib.util.spec_from_file_location('training_example',
    REPO_ROOT / 'examples/components/backend_training.py')
example = importlib.util.module_from_spec(spec)
sys.modules[spec.name] = example
spec.loader.exec_module(example)


class TrainingExampleTests(unittest.TestCase):
    def test_real_training_changes_weight_and_reduces_loss(self):
        output = example.main()
        self.assertTrue(output['training_performed'])
        self.assertFalse(output['llm_training'])
        self.assertFalse(output['simulated'])
        self.assertLess(output['loss_after'], output['loss_before'])
        self.assertAlmostEqual(output['checkpoint']['weight'], 2.0, places=4)
        self.assertAlmostEqual(output['prediction_at_3'], 6.0, places=4)
        self.assertEqual(output['active_model'], output['training_result']['model'])

    def test_worker_failure_is_terminal_and_duplicate_submit_reuses_job(self):
        artifacts = InMemoryArtifactStore()
        invalid = artifacts.put(b'not valid training JSON', kind='training_data')
        learner = example.SubprocessActorLearner(artifacts)
        self.addCleanup(learner.close)
        request = ActorTrainingRequest(example.TrainingOnlyPolicy.model, (invalid,),
                                       'toy-recipe', method='linear_regression')
        job = learner.submit(request, idempotency_key='same')
        self.assertEqual(learner.submit(request, idempotency_key='same'), job)
        with self.assertRaisesRegex(ValueError, 'different request'):
            learner.submit(replace(request, data_refs=()), idempotency_key='same')
        deadline = time.monotonic() + 10
        outcome = learner.poll(job)
        while not outcome.terminal and time.monotonic() < deadline:
            time.sleep(0.01)
            outcome = learner.poll(job)
        self.assertEqual(outcome.status, JobStatus.FAILED)
        self.assertIn('Worker exited', outcome.error)
        self.assertEqual(learner.poll(job), outcome)
        with self.assertRaises(RuntimeError):
            learner.result(job)
        learner.close()
        self.assertIsNotNone(learner._jobs[job].process.poll())
