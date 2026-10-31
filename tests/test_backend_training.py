"""Whole-job training delegation does not invoke the built-in rollout sequencer."""
from dataclasses import replace
import json
import unittest
from unittest.mock import Mock

from rsisys.experience.retrieval import BasicExperienceSystem
from rsisys.experience.storage import InMemoryExperienceStore
from rsisys.loops.contracts.candidate import Candidate
from rsisys.loops.contracts.directive import Directive
from rsisys.loops.contracts.jobs import JobStatus
from rsisys.loops.contracts.model_service import ModelRef, GenerationResponse
from rsisys.loops.contracts.requests import TrainSpec
from rsisys.loops.contracts.training import TrainingResult
from rsisys.loops.decision_making.action_space import DemoDecisionActionSpace
from rsisys.loops.decision_making.action_space.directives import DirectiveDefinitions, training_directive
from rsisys.loops.decision_making.harness.engine import DefaultDecisionHarness
from rsisys.loops.decision_making.harness.policy import ModelDecisionPolicy
from rsisys.loops.decision_making.loop import DecisionMakingLoop
from rsisys.loops.exploration.action_space import EchoActionSpace
from rsisys.loops.exploration.harness.engine import DefaultExplorationHarness
from rsisys.loops.exploration.harness.learner.command import TrainingCommandHandler
from rsisys.loops.exploration.harness.policy import ModelActorPolicy
from rsisys.loops.exploration.loop import ExplorationLoop
from rsisys.runtime.jobs import MockJobQueue
from rsisys.runtime.runner import Runner

MODEL = ModelRef('test', 'actor', 'base')


class ExternalFixture:
    """Protocol fixture with backend-owned data/rollout, no real training."""
    def __init__(self):
        self.jobs = MockJobQueue('external', pending_polls=1)
        self.keys = []
        self.requests = []
        self.lose_response = False

    def submit(self, request, *, idempotency_key):
        self.keys.append(idempotency_key)
        self.requests.append(request)
        def train():
            candidate = Candidate('candidate', request.base.revision, {}, ('actor',))
            return TrainingResult(request.base, replace(request.base, revision='trained'), candidate, simulated=True)
        job = self.jobs.submit(request, train, idempotency_key=idempotency_key)
        if self.lose_response:
            self.lose_response = False
            raise OSError('submission response lost')
        return job

    def poll(self, job):
        return self.jobs.poll(job)

    def cancel(self, job):
        self.jobs.cancel(job)

    def result(self, job):
        return self.jobs.result(job)


class BackendTrainingTests(unittest.TestCase):
    def make(self, method='rl', apply=False):
        backend = ExternalFixture()
        experience = BasicExperienceSystem(InMemoryExperienceStore())
        actor = ModelActorPolicy(Mock(), MODEL)
        actor.act = Mock(side_effect=AssertionError('No local rollout expected'))
        environment = Mock()
        harness = DefaultExplorationHarness(actor, environment, experience, learner=backend,
            command_handlers={'train': TrainingCommandHandler()})
        model = Mock(generate=Mock(return_value=GenerationResponse(json.dumps({
            'kind': 'explore', 'strategy': 'delegate', 'mode': 'train',
            'arguments': {'recipe_ref': 'external-config', 'method': method, 'apply_result': apply},
        }), MODEL, False)))
        catalog = DirectiveDefinitions((training_directive(),))
        dm = DecisionMakingLoop(DefaultDecisionHarness(ModelDecisionPolicy(model, MODEL, directives=catalog),
            Mock(), experience), DemoDecisionActionSpace(directives=catalog))
        exploration = ExplorationLoop(harness, EchoActionSpace())
        return Runner(dm, exploration), backend

    def finish(self, runner):
        for _ in range(8):
            step = runner.step('task')
            if step.kind == 'completed':
                return step
        self.fail('job did not terminate')

    def test_backend_owned_rl_and_other_methods_need_no_local_trajectory(self):
        for method in ('rl', 'dpo', 'sft', 'custom'):
            with self.subTest(method=method):
                runner, backend = self.make(method)
                result = self.finish(runner)
                self.assertEqual(result.event['feedback']['status'], 'completed')
                request, = backend.requests
                self.assertEqual(request.method, method)
                self.assertEqual(request.trajectory_refs, ())
                self.assertEqual(request.data_refs, ())
                harness = runner.exploration.harness
                self.assertIsNone(harness.state.rl)
                harness.policy.act.assert_not_called()
                self.assertFalse(harness.environment.mock_calls)
                self.assertEqual(harness.policy.model, MODEL)
                self.assertTrue(harness.experience.retrieve(kinds=('training_run',)))

    def test_uncertain_submit_retries_same_job_and_applies_once(self):
        runner, backend = self.make(apply=True)
        backend.lose_response = True
        update = runner.exploration.harness.policy.update_model
        runner.exploration.harness.policy.update_model = Mock(wraps=update)
        with self.assertRaises(OSError):
            runner.step('task')
        self.finish(runner)
        self.assertEqual(len(backend.keys), 2)
        self.assertEqual(backend.keys[0], backend.keys[1])
        self.assertEqual(backend.requests[0], backend.requests[1])
        runner.exploration.harness.policy.update_model.assert_called_once()
        self.assertEqual(runner.exploration.harness.policy.model.revision, 'trained')

    def test_cancel_uncertain_submit_resolves_then_cancels_without_apply(self):
        runner, backend = self.make(apply=True)
        backend.lose_response = True
        with self.assertRaises(OSError):
            runner.step('task')
        runner.cancel()
        result = self.finish(runner)
        self.assertEqual(result.event['feedback']['status'], 'cancelled')
        self.assertEqual(runner.exploration.harness.policy.model, MODEL)
        job, = result.event['feedback']['jobs']
        self.assertEqual(len(set(backend.keys)), 1)

    def test_backend_validation_failure_does_not_claim_success(self):
        runner, backend = self.make()
        backend.submit = Mock(side_effect=ValueError('unsupported recipe'))
        with self.assertRaisesRegex(ValueError, 'unsupported recipe'):
            runner.step('task')
        self.assertTrue(runner.busy)
        self.assertEqual(runner.exploration.harness.policy.model, MODEL)

    def test_builtin_sft_still_validates_inputs_before_submission(self):
        from rsisys.loops.contracts.requests import SFTSpec
        runner, backend = self.make()
        directive = Directive('empty-sft', 'task', 'strategy', mode='sft', payload=SFTSpec('recipe'))
        with self.assertRaisesRegex(ValueError, 'requires data artifacts'):
            runner.exploration.accept(directive)
        self.assertEqual(runner.exploration.inspect('empty-sft').feedback.status, 'failed')
        self.assertEqual(backend.requests, [])

    def test_decision_training_external_data_is_preserved_in_evidence(self):
        from rsisys.loops.decision_making.harness.learner import DecisionTrainingRequest
        from rsisys.loops.decision_making.harness.learner.backends.mock import MockDecisionLearner
        from rsisys.runtime.artifacts import InMemoryArtifactStore
        artifacts = InMemoryArtifactStore()
        data = artifacts.put(b'external samples', kind='training_data')
        experience = BasicExperienceSystem(InMemoryExperienceStore())
        harness = DefaultDecisionHarness(Mock(), Mock(), experience,
            learner=MockDecisionLearner(artifacts, pending_polls=0))
        job = harness.submit_learning(DecisionTrainingRequest(MODEL, (), 'recipe', (data,)),
                                      idempotency_key='decision-data')
        self.assertEqual(harness.poll_learning(job).status, JobStatus.SUCCEEDED)
        record, = experience.retrieve(kinds=('training_run',))
        self.assertIn(data.artifact_id, record.source_refs)
        self.assertEqual(record.details['request']['experience_refs'], ())

    def test_train_spec_validation(self):
        for args in (('', 'rl'), ('recipe', ''), ('recipe', 1)):
            with self.assertRaises(ValueError):
                TrainSpec(*args)
        with self.assertRaises(TypeError):
            TrainSpec('recipe', 'rl', apply_result='yes')


if __name__ == '__main__':
    unittest.main()
