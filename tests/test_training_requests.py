from dataclasses import asdict, replace
import json
import unittest

from rsisys.loops.contracts.jobs import JobStatus
from rsisys.loops.contracts.model_service import ModelRef
from rsisys.loops.decision_making.harness.learner import DecisionTrainingRequest
from rsisys.loops.decision_making.harness.learner.backends.mock import MockDecisionLearner
from rsisys.loops.exploration.harness.learner import ActorTrainingRequest
from rsisys.loops.exploration.harness.learner.backends.mock import MockActorLearner
from rsisys.runtime.artifacts import InMemoryArtifactStore


class TrainingRequestTests(unittest.TestCase):
    def setUp(self):
        self.base = ModelRef('external', 'model', 'v1')
        self.artifacts = InMemoryArtifactStore()
        self.data = self.artifacts.put(b'external dataset', kind='training_data')

    def test_custom_methods_are_backend_defined(self):
        actor = ActorTrainingRequest(self.base, (self.data,), 'external-recipe', method='dpo')
        decision = DecisionTrainingRequest(self.base, (), 'external-recipe', method='distillation')
        self.assertEqual(actor.method, 'dpo')
        self.assertEqual(decision.method, 'distillation')

    def test_external_rl_can_collect_its_own_trajectories(self):
        request = ActorTrainingRequest(self.base, (), 'external-rollouts', method='rl')
        self.assertEqual(request.data_refs, ())
        self.assertEqual(request.trajectory_refs, ())

    def test_external_sft_can_resolve_data_from_recipe(self):
        request = ActorTrainingRequest(self.base, (), 'external-dataset', method='sft')
        self.assertEqual(request.data_refs, ())

    def test_decision_preserves_old_positional_arguments_and_defaults(self):
        request = DecisionTrainingRequest(self.base, ('experience-1',), 'recipe')
        self.assertEqual(request.base, self.base)
        self.assertEqual(request.experience_refs, ('experience-1',))
        self.assertEqual(request.recipe_ref, 'recipe')
        self.assertEqual(request.data_refs, ())
        self.assertEqual(request.method, 'sft')

    def test_decision_accepts_external_data_without_experience(self):
        request = DecisionTrainingRequest(self.base, (), 'recipe', (self.data,), 'sft')
        learner = MockDecisionLearner(self.artifacts, pending_polls=0)
        job = learner.submit(request, idempotency_key='external-data')
        outcome = learner.poll(job)
        self.assertEqual(outcome.status, JobStatus.SUCCEEDED)
        result = learner.result(job)
        manifest = json.loads(self.artifacts.get(result.candidate.changes['model']))
        self.assertEqual(manifest['request']['data_refs'], [asdict(self.data)])
        self.assertEqual(manifest['request']['experience_refs'], [])
        self.assertEqual(manifest['request']['method'], 'sft')

    def test_decision_mock_validates_every_external_artifact(self):
        for invalid in (replace(self.data, digest='invalid'), replace(self.data, artifact_id='missing')):
            with self.subTest(invalid=invalid):
                learner = MockDecisionLearner(self.artifacts, pending_polls=0)
                request = DecisionTrainingRequest(self.base, (), 'recipe', (self.data, invalid))
                job = learner.submit(request, idempotency_key='invalid-data')
                outcome = learner.poll(job)
                self.assertEqual(outcome.status, JobStatus.FAILED)
                self.assertIsNone(outcome.result)
                self.assertTrue(outcome.error)
                with self.assertRaises(RuntimeError):
                    learner.result(job)

    def test_request_structure_is_validated_for_both_learners(self):
        requests = (
            ActorTrainingRequest(self.base, (), 'recipe'),
            DecisionTrainingRequest(self.base, (), 'recipe'),
        )
        invalid_fields = {
            'base': (None, 'model', asdict(self.base)),
            'method': (None, 42, '', '  '),
            'recipe_ref': (None, 42, '', '  '),
            'data_refs': (None, [], [self.data], self.data, ('artifact-id',), (self.data, None)),
        }
        for request in requests:
            for field, values in invalid_fields.items():
                for value in values:
                    with self.subTest(request=type(request).__name__, field=field, value=value):
                        with self.assertRaises(ValueError):
                            replace(request, **{field: value})

    def test_record_refs_require_tuples_of_strings(self):
        requests = (
            (ActorTrainingRequest(self.base, (), 'recipe'), 'trajectory_refs'),
            (DecisionTrainingRequest(self.base, (), 'recipe'), 'experience_refs'),
        )
        for request, field in requests:
            for value in (None, [], ['record'], 'record', (self.data,), ('record', 42)):
                with self.subTest(field=field, value=value):
                    with self.assertRaises(ValueError):
                        replace(request, **{field: value})
            self.assertEqual(getattr(replace(request, **{field: ('record',)}), field), ('record',))

    def test_actor_mock_rejects_unsupported_methods_on_submit(self):
        request = ActorTrainingRequest(self.base, (self.data,), 'recipe', method='dpo')
        learner = MockActorLearner(self.artifacts)
        with self.assertRaisesRegex(ValueError, 'MockActorLearner only supports sft or rl'):
            learner.submit(request, idempotency_key='unsupported')

    def test_actor_mock_requires_its_local_training_inputs_on_submit(self):
        learner = MockActorLearner(self.artifacts)
        for method, message in (('sft', 'SFT requires data artifacts'), ('rl', 'RL requires trajectory records')):
            with self.subTest(method=method):
                request = ActorTrainingRequest(self.base, (), 'recipe', method=method)
                with self.assertRaisesRegex(ValueError, message):
                    learner.submit(request, idempotency_key=method)

    def test_actor_mock_keeps_builtin_sft_and_rl(self):
        requests = (
            ActorTrainingRequest(self.base, (self.data,), 'recipe'),
            ActorTrainingRequest(self.base, (), 'recipe', 'rl', ('trajectory-1',)),
        )
        learner = MockActorLearner(self.artifacts, pending_polls=0)
        for request in requests:
            with self.subTest(method=request.method):
                job = learner.submit(request, idempotency_key=request.method)
                outcome = learner.poll(job)
                self.assertEqual(outcome.status, JobStatus.SUCCEEDED)
                self.assertTrue(outcome.result.simulated)
                self.assertEqual(outcome.result.base, self.base)


if __name__ == '__main__':
    unittest.main()
