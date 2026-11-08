"""Explicit evaluation inputs survive dispatch, backend retries and evidence writes."""

from copy import deepcopy
from dataclasses import asdict, replace
import json
import unittest
from unittest.mock import Mock

from rsisys.experience.retrieval import BasicExperienceSystem
from rsisys.experience.storage import InMemoryExperienceStore
from rsisys.loops.contracts.candidate import Candidate
from rsisys.loops.contracts.jobs import JobStatus
from rsisys.loops.contracts.lifecycle import ExecutionStatus
from rsisys.loops.contracts.model_service import GenerationResponse, ModelRef
from rsisys.loops.decision_making.action_space import DemoDecisionActionSpace, OperationResult
from rsisys.loops.decision_making.harness.engine import DefaultDecisionHarness
from rsisys.loops.decision_making.harness.policy import ModelDecisionPolicy
from rsisys.loops.decision_making.harness.tools import BoundDecisionTools
from rsisys.loops.decision_making.loop import DecisionMakingLoop
from rsisys.loops.exploration.action_space import EchoActionSpace
from rsisys.loops.exploration.harness.engine import DefaultExplorationHarness
from rsisys.loops.exploration.harness.evaluation import EvaluationRequest
from rsisys.loops.exploration.harness.evaluation.mock import MockEvaluator
from rsisys.loops.exploration.harness.learner.backends.mock import MockActorLearner
from rsisys.loops.exploration.harness.policy import ModelActorPolicy
from rsisys.loops.exploration.loop import ExplorationLoop
from rsisys.runtime.artifacts import InMemoryArtifactStore
from rsisys.runtime.runner import Runner


class TrackedArtifacts(InMemoryArtifactStore):
    def __init__(self):
        super().__init__()
        self.created = {}
        self.reads = []

    def put(self, content, *, kind):
        ref = super().put(content, kind=kind)
        self.created[ref.artifact_id] = ref
        return ref

    def get(self, ref):
        self.reads.append(ref)
        return super().get(ref)


class EvaluationInputTests(unittest.TestCase):
    def setUp(self):
        self.artifacts = TrackedArtifacts()
        model = self.artifacts.put(b'model fixture', kind='candidate_model')
        self.candidate = Candidate('candidate', 'base', {'model': model}, ('actor',))
        self.data = (
            self.artifacts.put(b'validation A', kind='evaluation_data'),
            self.artifacts.put(b'validation B', kind='evaluation_data'),
        )
        self.evaluator = MockEvaluator(self.artifacts, pending_polls=0)

    def report(self, result):
        ref = self.artifacts.created[result.evidence_refs[0]]
        self.assertEqual(ref.kind, 'mock_evaluation_report')
        return json.loads(self.artifacts.get(ref))

    def test_legacy_two_argument_request_still_evaluates_without_explicit_data(self):
        request = EvaluationRequest(self.candidate, 'protocol')
        self.assertEqual(request.data_refs, ())
        job = self.evaluator.submit(request, idempotency_key='legacy')
        outcome = self.evaluator.poll(job)
        self.assertEqual(outcome.status, JobStatus.SUCCEEDED)
        self.assertEqual(self.report(outcome.result)['request']['data_refs'], [])

    def test_data_refs_require_a_tuple_of_artifact_references(self):
        for refs in (None, 'artifact-id', list(self.data), ('artifact-id',), ({},), (self.data[0], None)):
            with self.subTest(refs=refs), self.assertRaisesRegex(TypeError, 'data_refs'):
                EvaluationRequest(self.candidate, 'protocol', refs)

    def test_direct_request_reads_every_input_and_records_refs_without_scores(self):
        request = EvaluationRequest(self.candidate, 'protocol', self.data)
        job = self.evaluator.submit(request, idempotency_key='direct')
        outcome = self.evaluator.poll(job)
        self.assertEqual(outcome.status, JobStatus.SUCCEEDED)
        self.assertEqual(self.artifacts.reads, [*self.candidate.changes.values(), *self.data])
        report = self.report(outcome.result)
        self.assertEqual(report['request'], json.loads(json.dumps(asdict(request))))
        self.assertTrue(report['simulated'])
        self.assertFalse(report['quality_evaluated'])
        self.assertEqual(outcome.result.metrics, {})
        self.assertTrue(outcome.result.simulated)
        self.assertEqual(self.evaluator.poll(job), outcome)
        self.assertEqual(self.artifacts.reads.count(self.data[0]), 1)
        self.assertEqual(self.artifacts.reads.count(self.data[1]), 1)

    def test_missing_or_mismatched_input_fails_instead_of_emitting_success_report(self):
        for invalid, expected_error in (
            (replace(self.data[0], artifact_id='missing'), 'KeyError'),
            (replace(self.data[0], digest='wrong digest'), 'ValueError'),
        ):
            with self.subTest(error=expected_error):
                request = EvaluationRequest(self.candidate, 'protocol', (invalid,))
                job = self.evaluator.submit(request, idempotency_key=expected_error)
                outcome = self.evaluator.poll(job)
                self.assertEqual(outcome.status, JobStatus.FAILED)
                self.assertIn(expected_error, outcome.error)
                self.assertIsNone(outcome.result)
        self.assertFalse(any(ref.kind == 'mock_evaluation_report' for ref in self.artifacts.created.values()))

    def test_idempotency_key_cannot_be_reused_with_different_evaluation_inputs(self):
        request = EvaluationRequest(self.candidate, 'protocol', (self.data[0],))
        job = self.evaluator.submit(request, idempotency_key='same')
        self.assertEqual(self.evaluator.submit(deepcopy(request), idempotency_key='same'), job)
        with self.assertRaisesRegex(ValueError, 'different request'):
            self.evaluator.submit(replace(request, data_refs=(self.data[1],)), idempotency_key='same')


class EvaluationDirectiveInputTests(unittest.TestCase):
    def setUp(self):
        self.artifacts = TrackedArtifacts()
        self.store = InMemoryExperienceStore()
        self.experience = BasicExperienceSystem(self.store)
        self.prepared = {}
        self.contexts = []
        model = ModelRef('test', 'model', 'base')

        def prepare(arguments):
            text = arguments['text']
            ref = self.artifacts.put(text.encode(), kind='input_data')
            self.prepared[text] = ref
            return OperationResult('Prepared data', (ref,), simulated=True)

        def generate(request):
            context = json.loads(request.prompt)
            self.contexts.append(context)
            step = len(self.contexts)
            if step in (1, 3, 4):
                text = {1: 'training', 3: 'validation', 4: 'unselected'}[step]
                payload = {'kind': 'invoke', 'operation': 'prepare_data', 'arguments': {'text': text}}
            elif step == 2:
                payload = {'kind': 'explore', 'mode': 'sft', 'strategy': 'train', 'recipe_ref': 'recipe',
                           'input_ids': [self.prepared['training'].artifact_id]}
            elif step == 5:
                payload = {'kind': 'explore', 'mode': 'evaluate', 'strategy': 'validate',
                           'candidate_id': context['last_feedback']['candidates'][0]['candidate_id'],
                           'protocol_ref': 'validation-protocol',
                           'input_ids': [self.prepared['validation'].artifact_id]}
            else:
                payload = {'kind': 'finish'}
            return GenerationResponse(json.dumps(payload), request.model, simulated=True)

        backend = Mock(generate=Mock(side_effect=generate))
        self.decision = DecisionMakingLoop(
            DefaultDecisionHarness(ModelDecisionPolicy(backend, model),
                                   BoundDecisionTools({'prepare_data': prepare}), self.experience),
            DemoDecisionActionSpace(directive_modes=('sft', 'evaluate')),
        )
        actor = ModelActorPolicy(Mock(), model)
        actor.act = Mock(side_effect=AssertionError('Training/evaluation must not invoke Actor.act'))
        self.evaluator = MockEvaluator(self.artifacts, pending_polls=0)
        self.evaluator.submit = Mock(wraps=self.evaluator.submit)
        self.exploration = ExplorationLoop(
            DefaultExplorationHarness(actor, Mock(), self.experience,
                                      learner=MockActorLearner(self.artifacts, pending_polls=0),
                                      evaluator=self.evaluator),
            EchoActionSpace(),
        )
        self.runner = Runner(self.decision, self.exploration)

    def finish_round(self):
        for _ in range(20):
            result = self.runner.run('train and validate', 1)
            if result.status != 'waiting':
                self.assertEqual(result.status, 'completed')
                return self.decision.harness.state.last_feedback
        self.fail('Directive did not complete')

    def record_for(self, feedback):
        self.assertEqual(len(feedback.evidence_refs), 1)
        return next(record for record in self.store.records() if record.record_id == feedback.evidence_refs[0])

    def assert_evaluation_evidence(self, training_feedback, evaluation_feedback):
        training = self.record_for(training_feedback)
        evaluation = self.record_for(evaluation_feedback)
        self.assertEqual(training.kind, 'training_run')
        self.assertEqual(training.details['request']['data_refs'], (asdict(self.prepared['training']),))
        self.assertEqual(evaluation.kind, 'evaluation')
        self.assertEqual(evaluation.details['request']['data_refs'], (asdict(self.prepared['validation']),))
        self.assertEqual(evaluation_feedback.status, ExecutionStatus.COMPLETED)
        self.assertEqual(evaluation_feedback.quality_metrics, {})
        self.assertTrue(evaluation_feedback.simulated)
        report_ref = self.artifacts.created[evaluation.details['outcome']['result']['evidence_refs'][0]]
        report = json.loads(self.artifacts.get(report_ref))
        self.assertEqual(report['request']['data_refs'], [asdict(self.prepared['validation'])])
        self.assertFalse(report['quality_evaluated'])
        improvements = [record for record in self.store.records() if record.kind == 'improvement']
        self.assertEqual([record.source_refs for record in improvements],
                         [training_feedback.evidence_refs, evaluation_feedback.evidence_refs])
        self.exploration.harness.policy.act.assert_not_called()
        self.assertEqual(self.exploration.harness.environment.mock_calls, [])

    def test_runner_model_input_ids_select_validation_data_for_backend_and_feedback(self):
        training = self.finish_round()
        evaluation = self.finish_round()
        request = self.evaluator.submit.call_args.args[0]
        self.assertEqual(request.data_refs, (self.prepared['validation'],))
        self.assertEqual(request.candidate, training.candidates[0])
        self.assertEqual(request.protocol_ref, 'validation-protocol')
        self.evaluator.submit.assert_called_once()
        self.assertEqual(self.artifacts.reads.count(self.prepared['validation']), 1)
        self.assertNotIn(self.prepared['unselected'], self.artifacts.reads)
        self.assertEqual(len(self.contexts[4]['operation_results']), 2)
        self.assertEqual(len(self.contexts), 5)
        self.assert_evaluation_evidence(training, evaluation)

    def test_lost_submit_response_retries_same_request_key_and_validation_data(self):
        training = self.finish_round()
        submit = self.evaluator.submit
        attempts = []

        def lose_response(request, *, idempotency_key):
            job = submit(request, idempotency_key=idempotency_key)
            attempts.append((deepcopy(request), idempotency_key, job))
            if len(attempts) == 1:
                raise OSError('evaluation submit response lost')
            return job

        self.evaluator.submit = lose_response
        with self.assertRaisesRegex(OSError, 'submit response lost'):
            self.finish_round()
        self.assertTrue(self.runner.busy)
        pending = self.decision.pending()
        self.assertEqual(pending.input_refs, (self.prepared['validation'],))
        evaluation = self.finish_round()
        self.assertEqual(len(attempts), 2)
        self.assertEqual(attempts[0], attempts[1])
        self.assertEqual(attempts[0][0].data_refs, pending.input_refs)
        self.assertEqual(attempts[0][1], f'directive:{pending.directive_id}')
        self.assertEqual(len(self.contexts), 5)
        self.assertEqual(self.artifacts.reads.count(self.prepared['validation']), 1)
        reports = [ref for ref in self.artifacts.created.values() if ref.kind == 'mock_evaluation_report']
        self.assertEqual(len(reports), 1)
        self.assert_evaluation_evidence(training, evaluation)

    def test_evaluation_record_commit_response_loss_keeps_input_refs_and_one_record(self):
        training = self.finish_round()
        record = self.experience.record
        attempts = []

        def lose_response(value):
            record(value)
            if value.kind == 'evaluation':
                attempts.append(deepcopy(value))
                if len(attempts) == 1:
                    raise OSError('evaluation record response lost')

        self.experience.record = lose_response
        with self.assertRaisesRegex(OSError, 'record response lost'):
            self.finish_round()
        evaluation = self.finish_round()
        self.assertEqual(len(attempts), 2)
        self.assertEqual(attempts[0], attempts[1])
        self.assertEqual(len([value for value in self.store.records() if value.kind == 'evaluation']), 1)
        self.evaluator.submit.assert_called_once()
        self.assertEqual(self.artifacts.reads.count(self.prepared['validation']), 1)
        self.assert_evaluation_evidence(training, evaluation)


if __name__ == '__main__':
    unittest.main()
