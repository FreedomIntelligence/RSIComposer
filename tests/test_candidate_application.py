"""Retained candidates can be evaluated and explicitly applied across rounds."""
from copy import deepcopy
from dataclasses import replace
import json
import unittest
from unittest.mock import Mock

from rsisys.experience.retrieval import BasicExperienceSystem
from rsisys.experience.storage import InMemoryExperienceStore
from rsisys.loops.contracts.artifacts import ArtifactRef
from rsisys.loops.contracts.candidate import Candidate
from rsisys.loops.contracts.directive import Directive
from rsisys.loops.contracts.jobs import JobStatus
from rsisys.loops.contracts.model_service import ModelRef
from rsisys.loops.contracts.requests import ApplyCandidateSpec, EvaluationSpec, TrainSpec
from rsisys.loops.contracts.training import ModelActivationPending, TrainingResult
from rsisys.loops.exploration.action_space import EchoActionSpace
from rsisys.loops.exploration.harness.engine import DefaultExplorationHarness
from rsisys.loops.exploration.harness.evaluation import EvaluationResult
from rsisys.loops.exploration.harness.learner import ActorTrainingRequest
from rsisys.loops.exploration.harness.learner.command import ApplyCandidateHandler, TrainingCommandHandler
from rsisys.loops.exploration.harness.rl import RLPlan, RLState
from rsisys.runtime.jobs import MockJobQueue
from rsisys.snapshot.codec import decode, encode


BASE = ModelRef('service', 'model', 'base')
NEW = replace(BASE, revision='trained')
RESULT = TrainingResult(BASE, NEW, Candidate('candidate', BASE.revision,
    {'weights': ArtifactRef('weights', 'checkpoint', 'file:///weights')}, ('actor',)), {'loss': 0.5})
ACTIONS = EchoActionSpace()


class Policy:
    def __init__(self):
        self.model = BASE
        self.binds = 0

    def update_model(self, model):
        self.model = model
        self.binds += 1


class Backend:
    def __init__(self, output):
        self.output = output
        self.queue = MockJobQueue('fixture', pending_polls=0)
        self.submissions = 0

    def submit(self, request, *, idempotency_key):
        self.submissions += 1
        return self.queue.submit(request, lambda: self.output, idempotency_key=idempotency_key)

    def poll(self, job):
        return self.queue.poll(job)

    def cancel(self, job):
        self.queue.cancel(job)


class CandidateApplicationTests(unittest.TestCase):
    def build(self, *, learner=True):
        store = InMemoryExperienceStore()
        harness = DefaultExplorationHarness(Policy(), Mock(), BasicExperienceSystem(store),
            learner=Backend(deepcopy(RESULT)) if learner else None,
            model_activator=Mock(activate=Mock(return_value=NEW)),
            command_handlers={'train': TrainingCommandHandler(), 'apply_candidate': ApplyCandidateHandler()})
        return harness, store

    def train(self, harness, key='training'):
        job = harness.submit_learning(ActorTrainingRequest(BASE, (), 'config', method='custom'),
                                      idempotency_key=key)
        harness.poll_learning(job)
        return job

    def directive(self, directive_id='apply'):
        return Directive(directive_id, 'task', 'selected by policy', mode='apply_candidate',
                         payload=ApplyCandidateSpec('candidate'))

    def test_spec_validation_and_opt_in(self):
        for value in (None, 1, False, '', '  ', [], {}):
            with self.subTest(value=value), self.assertRaises(ValueError):
                ApplyCandidateSpec(value)
        harness, _ = self.build()
        self.assertEqual(ApplyCandidateSpec('candidate').candidate_id, 'candidate')
        default = DefaultExplorationHarness(Policy(), Mock(), harness.experience)
        self.assertNotIn('apply_candidate', default.commands.handlers)

    def test_train_evaluate_twice_restore_then_apply_without_training_jobs(self):
        harness, store = self.build()
        harness.accept(Directive('train', 'task', '', mode='train', payload=TrainSpec('config', 'custom')))
        harness.advance('train', ACTIONS)
        trained = harness.advance('train', ACTIONS).feedback
        harness.acknowledge('train')
        self.assertEqual(trained.candidates, (RESULT.candidate,))
        for index, score in enumerate((0.1, 0.9)):
            protocol = f'protocol-{index}'
            backend = Backend(EvaluationResult('candidate', protocol, {'quality': score}, ('evidence',)))
            harness.evaluation.backend = backend
            directive = Directive(f'eval-{index}', 'task', '', mode='evaluate',
                                  payload=EvaluationSpec(trained.candidates[0], protocol))
            harness.accept(directive)
            harness.advance(directive.directive_id, ACTIONS)
            feedback = harness.advance(directive.directive_id, ACTIONS).feedback
            self.assertEqual(feedback.quality_metrics, {'quality': score})
            harness.acknowledge(directive.directive_id)
        self.assertEqual(harness.policy.model, BASE)
        harness.model_activator.activate.assert_not_called()
        self.assertEqual(harness.learner.submissions, 1)
        self.assertEqual([r.kind for r in store.records()], ['training_run', 'evaluation', 'evaluation'])

        restored, restored_store = self.build(learner=False)
        restored.state = decode(json.loads(json.dumps(encode(harness.state))))
        self.assertEqual(restored.state.candidate_results, {'candidate': RESULT})
        restored.accept(self.directive())
        feedback = restored.advance('apply', ACTIONS).feedback
        self.assertEqual(feedback.status, 'completed')
        self.assertEqual(feedback.adopted_candidate_ids, ('candidate',))
        self.assertEqual(feedback.candidates, (RESULT.candidate,))
        self.assertEqual(feedback.artifact_refs, tuple(RESULT.candidate.changes.values()))
        self.assertEqual(feedback.evidence_refs, (restored_store.records()[0].record_id,))
        self.assertEqual(restored.policy.model, NEW)
        self.assertEqual(restored.state.applied_candidates, {'candidate'})
        restored.acknowledge('apply')
        restored.state = decode(encode(restored.state))
        self.assertEqual(restored.apply_candidate('candidate'), NEW)
        self.assertEqual(restored.policy.binds, 1)
        self.assertFalse(restored.training.busy)

    def test_result_identity_is_checked_before_recording_and_saved_as_copy(self):
        harness, store = self.build()
        first = self.train(harness)
        harness.poll_learning(first)
        self.train(harness, 'same-content')
        self.assertEqual(len(store.records()), 2)
        read = harness.candidate_result('candidate')
        read.metrics['loss'] = 99
        read.candidate.changes.clear()
        self.assertEqual(harness.candidate_result('candidate'), RESULT)
        for index, conflicting in enumerate((replace(RESULT, metrics={'loss': 1}),
                replace(RESULT, model=replace(NEW, revision='other')))):
            harness.learner.output = conflicting
            with self.subTest(index=index), self.assertRaisesRegex(ValueError, 'Candidate ID reused'):
                self.train(harness, f'conflict-{index}')
            self.assertEqual(harness.candidate_result('candidate'), RESULT)
            self.assertEqual(len(store.records()), 2)

    def test_training_record_response_loss_keeps_candidate_and_record_identity(self):
        harness, store = self.build()
        record = harness.experience.record
        attempts = []

        def lose_response(value):
            attempts.append(deepcopy(value))
            record(value)
            if len(attempts) == 1:
                raise OSError('response lost')

        harness.experience.record = lose_response
        job = harness.submit_learning(ActorTrainingRequest(BASE, (), 'config'), idempotency_key='train')
        with self.assertRaises(OSError):
            harness.poll_learning(job)
        self.assertEqual(harness.candidate_result('candidate'), RESULT)
        harness.poll_learning(job)
        harness.poll_learning(job)
        self.assertEqual(attempts[0], attempts[1])
        self.assertEqual(store.records(), (attempts[0],))
        self.assertEqual(harness.learner.submissions, 1)

    def test_cancelled_training_and_invalid_results_do_not_create_candidates(self):
        harness, _ = self.build()
        job = harness.submit_learning(ActorTrainingRequest(BASE, (), 'config'), idempotency_key='cancel')
        self.assertEqual(harness.cancel_learning(job).status, JobStatus.CANCELLED)
        self.assertEqual(harness.state.candidate_results, {})
        for output in (replace(RESULT, candidate=replace(RESULT.candidate, affected_roles=('decision_maker',))),
                replace(RESULT, base=replace(BASE, model_id='wrong'))):
            harness, store = self.build()
            harness.learner.output = output
            with self.assertRaises(ValueError):
                self.train(harness)
            self.assertEqual(harness.state.candidate_results, {})
            self.assertEqual(store.records(), ())

    def test_job_and_candidate_application_share_activation_key_and_deduplicate(self):
        for first_by_job in (False, True):
            with self.subTest(first_by_job=first_by_job):
                harness, _ = self.build()
                job = self.train(harness)
                harness.model_activator.activate.side_effect = [ModelActivationPending('loading'), NEW]
                by_job = lambda: harness.apply_learning(job)
                by_candidate = lambda: harness.apply_candidate('candidate')
                first, second = (by_job, by_candidate) if first_by_job else (by_candidate, by_job)
                with self.assertRaises(ModelActivationPending):
                    first()
                self.assertEqual(harness.policy.model, BASE)
                self.assertEqual(second(), NEW)
                self.assertEqual(first(), NEW)
                calls = harness.model_activator.activate.call_args_list
                self.assertEqual(calls[0], calls[1])
                self.assertEqual(len(calls), 2)
                self.assertEqual(harness.policy.binds, 1)
                self.assertIn(job, harness.state.applied_training)

    def test_prepare_only_checks_payload_and_presence_and_prestart_cancel_has_no_effect(self):
        harness, store = self.build(learner=False)
        harness.state.candidate_results['candidate'] = deepcopy(RESULT)
        harness.policy = object()  # Not updatable: prepare must not attempt application.
        harness.model_activator = None
        harness.accept(self.directive())
        harness.cancel('apply')
        feedback = harness.advance('apply', ACTIONS).feedback
        self.assertEqual(feedback.status, 'cancelled')
        self.assertEqual(feedback.adopted_candidate_ids, ())
        self.assertEqual(feedback.candidates, (RESULT.candidate,))
        self.assertEqual(store.records(), ())
        self.assertEqual(harness.state.applied_candidates, set())
        for payload in (ApplyCandidateSpec('missing'), TrainSpec('config', 'custom')):
            harness, _ = self.build(learner=False)
            with self.subTest(payload=payload), self.assertRaises(ValueError):
                harness.accept(replace(self.directive(), payload=payload))

    def test_pending_command_activation_retry_or_cancel(self):
        for cancel in (False, True):
            with self.subTest(cancel=cancel):
                harness, store = self.build()
                self.train(harness)
                harness.model_activator.activate.side_effect = [ModelActivationPending('loading'), NEW]
                harness.accept(self.directive())
                with self.assertRaises(ModelActivationPending):
                    harness.advance('apply', ACTIONS)
                self.assertFalse(harness.inspect('apply').terminal)
                self.assertEqual(harness.policy.model, BASE)
                if cancel:
                    harness.cancel('apply')
                    with self.assertRaisesRegex(RuntimeError, 'cancelled'):
                        harness.apply_candidate_for_command('candidate', directive_id='apply')
                feedback = harness.advance('apply', ACTIONS).feedback
                self.assertEqual(feedback.status, 'cancelled' if cancel else 'completed')
                self.assertEqual(feedback.adopted_candidate_ids, () if cancel else ('candidate',))
                self.assertEqual(feedback.candidates, (RESULT.candidate,))
                self.assertEqual(harness.policy.model, BASE if cancel else NEW)
                self.assertEqual(harness.model_activator.activate.call_count, 1 if cancel else 2)
                self.assertEqual(len(store.records()), 2)
                if not cancel:
                    calls = harness.model_activator.activate.call_args_list
                    self.assertEqual(calls[0], calls[1])

    def test_application_record_retry_never_rebinds_and_preserves_committed_status(self):
        for after_write in (False, True):
            with self.subTest(after_write=after_write):
                harness, store = self.build()
                self.train(harness)
                record, attempts = harness.experience.record, []

                def flaky_record(value):
                    attempts.append(deepcopy(value))
                    if after_write or len(attempts) > 1:
                        record(value)
                    if len(attempts) == 1:
                        raise OSError('record unavailable')

                harness.experience.record = flaky_record
                harness.accept(self.directive())
                with self.assertRaises(OSError):
                    harness.advance('apply', ACTIONS)
                self.assertEqual(harness.policy.model, NEW)
                self.assertEqual(harness.inspect('apply').phase, 'recording')
                pending = deepcopy(harness.commands.receipts['apply'].record)
                harness.cancel('apply')
                feedback = harness.advance('apply', ACTIONS).feedback
                self.assertEqual(feedback.status, 'completed')
                self.assertEqual(attempts, [pending, pending])
                self.assertEqual(feedback.evidence_refs, (pending.record_id,))
                self.assertEqual(len(store.records()), 2)
                self.assertEqual(harness.policy.binds, 1)
                harness.model_activator.activate.assert_called_once()
                self.assertEqual(harness.advance('apply', ACTIONS).feedback, feedback)

    def test_application_checks_control_training_staleness_and_readiness(self):
        for blocker in ('directive', 'episode', 'interaction', 'rl', 'training', 'stale',
                        'unsupported', 'no-activator', 'wrong-ready', 'activation-failed'):
            with self.subTest(blocker=blocker):
                harness, _ = self.build()
                self.train(harness)
                activator = harness.model_activator
                if blocker == 'directive':
                    harness.accept(self.directive())
                elif blocker == 'episode':
                    harness.state.pending_episode = Mock()
                elif blocker == 'interaction':
                    harness.state.active_directive_id = 'executing'
                elif blocker == 'rl':
                    harness.state.rl = RLState(RLPlan(Directive('rollout', 'task', ''), 'config', 1))
                elif blocker == 'training':
                    harness.submit_learning(ActorTrainingRequest(BASE, (), 'config'), idempotency_key='other')
                elif blocker == 'stale':
                    harness.policy.model = replace(BASE, revision='stale')
                elif blocker == 'unsupported':
                    harness.policy = object()
                elif blocker == 'no-activator':
                    harness.model_activator = None
                elif blocker == 'wrong-ready':
                    activator.activate.return_value = BASE
                else:
                    activator.activate.side_effect = ValueError('rejected')
                with self.assertRaises((RuntimeError, ValueError, TypeError)):
                    harness.apply_candidate('candidate')
                self.assertEqual(harness.state.applied_candidates, set())
                if blocker not in ('wrong-ready', 'activation-failed'):
                    activator.activate.assert_not_called()
                if blocker != 'unsupported':
                    self.assertEqual(harness.policy.binds, 0)

    def test_command_checks_stale_base_at_advance_and_adapter_requires_ownership(self):
        harness, _ = self.build()
        self.train(harness)
        with self.assertRaisesRegex(RuntimeError, 'owning directive'):
            harness.apply_candidate_for_command('candidate', directive_id='missing')
        harness.accept(self.directive())
        harness.policy.model = replace(BASE, revision='changed-after-prepare')
        with self.assertRaisesRegex(ValueError, 'stale'):
            harness.advance('apply', ACTIONS)
        harness.model_activator.activate.assert_not_called()
        self.assertEqual(harness.policy.binds, 0)


if __name__ == '__main__':
    unittest.main()
