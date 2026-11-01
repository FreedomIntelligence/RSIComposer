"""Decision-local receipts, version attribution and review bookkeeping (no services)."""

from copy import deepcopy
from dataclasses import asdict, replace
import json
import unittest
from unittest.mock import Mock, patch

from rsisys.experience.retrieval import BasicExperienceSystem
from rsisys.experience.storage import InMemoryExperienceStore
from rsisys.loops.contracts.artifacts import ArtifactRef
from rsisys.loops.contracts.candidate import Candidate
from rsisys.loops.contracts.coordination import ExecutionView, ExplorationProgress, ReviewDecision
from rsisys.loops.contracts.directive import Directive
from rsisys.loops.contracts.feedback import Feedback
from rsisys.loops.contracts.lifecycle import ExecutionStatus
from rsisys.loops.contracts.model_service import GenerationResponse, ModelRef, TokenUsage
from rsisys.loops.contracts.requests import EvaluationSpec
from rsisys.loops.decision_making.action_space import DemoDecisionActionSpace, OperationCall, OperationResult
from rsisys.loops.decision_making.harness.decision import Decision
from rsisys.loops.decision_making.harness.engine import DefaultDecisionHarness
from rsisys.loops.decision_making.harness.learner import DecisionTrainingRequest
from rsisys.loops.decision_making.harness.learner.backends.mock import MockDecisionLearner
from rsisys.loops.decision_making.harness.policy import ModelDecisionPolicy
from rsisys.loops.decision_making.harness.state import DecisionContext
from rsisys.runtime.artifacts import InMemoryArtifactStore


DECISION = ModelRef('test', 'decision', 'v1')
ACTOR = ModelRef('test', 'actor', 'v1')


class ModelPolicyInteractionTests(unittest.TestCase):
    def policy(self, payload, *, simulated=False, model=DECISION):
        backend = Mock(generate=Mock(return_value=GenerationResponse(json.dumps(payload), model, simulated)))
        return ModelDecisionPolicy(backend, DECISION), backend

    def test_only_selected_input_sources_propagate_simulation(self):
        real = ArtifactRef('real', 'dataset', 'memory://real')
        simulated = ArtifactRef('simulated', 'dataset', 'memory://simulated')
        feedback = Feedback('previous', ExecutionStatus.COMPLETED, 'mock', True, artifact_refs=(simulated,))
        context = DecisionContext('task', (), (), (OperationResult('real', (real,), False),),
                                  last_feedback=feedback)
        for ids, expected in ((['simulated'], True), (['real'], False), ([], False), (None, False)):
            with self.subTest(ids=ids):
                payload = {'kind': 'explore', 'strategy': 'execute', 'arguments': {}}
                if ids is not None:
                    payload['input_ids'] = ids
                policy, _ = self.policy(payload)
                decision = policy.decide(context)
                self.assertEqual(decision.directive.simulated, expected)
                self.assertEqual([r.artifact_id for r in decision.directive.input_refs],
                                 ids if ids is not None else ['real'])
        # An unused simulated preparation must not taint explicitly selected real input.
        context = replace(context, operation_results=(OperationResult('mock', (simulated,), True),),
                          last_feedback=replace(feedback, simulated=False, artifact_refs=(real,)))
        policy, _ = self.policy({'kind': 'explore', 'strategy': 'execute', 'input_ids': ['real']})
        self.assertFalse(policy.decide(context).directive.simulated)
        policy, _ = self.policy({'kind': 'explore', 'strategy': 'execute'})
        self.assertTrue(policy.decide(context).directive.simulated)
        policy, _ = self.policy({'kind': 'explore', 'strategy': 'execute', 'input_ids': []}, simulated=True)
        self.assertTrue(policy.decide(context).directive.simulated)

    def test_review_schedule_is_top_level_and_review_has_a_separate_prompt(self):
        for arguments in ({'arguments': {}}, {}):
            policy, backend = self.policy({'kind': 'explore', 'strategy': 'execute',
                                          'review_after_seconds': 2, **arguments})
            directive = policy.decide(DecisionContext('task', (), (), ())).directive
            self.assertEqual(directive.review_after_seconds, 2)
            backend.generate.return_value = GenerationResponse(
                '{"action":"cancel","next_review_seconds":null,"reason":"plateau"}', DECISION, False,
                TokenUsage(3, 2))
            progress = ExplorationProgress(directive.directive_id, 'running', directive,
                                           metrics={'loss': 0.4}, elapsed_seconds=5)
            result = policy.review(DecisionContext('task', (), (), (), execution=ExecutionView(ACTOR, progress)))
            self.assertEqual(result, ReviewDecision('cancel', None, 'plateau', False, TokenUsage(3, 2)))
            request = backend.generate.call_args.args[0]
            self.assertIn('Review the running execution', request.system)
            self.assertEqual(json.loads(request.prompt)['execution']['progress']['metrics'], {'loss': 0.4})

    def test_model_review_rejects_new_directives_invalid_schedules_and_wrong_versions(self):
        context = DecisionContext('task', (), (), ())
        for payload in ({'kind': 'explore'}, {'action': 'train'}, {'action': 'continue', 'next_review_seconds': 0},
                        {'action': 'cancel', 'arguments': {}}, []):
            with self.subTest(payload=payload), self.assertRaises(ValueError):
                self.policy(payload)[0].review(context)
        with self.assertRaisesRegex(ValueError, 'identity'):
            self.policy({'action': 'continue'}, model=ACTOR)[0].review(context)


class DecisionFixture(unittest.TestCase):
    def setUp(self):
        self.store = InMemoryExperienceStore()
        self.policy = ModelDecisionPolicy(Mock(), DECISION)
        self.policy.decide = Mock()
        self.harness = DefaultDecisionHarness(self.policy, Mock(), BasicExperienceSystem(self.store),
            learner=MockDecisionLearner(InMemoryArtifactStore(), pending_polls=0), memory_limit=1, experience_limit=1)
        self.space = DemoDecisionActionSpace()
        self.harness.update_execution(ExecutionView(ACTOR))

    def dispatch(self, *, payload=None, mode='execute'):
        directive = Directive(f'round-{self.harness.state.issued}', 'task', 'same', mode=mode, payload=payload)
        self.policy.decide.return_value = Decision('explore', directive=directive)
        self.harness.step('task', self.space)
        return directive

    def feedback(self, directive, **kwargs):
        return Feedback(directive.directive_id, ExecutionStatus.COMPLETED, 'done', False, **kwargs)


class DecisionReceiptTests(DecisionFixture):
    def test_invalid_feedback_is_rejected_before_any_recording(self):
        directive = self.dispatch()
        bad = self.feedback(directive)
        object.__setattr__(bad, 'status', 'running')
        for feedback in (object(), bad):
            before = deepcopy(self.harness.state)
            with self.subTest(feedback=feedback), patch.object(self.harness.memory, 'append') as append, \
                    patch.object(self.harness.experience, 'record') as record:
                with self.assertRaises(TypeError):
                    self.harness.receive_feedback(directive, feedback)
                append.assert_not_called()
                record.assert_not_called()
            self.assertEqual(self.harness.state, before)

    def test_evaluation_receipt_survives_short_history_and_replaces_other_protocol(self):
        candidate = Candidate('A', 'v1', {}, ('actor',))
        directive = self.dispatch()
        self.harness.receive_feedback(directive, self.feedback(directive, candidates=(candidate,)))
        directive = self.dispatch(mode='evaluate', payload=EvaluationSpec(candidate, 'first-protocol'))
        feedback = self.feedback(directive, quality_metrics={'score': 0.9}, evidence_refs=('evidence',))
        self.harness.receive_feedback(directive, feedback)
        self.harness.receive_feedback(directive, deepcopy(feedback))
        self.policy.decide.return_value = Decision('invoke', operation=OperationCall('prepare_data', {'text': 'data'}))
        self.harness.tools.execute.return_value = OperationResult('prepared', (), False)
        self.harness.step('task', self.space)
        # A fresh decision after local preparation still sees the evaluated candidate/protocol.
        self.policy.decide.return_value = Decision('finish')
        self.harness.step('task', self.space)
        context = self.policy.decide.call_args.args[0]
        self.assertEqual(context.history[0].kind, 'operation')
        receipt, = context.candidate_evaluations
        self.assertEqual((receipt.candidate_id, receipt.protocol_ref, receipt.quality_metrics),
                         ('A', 'first-protocol', {'score': 0.9}))
        receipt.quality_metrics.clear()
        self.assertEqual(self.harness.state.candidate_evaluations['A'].quality_metrics, {'score': 0.9})
        self.harness.state.finished = False
        directive = self.dispatch(mode='evaluate', payload=EvaluationSpec(candidate, 'other-protocol'))
        feedback = replace(self.feedback(directive), status=ExecutionStatus.FAILED, error='evaluation failed')
        self.harness.receive_feedback(directive, feedback)
        self.assertEqual(len(self.harness.state.candidate_evaluations), 1)
        latest = self.harness.state.candidate_evaluations['A']
        self.assertEqual((latest.protocol_ref, latest.status, latest.quality_metrics),
                         ('other-protocol', ExecutionStatus.FAILED, {}))
        self.assertEqual(self.store.records()[-1].details['evaluation'], asdict(latest))
        from rsisys.snapshot.codec import decode, encode
        self.assertEqual(decode(encode(self.harness.state)), self.harness.state)

    def test_failed_feedback_recording_does_not_publish_evaluation(self):
        candidate = Candidate('A', 'v1', {}, ('actor',))
        directive = self.dispatch(mode='evaluate', payload=EvaluationSpec(candidate, 'protocol'))
        feedback = self.feedback(directive, quality_metrics={'score': 1})
        with patch.object(self.harness.experience, 'record', side_effect=OSError('write failed')):
            with self.assertRaises(OSError):
                self.harness.receive_feedback(directive, feedback)
        self.assertEqual(self.harness.state.candidate_evaluations, {})
        self.harness.receive_feedback(directive, feedback)
        self.assertEqual(self.harness.state.candidate_evaluations['A'].quality_metrics, {'score': 1})
        self.assertEqual(len(self.harness.memory.history(kind='feedback')), 1)

    def test_old_directive_keeps_original_bindings_after_decision_model_update(self):
        directive = self.dispatch()
        job = self.harness.submit_learning(DecisionTrainingRequest(DECISION, (), 'mock'), idempotency_key='first')
        new_model = self.harness.apply_learning(job)
        self.harness.update_execution(ExecutionView(replace(ACTOR, revision='v2')))
        self.harness.receive_feedback(directive, self.feedback(directive))
        provenance = self.harness.state.directive_provenance[directive.directive_id]
        self.assertEqual((provenance.decision_model, provenance.execution_model), (DECISION, ACTOR))
        self.assertEqual(self.store.records()[-1].details['provenance'], asdict(provenance))
        next_directive = self.dispatch()
        self.assertEqual(self.harness.state.directive_provenance[next_directive.directive_id].decision_model, new_model)
        self.assertEqual(self.harness.apply_learning(job), new_model)
        second_job = self.harness.submit_learning(DecisionTrainingRequest(new_model, (), 'mock'), idempotency_key='second')
        latest_model = self.harness.apply_learning(second_job)
        with self.assertRaisesRegex(ValueError, 'stale'):
            self.harness.apply_learning(job)
        self.assertEqual(self.policy.model, latest_model)


class ReviewRecordingTests(DecisionFixture):
    def test_invalid_policy_review_is_rejected_before_recording(self):
        directive = self.dispatch()
        progress = ExplorationProgress(directive.directive_id, 'running', directive)
        bad = ReviewDecision()
        object.__setattr__(bad, 'action', 'explore')
        for result in (object(), bad):
            self.policy.review = Mock(return_value=result)
            with self.subTest(result=result), patch.object(self.harness.memory, 'append') as append, \
                    patch.object(self.harness.experience, 'record') as record:
                with self.assertRaises((TypeError, ValueError)):
                    self.harness.review('task', progress)
                append.assert_not_called()
                record.assert_not_called()
            self.assertIsNone(self.harness.state.pending_review_record)
            self.assertEqual(self.harness.state.measured_input_tokens, 0)

    def test_review_recording_retries_only_writes_before_and_after_commit(self):
        for target in ('memory', 'experience'):
            for after_commit in (False, True):
                with self.subTest(target=target, after_commit=after_commit):
                    self.setUp()
                    directive = self.dispatch()
                    self.policy.review = Mock(return_value=ReviewDecision('cancel', reason='plateau', usage=TokenUsage(7, 3)))
                    progress = ExplorationProgress(directive.directive_id, 'running', directive, metrics={'loss': 0.4})
                    owner = getattr(self.harness, target)
                    method = 'append' if target == 'memory' else 'record'
                    commit = getattr(owner, method)
                    def fail(value):
                        if after_commit:
                            commit(value)
                        raise OSError('write failed')
                    with patch.object(owner, method, side_effect=fail), self.assertRaises(OSError):
                        self.harness.review('task', progress)
                    ident = self.harness.state.pending_review_record.record_id
                    result = self.harness.review('task', replace(progress, metrics={'loss': 0.3}))
                    self.assertEqual(result.action, 'cancel')
                    self.policy.review.assert_called_once()
                    self.assertEqual(self.harness.state.measured_input_tokens, 7)
                    self.assertEqual(self.harness.state.measured_output_tokens, 3)
                    self.assertIsNone(self.harness.state.pending_review_record)
                    records = [r for r in self.store.records() if r.record_id == ident]
                    self.assertEqual(len(records), 1)
                    self.assertEqual(records[0].details['execution']['progress']['metrics'], {'loss': 0.4})
                    self.harness.tools.execute.assert_not_called()

    def test_terminal_feedback_flushes_obsolete_review_without_cancelling(self):
        directive = self.dispatch()
        self.policy.review = Mock(return_value=ReviewDecision('cancel'))
        progress = ExplorationProgress(directive.directive_id, 'running', directive)
        with patch.object(self.harness.experience, 'record', side_effect=OSError('write failed')):
            with self.assertRaises(OSError):
                self.harness.review('task', progress)
        # Validate delivery before flushing the pending review, too.
        bad = self.feedback(directive)
        object.__setattr__(bad, 'status', 'running')
        with patch.object(self.harness.experience, 'record') as record, self.assertRaises(TypeError):
            self.harness.receive_feedback(directive, bad)
        record.assert_not_called()
        self.assertIsNotNone(self.harness.state.pending_review_record)
        self.harness.receive_feedback(directive, self.feedback(directive))
        self.assertIsNone(self.harness.state.pending_review_record)
        self.assertIsNone(self.harness.pending())
        self.assertEqual([r.kind for r in self.store.records()], ['runtime_feedback', 'improvement'])
        self.policy.review.assert_called_once()
        self.harness.tools.execute.assert_not_called()
        self.dispatch()  # Completion leaves no stranded review blocking the next decision.

    def test_policy_without_review_disables_rescheduling_and_view_is_detached(self):
        directive = self.dispatch()
        class NoReview:
            pass
        self.harness.policy = NoReview()
        progress = ExplorationProgress(directive.directive_id, 'running', directive, metrics={'loss': 0.4})
        self.assertEqual(self.harness.review('task', progress), ReviewDecision())
        progress.metrics.clear()
        self.assertEqual(self.harness.state.execution.progress.metrics, {'loss': 0.4})
        with self.assertRaisesRegex(ValueError, 'terminal'):
            self.harness.review('task', replace(progress, feedback=self.feedback(directive)))


if __name__ == '__main__':
    unittest.main()
