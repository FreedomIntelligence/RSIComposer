"""Exploration-owned terminal validation, retained receipts and version boundaries."""
from copy import deepcopy
from dataclasses import replace
import unittest
from unittest.mock import Mock, patch

from rsisys.experience.retrieval import BasicExperienceSystem
from rsisys.experience.storage import InMemoryExperienceStore
from rsisys.loops.contracts.candidate import Candidate
from rsisys.loops.contracts.directive import Directive
from rsisys.loops.contracts.lifecycle import ExecutionStatus
from rsisys.loops.contracts.model_service import ModelRef
from rsisys.loops.contracts.requests import ApplyCandidateSpec, EvaluationSpec, RLSpec, TrainSpec
from rsisys.loops.contracts.training import TrainingResult
from rsisys.loops.exploration.action_space import ActionProposal, EchoActionSpace
from rsisys.loops.exploration.harness.commands import InvalidJobResult
from rsisys.loops.exploration.harness.engine import DefaultExplorationHarness
from rsisys.loops.exploration.harness.environment import Transition
from rsisys.loops.exploration.harness.evaluation import EvaluationResult
from rsisys.loops.exploration.harness.learner import ActorTrainingRequest
from rsisys.loops.exploration.harness.learner.command import ApplyCandidateHandler, TrainingCommandHandler
from rsisys.runtime.jobs import MockJobQueue

BASE = ModelRef('fixture', 'model', 'v0')
FIRST = replace(BASE, revision='v1')
RESULT = TrainingResult(BASE, FIRST, Candidate('first', BASE.revision, {}, ('actor',)), {'loss': 0.5})
ACTIONS = EchoActionSpace()


class Backend:
    def __init__(self, result):
        self.result = result
        self.queue = MockJobQueue('recovery', pending_polls=0)
        self.poll = Mock(side_effect=self.queue.poll)
        self.cancel = Mock(side_effect=self.queue.cancel)

    def submit(self, request, *, idempotency_key):
        return self.queue.submit(request, lambda: self.result, idempotency_key=idempotency_key)


class Policy:
    def __init__(self):
        self.model = BASE
        self.binds = 0

    def update_model(self, model):
        self.model = model
        self.binds += 1

    def act(self, context):
        return ActionProposal('echo', {'text': 'fixture'}, False, self.model)


class ExplorationRecoveryTests(unittest.TestCase):
    def build(self):
        store = InMemoryExperienceStore()
        harness = DefaultExplorationHarness(
            Policy(), Mock(reset=Mock(return_value='fixture'),
                           step=Mock(return_value=Transition('done', True, False))),
            BasicExperienceSystem(store), learner=Backend(deepcopy(RESULT)),
            evaluator=Backend(EvaluationResult('first', 'protocol', {'quality': 0.8}, ())),
            model_activator=Mock(activate=Mock(side_effect=lambda result, **kw: result.model)),
            command_handlers={'train': TrainingCommandHandler(), 'apply_candidate': ApplyCandidateHandler()},
        )
        return harness, store

    def directive(self, mode, ident='command'):
        payload = (TrainSpec('recipe', 'custom', apply_result=True) if mode == 'train'
                   else EvaluationSpec(RESULT.candidate, 'protocol'))
        return Directive(ident, 'task', '', mode=mode, payload=payload)

    def test_invalid_terminal_results_fail_once_and_release_after_ack(self):
        cases = (
            ('train', 'type', {'bad': 'result'}),
            ('train', 'base', replace(RESULT, base=replace(BASE, model_id='wrong'))),
            ('train', 'role', replace(RESULT, candidate=replace(RESULT.candidate, affected_roles=('decision_maker',)))),
            ('train', 'collision', replace(RESULT, metrics={'loss': 1})),
            ('evaluate', 'type', {'bad': 'result'}),
            ('evaluate', 'candidate', EvaluationResult('wrong', 'protocol', {}, ())),
            ('evaluate', 'protocol', EvaluationResult('first', 'wrong', {}, ())),
        )
        for mode, reason, output in cases:
            for cancel in (False, True):
                with self.subTest(mode=mode, reason=reason, cancel=cancel):
                    h, store = self.build()
                    backend = h.learner if mode == 'train' else h.evaluation.backend
                    backend.result = output
                    if reason == 'collision':
                        h.state.candidate_results['first'] = deepcopy(RESULT)
                    directive = self.directive(mode)
                    h.accept(directive)
                    job = h.advance('command', ACTIONS).job
                    if cancel:
                        # A public poll can cache the invalid terminal before cancellation.
                        poll = h.poll_learning if mode == 'train' else h.poll_evaluation
                        with self.assertRaises(InvalidJobResult):
                            poll(job)
                        h.cancel('command')
                    with self.assertRaises(InvalidJobResult):
                        h.advance('command', ACTIONS)
                    pending = h.pending()
                    self.assertEqual(pending.feedback.status, ExecutionStatus.FAILED)
                    self.assertEqual(pending.feedback.jobs, (job,))
                    self.assertTrue(pending.feedback.simulated)
                    self.assertFalse(pending.feedback.candidates)
                    self.assertFalse(pending.feedback.evidence_refs)
                    self.assertTrue(h.command_busy)
                    self.assertEqual(h.advance('command', ACTIONS), pending)
                    self.assertEqual(h.cancel('command'), pending)
                    backend.poll.assert_called_once()
                    self.assertEqual(h.policy.binds, 0)
                    self.assertEqual(store.records(), ())
                    h.acknowledge('command')
                    self.assertIsNone(h.pending())
                    h.accept(Directive('next', 'task', ''))
                    self.assertEqual(h.pending().directive_id, 'next')

    def test_transport_and_backend_value_errors_remain_retryable(self):
        for mode in ('train', 'evaluate'):
            for failure in (OSError('temporary transport failure'), ValueError('backend decoding failed')):
                with self.subTest(mode=mode, failure=type(failure).__name__):
                    h, _ = self.build()
                    backend = h.learner if mode == 'train' else h.evaluation.backend
                    h.accept(self.directive(mode))
                    h.advance('command', ACTIONS)
                    backend.poll.side_effect = failure
                    with self.assertRaises(type(failure)):
                        h.advance('command', ACTIONS)
                    self.assertFalse(h.pending().terminal)
                    backend.poll.side_effect = backend.queue.poll
                    self.assertEqual(h.advance('command', ACTIONS).feedback.status, ExecutionStatus.COMPLETED)

    def test_experience_failure_retries_same_record_without_repolling_or_early_binding(self):
        for mode in ('train', 'evaluate'):
            for failure in (OSError('store unavailable'), ValueError('store rejected write')):
                with self.subTest(mode=mode, failure=type(failure).__name__):
                    h, store = self.build()
                    h.accept(self.directive(mode))
                    h.advance('command', ACTIONS)
                    record, attempts = h.experience.record, []
                    def flaky(value):
                        attempts.append(deepcopy(value))
                        record(value)
                        if len(attempts) == 1:
                            raise failure  # Committed, but response was lost.
                    h.experience.record = flaky
                    with self.assertRaises(type(failure)):
                        h.advance('command', ACTIONS)
                    self.assertFalse(h.pending().terminal)
                    self.assertEqual(h.policy.binds, 0)
                    feedback = h.advance('command', ACTIONS).feedback
                    self.assertEqual(feedback.status, ExecutionStatus.COMPLETED)
                    self.assertEqual(attempts[0], attempts[1])
                    self.assertEqual(len(store.records()), 1)
                    (h.learner if mode == 'train' else h.evaluation.backend).poll.assert_called_once()

    def test_invalid_rl_result_does_not_leave_rl_ownership_stuck(self):
        for cancel in (False, True):
            with self.subTest(cancel=cancel):
                h, _ = self.build()
                h.learner.result = replace(RESULT, candidate=replace(RESULT.candidate, affected_roles=('decision_maker',)))
                h.accept(Directive('rl', 'task', '', mode='rl', payload=RLSpec('recipe')))
                h.advance('rl', ACTIONS)  # Start the local mock sequencer.
                h.advance('rl', ACTIONS)  # One mock rollout.
                job = h.advance('rl', ACTIONS).job
                if cancel:
                    with self.assertRaises(InvalidJobResult):
                        h.poll_learning(job)
                    h.cancel('rl')
                with self.assertRaises(InvalidJobResult):
                    h.advance('rl', ACTIONS)
                self.assertEqual(h.pending().feedback.status, ExecutionStatus.FAILED)
                self.assertTrue(h.state.rl.terminal)
                self.assertEqual(h.state.rl.updates_completed, 0)
                h.acknowledge('rl')
                h.accept(Directive('next', 'task', ''))

    def test_historical_candidate_rejects_new_application_but_replays_old_receipt(self):
        h, store = self.build()
        job = h.submit_learning(ActorTrainingRequest(BASE, (), 'recipe'), idempotency_key='first')
        h.poll_learning(job)
        original = Directive('apply-first', 'task', '', mode='apply_candidate', payload=ApplyCandidateSpec('first'))
        h.accept(original)
        first_receipt = h.advance('apply-first', ACTIONS)
        h.acknowledge('apply-first')
        self.assertEqual(h.apply_candidate('first'), FIRST)  # Same current binding remains idempotent.
        second = replace(FIRST, revision='v2')
        h.learner.result = TrainingResult(FIRST, second, Candidate('second', FIRST.revision, {}, ('actor',)))
        next_job = h.submit_learning(ActorTrainingRequest(FIRST, (), 'recipe'), idempotency_key='second')
        h.poll_learning(next_job)
        h.apply_candidate('second')
        records = store.records()
        for apply in (lambda: h.apply_candidate('first'), lambda: h.apply_learning(job)):
            with self.assertRaisesRegex(ValueError, 'stale'):
                apply()
        h.accept(replace(original, directive_id='apply-first-again'))
        with self.assertRaisesRegex(ValueError, 'stale'):
            h.advance('apply-first-again', ACTIONS)
        self.assertFalse(h.pending().terminal)
        self.assertEqual(store.records(), records)
        self.assertEqual(h.policy.model, second)
        self.assertEqual(h.policy.binds, 2)
        self.assertEqual(h.accept(original), first_receipt)
        self.assertEqual(h.advance('apply-first', ACTIONS), first_receipt)
        h.acknowledge('apply-first')  # An old ack cannot release the new owner.
        self.assertEqual(h.pending().directive_id, 'apply-first-again')
        h.cancel('apply-first-again')
        self.assertEqual(h.advance('apply-first-again', ACTIONS).feedback.status, ExecutionStatus.CANCELLED)

    def test_pending_and_execution_view_are_read_only_copies_with_frozen_terminal_elapsed(self):
        h, _ = self.build()
        self.assertIsNone(h.pending())
        self.assertEqual(h.execution_view().model, BASE)
        h.accept(self.directive('train'))
        self.assertIsNone(h.pending().elapsed_seconds)
        with patch('rsisys.loops.exploration.harness.commands.monotonic', return_value=10):
            h.advance('command', ACTIONS)
        with patch('rsisys.loops.exploration.harness.commands.monotonic', return_value=13):
            view = h.execution_view()
            self.assertEqual(view.progress.elapsed_seconds, 3)
            self.assertEqual(view.progress.metrics, {})
            h.learner.poll.assert_not_called()
            self.assertEqual(view.model, BASE)
            terminal = h.advance('command', ACTIONS)
        self.assertEqual(h.execution_view().model, FIRST)
        terminal.metrics['loss'] = 999
        terminal.feedback.training_metrics.clear()
        with patch('rsisys.loops.exploration.harness.commands.monotonic', return_value=99):
            pending = h.pending()
            self.assertEqual(pending.elapsed_seconds, 3)
            self.assertEqual(pending.metrics, {'loss': 0.5})
            self.assertEqual(pending.feedback.training_metrics, {'loss': 0.5})
        h.learner.poll.assert_called_once()
        h.acknowledge('command')
        self.assertIsNone(h.pending())
        self.assertIsNone(h.execution_view().progress)
        h.policy = object()
        self.assertIsNone(h.execution_view().model)


if __name__ == '__main__':
    unittest.main()
