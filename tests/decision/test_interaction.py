"""Cross-loop failure boundaries and opt-in decision reviews."""
from tests import ROOT as REPO_ROOT
from dataclasses import replace
from pathlib import Path
import unittest
import tempfile

from rsisys.config import DemoConfig, build_demo
from rsisys.loops.contracts.coordination import ExplorationProgress, LoopStepResult, ReviewDecision
from rsisys.loops.contracts.directive import Directive
from rsisys.loops.contracts.feedback import Feedback
from rsisys.loops.contracts.jobs import JobStatus
from rsisys.loops.contracts.lifecycle import ExecutionStatus
from rsisys.loops.decision_making.harness.decision import Decision
from rsisys.loops.exploration.harness.commands import CommandHandler, ExplorationCommands
from rsisys.runtime.runner import Runner
from rsisys.application import Application, ApplicationConfig
from rsisys.loops.contracts.requests import SFTSpec, EvaluationSpec
from rsisys.snapshot.local import LocalSnapshotCapture, restore_snapshot


class SlowCommand(CommandHandler):
    def prepare(self, harness, state):
        pass

    def advance(self, harness, state, action_space):
        state.phase = 'running'
        if state.cancel_requested:
            return Feedback(state.directive.directive_id, ExecutionStatus.CANCELLED, 'cancelled', True)


class ReviewingPolicy:
    def __init__(self, directive, *reviews):
        self.directive = directive
        self.reviews = iter(reviews)
        self.decisions = 0
        self.contexts = []

    def decide(self, context):
        self.decisions += 1
        return (Decision('explore', directive=self.directive) if self.decisions == 1
                else Decision('finish'))

    def review(self, context):
        self.contexts.append(context)
        return next(self.reviews)


class InteractionTests(unittest.TestCase):
    def setUp(self):
        root = REPO_ROOT
        self.runner, self.store = build_demo(DemoConfig.load(root / 'configs/demo.json'))
        self.dm, self.ex = self.runner.decision_making, self.runner.exploration
        self.now = 0.0
        self.runner = Runner(self.dm, self.ex, clock=lambda: self.now)

    def slow(self, *reviews, delay=5):
        directive = Directive('slow', 'task', 'long task', review_after_seconds=delay)
        policy = ReviewingPolicy(directive, *reviews)
        self.dm.harness.policy = policy
        self.ex.harness.commands = ExplorationCommands(self.ex.harness, {'execute': SlowCommand()})
        return policy

    def test_recreated_runner_recovers_feedback_delivered_before_exploration_ack(self):
        policy = ReviewingPolicy(Directive('once', 'task', 'run'))
        self.dm.harness.policy = policy
        ack = self.ex.acknowledge
        def lost(ident):
            raise OSError('ack unavailable')
        self.ex.acknowledge = lost
        with self.assertRaisesRegex(OSError, 'ack unavailable'):
            self.runner.step('task')
        self.assertIsNone(self.dm.pending())
        count = self.ex.harness.environment.sandbox.execution_count
        self.ex.acknowledge = ack
        self.runner = Runner(self.dm, self.ex)
        self.assertTrue(self.runner.busy)
        self.assertEqual(self.runner.step('task').kind, 'completed')
        self.assertFalse(self.runner.busy)
        self.assertEqual(policy.decisions, 1)
        self.assertEqual(count, self.ex.harness.environment.sandbox.execution_count)
        self.assertEqual(self.dm.harness.state.completed, 1)
        self.assertEqual(self.runner.step('task').kind, 'finished')

    def test_recreated_runner_recovers_lost_receive_response(self):
        self.dm.harness.policy = ReviewingPolicy(Directive('once', 'task', 'run'))
        receive = self.dm.receive_feedback
        def lost(directive, feedback):
            receive(directive, feedback)
            raise OSError('receive response lost')
        self.dm.receive_feedback = lost
        with self.assertRaises(OSError):
            self.runner.step('task')
        self.dm.receive_feedback = receive
        self.runner = Runner(self.dm, self.ex)
        self.assertEqual(self.runner.step('task').kind, 'completed')
        self.assertEqual(self.dm.harness.state.completed, 1)

    def test_unowned_command_does_not_issue_another_decision(self):
        policy = self.slow()
        self.ex.accept(Directive('external', 'task', 'run'))
        with self.assertRaisesRegex(RuntimeError, 'unowned'):
            self.runner.step('task')
        self.assertEqual(policy.decisions, 0)

    def test_reviews_are_scheduled_separately_from_polling_and_can_cancel(self):
        policy = self.slow(ReviewDecision(next_review_seconds=7), ReviewDecision('cancel'))
        self.assertEqual(self.runner.step('task').kind, 'waiting')
        self.now = 4
        self.runner.step('task')
        self.assertEqual(len(policy.contexts), 0)
        self.now = 5
        step = self.runner.step('task')
        self.assertEqual(step.event['review']['action'], 'continue')
        self.assertEqual(step.decision_steps, 1)
        self.now = 11
        self.runner.step('task')
        self.assertEqual(len(policy.contexts), 1)
        self.now = 12
        self.assertEqual(self.runner.step('task').event['review']['action'], 'cancel')
        self.assertTrue(self.runner.busy)
        self.assertEqual(self.runner.step('task').kind, 'completed')
        self.assertEqual(self.dm.harness.state.last_feedback.status, ExecutionStatus.CANCELLED)
        self.assertEqual(policy.decisions, 1)
        context = policy.contexts[0]
        self.assertEqual(context.execution.model, self.ex.harness.policy.model)
        self.assertEqual(context.execution.progress.phase, 'running')
        self.assertEqual(len(policy.contexts), 2)
        self.assertEqual(self.runner.step('task').kind, 'finished')

    def test_none_review_interval_disables_further_reviews(self):
        policy = self.slow(ReviewDecision())
        self.runner.step('task')
        self.now = 6
        self.runner.step('task')
        self.now = 100
        self.runner.step('task')
        self.assertEqual(len(policy.contexts), 1)

    def test_no_opt_in_means_no_review(self):
        policy = self.slow(delay=None)
        self.runner.step('task')
        self.now = 100
        self.runner.step('task')
        self.assertEqual(len(policy.contexts), 0)

    def test_uncertain_cancel_retries_control_without_redeciding(self):
        policy = self.slow(ReviewDecision('cancel'))
        self.runner.step('task')
        cancel = self.ex.cancel
        calls = []
        def lost(ident):
            calls.append(ident)
            if len(calls) == 1:
                raise OSError('cancel unavailable')
            return cancel(ident)
        self.ex.cancel = lost
        self.now = 6
        with self.assertRaisesRegex(OSError, 'cancel unavailable'):
            self.runner.step('task')
        self.runner.step('task')
        self.assertEqual(calls, ['slow', 'slow'])
        self.assertEqual(len(policy.contexts), 1)
        self.assertEqual(self.dm.harness.state.last_feedback.status, ExecutionStatus.CANCELLED)

    def test_pending_cancel_is_retried_before_any_further_work(self):
        for manual in (False, True):
            with self.subTest(manual=manual):
                self.setUp()
                self.slow(ReviewDecision('cancel'))
                effect = []
                class CompletingCommand(SlowCommand):
                    def advance(self, harness, state, action_space):
                        if state.cancel_requested:
                            return super().advance(harness, state, action_space)
                        state.data['ticks'] = state.data.get('ticks', 0) + 1
                        if state.data['ticks'] >= 3:
                            effect.append('committed')
                            return Feedback('slow', ExecutionStatus.COMPLETED, 'done', True)
                self.ex.harness.commands = ExplorationCommands(self.ex.harness, {'execute': CompletingCommand()})
                self.runner.step('task')
                cancel = self.ex.cancel
                calls = []
                def lost(ident):
                    calls.append(ident)
                    if len(calls) == 1:
                        raise OSError('cancel unavailable')
                    return cancel(ident)
                self.ex.cancel = lost
                self.now = 6
                with self.assertRaisesRegex(OSError, 'cancel unavailable'):
                    if manual:
                        self.runner.cancel()
                    else:
                        self.runner.step('task')
                self.assertEqual(self.runner.step('task').kind, 'completed')
                self.assertEqual(calls, ['slow', 'slow'])
                self.assertEqual(effect, [])
                self.assertEqual(self.dm.harness.state.last_feedback.status, ExecutionStatus.CANCELLED)

    def test_feedback_rejects_nonterminal_status_before_delivery(self):
        for status in (JobStatus.RUNNING, 'running', JobStatus.FAILED):
            with self.subTest(status=status), self.assertRaises(TypeError):
                Feedback('bad', status, 'not terminal feedback', True)
        with self.assertRaises(ValueError):
            Feedback('bad', ExecutionStatus.FAILED, 'failed', True)
        with self.assertRaises(ValueError):
            Feedback('bad', ExecutionStatus.COMPLETED, 'done', True, error='bad')

    def test_review_record_failure_is_retried_before_further_work(self):
        policy = self.slow(ReviewDecision('cancel'))
        effect = []
        class CompletingCommand(SlowCommand):
            def advance(self, harness, state, action_space):
                if state.cancel_requested:
                    return super().advance(harness, state, action_space)
                state.data['ticks'] = state.data.get('ticks', 0) + 1
                if state.data['ticks'] >= 3:
                    effect.append('committed')
                    return Feedback('slow', ExecutionStatus.COMPLETED, 'done', True)
        self.ex.harness.commands = ExplorationCommands(self.ex.harness, {'execute': CompletingCommand()})
        self.runner.step('task')
        record = self.dm.harness.experience.record
        failed = False
        def flaky(value):
            nonlocal failed
            if value.details.get('event') == 'review' and not failed:
                failed = True
                raise OSError('review evidence unavailable')
            return record(value)
        self.dm.harness.experience.record = flaky
        self.now = 6
        with self.assertRaisesRegex(OSError, 'review evidence unavailable'):
            self.runner.step('task')
        self.assertEqual(self.runner.step('task').kind, 'completed')
        self.assertEqual(effect, [])
        self.assertEqual(len(policy.contexts), 1)
        self.assertEqual(self.dm.harness.state.last_feedback.status, ExecutionStatus.CANCELLED)

    def test_progress_and_dispatch_validate_identity(self):
        directive = Directive('one', 'task', 'run')
        with self.assertRaises(ValueError):
            ExplorationProgress('two', 'running', directive)
        with self.assertRaises(ValueError):
            LoopStepResult('dispatch')
        with self.assertRaises(ValueError):
            LoopStepResult('finished', directive=directive)

    def test_review_intervals_are_positive_and_finite(self):
        for value in (0, -1, float('inf'), float('nan'), True, '5'):
            with self.subTest(value=value):
                with self.assertRaises(ValueError):
                    Directive('bad', 'task', 'run', review_after_seconds=value)
                with self.assertRaises(ValueError):
                    ReviewDecision(next_review_seconds=value)
        with self.assertRaises(ValueError):
            ReviewDecision('replace')

    def test_quiescent_snapshot_restores_evaluation_and_version_provenance(self):
        config_path = REPO_ROOT / 'configs/local.json'
        with tempfile.TemporaryDirectory() as tmp:
            config = replace(ApplicationConfig.load(config_path), learning='mock',
                             data_dir=Path(tmp) / 'source')
            with Application(config) as app:
                decision = app.runner.decision_making.harness
                data = app.artifacts.put(b'data', kind='training_data')
                train = Directive('train', 'task', 'train', mode='sft', input_refs=(data,),
                                  payload=SFTSpec('recipe'))
                decision.policy.decide = lambda context: Decision('explore', directive=train)
                for _ in range(10):
                    if app.runner.step('task').kind == 'completed':
                        break
                candidate = decision.state.last_feedback.candidates[0]
                evaluation = Directive('eval', 'task', 'measure', mode='evaluate', input_refs=(data,),
                                       payload=EvaluationSpec(candidate, 'protocol'))
                decision.policy.decide = lambda context: Decision('explore', directive=evaluation)
                for _ in range(10):
                    if app.runner.step('task').kind == 'completed':
                        break
                expected = decision.state.candidate_evaluations[candidate.candidate_id]
                provenance = dict(decision.state.directive_provenance)
                manifest = LocalSnapshotCapture(app).capture('evaluated')
                directory = config.data_dir / 'snapshots' / manifest.snapshot_id
            with restore_snapshot(directory, Path(tmp) / 'restored') as restored:
                state = restored.runner.decision_making.harness.state
                self.assertEqual(state.candidate_evaluations[candidate.candidate_id], expected)
                self.assertEqual(state.directive_provenance, provenance)
                self.assertEqual(state.execution.model, restored.runner.exploration.harness.policy.model)
                self.assertIsNone(state.execution.progress)


if __name__ == '__main__':
    unittest.main()
