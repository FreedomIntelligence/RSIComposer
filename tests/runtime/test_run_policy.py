"""Run lifetime contracts, driven by a fake clock and existing loop endpoints."""
from tests import ROOT as REPO_ROOT
from dataclasses import replace
from pathlib import Path
from tempfile import TemporaryDirectory
import unittest
from unittest.mock import patch

from rsisys.config import DemoConfig, build_demo
from rsisys.application import Application, ApplicationConfig
from rsisys.loops.contracts.directive import Directive
from rsisys.loops.contracts.feedback import Feedback
from rsisys.loops.contracts.lifecycle import ExecutionStatus
from rsisys.loops.contracts.requests import SFTSpec, EvaluationSpec
from rsisys.loops.decision_making.harness.decision import Decision
from rsisys.loops.decision_making.action_space import OperationCall
from rsisys.runtime.runner import Runner
from rsisys.runtime.run_policy import (RunContext, RunDecision, ResearcherRunPolicy, TimeRunPolicy,
                                      GoalRunPolicy, MetricGoal, ExperimentOutcome)
from rsisys.snapshot.local import LocalSnapshotCapture
from examples.components.parallel_experiments import build_demo as parallel_demo

CONFIG = REPO_ROOT / 'configs/demo.json'


class Clock:
    now = 0.0
    def __call__(self):
        return self.now


class Policy:
    def __init__(self, *decisions):
        self.decisions = iter(decisions)
        self.contexts = []
    def decide(self, context):
        self.contexts.append(context)
        return next(self.decisions, Decision('finish'))


class RunPolicyTests(unittest.TestCase):
    def make(self, policy=None, *decisions, cleanup=30):
        assembly, _ = build_demo(DemoConfig.load(CONFIG), with_learning=True)
        self.addCleanup(assembly.close)
        clock = Clock()
        runner = Runner(assembly.decision_making, assembly.exploration, clock=clock,
                        run_policy=policy, cleanup_timeout_seconds=cleanup)
        actor = Policy(*decisions)
        runner.decision_making.harness.policy = actor
        return runner, clock, actor

    def train(self, runner, ident='train'):
        data = runner.decision_making.harness.tools.artifacts.put(b'data', kind='training_data')
        return Directive(ident, 'task', 'train', mode='sft', input_refs=(data,), payload=SFTSpec('recipe'))

    def test_default_finish_stops_and_is_terminal(self):
        runner, clock, policy = self.make()
        result = runner.step('task')
        self.assertEqual(result.kind, 'finished')
        self.assertEqual(result.event['run_stop']['reason'], 'researcher_finished')
        self.assertTrue(result.event['run_stop']['shutdown_complete'])
        clock.now = 100
        self.assertEqual(runner.step('task'), result)
        self.assertEqual(len(policy.contexts), 1)

    def test_time_finish_restarts_without_spin_or_resetting_memory(self):
        runner, clock, policy = self.make(TimeRunPolicy(10, restart_delay_seconds=2))
        self.assertEqual(runner.step('task').kind, 'waiting')
        history = runner.decision_making.harness.memory.history()
        for _ in range(10):
            self.assertEqual(runner.step('task').kind, 'waiting')
        self.assertEqual(len(policy.contexts), 1)
        clock.now = 2
        self.assertEqual(runner.step('task').event['run_control'], 'research_restarted')
        self.assertEqual(runner.decision_making.harness.memory.history(), history)
        self.assertEqual(runner.step('task').kind, 'waiting')
        self.assertEqual(policy.contexts[-1].objective, 'task')
        self.assertEqual(policy.contexts[-1].run_control['research_round'], 2)
        clock.now = 10
        result = runner.run('task', 5)
        self.assertEqual(result.stop_reason, 'budget_exhausted')
        self.assertTrue(result.shutdown_complete)
        self.assertEqual(len(policy.contexts), 2)

    def test_goal_restarts_until_external_predicate_succeeds(self):
        reached = [False]
        runner, clock, policy = self.make(GoalRunPolicy(lambda context: reached[0]))
        self.assertEqual(runner.step('task').kind, 'waiting')
        clock.now = 1
        runner.step('task')
        runner.step('task')
        reached[0] = True
        result = runner.step('task')
        self.assertEqual(result.event['run_stop']['reason'], 'goal_reached')
        self.assertEqual(len(policy.contexts), 2)

    def test_goal_deadline_overrides_simultaneous_success(self):
        policy = GoalRunPolicy(lambda context: True, max_seconds=5)
        context = RunContext('task', 5, 1, False)
        self.assertEqual(policy.evaluate(context).reason, 'budget_exhausted')
        self.assertEqual(policy.evaluate(context).shutdown, 'cancel')

    def test_deadline_during_decision_prevents_training_submission(self):
        runner, clock, _ = self.make(TimeRunPolicy(2))
        directive = self.train(runner)
        class SlowPolicy:
            def decide(self, context):
                clock.now = 3
                return Decision('explore', directive=directive)
        runner.decision_making.harness.policy = SlowPolicy()
        result = runner.step('task')
        self.assertEqual(result.event['run_stop']['reason'], 'budget_exhausted')
        self.assertEqual(runner.exploration.harness.commands.receipts, {})
        self.assertEqual(result.kind, 'finished')
        self.assertEqual(runner.decision_making.harness.state.issued, 0)
        self.assertIsNone(runner.decision_making.harness.state.last_feedback)

    def test_deadline_during_model_call_blocks_tool_execution(self):
        runner, clock, _ = self.make(TimeRunPolicy(2))
        class SlowPolicy:
            def decide(self, context):
                clock.now = 3
                return Decision('invoke', operation=OperationCall('prepare_data', {'text': 'data'}))
        harness = runner.decision_making.harness
        harness.policy = SlowPolicy()
        with patch.object(harness.tools, 'execute', wraps=harness.tools.execute) as execute:
            result = runner.step('task')
        self.assertEqual(result.event['run_stop']['reason'], 'budget_exhausted')
        self.assertTrue(result.event['run_stop']['shutdown_complete'])
        execute.assert_not_called()
        self.assertIsNone(harness.pending_research_job())

    def test_review_tool_cannot_start_after_model_consumes_remaining_time(self):
        from rsisys.loops.contracts.jobs import JobOutcome, JobStatus
        runner, clock, _ = self.make(TimeRunPolicy(2))
        directive = replace(self.train(runner), review_after_seconds=.1)
        class ReviewingPolicy(Policy):
            def review(self, context):
                clock.now = 3
                return Decision('invoke', operation=OperationCall('prepare_data', {'text': 'late'}))
        harness = runner.decision_making.harness
        harness.policy = ReviewingPolicy(Decision('explore', directive=directive))
        from rsisys.loops.decision_making.action_space.access import ActionPermissions
        runner.decision_making.action_space.permissions = ActionPermissions.unrestricted()
        runner.exploration.harness.training.backend.poll = lambda job: JobOutcome(job, JobStatus.RUNNING)
        runner.step('task')
        clock.now = 1
        with patch.object(harness.tools, 'execute', wraps=harness.tools.execute) as execute:
            result = runner.step('task')
        execute.assert_not_called()
        self.assertEqual(result.event['run_stop']['reason'], 'budget_exhausted')

    def test_stop_flushes_existing_tool_receipt_or_reports_it_unresolved(self):
        for recovers in (True, False):
            with self.subTest(recovers=recovers):
                runner, clock, _ = self.make(TimeRunPolicy(2), cleanup=3)
                harness = runner.decision_making.harness
                harness.policy = Policy(Decision('invoke', operation=OperationCall('prepare_data', {'text': 'data'})))
                record = harness.experience.record
                def fail(value):
                    raise OSError('record store unavailable')
                harness.experience.record = fail
                with patch.object(harness.tools, 'execute', wraps=harness.tools.execute) as execute:
                    with self.assertRaises(OSError):
                        runner.step('task')
                    self.assertTrue(harness.pending_records())
                    clock.now = 2
                    if recovers:
                        harness.experience.record = record
                        result = runner.step('task')
                        self.assertTrue(result.event['run_stop']['shutdown_complete'])
                        self.assertFalse(harness.pending_records())
                    else:
                        with self.assertRaises(OSError): runner.step('task')
                        clock.now = 5
                        result = runner.step('task')
                        self.assertFalse(result.event['run_stop']['shutdown_complete'])
                        self.assertTrue(result.event['run_stop']['unresolved_records'])
                    self.assertEqual(execute.call_count, 1)

    def test_parallel_stop_cancels_every_target_and_delivers_feedback(self):
        runner, children, _ = parallel_demo()
        clock = Clock()
        runner._clock, runner.run_policy = clock, TimeRunPolicy(5)
        runner.step('task')
        runner.step('task')
        self.assertEqual(set(children), {'slow', 'fast'})
        clock.now = 5
        for _ in range(20):
            result = runner.step('task')
            if result.kind == 'finished':
                break
        self.assertEqual(result.kind, 'finished')
        self.assertTrue(result.event['run_stop']['shutdown_complete'])
        self.assertEqual(runner.decision_making.harness.state.completed, 2)
        self.assertEqual(len(runner._outcomes), 2)
        self.assertEqual(set(children), {'slow', 'fast'})
        self.assertFalse(runner.busy)

    def test_cleanup_timeout_reports_uncancelled_experiment(self):
        runner, clock, _ = self.make(TimeRunPolicy(2), cleanup=3)
        directive = self.train(runner)
        runner.decision_making.harness.policy = Policy(Decision('explore', directive=directive))
        runner.step('task')
        def lost(ident):
            raise OSError('cancel receipt unavailable')
        runner.exploration.cancel = lost
        clock.now = 2
        with self.assertRaises(OSError):
            runner.step('task')
        clock.now = 5
        result = runner.step('task')
        self.assertEqual(result.kind, 'finished')
        self.assertFalse(result.event['run_stop']['shutdown_complete'])
        self.assertTrue(result.event['run_stop']['cleanup_timed_out'])
        self.assertEqual(result.event['run_stop']['unresolved_directives'], ['train'])
        self.assertTrue(runner.busy)

    def test_retry_of_accepted_restart_does_not_duplicate_cycle(self):
        runner, clock, policy = self.make(TimeRunPolicy(10))
        runner.step('task')
        original = runner.decision_making.restart_research
        calls = []
        def lose_once(context):
            calls.append(context)
            original(context)
            if len(calls) == 1:
                raise OSError('restart acknowledgement lost')
        runner.decision_making.restart_research = lose_once
        clock.now = 1
        with self.assertRaises(OSError):
            runner.step('task')
        clock.now = 2
        runner.step('task')
        self.assertEqual(calls[0], calls[1])
        self.assertEqual(len(runner.decision_making.harness.state.research_cycles), 1)
        runner.step('task')
        self.assertEqual(len(policy.contexts), 2)

    def test_custom_run_policy_and_wrong_return_type(self):
        class Custom:
            def evaluate(self, context):
                return RunDecision('stop', 'external_stop', evidence_refs=('report:1',))
        runner, _, policy = self.make(Custom())
        result = runner.step('task')
        self.assertEqual(result.event['run_stop']['evidence_refs'], ['report:1'])
        self.assertEqual(policy.contexts, [])
        class Invalid:
            def evaluate(self, context): return True
        runner, _, _ = self.make(Invalid())
        with self.assertRaisesRegex(TypeError, 'RunDecision'):
            runner.step('task')

    def test_metric_goal_ignores_mock_failed_nan_and_stale_evaluations(self):
        runner, _, _ = self.make()
        runner.decision_making.harness.policy = Policy(Decision('explore', directive=self.train(runner)))
        for _ in range(10):
            result = runner.step('task')
            if result.kind == 'completed': break
        candidate = runner.decision_making.harness.state.last_feedback.candidates[0]
        directive = Directive('eval', 'task', 'evaluate', mode='evaluate', payload=EvaluationSpec(candidate, 'p'))
        feedback = Feedback('eval', ExecutionStatus.COMPLETED, 'evaluated', False, quality_metrics={'score': .9})
        outcome = ExperimentOutcome(directive, feedback)
        context = RunContext('task', 1, 1, False, (outcome,))
        goal = MetricGoal('score', .8, 'p')
        self.assertTrue(goal(context))
        for bad in (replace(feedback, simulated=True),
                    replace(feedback, status=ExecutionStatus.FAILED, error='failed')):
            latest = ExperimentOutcome(replace(directive, directive_id='eval2'), replace(bad, directive_id='eval2'))
            self.assertFalse(goal(replace(context, outcomes=(outcome, latest))))
        # Constructors now reject nonfinite metrics. Keep the goal's defensive
        # check for a legacy/in-place-corrupted receipt that bypassed admission.
        corrupted = replace(feedback, directive_id='eval2')
        corrupted.quality_metrics['score'] = float('nan')
        latest = ExperimentOutcome(replace(directive, directive_id='eval2'), corrupted)
        self.assertFalse(goal(replace(context, outcomes=(outcome, latest))))
        self.assertFalse(MetricGoal('score', .8, 'different')(context))

    def test_goal_observes_delivered_feedback_and_counts_last_completion(self):
        runner, _, _ = self.make(GoalRunPolicy(lambda context: bool(context.outcomes)))
        runner.decision_making.harness.policy = Policy(Decision('explore', directive=self.train(runner)))
        for _ in range(10):
            result = runner.run('task', 9)
            if result.status == 'finished': break
        self.assertEqual(result.stop_reason, 'goal_reached')
        self.assertEqual(result.rounds_completed, 1)
        self.assertFalse(runner.busy)

    def test_default_assembly_accepts_run_policy(self):
        runner, _ = build_demo(DemoConfig.load(CONFIG), run_policy=TimeRunPolicy(5))
        self.addCleanup(runner.close)
        self.assertIsInstance(runner.run_policy, TimeRunPolicy)

    def test_snapshot_rejects_nondefault_run_policy(self):
        with TemporaryDirectory() as directory:
            config = ApplicationConfig.load(REPO_ROOT / 'configs/local.json')
            config = replace(config, data_dir=Path(directory) / 'run')
            with Application(config, run_policy=TimeRunPolicy(5)) as app:
                with self.assertRaisesRegex(TypeError, 'run policies'):
                    LocalSnapshotCapture(app).capture('policy')

    def test_parallel_finish_drains_then_restarts_same_decision_harness(self):
        runner, children, _ = parallel_demo()
        clock = Clock()
        runner._clock, runner.run_policy = clock, TimeRunPolicy(100)
        runner.step('task')
        runner.step('task')
        policy = Policy()
        harness = runner.decision_making.harness
        harness.policy = policy
        for _ in range(20):
            runner.step('task')
            if not runner.busy:
                break
        self.assertEqual(harness.state.completed, 2)
        self.assertEqual(len(policy.contexts), 1)
        clock.now = 1
        result = runner.step('task')
        # A finish can be observed while draining; the cooldown starts once idle.
        if result.event.get('run_control') != 'research_restarted':
            clock.now = 2
            result = runner.step('task')
        self.assertEqual(result.event['run_control'], 'research_restarted')
        self.assertIs(runner.decision_making.harness, harness)
        self.assertEqual(harness.state.completed, 2)
        runner.step('task')
        self.assertEqual(len(policy.contexts), 2)
        self.assertTrue(all(child.commands.active is None for child in children.values()))

    def test_pending_research_job_is_reported_until_resume_or_cleanup_timeout(self):
        from rsisys.loops.contracts.jobs import JobRef
        job = JobRef('job', 'external')
        runner, clock, _ = self.make(TimeRunPolicy(2), Decision('wait', job=job), cleanup=3)
        self.assertEqual(runner.step('task').job, job)
        clock.now = 2
        self.assertEqual(runner.step('task').kind, 'waiting')
        clock.now = 5
        result = runner.step('task')
        self.assertFalse(result.event['run_stop']['shutdown_complete'])
        self.assertEqual(result.event['run_stop']['unresolved_research_job']['job_id'], job.job_id)

    def test_invalid_modes_budgets_and_goal_outputs(self):
        for value in (0, -1, float('nan'), True):
            with self.assertRaises(ValueError): TimeRunPolicy(value)
        with self.assertRaises(ValueError): TimeRunPolicy(5, restart_delay_seconds=0)
        with self.assertRaises(TypeError): GoalRunPolicy(None)
        with self.assertRaises(TypeError): GoalRunPolicy(lambda c: 1).evaluate(RunContext('t', 0, 1, False))


if __name__ == '__main__':
    unittest.main()
