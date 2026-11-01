"""Real toy controller/HTTP adapters with CPU fixtures and a virtual Runner clock."""
from contextlib import redirect_stderr, redirect_stdout
from dataclasses import replace
import io
import json
import os
from pathlib import Path
import subprocess
import sys
import tempfile
import time
from types import SimpleNamespace
import unittest
from unittest.mock import patch

from test_qwen_toy import Services, servers
from examples.qwen_toy import run as controller
from examples.qwen_toy.adapters import PROTOCOL
from rsisys.runtime.run_policy import GoalRunPolicy, ResearcherRunPolicy, TimeRunPolicy
from rsisys.runtime.runner import Runner


class QwenRunPolicyTests(unittest.TestCase):
    def args(self, *extra):
        return controller.parse_args(['--baseline-url', 'http://localhost:1',
                                      '--worker-url', 'http://localhost:2', *extra])

    def execute(self, service, *extra, poll_interval=0.005, max_steps=100, finish_with_feedback=False):
        clock = SimpleNamespace(now=0.0, sleeps=[], steps=[], policies=[])

        def sleep(seconds):
            clock.sleeps.append(seconds)
            clock.now += seconds

        def runner_factory(*args, **kwargs):
            self.assertEqual(kwargs['cleanup_timeout_seconds'], 30)
            clock.policies.append(kwargs['run_policy'])
            runner = Runner(*args, **kwargs, clock=lambda: clock.now)
            step = runner.step

            def recorded_step(objective):
                result = step(objective)
                if (finish_with_feedback and result.kind == 'completed'
                        and result.event.get('run_stop', {}).get('shutdown_complete')):
                    result = replace(result, kind='finished', directive_id=None)
                clock.steps.append((clock.now, result))
                return result

            runner.step = recorded_step
            return runner

        with tempfile.TemporaryDirectory() as directory, servers(service) as urls:
            root = Path(directory)
            args = controller.parse_args([
                '--baseline-url', urls[0], '--worker-url', urls[1], '--run-dir', str(root),
                '--poll-interval', str(poll_interval), '--timeout', '10', '--max-steps', str(max_steps),
                '--restart-delay-seconds', '0.01', '--max-trainings', '1', *extra])
            # Only controller sleep and Runner elapsed time are virtual. HTTP and
            # the hard service deadline retain real clocks and real loopback I/O.
            with patch.object(controller, 'Runner', side_effect=runner_factory), patch.object(
                    controller, 'time', SimpleNamespace(monotonic=time.monotonic, sleep=sleep)), \
                    redirect_stdout(io.StringIO()):
                result = controller.run(args)
            self.assertEqual(json.loads((root / 'result.json').read_text()), result)
            monitor = json.loads((root / 'monitor.json').read_text())
            self.assertEqual(monitor['run_stop'], result['run_stop'])
            self.assertEqual(monitor['stop_reason'], result['stop_reason'])
            self.assertEqual(monitor['goal_reached'], result['goal_reached'])
            self.assertEqual(monitor['remote_may_run'], result['remote_may_run'])
            self.assertEqual(result['remote_may_run'], result['remote_job_may_still_run'])
            rounds = [json.loads(path.read_text()) for path in sorted(root.glob('round-*.json'))]
        return result, monitor, clock, rounds

    def test_cli_policies_defaults_and_goal_protocol(self):
        self.assertIsInstance(controller.make_run_policy(self.args()), ResearcherRunPolicy)
        timed = controller.make_run_policy(self.args('--run-mode', 'time', '--run-seconds', '20'))
        self.assertIsInstance(timed, TimeRunPolicy)
        self.assertEqual((timed.seconds, timed.shutdown, timed.restart_delay_seconds), (20, 'cancel', 1))
        for direction in ('higher', 'lower'):
            goal = controller.make_run_policy(self.args('--run-mode', 'goal', '--goal-metric', 'candidate_nll',
                '--goal-threshold', '0', '--goal-direction', direction, '--run-seconds', '30'))
            self.assertIsInstance(goal, GoalRunPolicy)
            self.assertEqual((goal.max_seconds, goal.shutdown), (30, 'drain'))
            self.assertEqual(goal.goal.protocol_ref, PROTOCOL)
            self.assertEqual(goal.goal.higher_is_better, direction == 'higher')
            self.assertFalse(goal.goal.allow_simulated)
        unbounded = controller.make_run_policy(self.args('--run-mode', 'goal', '--goal-metric', 'delta_nll',
                                                          '--goal-threshold', '-1'))
        self.assertIsNone(unbounded.max_seconds)

    def test_cli_rejects_missing_invalid_or_ignored_policy_arguments(self):
        invalid = [('--run-mode', 'unknown'), ('--run-mode', 'time'), ('--run-mode', 'goal'),
                   ('--run-mode', 'goal', '--goal-metric', 'delta_nll'),
                   ('--run-mode', 'goal', '--goal-threshold', '0'),
                   ('--run-mode', 'goal', '--goal-metric', ' ', '--goal-threshold', '0'),
                   ('--run-seconds', '1'), ('--goal-metric', 'delta_nll'),
                   ('--goal-direction', 'invalid')]
        for value in ('0', '-1', 'nan', 'inf', '-inf'):
            invalid.extend([('--run-mode', 'time', f'--run-seconds={value}'),
                            (f'--restart-delay-seconds={value}',)])
        for value in ('nan', 'inf', '-inf'):
            invalid.append(('--run-mode', 'goal', '--goal-metric', 'delta_nll', f'--goal-threshold={value}'))
        for options in invalid:
            with self.subTest(options=options), redirect_stderr(io.StringIO()), self.assertRaises(SystemExit):
                self.args(*options)

    def test_default_finish_remains_completed_without_claiming_goal(self):
        result, monitor, clock, _ = self.execute(Services(plan=['execute', 'finish']))
        self.assertEqual(result['status'], 'completed')
        self.assertEqual(result['run_stop']['reason'], 'researcher_finished')
        self.assertTrue(result['run_stop']['shutdown_complete'])
        self.assertFalse(result['goal_reached'])
        self.assertFalse(result['remote_may_run'])
        self.assertEqual(result['rounds_completed'], 1)
        self.assertIsInstance(clock.policies[0], ResearcherRunPolicy)
        self.assertEqual(monitor['workflow']['completed_directives'], 1)

    def test_time_finish_restarts_after_delay_and_expires_without_goal(self):
        service = Services(plan=['finish'] * 20)
        result, monitor, clock, _ = self.execute(service, '--run-mode', 'time', '--run-seconds', '0.04')
        self.assertEqual(result['status'], 'stopped')
        self.assertEqual(result['stop_reason'], 'budget_exhausted')
        self.assertEqual(result['run_stop']['reason'], 'budget_exhausted')
        self.assertTrue(result['run_stop']['shutdown_complete'])
        self.assertFalse(result['goal_reached'])
        self.assertFalse(result['remote_may_run'])
        self.assertGreater(len(service.decisions), 1)
        self.assertEqual(result['model_calls'], len(service.decisions))
        finishes = [at for at, step in clock.steps if step.event.get('run_control') == 'researcher_finished']
        restarts = [at for at, step in clock.steps if step.event.get('run_control') == 'research_restarted']
        for finish, restart in zip(finishes, restarts):
            self.assertGreaterEqual(restart - finish + 1e-9, 0.01)
        self.assertEqual(len(clock.sleeps), sum(step.kind == 'waiting' for _, step in clock.steps))
        self.assertTrue(all(0 < value <= 0.005 for value in clock.sleeps))
        self.assertTrue(any(e['kind'] == 'run_stop' and e['details']['reason'] == 'budget_exhausted'
                            for e in monitor['events']))

    def test_goal_finish_restarts_and_final_evaluation_is_recorded_before_stop(self):
        for metric, threshold, direction, wrapped in (
                ('delta_nll', '0.5', 'higher', False), ('candidate_nll', '1.5', 'lower', False),
                ('delta_nll', '0.5', 'higher', True)):
            with self.subTest(direction=direction, wrapped=wrapped):
                service = Services(plan=['execute', 'finish', 'train', 'evaluate'])
                result, monitor, clock, rounds = self.execute(service, '--run-mode', 'goal',
                    '--goal-metric', metric, '--goal-threshold', threshold, '--goal-direction', direction,
                    finish_with_feedback=wrapped)
                self.assertEqual(result['status'], 'completed')
                self.assertTrue(result['goal_reached'])
                self.assertEqual(result['run_stop']['reason'], 'goal_reached')
                self.assertEqual(result['run_stop']['shutdown'], 'drain')
                self.assertEqual([e['mode'] for e in result['experiments']], ['execute', 'train', 'evaluate'])
                self.assertEqual(result['experiments'][-1]['metrics']['delta_nll'], 0.5)
                self.assertEqual(result['metrics']['candidate_nll'], 1.5)
                self.assertEqual(result['candidate_id'], 'candidate-1')
                self.assertEqual(result['model_calls'], 4)
                self.assertFalse(result['applied'])
                self.assertEqual(result['actor_revision'], 'baseline')
                self.assertEqual(monitor['candidates'][0]['status'], 'proposed')
                self.assertEqual(monitor['candidates'][0]['metrics']['delta_nll'], 0.5)
                self.assertEqual(monitor['current']['mode'], 'evaluate')
                self.assertEqual(monitor['current']['status'], 'completed')
                self.assertEqual(monitor['decisions'][-1]['status'], 'completed')
                self.assertEqual(monitor['workflow']['completed_directives'], 3)
                self.assertEqual(rounds, result['round_results'])
                terminal = clock.steps[-1][1]
                self.assertEqual(terminal.kind, 'finished')
                feedback_step = terminal if wrapped else clock.steps[-2][1]
                self.assertIn('feedback', feedback_step.event)
                self.assertEqual(feedback_step.event['run_stop']['reason'], 'goal_reached')
                self.assertEqual(feedback_step.kind, 'finished' if wrapped else 'completed')
                self.assertEqual(service.contexts[-1]['run_control']['research_round'], 2)

    def test_goal_time_limit_is_not_goal_completion(self):
        result, _, _, _ = self.execute(Services(plan=['finish'] * 20), '--run-mode', 'goal',
            '--goal-metric', 'delta_nll', '--goal-threshold', '0.5', '--run-seconds', '0.02')
        self.assertEqual(result['status'], 'stopped')
        self.assertEqual(result['run_stop']['reason'], 'budget_exhausted')
        self.assertFalse(result['goal_reached'])

    def test_restart_keeps_model_training_and_directive_budget_totals(self):
        cases = [(['finish'] * 3, ('--max-model-calls', '2'), 'timed_out', 2, 0, 0),
                 (['train', 'finish'] + ['train'] * 3, (), 'failed', 5, 1, 1),
                 (['execute', 'finish'] + ['execute'] * 3, ('--max-directives', '1'), 'failed', 5, 0, 1)]
        for plan, limits, status, calls, trainings, completed in cases:
            with self.subTest(limits=limits):
                service = Services(plan=plan)
                result, _, _, _ = self.execute(service, '--run-mode', 'time', '--run-seconds', '100', *limits)
                self.assertEqual(result['status'], status)
                self.assertEqual(result['model_calls'], calls)
                self.assertEqual(result['training_submissions'], trainings)
                self.assertEqual(result['completed_directives'], completed)
                self.assertFalse(result['goal_reached'])
                self.assertEqual(len(service.jobs), trainings)

    def test_time_stop_cancels_and_records_final_cancelled_feedback(self):
        service = Services(plan=['train'], forever=True)
        result, monitor, clock, _ = self.execute(service, '--run-mode', 'time', '--run-seconds', '0.01')
        self.assertEqual(result['run_stop']['reason'], 'budget_exhausted')
        self.assertEqual(result['status'], 'stopped')
        self.assertEqual(result['run_stop']['shutdown'], 'cancel')
        self.assertTrue(result['run_stop']['shutdown_complete'])
        self.assertTrue(result['cancel_requested'])
        self.assertTrue(result['cancellation_confirmed'])
        self.assertFalse(result['remote_may_run'])
        self.assertEqual(result['experiments'][0]['status'], 'cancelled')
        self.assertEqual(monitor['current']['status'], 'cancelled')
        self.assertEqual(monitor['workflow']['completed_directives'], 1)
        self.assertIn('feedback', clock.steps[-2][1].event)

    def test_unconfirmed_policy_cleanup_reports_remote_uncertainty(self):
        service = Services(plan=['train'], forever=True, cancel_pending=True)
        result, monitor, _, _ = self.execute(service, '--run-mode', 'time', '--run-seconds', '0.01',
                                             poll_interval=5)
        self.assertEqual(result['status'], 'timed_out')
        self.assertFalse(result['goal_reached'])
        self.assertFalse(result['run_stop']['shutdown_complete'])
        self.assertTrue(result['run_stop']['cleanup_timed_out'])
        self.assertTrue(result['run_stop']['unresolved_directives'])
        self.assertIsNone(result['run_stop']['unresolved_research_job'])
        self.assertTrue(result['remote_may_run'])
        self.assertTrue(result['cancel_requested'])
        self.assertFalse(result['cancellation_confirmed'])
        self.assertEqual(result['experiments'], [])
        self.assertEqual(monitor['status'], 'timed_out')

    def test_hard_controller_deadline_still_bounds_time_mode(self):
        # Expire after submission, not during startup on a busy host. Keep real
        # loopback requests and an independent real cleanup deadline.
        deadline_type, call = controller.Deadline, controller.HTTPClient.call
        deadlines = []

        def make_deadline(seconds):
            deadline = deadline_type(seconds)
            deadlines.append(deadline)
            return deadline

        def expire_on_poll(client, path, payload=None):
            if path.startswith('/jobs/') and client.deadline is deadlines[0]:
                client.deadline.end = time.monotonic() - 1
            return call(client, path, payload)

        with patch.object(controller, 'Deadline', side_effect=make_deadline), patch.object(
                controller.HTTPClient, 'call', new=expire_on_poll):
            result, _, _, _ = self.execute(Services(plan=['train'], forever=True),
                '--run-mode', 'time', '--run-seconds', '100', '--timeout', '10')
        self.assertEqual(result['status'], 'timed_out')
        self.assertEqual(result['error_code'], 'TimeoutError')
        self.assertIsNone(result['run_stop'])
        self.assertFalse(result['goal_reached'])
        self.assertTrue(result['cancellation_confirmed'])

    def test_cli_exit_codes_distinguish_policy_stop_and_hard_budget_failure(self):
        project = Path(controller.__file__).resolve().parents[2]
        cases = [(['finish'], (), 'completed', 0),
                 (['finish'] * 100, ('--run-mode', 'time', '--run-seconds', '0.05'), 'stopped', 0),
                 (['execute', 'finish'], ('--max-model-calls', '1'), 'timed_out', 1)]
        for plan, options, status, code in cases:
            with self.subTest(status=status), tempfile.TemporaryDirectory() as directory, \
                    servers(Services(plan=plan)) as urls:
                process = subprocess.run([sys.executable, str(project / 'examples/qwen_toy/run.py'),
                    '--baseline-url', urls[0], '--worker-url', urls[1], '--run-dir', directory,
                    '--poll-interval', '0.001', '--restart-delay-seconds', '0.005', '--timeout', '10',
                    '--max-model-calls', '100', *options], cwd=project,
                    env={**os.environ, 'PYTHONPATH': str(project / 'src')},
                    capture_output=True, text=True, timeout=15)
                self.assertEqual(process.returncode, code, process.stdout + process.stderr)
                result = json.loads(process.stdout.splitlines()[-1])
                self.assertEqual(result['status'], status)
                self.assertFalse(result['goal_reached'])


if __name__ == '__main__':
    unittest.main()
