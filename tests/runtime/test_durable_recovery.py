"""Restart/fault boundaries, not just serialization of idle snapshots."""
from tests import ROOT as REPO_ROOT

from dataclasses import replace
import json
import os
from pathlib import Path
import subprocess
import sys
import tempfile
import time
import unittest
from unittest.mock import patch

from rsisys.application import Application, ApplicationConfig
from rsisys.assembly import build_default_decision
from rsisys.loops.contracts.coordination import DirectiveUpdate
from rsisys.loops.contracts.directive import Directive
from rsisys.loops.contracts.jobs import JobRef, JobStatus
from rsisys.loops.contracts.requests import SFTSpec
from rsisys.loops.decision_making.harness.decision import Decision
from rsisys.loops.decision_making.harness.engine import DefaultDecisionHarness
from rsisys.runtime.recovery import RecoveryRequired, DurableRecovery
from rsisys.runtime.run_policy import TimeRunPolicy, GoalRunPolicy, MetricGoal
from rsisys.snapshot.local import LocalSnapshotCapture, restore_snapshot

ROOT = REPO_ROOT


class ExtensionHarness(DefaultDecisionHarness):
    recovery_id = 'test-session/v1'

    def export_recovery_state(self):
        return {'session': self.session}

    def restore_recovery_state(self, state):
        self.session = state['session']


def extension_factory(context, exploration):
    loop = build_default_decision(context, exploration)
    loop.harness.__class__ = ExtensionHarness
    loop.harness.session = 'original-session'
    return loop


class DurableRecoveryTests(unittest.TestCase):
    def setUp(self):
        self.directory = tempfile.TemporaryDirectory()
        self.addCleanup(self.directory.cleanup)
        self.root = Path(self.directory.name).resolve()
        self.config = replace(ApplicationConfig.load(ROOT / 'configs/local.json'),
                              data_dir=self.root / 'run', execution='mock', learning='mock')

    def start(self, **kwargs):
        return Application(self.config, durable=True, **kwargs)

    def tick(self, app):
        return app.runner.step(app.config.objective)

    def training(self, app):
        data = app.artifacts.put(b'data', kind='training_data')
        directive = Directive('train', self.config.objective, 'train', mode='sft', input_refs=(data,),
                              payload=SFTSpec('recipe'), review_after_seconds=100)
        with patch.object(app.runner.decision_making.harness.policy, 'decide',
                          return_value=Decision('explore', directive=directive)):
            result = self.tick(app)
        self.assertEqual(result.kind, 'waiting')
        return directive, result.job

    def test_restart_active_mock_job_retains_identity_then_completes_without_resubmit(self):
        with self.start() as app:
            directive, job = self.training(app)
            self.assertTrue(app.runner.busy)
        with Application.resume(self.config.data_dir) as app:
            self.assertEqual(app.runner.exploration.pending().job, job)
            backend = app.runner.exploration.harness.training.backend
            with patch.object(backend, 'submit', side_effect=AssertionError('resubmitted')):
                self.assertEqual(self.tick(app).kind, 'waiting')
                result = self.tick(app)
            self.assertEqual(result.kind, 'completed')
            self.assertEqual(app.runner._outcomes[directive.directive_id].feedback.jobs, (job,))
            self.assertEqual(len(backend.jobs._jobs), 1)
        with Application.resume(self.config.data_dir) as app:
            self.assertEqual(len(app.runner._outcomes), 1)
            self.assertEqual(app.runner.exploration.harness.training.poll(job).status, JobStatus.SUCCEEDED)

    def test_cancel_job_survives_restart_and_does_not_resubmit(self):
        with self.start() as app:
            directive, job = self.training(app)
            app.runner.cancel(directive.directive_id)
        with Application.resume(self.config.data_dir) as app:
            actor = app.runner.exploration.harness
            self.assertTrue(actor.commands.receipts[directive.directive_id].cancel_requested)
            with patch.object(actor.training.backend, 'submit', side_effect=AssertionError('resubmitted')):
                self.assertEqual(self.tick(app).kind, 'completed')
            self.assertEqual(app.runner._outcomes['train'].feedback.status.value, 'cancelled')

    def test_deadline_counts_downtime_and_preserves_elapsed(self):
        with self.start(run_policy=TimeRunPolicy(10)) as app:
            app.runner._clock = lambda: 100.0
            app.recovery.wall_clock = lambda: time.time() - 20
            self.tick(app)
        with Application.resume(self.config.data_dir) as app:
            self.assertGreaterEqual(app.runner._run_context(self.config.objective).elapsed_seconds, 20)
            with patch.object(app.runner.decision_making.harness.policy, 'decide',
                              side_effect=AssertionError('new decision after deadline')):
                result = self.tick(app)
            self.assertEqual(result.event['run_stop']['reason'], 'budget_exhausted')
            self.assertTrue(result.event['run_stop']['shutdown_complete'])

    def test_cycles_restart_delay_outcomes_and_terminal_state_survive(self):
        with self.start(run_policy=TimeRunPolicy(200, restart_delay_seconds=5)) as app:
            self.tick(app)
            self.tick(app)
            self.assertEqual(len(app.runner._outcomes), 1)
            with patch.object(app.runner.decision_making.harness.policy, 'decide', return_value=Decision('finish')):
                self.assertEqual(self.tick(app).kind, 'waiting')
        with Application.resume(self.config.data_dir) as app:
            self.assertEqual(len(app.runner._outcomes), 1)
            self.assertEqual(self.tick(app).event['run_control'], 'restart_wait')
            restart_time = app.runner._restart_at + 1
            app.runner._clock = lambda: restart_time
            result = self.tick(app)
            self.assertEqual(result.event['research_round'], 2)
        with Application.resume(self.config.data_dir) as app:
            self.assertEqual(app.runner._research_round, 2)
            self.assertIn(2, app.runner.decision_making.harness.state.research_cycles)

    def test_metric_goal_and_cleanup_configuration_round_trip(self):
        policy = GoalRunPolicy(MetricGoal('score', .9, 'held-out'), max_seconds=123)
        with self.start(run_policy=policy, cleanup_timeout_seconds=7) as app:
            self.tick(app)
        with Application.resume(self.config.data_dir) as app:
            self.assertEqual(app.runner.run_policy, policy)
            self.assertEqual(app.runner.cleanup_timeout_seconds, 7)
        with self.assertRaisesRegex(ValueError, 'replace'):
            Application.resume(self.config.data_dir, run_policy=TimeRunPolicy(999))

    def test_custom_goal_requires_contract_before_first_effect(self):
        with self.assertRaisesRegex(TypeError, 'recovery_contract'):
            self.start(run_policy=GoalRunPolicy(lambda context: False))
        self.assertFalse((self.config.data_dir / 'recovery' / 'checkpoint.json').exists())

    def test_uncertain_feedback_ack_retains_receipts_and_outbox_until_reconciled(self):
        with self.start() as app:
            self.tick(app)
            acknowledge = app.runner.exploration.acknowledge
            def lost_ack(ident):
                acknowledge(ident)
                raise ConnectionError('ack response lost')
            with patch.object(app.runner.exploration, 'acknowledge', side_effect=lost_ack):
                with self.assertRaises(ConnectionError):
                    self.tick(app)
            feedback = app.runner._feedback
            self.assertIsNotNone(feedback)
            records = len(app.store.records())
            with self.assertRaises(RecoveryRequired):
                self.tick(app)
        with Application.resume(self.config.data_dir) as app:
            self.assertEqual(app.runner._feedback, feedback)
            dm = app.runner.decision_making.harness
            self.assertIn(feedback.directive_id, dm.state.delivered_feedback)
            with self.assertRaises(RecoveryRequired):
                app.runner.cancel()
            def verify(application, incident):
                self.assertEqual(application.runner.exploration.inspect(feedback.directive_id).feedback, feedback)
                self.assertIn(feedback.directive_id, dm.state.delivered_feedback)
            app.recovery.reconcile(verify, evidence='Matched terminal exploration receipt and delivered feedback')
            self.assertEqual(self.tick(app).kind, 'completed')
            self.assertEqual(len(app.store.records()), records)
            self.assertEqual(len(app.runner._outcomes), 1)
        with Application.resume(self.config.data_dir) as app:
            self.assertIsNone(app.recovery.intervention)
            self.assertEqual(len(app.runner._outcomes), 1)

    def test_checkpoint_failure_after_effect_does_not_allow_repeat(self):
        with self.start() as app:
            with patch.object(app.recovery, '_checkpoint', side_effect=OSError('disk full')):
                with self.assertRaises(OSError):
                    self.tick(app)
            self.assertEqual(len(app.artifacts.refs()), 1)
            with self.assertRaises(RecoveryRequired):
                self.tick(app)
        with Application.resume(self.config.data_dir) as app:
            self.assertEqual(app.recovery.intervention['operation'], 'step')
            self.assertEqual(len(app.artifacts.refs()), 1)
            with self.assertRaises(RecoveryRequired):
                app.run()

    def test_process_death_after_tool_effect_is_not_replayed(self):
        config_path = self.root / 'config.json'
        config_path.write_text(json.dumps(self.config.to_dict()))
        script = '''
import os, sys
from pathlib import Path
from unittest.mock import patch
from rsisys.application import Application, ApplicationConfig
from rsisys.loops.decision_making.harness.tools.local import LocalDecisionTools
app = Application(ApplicationConfig.load(Path(sys.argv[1])), durable=True)
original = LocalDecisionTools.execute
def crash(self, call):
    original(self, call)
    os._exit(73)
with patch.object(LocalDecisionTools, 'execute', crash):
    app.runner.step(app.config.objective)
'''
        process = subprocess.run([sys.executable, '-c', script, str(config_path)],
                                 cwd=ROOT, env={**os.environ, 'PYTHONPATH': str(ROOT / 'src')},
                                 capture_output=True, text=True, timeout=15)
        self.assertEqual(process.returncode, 73, process.stderr)
        with Application.resume(self.config.data_dir) as app:
            self.assertEqual(len(app.artifacts.refs()), 1)
            with self.assertRaises(RecoveryRequired):
                self.tick(app)
            self.assertEqual(len(app.artifacts.refs()), 1)

    def test_successful_process_exit_reopens_at_next_step(self):
        config_path = self.root / 'config.json'
        config_path.write_text(json.dumps(self.config.to_dict()))
        script = '''
import os, sys
from pathlib import Path
from rsisys.application import Application, ApplicationConfig
app = Application(ApplicationConfig.load(Path(sys.argv[1])), durable=True)
app.runner.step(app.config.objective)
os._exit(0)
'''
        process = subprocess.run([sys.executable, '-c', script, str(config_path)], cwd=ROOT,
                                 env={**os.environ, 'PYTHONPATH': str(ROOT / 'src')},
                                 capture_output=True, text=True, timeout=15)
        self.assertEqual(process.returncode, 0, process.stderr)
        with Application.resume(self.config.data_dir) as app:
            self.assertEqual(self.tick(app).kind, 'completed')
            self.assertEqual(len(app.runner.decision_making.harness.memory.history(kind='operation')), 1)

    def test_failed_reconciliation_remains_blocked_after_restart(self):
        with self.start() as app:
            with patch.object(app.runner, '_transport_step', side_effect=RuntimeError('interrupted')):
                with self.assertRaises(RuntimeError):
                    self.tick(app)
            def bad_repair(application, incident):
                raise ConnectionError('cannot verify')
            with self.assertRaises(ConnectionError):
                app.recovery.reconcile(bad_repair, evidence='attempted verification')
        with Application.resume(self.config.data_dir) as app:
            self.assertEqual(app.recovery.intervention['operation'], 'reconcile')
            with self.assertRaises(RecoveryRequired):
                self.tick(app)

    def test_session_extension_rebinds_only_with_explicit_factory(self):
        with self.start(decision_factory=extension_factory) as app:
            app.runner.decision_making.harness.session = 'persistent-session'
            self.tick(app)
        with self.assertRaisesRegex(TypeError, 'original module assembly|recovery_id'):
            Application.resume(self.config.data_dir)
        with Application.resume(self.config.data_dir, decision_factory=extension_factory) as app:
            self.assertEqual(app.runner.decision_making.harness.session, 'persistent-session')
            self.assertEqual(self.tick(app).kind, 'completed')

    def test_agenda_archive_and_compactor_survive_recovery_and_snapshot(self):
        from rsisys.loops.decision_making.harness.context import BasicContextCompactor
        with self.start() as app:
            dm = app.runner.decision_making.harness
            dm.research_agenda.patch(expected_revision=0, upserts=[{'id': 'a', 'kind': 'agenda', 'text': 'Check evidence'}],
                                     remove_ids=[], reason='research plan', evidence_refs=[])
            dm.context_compactor = BasicContextCompactor(trigger_tokens=90000, target_tokens=30000)
            self.tick(app)
            archive = dm.research_agenda.export()
            manifest = LocalSnapshotCapture(app).capture('agenda')
            snapshot = self.config.data_dir / 'snapshots' / manifest.snapshot_id
        for constructor in (lambda: Application.resume(self.config.data_dir),
                            lambda: restore_snapshot(snapshot, self.root / 'fork')):
            with constructor() as app:
                dm = app.runner.decision_making.harness
                self.assertEqual(dm.research_agenda.export(), archive)
                self.assertEqual(dm.context_compactor.trigger_tokens, 90000)
                self.assertEqual(dm.context_compactor.target_tokens, 30000)
                self.assertIs(dm.tools.research_tools.agenda, dm.research_agenda)
                self.assertTrue((app.config.data_dir / 'decision_research.json').exists())

    def test_workspace_lock_and_checkpoint_integrity(self):
        with self.start():
            with self.assertRaisesRegex(RuntimeError, 'another process'):
                Application.resume(self.config.data_dir)
        path = self.config.data_dir / 'recovery' / 'checkpoint.json'
        value = json.loads(path.read_text())
        value['sha256'] = 'invalid'
        path.write_text(json.dumps(value))
        with self.assertRaisesRegex(ValueError, 'digest'):
            Application.resume(self.config.data_dir)

    def test_update_uses_same_fail_closed_boundary(self):
        with self.start() as app:
            directive, job = self.training(app)
            with self.assertRaises(NotImplementedError):
                app.runner.update(DirectiveUpdate(directive.directive_id, 0, 'update1', {'lr': .1}))
        with Application.resume(self.config.data_dir) as app:
            self.assertEqual(app.recovery.intervention['operation'], 'update')
            self.assertEqual(app.runner.exploration.pending().job, job)
            with self.assertRaises(RecoveryRequired):
                self.tick(app)

    def test_first_interrupted_step_retains_budget_start_from_journal(self):
        with self.start(run_policy=TimeRunPolicy(10)) as app:
            app.recovery.wall_clock = lambda: time.time() - 30
            with patch.object(app.recovery, '_checkpoint', side_effect=OSError('disk unavailable')):
                with self.assertRaises(OSError):
                    self.tick(app)
        with Application.resume(self.config.data_dir) as app:
            self.assertGreaterEqual(app.runner._run_context(self.config.objective).elapsed_seconds, 30)
            with self.assertRaisesRegex(RuntimeError, 'Reconcile'):
                LocalSnapshotCapture(app).capture('cannot-bypass-intervention')

    def test_cleanup_deadline_and_terminal_result_survive_restart(self):
        with self.start(run_policy=TimeRunPolicy(1, shutdown='drain'), cleanup_timeout_seconds=3) as app:
            directive, job = self.training(app)
            app.runner._run_started -= 2
            app.recovery.wall_clock = lambda: time.time() - 10
            result = self.tick(app)
            self.assertEqual(result.kind, 'waiting')
            self.assertIsNotNone(app.runner._stop_decision)
        with Application.resume(self.config.data_dir) as app:
            result = self.tick(app)
            self.assertEqual(result.kind, 'finished')
            self.assertTrue(result.event['run_stop']['cleanup_timed_out'])
            self.assertFalse(result.event['run_stop']['shutdown_complete'])
            self.assertEqual(result.event['run_stop']['unresolved_directives'], [directive.directive_id])
        with Application.resume(self.config.data_dir) as app:
            self.assertEqual(self.tick(app), result)

    def test_review_deadline_rebases_without_reset(self):
        with self.start() as app:
            self.training(app)
            app.recovery.wall_clock = lambda: time.time() - 50
            self.tick(app)
            remaining = app.runner._review_at - app.runner._clock()
        with Application.resume(self.config.data_dir) as app:
            restored = app.runner._review_at - app.runner._clock()
            self.assertAlmostEqual(restored, remaining - 50, delta=2)

    def test_external_contract_explicitly_restores_custom_policy_state(self):
        from rsisys.runtime.recovery import BuiltinRecovery
        from rsisys.runtime.run_policy import RunDecision
        class CustomPolicy:
            def __init__(self, seconds): self.seconds = seconds
            def evaluate(self, context):
                return RunDecision('stop', 'custom_budget') if context.elapsed_seconds >= self.seconds else RunDecision()
        class Contract:
            contract_id = 'test-custom/v1'
            def capture(self, app):
                return {'seconds': app.runner.run_policy.seconds, 'builtin': BuiltinRecovery().capture(app)}
            def restore(self, app, state):
                app.runner.run_policy.seconds = state['seconds']
                BuiltinRecovery().restore(app, state['builtin'])
        with self.start(run_policy=CustomPolicy(100), recovery_contract=Contract()) as app:
            self.tick(app)
        with self.assertRaisesRegex(TypeError, 'original recovery contract'):
            Application.resume(self.config.data_dir, run_policy=CustomPolicy(200))
        with Application.resume(self.config.data_dir, run_policy=CustomPolicy(200), recovery_contract=Contract()) as app:
            self.assertEqual(app.runner.run_policy.seconds, 100)
            self.assertEqual(self.tick(app).kind, 'completed')


class ResearchCancellationTests(unittest.TestCase):
    def test_cancel_hook_observes_clearance_and_no_hook_does_not_claim_success(self):
        from rsisys.loops.decision_making.loop import DecisionMakingLoop
        from rsisys.runtime.runner import Runner
        class Research:
            job = JobRef('session', 'inference')
            def pending(self): return None
            def pending_research_job(self): return self.job
            def cancel_research(self): self.job = None
        class Exploration:
            def pending(self): return None
        research = Research()
        loop = DecisionMakingLoop(research, None)
        runner = Runner(loop, Exploration(), run_policy=TimeRunPolicy(1), clock=lambda: 2)
        runner._run_started = 0
        result = runner.step('task')
        self.assertTrue(result.event['run_stop']['shutdown_complete'])
        class Uncancellable(Research):
            cancel_research = None
        runner = Runner(DecisionMakingLoop(Uncancellable(), None), Exploration(),
                        run_policy=TimeRunPolicy(1), clock=lambda: 2)
        runner._run_started = 0
        self.assertEqual(runner.step('task').kind, 'waiting')
        runner._clock = lambda: 40
        result = runner.step('task')
        self.assertFalse(result.event['run_stop']['shutdown_complete'])
        self.assertTrue(result.event['run_stop']['cleanup_timed_out'])
