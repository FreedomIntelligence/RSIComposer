"""Integration boundaries between default harness memory and system improvement evidence."""

from dataclasses import asdict, replace
import json
from pathlib import Path
import sqlite3
import tempfile
import unittest
from unittest.mock import patch

from rsisys.application import Application, ApplicationConfig
from rsisys.config import DemoConfig, build_demo
from rsisys.experience.records import ExperienceRecord
from rsisys.experience.sqlite import SQLiteExperienceStore
from rsisys.loops.contracts.directive import Directive
from rsisys.loops.contracts.requests import RLSpec
from rsisys.loops.decision_making.harness.decision import Decision
from unittest.mock import Mock
from rsisys.loops._memory import Memory
from rsisys.loops.contracts.model_service import BackendBinding, ModelRef
from rsisys.loops.decision_making.harness.learner import DecisionTrainingRequest
from rsisys.loops.exploration.harness.learner import ActorTrainingRequest
from rsisys.loops.exploration.harness.memory import ExplorationMemory
from rsisys.snapshot.local import LocalSnapshotCapture, restore_snapshot

CONFIG = Path(__file__).resolve().parents[1] / 'configs/demo.json'


class ExperienceBoundaryTests(unittest.TestCase):
    def setUp(self):
        self.runner, self.store = build_demo(DemoConfig.load(CONFIG), with_learning=True)
        self.decision = self.runner.decision_making
        self.actor = self.runner.exploration

    def test_task_content_stays_local_and_feedback_references_system_evidence(self):
        self.runner.run('task-payload-that-is-not-system-feedback', 1)
        history = self.actor.harness.memory.history(kind='trajectory')
        self.assertIn('task-payload-that-is-not-system-feedback', json.dumps(asdict(history[0])))
        system = self.store.records()
        self.assertNotIn('task-payload-that-is-not-system-feedback', json.dumps([asdict(r) for r in system]))
        record = next(r for r in system if r.record_id == self.decision.harness.state.last_feedback.evidence_refs[0])
        self.assertEqual(record.kind, 'runtime_feedback')
        self.assertEqual(record.details['trajectory_refs'], (history[0].record_id,))
        self.assertGreaterEqual(record.details['execution_metrics']['elapsed_seconds'], 0)
        self.assertEqual(record.details['execution_metrics']['steps'], 1)
        self.assertNotIn('transition', record.details)
        self.assertFalse(hasattr(self.decision.harness.experience, 'remember'))

    def test_decision_reads_all_feedback_sources_but_memories_do_not_leak(self):
        self.actor.harness.memory.remember(Memory('same-id', 'procedural', 'actor-only-note'))
        self.decision.harness.memory.remember(Memory('same-id', 'semantic', 'decision-only-note'))
        self.actor.harness.run(Directive('prior', 'a task', 'execute'), action_space=self.actor.action_space)
        with patch.object(self.decision.harness.policy, 'decide', wraps=self.decision.harness.policy.decide) as decide:
            self.decision.step('research')
        context = decide.call_args.args[0]
        self.assertTrue(any(record.scope == 'actor' for record in context.experience))
        self.assertEqual([memory.content for memory in context.memories], ['decision-only-note'])
        with patch.object(self.actor.harness.policy, 'act', wraps=self.actor.harness.policy.act) as act:
            self.actor.harness.run(Directive('next', 'another task', 'execute'), action_space=self.actor.action_space)
        context = act.call_args.args[0]
        self.assertEqual(context.history, ())
        self.assertEqual([memory.content for memory in context.memories], ['actor-only-note'])
        self.assertFalse(hasattr(context, 'experience'))

    def test_rl_uses_local_trajectories_and_rejects_system_record_ids(self):
        feedback = self.actor.harness.run(Directive('rollout', 'task', 'explore'), action_space=self.actor.action_space)
        request = ActorTrainingRequest(self.actor.harness.policy.model, (), 'recipe', 'rl', feedback.evidence_refs)
        with self.assertRaisesRegex(ValueError, 'trajectory history'):
            self.actor.harness.submit_learning(request, idempotency_key='wrong-store')
        refs = tuple(r.record_id for r in self.actor.harness.memory.history(task_id='rollout'))
        request = replace(request, trajectory_refs=refs)
        job = self.actor.harness.submit_learning(request, idempotency_key='right-store')
        self.assertEqual(self.actor.harness.training.request(job).trajectory_refs, refs)
        # Decision learning may explicitly select evidence from either loop.
        decision_job = self.decision.harness.submit_learning(
            DecisionTrainingRequest(self.decision.harness.policy.model, feedback.evidence_refs, 'diagnose'),
            idempotency_key='decision-from-actor-feedback',
        )
        self.assertEqual(self.decision.harness.training.request(decision_job).experience_refs, feedback.evidence_refs)

    def test_feedback_memory_failure_retries_without_reexecuting_actor(self):
        append = self.decision.harness.memory.append
        failed = False
        def lose_ack(entry):
            nonlocal failed
            append(entry)
            if entry.kind == 'feedback' and not failed:
                failed = True
                raise OSError('memory response lost')
        self.decision.harness.memory.append = lose_ack
        with self.assertRaisesRegex(OSError, 'memory response lost'):
            self.runner.run('task', 1)
        count = self.actor.harness.environment.sandbox.execution_count
        self.assertEqual(self.runner.run('task', 1).status, 'completed')
        self.assertEqual(self.actor.harness.environment.sandbox.execution_count, count)
        self.assertEqual(len(self.decision.harness.memory.history(kind='feedback')), 1)
        self.assertEqual(len([r for r in self.store.records() if r.kind == 'improvement']), 1)

    def test_actor_feedback_commit_response_loss_retries_without_reexecution(self):
        record = self.actor.harness.experience.record
        failed = False
        def lose_ack(value):
            nonlocal failed
            record(value)
            if value.scope == 'actor' and not failed:
                failed = True
                raise OSError('actor response lost')
        self.actor.harness.experience.record = lose_ack
        with self.assertRaisesRegex(OSError, 'actor response lost'):
            self.runner.run('task', 1)
        count = self.actor.harness.environment.sandbox.execution_count
        saved = self.actor.harness.state.pending_episode.record
        self.assertEqual(self.runner.run('task', 1).status, 'completed')
        self.assertEqual(self.actor.harness.environment.sandbox.execution_count, count)
        self.assertEqual(len([r for r in self.store.records() if r.scope == 'actor']), 1)
        self.assertEqual(self.decision.harness.state.last_feedback.evidence_refs, (saved.record_id,))
        self.assertEqual(self.actor.harness.state.completed, 1)

    def test_rl_pending_feedback_survives_storage_failure_and_cancellation(self):
        for cancel in (False, True):
            with self.subTest(cancel=cancel):
                runner, store = build_demo(DemoConfig.load(CONFIG), with_learning=True)
                dm, actor = runner.decision_making, runner.exploration
                directive = Directive('rl-persist', 'task', 'improve', mode='rl', payload=RLSpec('recipe', 1))
                dm.harness.policy = Mock(decide=Mock(return_value=Decision('explore', directive=directive)))
                runner.step('task')
                record = actor.harness.experience.record
                def unavailable(value):
                    if value.scope == 'actor':
                        raise OSError('offline')
                    record(value)
                actor.harness.experience.record = unavailable
                with self.assertRaisesRegex(OSError, 'offline'):
                    runner.step('task')
                self.assertFalse(actor.harness.state.rl.terminal)
                self.assertIsNotNone(actor.harness.state.pending_episode)
                count = actor.harness.environment.sandbox.execution_count
                if cancel:
                    runner.cancel()
                actor.harness.experience.record = record
                for _ in range(10):
                    result = runner.step('task')
                    if result.kind == 'completed':
                        break
                self.assertEqual(result.kind, 'completed')
                self.assertEqual(actor.harness.environment.sandbox.execution_count, count)
                self.assertIsNone(actor.harness.state.pending_episode)
                self.assertEqual(actor.harness.state.completed, 1)
                self.assertEqual(len([r for r in store.records() if r.kind == 'runtime_feedback']), 1)
                self.assertEqual(actor.harness.state.rl.updates_completed, 0 if cancel else 1)

    def test_cancel_after_execution_does_not_discard_pending_evidence(self):
        record = self.actor.harness.experience.record
        def unavailable(value):
            if value.scope == 'actor':
                raise OSError('offline')
            record(value)
        self.actor.harness.experience.record = unavailable
        with self.assertRaisesRegex(OSError, 'offline'):
            self.runner.run('task', 1)
        count = self.actor.harness.environment.sandbox.execution_count
        self.runner.cancel()
        self.actor.harness.experience.record = record
        self.assertEqual(self.runner.run('task', 1).status, 'completed')
        self.assertEqual(self.actor.harness.environment.sandbox.execution_count, count)
        self.assertIsNone(self.actor.harness.state.pending_episode)
        self.assertTrue(self.decision.harness.state.last_feedback.evidence_refs)

    def test_snapshot_restores_local_rl_inputs_and_independent_context_limits(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            model = ModelRef('mock', 'model', 'v1')
            config = ApplicationConfig('task', root/'run', {'mock': BackendBinding('mock', 'mock')},
                                       {'decision_maker': model, 'actor': model}, learning='mock')
            with Application(config) as app:
                app.run()
                dm, actor = app.runner.decision_making, app.runner.exploration
                dm.harness.experience_limit, dm.harness.memory_limit, actor.harness.memory_limit = 2, 3, 4
                refs = tuple(r.record_id for r in actor.harness.memory.history(kind='trajectory'))
                manifest = LocalSnapshotCapture(app).capture('memory')
                snapshot = config.data_dir/'snapshots'/manifest.snapshot_id
            with restore_snapshot(snapshot, root/'restored') as app:
                dm, actor = app.runner.decision_making, app.runner.exploration
                self.assertEqual((dm.harness.experience_limit, dm.harness.memory_limit, actor.harness.memory_limit), (2, 3, 4))
                self.assertEqual(tuple(r.record_id for r in actor.harness.memory.history()), refs)
                actor.harness.submit_learning(ActorTrainingRequest(actor.harness.policy.model, (), 'recipe', 'rl', refs),
                                      idempotency_key='restored-trajectories')

    def test_snapshot_rejects_both_pending_record_writes(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            model = ModelRef('mock', 'model', 'v1')
            for owner in ('decision_maker', 'actor'):
                config = ApplicationConfig('task', root/owner, {'mock': BackendBinding('mock', 'mock')},
                                           {'decision_maker': model, 'actor': model})
                with Application(config) as app:
                    record = app.experience.record
                    def offline(value):
                        raise OSError('record offline')
                    app.experience.record = offline
                    with self.assertRaisesRegex(OSError, 'record offline'):
                        if owner == 'decision_maker':
                            app.runner.decision_making.step('task')
                        else:
                            app.runner.exploration.harness.run(Directive('pending', 'task', 'execute'), action_space=app.runner.exploration.action_space)
                    with self.assertRaisesRegex(RuntimeError, 'pending'):
                        LocalSnapshotCapture(app).capture('unfinished-write')
                    app.experience.record = record
                    if owner == 'decision_maker':
                        app.runner.decision_making.step('task')
                    else:
                        app.runner.exploration.harness.run(Directive('pending', 'task', 'execute'), action_space=app.runner.exploration.action_space)
                    LocalSnapshotCapture(app).capture('write-completed')

    def test_old_experience_schema_is_rejected_without_discarding_memory(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory)/'old.sqlite3'
            connection = sqlite3.connect(path)
            connection.executescript("CREATE TABLE metadata (key TEXT PRIMARY KEY, value TEXT NOT NULL);"
                                     "INSERT INTO metadata VALUES ('schema_version', '1');"
                                     "CREATE TABLE memory (content TEXT); INSERT INTO memory VALUES ('old note');")
            connection.close()
            with self.assertRaisesRegex(ValueError, 'schema'):
                SQLiteExperienceStore(path)
            connection = sqlite3.connect(path)
            self.assertEqual(connection.execute('SELECT content FROM memory').fetchone()[0], 'old note')
            connection.close()

    def test_custom_memory_works_in_loop_but_needs_explicit_snapshot_support(self):
        class CustomMemory(ExplorationMemory):
            pass
        self.actor.harness.memory = CustomMemory()
        self.actor.harness.run(Directive('custom', 'task', 'execute'), action_space=self.actor.action_space)
        self.assertEqual(len(self.actor.harness.memory.history()), 1)
        with tempfile.TemporaryDirectory() as directory:
            model = ModelRef('mock', 'model', 'v1')
            config = ApplicationConfig('task', Path(directory)/'run', {'mock': BackendBinding('mock', 'mock')},
                                       {'decision_maker': model, 'actor': model})
            with Application(config) as app:
                app.runner.exploration.harness.memory = CustomMemory()
                with self.assertRaisesRegex(TypeError, 'memories'):
                    LocalSnapshotCapture(app).capture('custom')


if __name__ == '__main__':
    unittest.main()
