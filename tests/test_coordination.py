from copy import deepcopy
from dataclasses import replace
from pathlib import Path
import unittest
import json
import tempfile
from unittest.mock import Mock

from rsisys.config import DemoConfig, build_demo
from rsisys.application import Application, ApplicationConfig
from rsisys.snapshot.local import LocalSnapshotCapture, restore_snapshot
from rsisys.loops.contracts.model_service import GenerationResponse
from rsisys.loops.contracts.directive import Directive
from rsisys.loops.contracts.requests import SFTSpec, RLSpec, EvaluationSpec
from rsisys.loops.contracts.lifecycle import ExecutionStatus
from rsisys.loops.contracts.jobs import JobOutcome, JobStatus
from rsisys.loops.decision_making.decision import Decision
from rsisys.loops.exploration.learner import ActorTrainingRequest
from rsisys.runtime.runner import Runner

CONFIG = Path(__file__).resolve().parents[1] / 'configs/demo.json'


class SequencePolicy:
    def __init__(self, *decisions):
        self.decisions = iter(decisions)
        self.calls = 0

    def decide(self, context):
        self.calls += 1
        return next(self.decisions)


class CoordinationTests(unittest.TestCase):
    def setUp(self):
        self.runner, self.store = build_demo(DemoConfig.load(CONFIG), with_learning=True)
        self.dm, self.actor = self.runner.decision_making, self.runner.exploration
        self.artifacts = self.dm.space.artifacts
        self.data = self.artifacts.put(b'data', kind='training_data')

    def dispatch(self, directive):
        self.dm.policy = SequencePolicy(Decision('explore', directive=directive), Decision('finish'))

    def sft(self, ident='sft', apply=False):
        return Directive(ident, 'task', 'train', mode='sft', input_refs=(self.data,),
                         payload=SFTSpec('recipe', apply_result=apply))

    def finish(self):
        for _ in range(40):
            result = self.runner.run('task', 1)
            if result.status != 'waiting':
                return result
        self.fail('Command did not reach a terminal delivery')

    def test_sft_waits_without_redeciding_then_delivers_candidate(self):
        self.dispatch(self.sft())
        base = self.actor.policy.model
        first = self.runner.run('task', 1)
        self.assertEqual(first.status, 'waiting')
        self.assertEqual(first.waiting_directive_id, 'sft')
        self.assertIsNotNone(first.waiting_job)
        self.assertEqual(self.dm.policy.calls, 1)
        self.assertEqual(self.finish().status, 'completed')
        self.assertEqual(self.dm.policy.calls, 1)
        self.assertEqual(self.actor.policy.model, base)
        feedback = self.dm.state.last_feedback
        self.assertEqual(feedback.status, ExecutionStatus.COMPLETED)
        self.assertEqual(len(feedback.candidates), 1)
        self.assertEqual(len(feedback.evidence_refs), 1)
        self.assertTrue(feedback.simulated)
        self.assertEqual(self.runner.run('task', 1).status, 'finished')

    def test_sft_explicit_apply_and_evaluate_are_separate_commands(self):
        self.dispatch(self.sft(apply=True))
        base = self.actor.policy.model
        self.finish()
        self.assertNotEqual(base, self.actor.policy.model)
        candidate = self.dm.state.last_feedback.candidates[0]
        self.dispatch(Directive('eval', 'task', 'measure', mode='evaluate',
                                payload=EvaluationSpec(candidate, 'protocol')))
        self.finish()
        feedback = self.dm.state.last_feedback
        self.assertEqual(feedback.directive_id, 'eval')
        self.assertEqual(feedback.quality_metrics, {})
        self.assertEqual(len(feedback.evidence_refs), 1)
        self.assertTrue(feedback.simulated)
        record = next(r for r in self.store.records() if r.record_id == feedback.evidence_refs[0])
        self.assertEqual(record.kind, 'evaluation')

    def test_rl_updates_within_one_directive_and_aggregates_evidence(self):
        self.dispatch(Directive('rl', 'task', 'improve', mode='rl', payload=RLSpec('recipe', 2)))
        base = self.actor.policy.model
        self.assertEqual(self.finish().status, 'completed')
        feedback = self.dm.state.last_feedback
        self.assertEqual(feedback.directive_id, 'rl')
        self.assertEqual(self.dm.state.completed, 1)
        self.assertEqual(self.dm.policy.calls, 1)
        self.assertEqual(len(feedback.candidates), 2)
        self.assertEqual(len(feedback.jobs), 2)
        self.assertEqual(len(feedback.evidence_refs), 4)
        self.assertTrue(feedback.simulated)
        trajectories = self.actor.memory.history(kind='trajectory')
        self.assertEqual(trajectories[0].details['model']['revision'], base.revision)
        self.assertNotEqual(trajectories[0].details['model'], trajectories[1].details['model'])

    def test_feedback_write_after_commit_retries_delivery_without_reexecution(self):
        self.dispatch(Directive('once', 'task', 'execute'))
        record = self.dm.experience.record
        failed = False
        def flaky(value):
            nonlocal failed
            record(value)
            if value.kind == 'improvement' and not failed:
                failed = True
                raise OSError('commit response lost')
        self.dm.experience.record = flaky
        with self.assertRaisesRegex(OSError, 'commit response lost'):
            self.runner.run('task', 1)
        count = self.actor.environment.sandbox.execution_count
        self.assertTrue(self.runner.busy)
        self.assertEqual(self.finish().status, 'completed')
        self.assertEqual(count, self.actor.environment.sandbox.execution_count)
        self.assertEqual(self.dm.state.completed, 1)
        self.assertEqual(len([r for r in self.store.records() if r.kind == 'improvement']), 1)

    def test_lost_delivery_ack_is_idempotent_and_blocks_new_decisions(self):
        self.dispatch(Directive('once', 'task', 'execute'))
        receive = self.dm.receive_feedback
        calls = 0
        def flaky(directive, feedback):
            nonlocal calls
            receive(directive, feedback)
            calls += 1
            if calls == 1:
                raise OSError('ack lost')
        self.dm.receive_feedback = flaky
        with self.assertRaises(OSError):
            self.runner.run('task', 1)
        self.finish()
        self.assertEqual(self.dm.policy.calls, 1)
        self.assertEqual(self.dm.state.completed, 1)
        self.assertEqual(self.actor.environment.sandbox.execution_count, 1)

    def test_uncertain_submit_reuses_request_key_and_does_not_restart(self):
        self.dispatch(self.sft())
        submit = self.actor.learner.submit
        calls = []
        def lost(request, *, idempotency_key):
            calls.append((deepcopy(request), idempotency_key))
            job = submit(request, idempotency_key=idempotency_key)
            if len(calls) == 1:
                raise OSError('submit response lost')
            return job
        self.actor.learner.submit = lost
        with self.assertRaises(OSError):
            self.runner.run('task', 1)
        self.assertTrue(self.runner.busy)
        with self.assertRaisesRegex(RuntimeError, 'directive owns'):
            self.actor.run(Directive('outside', 'task', 'run'))
        self.finish()
        self.assertEqual(calls[0], calls[1])
        self.assertEqual(len(self.actor.learner.jobs._jobs), 1)
        self.assertEqual(self.dm.policy.calls, 1)

    def test_cancel_after_lost_submit_resolves_then_cancels_original_job(self):
        self.dispatch(self.sft(apply=True))
        base = self.actor.policy.model
        submit = self.actor.learner.submit
        calls = 0
        def lost(request, *, idempotency_key):
            nonlocal calls
            job = submit(request, idempotency_key=idempotency_key)
            calls += 1
            if calls == 1:
                raise OSError('lost')
            return job
        self.actor.learner.submit = lost
        with self.assertRaises(OSError):
            self.runner.run('task', 1)
        self.assertFalse(self.runner.cancel().terminal)
        self.finish()
        self.assertEqual(self.dm.state.last_feedback.status, ExecutionStatus.CANCELLED)
        self.assertEqual(self.actor.policy.model, base)
        self.assertEqual(len(self.actor.learner.jobs._jobs), 1)

    def test_cancel_racing_training_success_never_applies_weights(self):
        self.dispatch(self.sft(apply=True))
        base = self.actor.policy.model
        backend = self.actor.learner
        backend.cancel = lambda job: None
        self.runner.run('task', 1)
        self.runner.cancel()
        self.finish()
        self.assertEqual(self.dm.state.last_feedback.status, ExecutionStatus.CANCELLED)
        self.assertEqual(self.actor.policy.model, base)
        self.assertEqual(len(self.dm.state.last_feedback.candidates), 1)

    def test_cancel_rl_retains_completed_updates_and_does_not_start_another_rollout(self):
        self.dispatch(Directive('rl', 'task', 'improve', mode='rl', payload=RLSpec('recipe', 3)))
        for _ in range(12):
            self.runner.step('task')
            if self.actor.state.rl.updates_completed == 1:
                break
        self.assertEqual(self.actor.state.rl.updates_completed, 1)
        count = self.actor.environment.sandbox.execution_count
        self.runner.cancel()
        self.finish()
        self.assertEqual(self.dm.state.last_feedback.status, ExecutionStatus.CANCELLED)
        self.assertEqual(len(self.dm.state.last_feedback.candidates), 1)
        self.assertEqual(self.actor.environment.sandbox.execution_count, count)

    def test_training_failure_delivers_terminal_feedback_and_allows_next_decision(self):
        self.dispatch(self.sft(apply=True))
        self.actor.learner.poll = lambda job: JobOutcome(job, JobStatus.FAILED, error='backend failed', simulated=True)
        self.finish()
        self.assertEqual(self.dm.state.last_feedback.status, ExecutionStatus.FAILED)
        self.assertEqual(self.runner.run('task', 1).status, 'finished')

    def test_command_identity_and_ownership(self):
        directive = self.sft()
        self.actor.accept(directive)
        self.actor.accept(directive)
        with self.assertRaisesRegex(ValueError, 'different content'):
            self.actor.accept(replace(directive, strategy='different'))
        with self.assertRaisesRegex(RuntimeError, 'busy'):
            self.actor.accept(replace(directive, directive_id='second'))
        with self.assertRaises(RuntimeError):
            self.actor.submit_learning(ActorTrainingRequest(self.actor.policy.model, (self.data,), 'r'), idempotency_key='outside')
        self.actor.cancel(directive.directive_id)
        progress = self.actor.advance(directive.directive_id)
        self.assertTrue(progress.terminal)
        self.assertEqual(len(self.actor.learner.jobs._jobs), 0)
        self.actor.acknowledge(directive.directive_id)
        self.assertTrue(self.actor.accept(directive).terminal)

    def test_runner_only_requires_public_endpoints(self):
        # Deliberately expose no .state, policy, learner or environment to the Runner.
        dm, actor = self.dm, self.actor
        class DecisionPort:
            step = staticmethod(dm.step)
            pending = staticmethod(dm.pending)
            receive_feedback = staticmethod(dm.receive_feedback)
        class ActorPort:
            accept = staticmethod(actor.accept)
            advance = staticmethod(actor.advance)
            inspect = staticmethod(actor.inspect)
            cancel = staticmethod(actor.cancel)
            acknowledge = staticmethod(actor.acknowledge)
        runner = Runner(DecisionPort(), ActorPort())
        self.assertEqual(runner.run('task', 1).status, 'completed')

    def test_recreated_runner_can_continue_pending_in_process_command(self):
        self.dispatch(self.sft())
        self.runner.run('task', 1)
        self.runner = Runner(self.dm, self.actor)
        self.assertEqual(self.finish().status, 'completed')
        self.assertEqual(self.dm.policy.calls, 1)


    def test_reusing_a_completed_directive_id_is_rejected_before_dispatch(self):
        directive = Directive('once', 'task', 'execute')
        self.dispatch(directive)
        self.finish()
        self.dispatch(replace(directive, strategy='new content'))
        with self.assertRaisesRegex(ValueError, 'fresh directive ID'):
            self.runner.step('task')
        self.assertEqual(self.actor.environment.sandbox.execution_count, 1)
        self.assertIsNone(self.dm.pending())

    def test_feedback_rejects_mutated_directive_and_wrong_feedback_identity(self):
        directive = self.sft()
        self.dispatch(directive)
        self.runner.step('task')
        while not self.actor.inspect(directive.directive_id).terminal:
            self.actor.advance(directive.directive_id)
        feedback = self.actor.inspect(directive.directive_id).feedback
        with self.assertRaisesRegex(ValueError, 'content differs'):
            self.dm.receive_feedback(replace(directive, strategy='changed'), feedback)
        with self.assertRaisesRegex(ValueError, 'does not match'):
            self.dm.receive_feedback(directive, replace(feedback, directive_id='other'))
        self.assertIsNotNone(self.dm.pending())
        self.finish()

    def test_rl_uncertain_submit_and_cancel_preserve_one_rollout_and_job(self):
        self.dispatch(Directive('rl', 'task', 'improve', mode='rl', payload=RLSpec('recipe', 2)))
        submit = self.actor.learner.submit
        calls = 0
        def lost(request, *, idempotency_key):
            nonlocal calls
            job = submit(request, idempotency_key=idempotency_key)
            calls += 1
            if calls == 1:
                raise OSError('lost RL submit')
            return job
        self.actor.learner.submit = lost
        self.runner.step('task')  # initialize RL
        self.runner.step('task')  # rollout
        with self.assertRaises(OSError):
            self.runner.step('task')
        self.runner.cancel()
        self.finish()
        self.assertEqual(self.actor.environment.sandbox.execution_count, 1)
        self.assertEqual(len(self.actor.learner.jobs._jobs), 1)
        self.assertEqual(self.dm.state.last_feedback.status, ExecutionStatus.CANCELLED)
        self.assertEqual(len(self.dm.state.last_feedback.jobs), 1)
        self.assertTrue(self.dm.state.last_feedback.simulated)

    def test_poll_failure_keeps_the_original_command_recoverable(self):
        self.dispatch(self.sft())
        poll = self.actor.learner.poll
        calls = 0
        def flaky(job):
            nonlocal calls
            calls += 1
            if calls == 1:
                raise OSError('poll temporarily unavailable')
            return poll(job)
        self.actor.learner.poll = flaky
        self.runner.step('task')
        with self.assertRaises(OSError):
            self.runner.step('task')
        self.assertTrue(self.runner.busy)
        self.finish()
        self.assertEqual(len(self.actor.learner.jobs._jobs), 1)
        self.assertEqual(self.dm.policy.calls, 1)

    def test_invalid_learning_payload_becomes_failure_feedback_before_submission(self):
        self.dispatch(Directive('invalid', 'task', 'train', mode='sft', payload=RLSpec('recipe')))
        with self.assertRaisesRegex(ValueError, 'SFTSpec'):
            self.runner.step('task')
        self.assertEqual(self.dm.state.last_feedback.status, ExecutionStatus.FAILED)
        self.assertFalse(self.runner.busy)
        self.assertEqual(len(self.actor.learner.jobs._jobs), 0)

    def test_model_policy_can_train_and_evaluate_using_explicit_feedback(self):
        self.dm.step('task')  # prepare an artifact with the normal policy
        calls = []
        def generate(request):
            context = json.loads(request.prompt)
            calls.append(context)
            if len(calls) == 1:
                payload = {'kind': 'explore', 'strategy': 'train', 'mode': 'sft', 'recipe_ref': 'r',
                           'input_ids': [context['operation_results'][0]['artifacts'][0]['artifact_id']]}
            elif len(calls) == 2:
                candidate = context['last_feedback']['candidates'][0]['candidate_id']
                payload = {'kind': 'explore', 'strategy': 'measure', 'mode': 'evaluate',
                           'candidate_id': candidate, 'protocol_ref': 'p'}
            else:
                payload = {'kind': 'finish'}
            return GenerationResponse(json.dumps(payload), request.model, True)
        self.dm.policy.backend = Mock(generate=generate)
        self.finish()
        self.finish()
        self.assertEqual(self.runner.run('task', 1).status, 'finished')
        self.assertEqual(len(calls), 3)
        self.assertEqual(calls[2]['last_feedback']['quality_metrics'], {})

    def test_snapshot_rejects_unacknowledged_delivery_then_restores_finished_state(self):
        with tempfile.TemporaryDirectory() as tmp:
            config = ApplicationConfig.load(CONFIG.parent / 'local.json')
            config = replace(config, data_dir=Path(tmp) / 'source')
            with Application(config) as app:
                receive = app.runner.decision_making.receive_feedback
                def lost(directive, feedback):
                    receive(directive, feedback)
                    raise OSError('ack lost')
                app.runner.decision_making.receive_feedback = lost
                with self.assertRaises(OSError):
                    app.run()
                self.assertIsNone(app.runner.decision_making.pending())
                with self.assertRaisesRegex(RuntimeError, 'pending coordination'):
                    LocalSnapshotCapture(app).capture('too-early')
                app.runner.decision_making.receive_feedback = receive
                app.run()
                manifest = LocalSnapshotCapture(app).capture('done')
                directory = config.data_dir / 'snapshots' / manifest.snapshot_id
            with restore_snapshot(directory, Path(tmp) / 'restored') as restored:
                self.assertEqual(restored.runner.decision_making.state.completed, 1)
                self.assertEqual(len(restored.runner.decision_making.state.issued_directive_ids), 1)
                self.assertEqual(restored.run().status, 'completed')


if __name__ == '__main__':
    unittest.main()
