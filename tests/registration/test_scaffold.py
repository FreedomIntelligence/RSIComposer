from tests import ROOT as REPO_ROOT
from dataclasses import replace
from pathlib import Path
import unittest
from unittest.mock import Mock

from rsisys.config import DemoConfig, build_demo
from rsisys.loops.contracts.directive import Directive
from rsisys.loops.contracts.jobs import JobRef, JobStatus
from rsisys.loops.contracts.lifecycle import ExecutionStatus
from rsisys.loops.contracts.model_service import BackendBinding, ModelRef
from rsisys.loops.decision_making.harness.decision import Decision
from rsisys.loops.decision_making.action_space import OperationCall, OperationResult
from rsisys.loops.decision_making.harness.learner import DecisionTrainingRequest, UnimplementedDecisionLearner
from rsisys.loops.exploration.harness.environment import Transition
from rsisys.loops.exploration.harness.learner import ActorTrainingRequest, UnimplementedActorLearner
from rsisys.runtime.artifacts import InMemoryArtifactStore
from rsisys.runtime.model import MockModelBackend, RemoteModelBackend
from rsisys.runtime.sandbox import ExecRequest, MockSandboxBackend, SandboxSpec
from rsisys.snapshot.capture import UnimplementedCapture

CONFIG = REPO_ROOT / 'configs/demo.json'


class RecordingBackend(MockModelBackend):
    def __init__(self):
        self.requests = []

    def generate(self, request):
        self.requests.append(request)
        return super().generate(request)


class SequencePolicy:
    def __init__(self, *decisions):
        self.decisions = iter(decisions)
        self.calls = 0

    def decide(self, context):
        self.calls += 1
        return next(self.decisions)


class TwoStepEnvironment:
    def __init__(self):
        self.steps = 0
        self.closed = False

    def reset(self, directive):
        self.steps = 0
        return 'start'

    def step(self, action):
        self.steps += 1
        return Transition(f'step {self.steps}', self.steps == 2, True)

    def close(self):
        self.closed = True


class FailedCommandSandbox(MockSandboxBackend):
    def exec(self, sandbox, request):
        return super().exec(sandbox, ExecRequest(('unsupported',)))


class ScaffoldTests(unittest.TestCase):
    def setUp(self):
        self.config = DemoConfig.load(CONFIG)
        self.runner, self.store = build_demo(self.config)

    def dispatch(self):
        self.assertEqual(self.runner.decision_making.step('task').kind, 'progressed')
        step = self.runner.decision_making.step('task')
        self.assertEqual(step.kind, 'dispatch')
        return step.directive

    def test_preparation_and_artifact_handoff_before_actor(self):
        decision, exploration = self.runner.decision_making, self.runner.exploration
        step = decision.step('input content')
        self.assertEqual(step.kind, 'progressed')
        self.assertEqual(exploration.harness.state.completed, 0)
        ref = step.operation_result.artifacts[0]
        self.assertEqual(decision.harness.tools.artifacts.get(ref), b'input content\n')
        directive = decision.step('input content').directive
        self.assertEqual(directive.input_refs, (ref,))
        feedback = exploration.harness.run(directive, action_space=exploration.action_space)
        decision.receive_feedback(directive, feedback)
        self.assertIn('input content', feedback.summary)
        self.assertEqual(feedback.status, ExecutionStatus.COMPLETED)
        self.assertEqual(len(self.store.records()), 3)
        self.assertEqual(decision.harness.tools.sandbox.execution_count, 2)
        self.assertEqual(decision.harness.tools.sandbox.active_count, 0)
        self.assertEqual(exploration.harness.environment.artifacts.get(feedback.artifact_refs[0]).decode(), feedback.summary)

    def test_feedback_reaches_next_decision_without_fake_metrics(self):
        backend = RecordingBackend()
        self.runner.decision_making.harness.policy.backend = backend
        result = self.runner.run('test', 2)
        self.assertEqual(result.status, 'completed')
        self.assertEqual(result.rounds_completed, 2)
        self.assertEqual(result.decision_steps, 4)
        self.assertEqual(len(self.store.records()), 6)
        self.assertIn('prepared inputs', backend.requests[2].prompt)
        self.assertNotIn('prepared inputs', backend.requests[0].prompt)
        for event in result.events:
            if event['kind'] == 'dispatch':
                self.assertEqual(event['directive']['directive_id'], event['feedback']['directive_id'])
                self.assertTrue(event['feedback']['simulated'])
                self.assertEqual(event['feedback']['quality_metrics'], {})
        self.assertEqual(len(self.runner.exploration.harness.experience.retrieve('actor')), 2)
        self.assertEqual(len(self.runner.decision_making.harness.experience.retrieve('decision_maker')), 4)

    def test_wrong_and_duplicate_feedback_cannot_advance_loop(self):
        directive = self.dispatch()
        decision, exploration = self.runner.decision_making, self.runner.exploration
        with self.assertRaises(RuntimeError):
            decision.step('overlap')
        feedback = exploration.harness.run(directive, action_space=exploration.action_space)
        with self.assertRaises(ValueError):
            decision.receive_feedback(directive, replace(feedback, directive_id='wrong'))
        self.assertEqual(decision.harness.state.completed, 0)
        decision.receive_feedback(directive, feedback)
        decision.receive_feedback(directive, feedback)
        self.assertEqual(decision.harness.state.completed, 1)

    def test_action_validation_does_not_execute_and_releases_sandbox(self):
        sandbox = self.runner.exploration.harness.environment.sandbox
        with self.assertRaises(ValueError):
            self.runner.exploration.harness.run(Directive('test', 'task', 'strategy', allowed_actions=()), action_space=self.runner.exploration.action_space)
        self.assertEqual(sandbox.execution_count, 0)
        self.assertEqual(sandbox.active_count, 0)
        self.assertEqual(self.store.records()[0].kind, "runtime_feedback")
        self.assertEqual(self.store.records()[0].details["status"], "failed")
        self.assertIsNone(self.runner.exploration.harness.state.active_directive_id)

    def test_multistep_and_step_limit_are_distinct(self):
        environment = TwoStepEnvironment()
        self.runner.exploration.harness.environment = environment
        feedback = self.runner.exploration.harness.run(Directive('multi', 'task', 'strategy', max_steps=3), action_space=self.runner.exploration.action_space)
        self.assertEqual(feedback.status, ExecutionStatus.COMPLETED)
        self.assertEqual(len(feedback.evidence_refs), 1)
        self.assertEqual(environment.steps, 2)
        self.assertTrue(environment.closed)
        limited = self.runner.exploration.harness.run(Directive('limited', 'task', 'strategy', max_steps=1), action_space=self.runner.exploration.action_space)
        self.assertEqual(limited.status, ExecutionStatus.PARTIAL)
        self.assertEqual(len(limited.evidence_refs), 1)

    def test_actor_exception_reaches_decision_and_next_run_can_continue(self):
        actor = self.runner.exploration
        policy = actor.harness.policy
        error = ValueError('malformed model response')
        actor.harness.policy = Mock(act=Mock(side_effect=error))
        with self.assertRaises(ValueError) as caught:
            self.runner.run('test', 1)
        self.assertIs(caught.exception, error)
        decision = self.runner.decision_making
        self.assertIsNone(decision.harness.state.pending_directive_id)
        self.assertIsNone(actor.harness.state.active_directive_id)
        self.assertEqual(decision.harness.state.last_feedback.status, ExecutionStatus.FAILED)
        self.assertIn('malformed model response', decision.harness.state.last_feedback.error)
        self.assertEqual(actor.harness.environment.sandbox.active_count, 0)
        actor.harness.policy = policy
        self.assertEqual(self.runner.run('test', 1).status, 'completed')

    def test_failure_after_a_step_preserves_evidence_without_replaying_action(self):
        actor = self.runner.exploration
        environment = TwoStepEnvironment()
        actor.harness.environment = environment
        act = actor.harness.policy.act
        def fail_after_first_step(context):
            if environment.steps:
                raise ValueError('second response failed')
            return act(context)
        actor.harness.policy = Mock(act=Mock(side_effect=fail_after_first_step))
        decision = self.runner.decision_making
        decision.harness.policy = SequencePolicy(
            Decision('explore', directive=Directive('partial-error', 'task', 'strategy')),
            Decision('finish'),
        )
        with self.assertRaisesRegex(ValueError, 'second response failed'):
            self.runner.run('test', 1)
        feedback = decision.harness.state.last_feedback
        self.assertEqual(feedback.status, ExecutionStatus.FAILED)
        self.assertEqual(len(feedback.evidence_refs), 1)
        self.assertEqual(feedback.summary, 'step 1')
        self.assertTrue(feedback.simulated)
        self.assertTrue(environment.closed)
        self.assertEqual(self.runner.run('test', 1).status, 'finished')
        self.assertEqual(environment.steps, 1)

    def test_reset_and_cleanup_failures_release_loop_state(self):
        for phase in ('reset', 'close'):
            with self.subTest(phase=phase):
                runner, _ = build_demo(self.config)
                environment = TwoStepEnvironment()
                setattr(environment, phase, Mock(side_effect=OSError(f'{phase} failed')))
                runner.exploration.harness.environment = environment
                with self.assertRaisesRegex(OSError, f'{phase} failed'):
                    runner.run('test', 1)
                feedback = runner.decision_making.harness.state.last_feedback
                self.assertEqual(feedback.status, ExecutionStatus.FAILED)
                self.assertIn(f'{phase} failed', feedback.error)
                self.assertIsNone(runner.decision_making.harness.state.pending_directive_id)
                self.assertIsNone(runner.exploration.harness.state.active_directive_id)
                self.assertEqual(runner.exploration.harness.state.completed, 0)
                if phase == 'close':
                    self.assertEqual(len(feedback.evidence_refs), 1)

    def test_cleanup_error_does_not_mask_original_error(self):
        actor = self.runner.exploration
        original = ValueError('policy failed')
        actor.harness.policy = Mock(act=Mock(side_effect=original))
        environment = TwoStepEnvironment()
        environment.close = Mock(side_effect=OSError('cleanup failed'))
        actor.harness.environment = environment
        with self.assertRaises(ValueError) as caught:
            self.runner.run('test', 1)
        self.assertIs(caught.exception, original)
        self.assertIn('cleanup failed', original.__notes__[0])
        feedback = self.runner.decision_making.harness.state.last_feedback
        self.assertIn('policy failed', feedback.error)
        self.assertIn('cleanup failed', feedback.error)
        self.assertIsNone(self.runner.decision_making.harness.state.pending_directive_id)

    def test_unsupported_mode_returns_failure_to_decision_without_stale_feedback(self):
        self.runner.run('test', 1)
        decision = self.runner.decision_making
        directive = Directive('sft-error', 'task', 'strategy', mode='sft')
        from rsisys.loops.decision_making.action_space import DemoDecisionActionSpace
        decision.action_space = DemoDecisionActionSpace()
        decision.harness.policy = SequencePolicy(Decision('explore', directive=directive), Decision('finish'))
        with self.assertRaises(NotImplementedError):
            self.runner.run('test', 1)
        feedback = decision.harness.state.last_feedback
        self.assertEqual(feedback.directive_id, directive.directive_id)
        self.assertEqual(feedback.status, ExecutionStatus.FAILED)
        self.assertEqual(feedback.evidence_refs, ())
        self.assertEqual(self.runner.run('test', 1).status, 'finished')

    def test_actor_record_failure_preserves_artifacts_already_produced(self):
        actor = self.runner.exploration
        record = actor.harness.experience.record
        def fail_actor_record(experience):
            if experience.scope == 'actor':
                raise OSError('actor record failed')
            return record(experience)
        actor.harness.experience.record = fail_actor_record
        with self.assertRaisesRegex(OSError, 'actor record failed'):
            self.runner.run('test', 1)
        self.assertTrue(self.runner.busy)
        self.assertIsNone(self.runner.decision_making.harness.state.last_feedback)
        pending = actor.harness.state.pending_episode
        self.assertEqual(len(pending.feedback.artifact_refs), 1)
        self.assertEqual(actor.harness.environment.artifacts.get(pending.feedback.artifact_refs[0]).decode(),
                         pending.feedback.summary)
        executions = actor.harness.environment.sandbox.execution_count
        actor.harness.experience.record = record
        self.assertEqual(self.runner.run('test', 1).status, 'completed')
        self.assertEqual(actor.harness.environment.sandbox.execution_count, executions)
        feedback = self.runner.decision_making.harness.state.last_feedback
        self.assertEqual(feedback.status, ExecutionStatus.COMPLETED)
        self.assertEqual(feedback.evidence_refs, (pending.record.record_id,))

    def test_interrupt_is_propagated_with_cancelled_feedback(self):
        self.runner.exploration.harness.policy = Mock(act=Mock(side_effect=KeyboardInterrupt()))
        with self.assertRaises(KeyboardInterrupt):
            self.runner.run('test', 1)
        self.assertEqual(self.runner.decision_making.harness.state.last_feedback.status, ExecutionStatus.CANCELLED)
        self.assertIsNone(self.runner.decision_making.harness.state.pending_directive_id)
        self.assertIsNone(self.runner.exploration.harness.state.active_directive_id)
        self.assertEqual(self.runner.exploration.harness.environment.sandbox.active_count, 0)

    def test_failed_command_is_not_successful_task_feedback(self):
        sandbox = FailedCommandSandbox()
        self.runner.exploration.harness.environment.sandbox = sandbox
        feedback = self.runner.exploration.harness.run(Directive('failure', 'task', 'strategy'), action_space=self.runner.exploration.action_space)
        self.assertEqual(feedback.status, ExecutionStatus.FAILED)
        self.assertIn('Unsupported', feedback.error)
        self.assertEqual(len(feedback.artifact_refs), 1)
        self.assertEqual(sandbox.active_count, 0)

    def test_wait_does_not_poll_or_resubmit_and_resume_checks_identity(self):
        job = JobRef('pending', 'test')
        policy = SequencePolicy(Decision('wait', job=job), Decision('finish', reason='done'))
        decision = self.runner.decision_making
        decision.harness.policy = policy
        first = self.runner.run('test', 1)
        second = self.runner.run('test', 1)
        self.assertEqual((first.status, second.status), ('waiting', 'waiting'))
        self.assertEqual(policy.calls, 1)
        self.assertEqual(self.runner.exploration.harness.state.completed, 0)
        result = OperationResult('external result', (), True)
        with self.assertRaises(ValueError):
            decision.harness.resume(JobRef('other', 'test'), result)
        decision.harness.resume(job, result)
        with self.assertRaises(ValueError):
            decision.harness.resume(job, result)
        self.assertEqual(self.runner.run('test', 1).status, 'finished')

    def test_async_decision_operation_is_submitted_once(self):
        job = JobRef('operation', 'test')
        class AsyncTools:
            calls = 0
            def describe(self):
                return ()
            def execute(self, call):
                self.calls += 1
                return job
        decision = self.runner.decision_making
        decision.harness.tools = AsyncTools()
        decision.harness.policy = SequencePolicy(Decision('invoke', operation=OperationCall('prepare_data', {'text': 'input'})))
        self.assertEqual(self.runner.run('test', 1).waiting_job, job)
        self.assertEqual(self.runner.run('test', 1).waiting_job, job)
        self.assertEqual(decision.harness.tools.calls, 1)

    def test_decision_progress_is_bounded_and_finish_is_sticky(self):
        call = Decision('invoke', operation=OperationCall('prepare_data', {'text': 'input'}))
        decision = self.runner.decision_making
        decision.harness.policy = SequencePolicy(call, call, Decision('finish'))
        result = self.runner.run('test', 1, max_decision_steps=2)
        self.assertEqual(result.status, 'step_limit')
        self.assertEqual(result.rounds_completed, 0)
        self.assertEqual(self.runner.run('test', 1).status, 'finished')
        self.assertEqual(self.runner.run('test', 1).status, 'finished')
        self.assertEqual(decision.harness.policy.calls, 3)

    def test_shared_and_independent_model_bindings(self):
        decision, actor = self.runner.decision_making.harness.policy, self.runner.exploration.harness.policy
        self.assertIs(decision.backend, actor.backend)
        self.assertEqual(decision.model, actor.model)
        model = ModelRef('actor-only', 'other', 'v1')
        config = replace(self.config, backends={**self.config.backends, 'actor-only': BackendBinding('actor-only', 'mock')},
                         roles={**self.config.roles, 'actor': model})
        runner, _ = build_demo(config)
        self.assertIsNot(runner.decision_making.harness.policy.backend, runner.exploration.harness.policy.backend)
        self.assertEqual(runner.run('independent', 1).status, 'completed')

    def test_artifact_reference_integrity_and_lifetime(self):
        store = InMemoryArtifactStore()
        ref = store.put(b'data', kind='input')
        self.assertEqual(store.get(ref), b'data')
        with self.assertRaises(ValueError):
            store.get(replace(ref, digest='wrong'))
        with self.assertRaises(KeyError):
            InMemoryArtifactStore().get(ref)

    def test_closed_sandbox_cannot_execute_and_close_is_idempotent(self):
        sandbox = MockSandboxBackend()
        ref = sandbox.create(SandboxSpec())
        sandbox.close(ref)
        sandbox.close(ref)
        with self.assertRaises(ValueError):
            sandbox.exec(ref, ExecRequest(('echo', 'x')))
        self.assertFalse(sandbox.capabilities().snapshot)
        self.assertFalse(sandbox.capabilities().fork)

    def test_unimplemented_training_snapshot_and_remote_remain_explicit(self):
        with self.assertRaises(NotImplementedError):
            self.runner.exploration.harness.run(Directive('sft', 'task', 'strategy', mode='sft'), action_space=self.runner.exploration.action_space)
        model = self.config.roles['actor']
        for learner, request in (
            (UnimplementedActorLearner(), ActorTrainingRequest(model, (self.runner.decision_making.harness.tools.artifacts.put(b'data', kind='input'),), 'recipe')),
            (UnimplementedDecisionLearner(), DecisionTrainingRequest(model, ('fixture-experience',), 'recipe')),
        ):
            with self.assertRaises(NotImplementedError):
                learner.submit(request)
            with self.assertRaises(NotImplementedError):
                learner.result(JobRef('unavailable', 'none'))
        with self.assertRaises(NotImplementedError):
            UnimplementedCapture().capture('test')
        with self.assertRaises(ValueError):
            RemoteModelBackend(BackendBinding('remote', 'http'))

    def test_invalid_decision_payload_is_rejected(self):
        with self.assertRaises(ValueError):
            Decision('explore')
        with self.assertRaises(ValueError):
            Decision('finish', job=JobRef('extra', 'test'))


if __name__ == '__main__':
    unittest.main()
