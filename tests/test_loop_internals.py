from dataclasses import replace
from pathlib import Path
import unittest
from unittest.mock import Mock

from rsisys.config import DemoConfig, build_demo
from rsisys.loops.contracts.jobs import JobRef, JobStatus, JobOutcome
from rsisys.loops.contracts.directive import Directive
from rsisys.loops.contracts.lifecycle import ExecutionStatus
from rsisys.loops.decision_making.decision import Decision
from rsisys.loops.decision_making.decision_space import (
    OperationCall, OperationSpec, OperationResult, ModularDecisionSpace,
)
from rsisys.loops.decision_making.learner import DecisionTrainingRequest
from rsisys.loops.exploration.learner import ActorTrainingRequest
from rsisys.loops.exploration.evaluation import EvaluationRequest
from rsisys.loops.exploration.rl import RLPlan
from rsisys.loops.exploration.action_space import Action, ActionSpec, ActionProposal, ModularActionSpace
from rsisys.runtime.jobs import MockJobQueue

CONFIG = Path(__file__).resolve().parents[1] / 'configs/demo.json'


class LoopInternalsTests(unittest.TestCase):
    def setUp(self):
        self.runner, self.store = build_demo(DemoConfig.load(CONFIG), with_learning=True)
        self.decision = self.runner.decision_making
        self.actor = self.runner.exploration
        self.artifacts = self.decision.space.artifacts

    def finish(self, poll, job):
        for _ in range(8):
            outcome = poll(job)
            if outcome.terminal:
                return outcome
        self.fail('Fixture job failed to terminate')

    def decision_request(self):
        self.decision.step('prepare training experience')
        refs = tuple(r.record_id for r in self.store.records() if r.scope == 'decision_maker')
        return DecisionTrainingRequest(self.decision.policy.model, refs, 'mock-recipe')

    def actor_request(self):
        ref = self.artifacts.put(b'fixture dataset', kind='training_data')
        return ActorTrainingRequest(self.actor.policy.model, (ref,), 'mock-sft')

    def start_rl(self, updates=2):
        return self.actor.start_rl(RLPlan(Directive('rl', 'task', 'explore'), 'mock-rl', updates))

    def advance_rl(self):
        for _ in range(30):
            state = self.actor.step_rl()
            if state.terminal:
                return state
        self.fail('Fixture RL cycle failed to terminate')

    def test_local_decision_run_stops_before_actor_execution(self):
        result = self.decision.run('task')
        self.assertEqual(result.status, 'dispatch')
        self.assertEqual([s.kind for s in result.steps], ['progressed', 'dispatch'])
        self.assertEqual(self.actor.state.completed, 0)
        self.assertEqual(len(self.store.records()), 1)

    def test_local_decision_run_is_bounded(self):
        self.decision.policy = Mock(decide=Mock(return_value=Decision(
            'invoke', operation=OperationCall('prepare_data', {'text': 'data'}))))
        result = self.decision.run('task', max_steps=2)
        self.assertEqual(result.status, 'step_limit')
        self.assertEqual(len(result.steps), 2)
        self.assertEqual(self.decision.policy.decide.call_count, 2)

    def test_failed_operation_is_visible_to_next_decision(self):
        self.decision.policy = Mock(decide=Mock(side_effect=[
            Decision('invoke', operation=OperationCall('missing', {})), Decision('finish'),
        ]))
        with self.assertRaises(ValueError):
            self.decision.step('task')
        self.assertEqual(self.decision.state.operation_results[-1].status, ExecutionStatus.FAILED)
        self.assertEqual(self.decision.run('task').status, 'finished')
        context = self.decision.policy.decide.call_args.args[0]
        self.assertIn('Unsupported', context.operation_results[-1].error)

    def test_decision_learning_waits_deduplicates_records_and_updates_only_itself(self):
        request = self.decision_request()
        actor_model = self.actor.policy.model
        job = self.decision.submit_learning(request, idempotency_key='decision-training')
        self.assertEqual(self.decision.submit_learning(request, idempotency_key='decision-training'), job)
        self.assertEqual(self.decision.poll_learning(job).status, JobStatus.RUNNING)
        self.assertFalse(any(r.kind == 'training_run' for r in self.store.records()))
        outcome = self.finish(self.decision.poll_learning, job)
        self.assertTrue(outcome.simulated)
        self.assertEqual(outcome.result.metrics, {})
        self.assertEqual(self.decision.policy.model, request.base)
        self.decision.poll_learning(job)
        self.assertEqual(len([r for r in self.store.records() if r.kind == 'training_run']), 1)
        self.decision.apply_learning(job)
        self.decision.apply_learning(job)
        self.assertEqual(self.decision.policy.model, outcome.result.model)
        self.assertEqual(self.actor.policy.model, actor_model)

    def test_learning_rejects_unknown_experience_and_changed_idempotency_request(self):
        with self.assertRaises(ValueError):
            self.decision.submit_learning(DecisionTrainingRequest(
                self.decision.policy.model, ('nonexistent',), 'recipe'), idempotency_key='bad')
        request = self.decision_request()
        self.decision.submit_learning(request, idempotency_key='same')
        with self.assertRaises(ValueError):
            self.decision.submit_learning(replace(request, recipe_ref='different'), idempotency_key='same')

    def test_actor_sft_and_explicit_local_binding(self):
        request = self.actor_request()
        decision_model = self.decision.policy.model
        job = self.actor.submit_learning(request, idempotency_key='sft')
        with self.assertRaises(RuntimeError):
            self.actor.run(Directive('busy', 'task', 'strategy'))
        with self.assertRaises(RuntimeError):
            self.actor.learner.result(job)
        outcome = self.finish(self.actor.poll_learning, job)
        self.assertEqual(outcome.status, JobStatus.SUCCEEDED)
        self.assertEqual(self.actor.policy.model, request.base)
        self.actor.apply_learning(job)
        self.assertEqual(self.actor.policy.model, outcome.result.model)
        self.assertEqual(self.decision.policy.model, decision_model)
        self.assertEqual(self.actor.run(Directive('after-sft', 'task', 'strategy')).status,
                         ExecutionStatus.COMPLETED)

    def test_cancelled_training_is_recorded_and_never_applied(self):
        job = self.actor.submit_learning(self.actor_request(), idempotency_key='cancel')
        outcome = self.actor.cancel_learning(job)
        self.assertEqual(outcome.status, JobStatus.CANCELLED)
        self.assertEqual(self.actor.poll_learning(job), outcome)
        with self.assertRaises(RuntimeError):
            self.actor.apply_learning(job)
        self.assertEqual(len([r for r in self.store.records() if r.kind == 'training_run']), 1)

    def test_failed_training_does_not_produce_a_model(self):
        request = self.actor_request()
        invalid = replace(request.data_refs[0], digest='invalid')
        job = self.actor.submit_learning(replace(request, data_refs=(invalid,)), idempotency_key='failure')
        outcome = self.finish(self.actor.poll_learning, job)
        self.assertEqual(outcome.status, JobStatus.FAILED)
        self.assertIsNone(outcome.result)
        self.assertEqual(self.actor.policy.model, request.base)
        self.assertIn('digest', outcome.error)

    def test_stale_training_cannot_overwrite_a_newer_policy(self):
        request = self.actor_request()
        job = self.actor.submit_learning(request, idempotency_key='stale')
        self.finish(self.actor.poll_learning, job)
        newer = replace(request.base, revision='another-version')
        self.actor.policy.update_model(newer)
        with self.assertRaisesRegex(ValueError, 'stale'):
            self.actor.apply_learning(job)
        self.assertEqual(self.actor.policy.model, newer)

    def test_rl_updates_model_between_rollouts_and_records_lineage(self):
        base = self.actor.policy.model
        decision_base = self.decision.policy.model
        self.start_rl(2)
        state = self.advance_rl()
        self.assertEqual(state.phase, 'completed')
        self.assertEqual(state.updates_completed, 2)
        self.assertEqual(state.results[0].base, base)
        self.assertEqual(state.results[1].base, state.results[0].model)
        self.assertEqual(self.actor.policy.model, state.results[1].model)
        self.assertEqual(self.decision.policy.model, decision_base)
        trajectories = self.actor.memory.history(kind='trajectory')
        self.assertEqual(len(trajectories), 2)
        self.assertEqual(trajectories[0].details['model']['revision'], base.revision)
        self.assertEqual(trajectories[1].details['model']['revision'], state.results[0].model.revision)
        self.assertNotEqual(trajectories[0].task_id, trajectories[1].task_id)
        records = len(self.store.records())
        self.assertEqual(self.actor.step_rl(), state)
        self.assertEqual(len(self.store.records()), records)

    def test_rl_owns_execution_and_model_updates_until_finished(self):
        self.start_rl()
        with self.assertRaises(RuntimeError):
            self.actor.run(Directive('overlap', 'task', 'strategy'))
        with self.assertRaises(RuntimeError):
            self.actor.submit_learning(self.actor_request(), idempotency_key='overlap')
        with self.assertRaises(RuntimeError):
            self.start_rl()

    def test_lost_submission_response_retries_same_job_without_repeating_rollout(self):
        self.start_rl(1)
        self.actor.step_rl()
        backend = self.actor.training.backend
        submit = backend.submit
        calls = []
        def lose_first_response(request, *, idempotency_key):
            job = submit(request, idempotency_key=idempotency_key)
            calls.append(job)
            if len(calls) == 1:
                raise ConnectionError('lost response')
            return job
        backend.submit = lose_first_response
        with self.assertRaises(ConnectionError):
            self.actor.step_rl()
        self.assertEqual(self.actor.state.rl.phase, 'submitting')
        self.assertEqual(self.advance_rl().phase, 'completed')
        self.assertEqual(calls[0], calls[1])
        self.assertEqual(len(self.actor.memory.history(kind='trajectory')), 1)

    def test_rl_failure_stops_without_applying_weights(self):
        base = self.actor.policy.model
        self.start_rl()
        self.actor.step_rl()
        backend = self.actor.training.backend
        backend.poll = lambda job: JobOutcome(job, JobStatus.FAILED, error='training failed', simulated=True)
        self.actor.step_rl()
        state = self.advance_rl()
        self.assertEqual(state.phase, 'failed')
        self.assertEqual(state.updates_completed, 0)
        self.assertEqual(self.actor.policy.model, base)

    def test_rl_cancellation_prevents_next_update(self):
        base = self.actor.policy.model
        self.start_rl()
        self.actor.step_rl()
        self.actor.step_rl()
        state = self.actor.cancel_rl()
        self.assertEqual(state.phase, 'cancelled')
        self.assertEqual(self.actor.step_rl().phase, 'cancelled')
        self.assertEqual(self.actor.policy.model, base)

    def test_rl_cancel_before_submission_does_not_create_job(self):
        self.start_rl()
        self.actor.step_rl()
        self.assertEqual(self.actor.cancel_rl().phase, 'cancelled')
        self.assertFalse(self.actor.training.busy)

    def test_evaluation_records_evidence_without_promoting_candidate(self):
        request = self.actor_request()
        job = self.actor.submit_learning(request, idempotency_key='sft')
        result = self.finish(self.actor.poll_learning, job).result
        evaluation = EvaluationRequest(result.candidate, 'mock-eval-protocol')
        eval_job = self.actor.submit_evaluation(evaluation, idempotency_key='eval')
        outcome = self.finish(self.actor.poll_evaluation, eval_job)
        self.assertEqual(outcome.result.candidate_id, result.candidate.candidate_id)
        self.assertEqual(outcome.result.metrics, {})
        self.assertTrue(outcome.result.evidence_refs)
        self.assertTrue(outcome.result.simulated)
        self.actor.poll_evaluation(eval_job)
        self.assertEqual(len([r for r in self.store.records() if r.kind == 'evaluation']), 1)
        self.assertEqual(self.actor.policy.model, request.base)

    def test_unknown_job_and_mismatched_outcome_are_rejected(self):
        with self.assertRaises(ValueError):
            self.actor.poll_learning(JobRef('foreign', 'foreign'))
        request = self.actor_request()
        backend = self.actor.training.backend
        backend.poll = lambda job: JobOutcome(JobRef('wrong', job.backend), JobStatus.RUNNING)
        job = self.actor.submit_learning(request, idempotency_key='mismatch')
        with self.assertRaises(ValueError):
            self.actor.poll_learning(job)

    def test_poll_transport_error_leaves_job_recoverable(self):
        backend = self.actor.training.backend
        poll = backend.poll
        calls = []
        def flaky(job):
            calls.append(job)
            if len(calls) == 1:
                raise ConnectionError('unreachable')
            return poll(job)
        backend.poll = flaky
        job = self.actor.submit_learning(self.actor_request(), idempotency_key='flaky')
        with self.assertRaises(ConnectionError):
            self.actor.poll_learning(job)
        self.assertTrue(self.actor.training.busy)
        self.assertEqual(self.finish(self.actor.poll_learning, job).status, JobStatus.SUCCEEDED)

    def test_terminal_record_failure_can_retry_without_rerunning_training(self):
        job = self.actor.submit_learning(self.actor_request(), idempotency_key='storage')
        record = self.actor.experience.record
        self.actor.experience.record = Mock(side_effect=OSError('store unavailable'))
        self.actor.poll_learning(job)
        with self.assertRaises(OSError):
            self.actor.poll_learning(job)
        self.actor.experience.record = record
        outcome = self.actor.poll_learning(job)
        self.assertEqual(outcome.status, JobStatus.SUCCEEDED)
        self.assertEqual(len([r for r in self.store.records() if r.kind == 'training_run']), 1)

    def test_delayed_cancellation_does_not_apply_a_racing_success(self):
        base = self.actor.policy.model
        backend = self.actor.training.backend
        backend.jobs.pending_polls = 2
        backend.cancel = Mock()  # The external backend acknowledges the request, but finishes first.
        self.start_rl(1)
        self.actor.step_rl()
        self.actor.step_rl()
        self.assertEqual(self.actor.cancel_rl().phase, 'training')
        self.assertEqual(self.advance_rl().phase, 'cancelled')
        self.assertEqual(self.actor.policy.model, base)
        self.assertEqual(backend.cancel.call_count, 1)
        self.assertEqual(self.actor.state.rl.updates_completed, 0)

    def test_cancel_after_lost_submission_resolves_original_job(self):
        base = self.actor.policy.model
        backend = self.actor.training.backend
        submit = backend.submit
        calls = []
        def lose_first_response(request, *, idempotency_key):
            job = submit(request, idempotency_key=idempotency_key)
            calls.append(job)
            if len(calls) == 1:
                raise ConnectionError('lost response')
            return job
        backend.submit = lose_first_response
        self.start_rl(1)
        self.actor.step_rl()
        with self.assertRaises(ConnectionError):
            self.actor.step_rl()
        self.assertEqual(self.actor.cancel_rl().phase, 'submitting')
        self.assertEqual(self.advance_rl().phase, 'cancelled')
        self.assertEqual(calls[0], calls[1])
        self.assertEqual(self.actor.policy.model, base)

    def test_wrong_role_training_result_is_not_recorded_or_applied(self):
        backend = self.actor.training.backend
        poll = backend.poll
        def wrong_role(job):
            outcome = poll(job)
            if outcome.status == JobStatus.SUCCEEDED:
                result = outcome.result
                return replace(outcome, result=replace(result, candidate=replace(
                    result.candidate, affected_roles=('decision_maker',))))
            return outcome
        backend.poll = wrong_role
        request = self.actor_request()
        job = self.actor.submit_learning(request, idempotency_key='wrong-role')
        with self.assertRaisesRegex(ValueError, 'role'):
            self.finish(self.actor.poll_learning, job)
        self.assertEqual(self.actor.policy.model, request.base)
        self.assertFalse(any(r.kind == 'training_run' for r in self.store.records()))

    def test_evaluation_protocol_mismatch_is_rejected(self):
        job = self.actor.submit_learning(self.actor_request(), idempotency_key='train')
        candidate = self.finish(self.actor.poll_learning, job).result.candidate
        backend = self.actor.evaluation.backend
        poll = backend.poll
        def wrong_protocol(job):
            outcome = poll(job)
            if outcome.status == JobStatus.SUCCEEDED:
                return replace(outcome, result=replace(outcome.result, protocol_ref='wrong'))
            return outcome
        backend.poll = wrong_protocol
        job = self.actor.submit_evaluation(EvaluationRequest(candidate, 'expected'), idempotency_key='eval')
        with self.assertRaisesRegex(ValueError, 'protocol'):
            self.finish(self.actor.poll_evaluation, job)
        self.assertFalse(any(r.kind == 'evaluation' for r in self.store.records()))

    def test_modular_spaces_dispatch_custom_capabilities(self):
        class Operation:
            spec = OperationSpec('literature', 'Search an injected corpus', {})
            def execute(self, arguments):
                if not isinstance(arguments.get('query'), str):
                    raise ValueError('query is required')
                return OperationResult(arguments['query'], (), True)
        space = ModularDecisionSpace((Operation(),))
        self.assertEqual(space.execute(OperationCall('literature', {'query': 'topic'})).summary, 'topic')
        with self.assertRaises(ValueError):
            space.execute(OperationCall('missing', {}))
        with self.assertRaises(ValueError):
            ModularDecisionSpace((Operation(), Operation()))
        class Definition:
            spec = ActionSpec('move', 'Move in an injected environment', {})
            def validate(self, arguments):
                if type(arguments.get('distance')) is not int:
                    raise ValueError('integer distance required')
                return Action('move', arguments=arguments)
        actions = ModularActionSpace((Definition(),))
        proposal = ActionProposal('move', {'distance': 2}, True)
        self.assertEqual(actions.validate(proposal, ('move',)).arguments, {'distance': 2})
        with self.assertRaises(ValueError):
            actions.validate(proposal, ())

    def test_context_limit_applies_to_decision_and_actor(self):
        self.runner.run('test', 3)
        self.decision.experience_limit = 1
        self.decision.memory_limit = 1
        self.actor.memory_limit = 1
        decide = self.decision.policy.decide
        act = self.actor.policy.act
        self.decision.policy.decide = Mock(wraps=decide)
        self.actor.policy.act = Mock(wraps=act)
        self.runner.run('test', 1)
        self.assertEqual(len(self.decision.policy.decide.call_args.args[0].experience), 1)
        self.assertEqual(len(self.decision.policy.decide.call_args.args[0].history), 1)
        self.assertEqual(self.actor.policy.act.call_args.args[0].history, ())
        step = self.actor.environment.step
        self.actor.environment.step = lambda action: replace(step(action), terminated=False)
        self.actor.run(Directive('limited-history', 'task', 'strategy', max_steps=3))
        context = self.actor.policy.act.call_args.args[0]
        self.assertEqual(len(context.history), 1)
        self.assertEqual(context.history[0].task_id, 'limited-history')

    def test_model_response_revision_must_match_request(self):
        backend = self.actor.policy.backend
        generate = backend.generate
        backend.generate = lambda request: replace(generate(request), model=replace(request.model, revision='wrong'))
        with self.assertRaisesRegex(ValueError, 'identity'):
            self.actor.run(Directive('wrong-model', 'task', 'strategy'))

    def test_mock_job_results_cannot_be_read_before_completion(self):
        queue = MockJobQueue('fixture', pending_polls=1)
        callback = Mock(return_value='result')
        job = queue.submit('request', callback, idempotency_key='key')
        with self.assertRaises(RuntimeError):
            queue.result(job)
        self.assertEqual(queue.poll(job).status, JobStatus.RUNNING)
        self.assertEqual(queue.poll(job).result, 'result')
        self.assertEqual(queue.poll(job).result, 'result')
        self.assertEqual(callback.call_count, 1)


if __name__ == '__main__':
    unittest.main()
