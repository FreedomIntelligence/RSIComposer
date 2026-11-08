"""Execution components work without a harness and remain replaceable when composed."""

from copy import deepcopy
from dataclasses import replace
import unittest
from unittest.mock import Mock

from rsisys.experience.retrieval import BasicExperienceSystem
from rsisys.experience.storage import InMemoryExperienceStore
from rsisys.loops.contracts.candidate import Candidate
from rsisys.loops.contracts.directive import Directive
from rsisys.loops.contracts.lifecycle import ExecutionStatus
from rsisys.loops.contracts.model_service import ModelRef, TokenUsage
from rsisys.loops.contracts.requests import EvaluationSpec, TrainSpec
from rsisys.loops.contracts.training import TrainingResult
from rsisys.loops.exploration.action_space import ActionProposal, EchoActionSpace
from rsisys.loops.exploration.harness.engine import DefaultExplorationHarness
from rsisys.loops.exploration.harness.environment import EpisodeExecution, Transition
from rsisys.loops.exploration.harness.evaluation import EvaluationExecution, EvaluationRequest, EvaluationResult
from rsisys.loops.exploration.harness.learner import ActorTrainingRequest, LearningExecution
from rsisys.loops.exploration.harness.learner.command import TrainingCommandHandler
from rsisys.loops.exploration.harness.memory import ExplorationMemory
from rsisys.loops.exploration.harness.rl import LocalRLExecution, RLPlan
from rsisys.loops.exploration.harness.state import ExplorationState
from rsisys.runtime.jobs import JobBook, InvalidJobResult, MockJobQueue
from rsisys.snapshot.codec import decode, encode


class Policy:
    def __init__(self):
        self.model = ModelRef('fixture', 'actor', 'base')
        self.contexts = []
        self.binds = []

    def act(self, context):
        self.contexts.append(context)
        return ActionProposal('echo', {'text': 'fixture'}, True, self.model, TokenUsage(2, 1))

    def update_model(self, model):
        self.model = model
        self.binds.append(model)


class Backend:
    def __init__(self, factory):
        self.factory = factory
        self.queue = MockJobQueue('execution', pending_polls=0)
        self.poll = Mock(side_effect=self.queue.poll)
        self.cancel = Mock(side_effect=self.queue.cancel)

    def submit(self, request, *, idempotency_key):
        return self.queue.submit(request, lambda: self.factory(request), idempotency_key=idempotency_key)


def trained(request):
    model = replace(request.base, revision=request.base.revision + '-trained')
    return TrainingResult(request.base, model,
                          Candidate(model.revision, request.base.revision, {}, ('actor',)), simulated=True)


def evaluated(request):
    return EvaluationResult(request.candidate.candidate_id, request.protocol_ref, {'quality': 0.75}, (), True)


class ExplorationExecutionTests(unittest.TestCase):
    def setUp(self):
        # Build each component directly. No DefaultExplorationHarness is involved.
        self.state = ExplorationState(instructions='standalone')
        self.policy, self.memory = Policy(), ExplorationMemory()
        self.store = InMemoryExperienceStore()
        self.experience = BasicExperienceSystem(self.store)
        self.environment = Mock(reset=Mock(return_value='start'),
                                step=Mock(return_value=Transition('done', True, True)))
        self.backend, self.evaluator = Backend(trained), Backend(evaluated)
        self.episodes = EpisodeExecution(self.policy, self.environment, self.memory,
                                         self.experience, self.state)
        self.learning = LearningExecution(self.policy, self.memory, self.experience,
                                          self.state, JobBook(self.backend))
        self.evaluations = EvaluationExecution(self.experience, self.state, JobBook(self.evaluator))
        self.rl = LocalRLExecution(self.episodes, self.learning, self.state)
        self.actions = EchoActionSpace()
        self.directive = Directive('episode', 'task', 'exercise the environment')

    def lose_first_record_response(self):
        record, attempts = self.experience.record, []

        def flaky(value):
            attempts.append(deepcopy(value))
            record(value)
            if len(attempts) == 1:
                raise OSError('lost committed record response')

        self.experience.record = flaky
        return attempts

    def finish_rl(self, execution):
        for _ in range(20):
            state = execution.step(self.actions)
            if state.terminal:
                return state
        self.fail('Local fixture did not terminate')

    def harness(self):
        return DefaultExplorationHarness(self.policy, self.environment, self.experience,
            self.backend, self.evaluator, memory=self.memory,
            command_handlers={'train': TrainingCommandHandler()})

    def test_episode_retries_committed_evidence_without_repeating_environment(self):
        attempts = self.lose_first_record_response()
        with self.assertRaises(OSError):
            self.episodes.run(self.directive, self.actions)
        self.assertEqual(self.state.completed, 0)
        with self.assertRaisesRegex(RuntimeError, 'pending'):
            self.episodes.run(replace(self.directive, directive_id='other'), self.actions)
        feedback = self.episodes.run(self.directive, self.actions)
        self.assertEqual(feedback.status, ExecutionStatus.COMPLETED)
        self.assertEqual(feedback.execution_metrics['measured_input_tokens'], 2)
        self.assertEqual(self.policy.contexts[0].instructions, 'standalone')
        self.assertEqual(self.state.completed, 1)
        self.assertEqual(attempts[0], attempts[1])
        self.assertEqual(len(self.store.records()), 1)
        self.environment.reset.assert_called_once()
        self.environment.step.assert_called_once()
        self.environment.close.assert_called_once()

    def test_episode_preserves_execution_failure_and_cleanup_evidence(self):
        self.environment.step.side_effect = ValueError('action failed')
        self.environment.close.side_effect = OSError('cleanup failed')
        with self.assertRaisesRegex(ValueError, 'action failed') as caught:
            self.episodes.run(self.directive, self.actions)
        self.assertIn('cleanup failed', caught.exception.__notes__[0])
        self.assertIsNone(self.state.active_directive_id)
        self.assertEqual(self.state.last_feedback.status, ExecutionStatus.FAILED)
        self.assertEqual(self.state.completed, 0)
        self.assertIn('cleanup failed', self.store.records()[0].details['error'])

    def test_learning_retries_evidence_before_binding_and_rejects_stale_candidates(self):
        request = ActorTrainingRequest(self.policy.model, (), 'recipe')
        job = self.learning.submit(request, idempotency_key='train')
        self.assertEqual(self.learning.submit(request, idempotency_key='train'), job)
        with self.assertRaises(RuntimeError):
            self.learning.submit(request, idempotency_key='overlap')
        attempts = self.lose_first_record_response()
        with self.assertRaises(OSError):
            self.learning.apply(job)
        self.assertEqual(self.policy.binds, [])
        model = self.learning.apply(job)
        self.assertEqual(self.learning.apply(job), model)
        self.assertEqual(self.policy.binds, [model])
        self.backend.poll.assert_called_once()
        self.assertEqual(attempts[0], attempts[1])
        candidate = self.learning.candidate_result(model.revision)
        candidate.metrics['mutated'] = 1
        self.assertEqual(self.learning.candidate_result(model.revision).metrics, {})
        self.policy.update_model(replace(model, revision='newer'))
        with self.assertRaisesRegex(ValueError, 'stale'):
            self.learning.apply_candidate(model.revision)

    def test_evaluation_retries_evidence_and_validates_terminal_identity(self):
        candidate = trained(ActorTrainingRequest(self.policy.model, (), 'recipe')).candidate
        request = EvaluationRequest(candidate, 'protocol')
        job = self.evaluations.submit(request, idempotency_key='evaluate')
        attempts = self.lose_first_record_response()
        with self.assertRaises(OSError):
            self.evaluations.poll(job)
        self.assertEqual(self.evaluations.poll(job).result.metrics, {'quality': 0.75})
        self.evaluator.poll.assert_called_once()
        self.assertEqual(attempts[0], attempts[1])
        self.evaluator.factory = lambda request: replace(evaluated(request), protocol_ref='wrong')
        bad = self.evaluations.submit(request, idempotency_key='bad')
        with self.assertRaises(InvalidJobResult):
            self.evaluations.poll(bad)
        self.assertEqual(len(self.store.records()), 1)

    def test_local_rl_sequences_components_and_preserves_model_lineage(self):
        self.rl.start(RLPlan(self.directive, 'recipe', 2))
        result = self.finish_rl(self.rl)
        self.assertEqual(result.phase, 'completed')
        self.assertEqual(result.updates_completed, 2)
        self.assertEqual(result.results[1].base, result.results[0].model)
        self.assertEqual(self.policy.model, result.results[1].model)
        self.assertEqual(len(self.memory.history(kind='trajectory')), 2)
        self.assertEqual(len(self.store.records()), 4)

    def test_local_rl_resolves_lost_submission_before_cancelling_without_binding(self):
        self.rl.start(RLPlan(self.directive, 'recipe', 1))
        self.rl.step(self.actions)
        submit, jobs = self.backend.submit, []

        def flaky(request, *, idempotency_key):
            jobs.append(submit(request, idempotency_key=idempotency_key))
            if len(jobs) == 1:
                raise OSError('lost submit response')
            return jobs[-1]

        self.backend.submit = flaky
        with self.assertRaises(OSError):
            self.rl.step(self.actions)
        self.assertFalse(self.rl.cancel().terminal)
        result = self.finish_rl(self.rl)
        self.assertEqual(result.phase, 'cancelled')
        self.assertEqual(jobs[0], jobs[1])
        self.assertEqual(self.policy.binds, [])
        self.assertEqual(len(self.memory.history(kind='trajectory')), 1)
        self.backend.cancel.assert_called_once_with(jobs[0])

    def test_restored_state_is_used_by_episode_learning_evaluation_and_rl(self):
        harness = self.harness()
        old = harness.state
        before = deepcopy(old)
        restored = decode(encode(ExplorationState(instructions='restored instructions', completed=5)))
        harness.state = restored
        harness.run(self.directive, self.actions)
        job = harness.submit_learning(ActorTrainingRequest(harness.policy.model, (), 'recipe'),
                                      idempotency_key='restored')
        result = harness.poll_learning(job).result
        harness.apply_learning(job)
        evaluation = harness.submit_evaluation(EvaluationRequest(result.candidate, 'protocol'),
                                               idempotency_key='restored-eval')
        harness.poll_evaluation(evaluation)
        harness.start_rl(RLPlan(self.directive, 'recipe', 1))
        self.assertEqual(self.finish_rl(harness.local_rl).phase, 'completed')
        self.assertEqual(old, before)
        self.assertEqual(restored.completed, 7)
        self.assertEqual(len(restored.recorded_training), 2)
        self.assertIn(evaluation, restored.recorded_evaluation)
        self.assertEqual(self.policy.contexts[-1].instructions, 'restored instructions')

    def test_replacing_execution_components_changes_commands_and_local_rl(self):
        harness = self.harness()
        environment = Mock(reset=Mock(return_value='replacement'),
                           step=Mock(return_value=Transition('replacement finished', True, True)))
        harness.episodes = EpisodeExecution(harness.policy, environment, harness.memory,
                                            harness.experience, harness.state)
        backend = Backend(lambda request: replace(trained(request), metrics={'replacement': 1}))
        harness.learning = LearningExecution(harness.policy, harness.memory, harness.experience,
                                              harness.state, JobBook(backend))
        evaluator = Backend(lambda request: replace(evaluated(request), metrics={'replacement': 2}))
        harness.evaluations = EvaluationExecution(harness.experience, harness.state, JobBook(evaluator))
        # Commands have no dependency on the old private harness execution hooks.
        harness._run_episode = harness._submit_learning = harness._apply_learning = Mock(
            side_effect=AssertionError('Private harness callback must not run'))
        harness.accept(self.directive)
        self.assertEqual(harness.advance(self.directive.directive_id, self.actions).feedback.summary,
                         'replacement finished')
        harness.acknowledge(self.directive.directive_id)
        train = Directive('train', 'task', '', mode='train', payload=TrainSpec('recipe', 'custom', apply_result=True))
        harness.accept(train)
        harness.advance('train', self.actions)
        result = harness.advance('train', self.actions).feedback
        self.assertEqual(result.training_metrics, {'replacement': 1})
        harness.acknowledge('train')
        evaluate = Directive('eval', 'task', '', mode='evaluate',
                             payload=EvaluationSpec(result.candidates[0], 'protocol'))
        harness.accept(evaluate)
        harness.advance('eval', self.actions)
        self.assertEqual(harness.advance('eval', self.actions).feedback.quality_metrics, {'replacement': 2})
        harness.acknowledge('eval')
        harness.start_rl(RLPlan(self.directive, 'recipe', 1))
        result = self.finish_rl(harness.local_rl)
        self.assertEqual(result.feedback[0].summary, 'replacement finished')
        self.assertEqual(result.results[0].metrics, {'replacement': 1})
        self.environment.reset.assert_not_called()
        self.backend.poll.assert_not_called()
        self.evaluator.poll.assert_not_called()


if __name__ == '__main__':
    unittest.main()
