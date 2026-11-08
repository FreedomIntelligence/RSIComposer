"""System directives bypass Actor actions; prompt receipts survive uncertain writes."""

from copy import deepcopy
from dataclasses import asdict, replace
from hashlib import sha256
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import Mock, patch

from rsisys.experience.retrieval import BasicExperienceSystem
from rsisys.experience.sqlite import SQLiteExperienceStore
from rsisys.experience.storage import InMemoryExperienceStore
from rsisys.loops.contracts.candidate import Candidate
from rsisys.loops.contracts.directive import Directive
from rsisys.loops.contracts.feedback import Feedback
from rsisys.loops.contracts.lifecycle import ExecutionStatus
from rsisys.loops.contracts.model_service import GenerationResponse, ModelRef
from rsisys.loops.contracts.requests import EvaluationSpec, PromptSpec, RLSpec, SFTSpec
from rsisys.loops.decision_making.action_space import DemoDecisionActionSpace
from rsisys.loops.decision_making.harness.engine import DefaultDecisionHarness
from rsisys.loops.decision_making.harness.policy import ModelDecisionPolicy
from rsisys.loops.decision_making.harness.state import DecisionContext
from rsisys.loops.decision_making.loop import DecisionMakingLoop
from rsisys.loops.exploration.action_space import ActionProposal, EchoActionSpace
from rsisys.loops.exploration.harness.engine import DefaultExplorationHarness
from rsisys.loops.exploration.harness.environment import Transition
from rsisys.loops.exploration.action_space.local import CommandActionSpace
from rsisys.loops.exploration.harness.evaluation.mock import MockEvaluator
from rsisys.loops.exploration.harness.learner.backends.mock import MockActorLearner
from rsisys.loops.exploration.harness.policy import ActorContext, ModelActorPolicy
from rsisys.loops.exploration.harness.state import ExplorationState
from rsisys.loops.exploration.loop import ExplorationLoop
from rsisys.runtime.artifacts import InMemoryArtifactStore


MODEL = ModelRef('test', 'actor', 'v1')


class CountingState(ExplorationState):
    """Detect reapplication even when writing the same prompt value."""

    def __setattr__(self, name, value):
        if name == 'instructions':
            self.instruction_writes = getattr(self, 'instruction_writes', 0) + 1
        super().__setattr__(name, value)


class SystemDirectiveTests(unittest.TestCase):
    def make_loop(self, *, policy=None, learner=None, evaluator=None, store=None):
        store = store if store is not None else InMemoryExperienceStore()
        policy = policy if policy is not None else Mock(
            act=Mock(side_effect=AssertionError('System command called Actor.act')))
        environment = Mock(reset=Mock(return_value='raw task observation'),
                           step=Mock(return_value=Transition('done', True, False)))
        harness = DefaultExplorationHarness(policy, environment, BasicExperienceSystem(store),
                                            learner=learner, evaluator=evaluator)
        return ExplorationLoop(harness, EchoActionSpace()), store

    def prompt(self, ident='prompt', instructions='Use concise Chinese.'):
        return Directive(ident, 'private objective', 'private strategy',
                         mode='configure_prompt', payload=PromptSpec(instructions))

    def finish(self, loop, directive):
        progress = loop.accept(directive)
        for _ in range(20):
            if progress.terminal:
                return progress.feedback
            progress = loop.advance(directive.directive_id)
        self.fail('Command did not terminate')

    def test_prompt_updates_only_state_and_emits_system_evidence(self):
        learner, evaluator = Mock(), Mock()
        loop, store = self.make_loop(learner=learner, evaluator=evaluator)
        directive = self.prompt(instructions='请用中文回答。')
        loop.accept(directive)
        self.assertEqual(loop.harness.state.instructions, '')
        feedback = loop.advance(directive.directive_id).feedback
        self.assertEqual(feedback.status, ExecutionStatus.COMPLETED)
        self.assertEqual(feedback.directive_id, directive.directive_id)
        self.assertEqual(loop.harness.state.instructions, directive.payload.instructions)
        record, = store.records()
        self.assertEqual(feedback.evidence_refs, (record.record_id,))
        self.assertEqual(record.directive_id, directive.directive_id)
        self.assertEqual(record.details['before']['sha256'], sha256(b'').hexdigest())
        self.assertEqual(record.details['after'], {
            'sha256': sha256(directive.payload.instructions.encode()).hexdigest(),
            'characters': len(directive.payload.instructions), 'configured': True,
        })
        self.assertFalse(record.source_refs)
        serialized = json.dumps(asdict(record), ensure_ascii=False)
        for raw in (directive.objective, directive.strategy, directive.payload.instructions):
            self.assertNotIn(raw, serialized)
        self.assertEqual(loop.harness.memory.history(), ())
        loop.harness.policy.act.assert_not_called()
        self.assertEqual(loop.harness.environment.mock_calls, [])
        self.assertEqual(learner.mock_calls, [])
        self.assertEqual(evaluator.mock_calls, [])

    def test_prompt_reaches_context_and_model_system_across_episodes_then_clears(self):
        backend = Mock(generate=Mock(return_value=GenerationResponse(
            '{"name":"echo","arguments":{"text":"done"}}', MODEL, False)))
        policy = ModelActorPolicy(backend, MODEL)
        loop, _ = self.make_loop(policy=policy)
        prompt = self.prompt()
        self.finish(loop, prompt)
        loop.acknowledge(prompt.directive_id)
        for ident in ('first', 'second'):
            loop.harness.run(Directive(ident, 'task', 'strategy'), loop.action_space)
            request = backend.generate.call_args.args[0]
            self.assertEqual(json.loads(request.prompt)['instructions'], prompt.payload.instructions)
            self.assertTrue(request.system.endswith('\n\n' + prompt.payload.instructions))
        cleared = self.prompt('clear', '')
        self.finish(loop, cleared)
        loop.acknowledge(cleared.directive_id)
        self.assertEqual(loop.harness.state.instructions, '')
        loop.harness.run(Directive('third', 'task', 'strategy'), loop.action_space)
        request = backend.generate.call_args.args[0]
        self.assertNotIn(prompt.payload.instructions, request.system)
        self.assertEqual(json.loads(request.prompt)['instructions'], '')
        self.assertEqual(ActorContext('', prompt, (), ()).instructions, '')

    def test_prompt_record_retry_preserves_payload_and_applies_only_once(self):
        for after_commit in (False, True):
            with self.subTest(after_commit=after_commit):
                loop, store = self.make_loop()
                state = CountingState(instructions='before')
                loop.harness.state = state
                baseline = state.instruction_writes
                directive = self.prompt()
                loop.accept(directive)
                original_record = loop.harness.experience.record
                captured = []

                def flaky_record(record):
                    captured.append(deepcopy(record))
                    if len(captured) <= 2:
                        if after_commit:
                            original_record(record)
                        raise OSError('uncertain system feedback write')
                    original_record(record)

                loop.harness.experience.record = flaky_record
                with self.assertRaises(OSError):
                    loop.advance(directive.directive_id)
                self.assertEqual(state.instructions, directive.payload.instructions)
                self.assertFalse(loop.inspect(directive.directive_id).terminal)
                self.assertEqual(loop.accept(deepcopy(directive)).phase, 'recording')
                with self.assertRaisesRegex(ValueError, 'different content'):
                    loop.accept(replace(directive, payload=PromptSpec('other')))
                with self.assertRaises(RuntimeError):
                    loop.accept(self.prompt('next'))
                with self.assertRaises(RuntimeError):
                    loop.acknowledge(directive.directive_id)
                with patch('rsisys.loops.exploration.harness.commands.uuid4', side_effect=AssertionError):
                    with self.assertRaises(OSError):
                        loop.advance(directive.directive_id)
                    # Cancellation after the state change cannot undo the command.
                    loop.cancel(directive.directive_id)
                    feedback = loop.advance(directive.directive_id).feedback
                self.assertEqual(state.instruction_writes, baseline + 1)
                self.assertTrue(all(record == captured[0] for record in captured))
                self.assertEqual(store.records(), (captured[0],))
                self.assertEqual(feedback.status, ExecutionStatus.COMPLETED)
                self.assertEqual(feedback.evidence_refs, (captured[0].record_id,))
                self.assertEqual(loop.advance(directive.directive_id).feedback, feedback)
                self.assertEqual(len(captured), 3)
                loop.harness.policy.act.assert_not_called()
                self.assertEqual(loop.harness.environment.mock_calls, [])

    def test_prompt_sqlite_commit_response_loss_deduplicates(self):
        with tempfile.TemporaryDirectory() as directory:
            store = SQLiteExperienceStore(Path(directory) / 'experience.sqlite3')
            self.addCleanup(store.close)
            loop, _ = self.make_loop(store=store)
            directive = self.prompt()
            loop.accept(directive)
            original_record = loop.harness.experience.record

            def lost_response(record):
                original_record(record)
                raise OSError('commit response lost')

            loop.harness.experience.record = lost_response
            with self.assertRaises(OSError):
                loop.advance(directive.directive_id)
            committed, = store.records()
            loop.harness.experience.record = original_record
            feedback = loop.advance(directive.directive_id).feedback
            self.assertEqual(store.records(), (committed,))
            self.assertEqual(feedback.evidence_refs, (committed.record_id,))

    def test_cancel_after_accept_prevents_prompt_change_and_is_retryable(self):
        loop, store = self.make_loop()
        loop.harness.state.instructions = 'keep this'
        directive = self.prompt()
        loop.accept(directive)
        loop.cancel(directive.directive_id)
        original_record = loop.harness.experience.record
        loop.harness.experience.record = Mock(side_effect=OSError('unavailable'))
        with self.assertRaises(OSError):
            loop.advance(directive.directive_id)
        self.assertEqual(loop.harness.state.instructions, 'keep this')
        loop.harness.experience.record = original_record
        feedback = loop.advance(directive.directive_id).feedback
        self.assertEqual(feedback.status, ExecutionStatus.CANCELLED)
        record, = store.records()
        self.assertEqual(record.details['before'], record.details['after'])
        self.assertEqual(feedback.evidence_refs, (record.record_id,))
        loop.harness.policy.act.assert_not_called()
        self.assertEqual(loop.harness.environment.mock_calls, [])

    def test_replay_preserves_instruction_id_without_overwriting_later_configuration(self):
        loop, store = self.make_loop()
        first, second = self.prompt('first', 'one'), self.prompt('second', 'two')
        feedback = self.finish(loop, first)
        loop.acknowledge(first.directive_id)
        self.finish(loop, second)
        self.assertEqual(loop.accept(first).feedback, feedback)
        self.assertEqual(loop.advance(first.directive_id).feedback, feedback)
        self.assertEqual(loop.harness.state.instructions, 'two')
        self.assertEqual(len(store.records()), 2)
        with self.assertRaises(ValueError):
            loop.accept(replace(first, payload=PromptSpec('changed')))

    def test_prompt_rejects_non_string_and_wrong_payload(self):
        for value in (None, 1, False, [], {}):
            with self.subTest(value=value), self.assertRaises(ValueError):
                PromptSpec(value)
        for payload in (None, SFTSpec('recipe')):
            loop, store = self.make_loop()
            with self.assertRaisesRegex(ValueError, 'PromptSpec'):
                loop.accept(replace(self.prompt(), payload=payload))
            self.assertEqual(loop.harness.state.instructions, '')
            self.assertEqual(store.records(), ())
            loop.harness.policy.act.assert_not_called()

    def test_sft_and_evaluation_never_call_actor_or_environment(self):
        artifacts = InMemoryArtifactStore()
        data = artifacts.put(b'data', kind='training_data')
        learner = MockActorLearner(artifacts, pending_polls=0)
        evaluator = MockEvaluator(artifacts, pending_polls=0)
        loop, store = self.make_loop(learner=learner, evaluator=evaluator)
        sft = Directive('sft', 'train', 'supervised', mode='sft', input_refs=(data,),
                        payload=SFTSpec('recipe', base=MODEL))
        training = self.finish(loop, sft)
        self.assertEqual(training.status, ExecutionStatus.COMPLETED)
        self.assertEqual(training.evidence_refs, (store.records()[0].record_id,))
        self.assertEqual(store.records()[0].kind, 'training_run')
        loop.acknowledge(sft.directive_id)
        evaluation = Directive('evaluation', 'assess', 'protocol', mode='evaluate',
                               payload=EvaluationSpec(training.candidates[0], 'protocol'))
        assessed = self.finish(loop, evaluation)
        self.assertEqual(assessed.status, ExecutionStatus.COMPLETED)
        self.assertEqual(assessed.evidence_refs, (store.records()[1].record_id,))
        self.assertEqual(store.records()[1].kind, 'evaluation')
        loop.harness.policy.act.assert_not_called()
        self.assertEqual(loop.harness.environment.mock_calls, [])
        self.assertEqual(loop.harness.memory.history(), ())

    def test_rl_uses_actor_only_for_rollout_not_for_learner_phases(self):
        backend = Mock(generate=Mock(return_value=GenerationResponse(
            '{"name":"echo","arguments":{"text":"rollout"}}', MODEL, False)))
        policy = ModelActorPolicy(backend, MODEL)
        learner = MockActorLearner(InMemoryArtifactStore(), pending_polls=0)
        loop, _ = self.make_loop(policy=policy, learner=learner)
        loop.harness.state.instructions = 'persistent rollout instruction'
        directive = Directive('rl', 'learn', 'rollout then train', mode='rl', payload=RLSpec('recipe'))
        loop.accept(directive)
        loop.advance(directive.directive_id)  # Plan creation does not invoke Actor.
        backend.generate.assert_not_called()
        loop.advance(directive.directive_id)  # Rollout is an ordinary execute episode.
        backend.generate.assert_called_once()
        request = backend.generate.call_args.args[0]
        self.assertEqual(json.loads(request.prompt)['directive']['mode'], 'execute')
        self.assertIn(loop.harness.state.instructions, request.system)
        with patch.object(policy, 'act', side_effect=AssertionError('Actor called during training')):
            loop.advance(directive.directive_id)  # Submit learner request.
            feedback = loop.advance(directive.directive_id).feedback
        self.assertEqual(feedback.status, ExecutionStatus.COMPLETED)
        self.assertNotEqual(policy.model, MODEL)
        self.assertEqual(loop.harness.environment.step.call_args.args[0].name, 'echo')

    def test_actor_action_spaces_reject_system_commands_even_if_directive_allows_names(self):
        for space in (EchoActionSpace(), CommandActionSpace()):
            for name in ('configure_prompt', 'prompt', 'train', 'sft', 'rl', 'evaluate'):
                with self.subTest(space=type(space).__name__, name=name):
                    self.assertNotIn(name, {spec.name for spec in space.describe()})
                    with self.assertRaises(ValueError):
                        space.validate(ActionProposal(name, {'instructions': 'override'}, False), (name,))


class ModelDecisionDirectiveTests(unittest.TestCase):
    def parse(self, payload, modes=('execute', 'configure_prompt'), feedback=None):
        backend = Mock(generate=Mock(return_value=GenerationResponse(json.dumps(payload), MODEL, False)))
        policy = ModelDecisionPolicy(backend, MODEL)
        context = DecisionContext('task', (), (), (), last_feedback=feedback, directive_modes=modes)
        return policy.decide(context)

    def test_prompt_json_is_dispatched_with_action_space_modes_and_stable_id(self):
        backend = Mock(generate=Mock(return_value=GenerationResponse(json.dumps({
            'kind': 'explore', 'mode': 'configure_prompt', 'strategy': 'configure',
            'instructions': 'Keep responses short.',
        }), MODEL, False)))
        experience = BasicExperienceSystem(InMemoryExperienceStore())
        tools = Mock()
        loop = DecisionMakingLoop(DefaultDecisionHarness(ModelDecisionPolicy(backend, MODEL), tools, experience),
                                  DemoDecisionActionSpace(directive_modes=('configure_prompt',)))
        result = loop.step('task')
        self.assertEqual(result.kind, 'dispatch')
        self.assertEqual(result.directive.payload, PromptSpec('Keep responses short.'))
        self.assertEqual(loop.pending(), result.directive)
        self.assertEqual(loop.harness.state.issued_directive_ids, {result.directive.directive_id})
        context = json.loads(backend.generate.call_args.args[0].prompt)
        self.assertEqual(context['directive_modes'], ['configure_prompt'])
        self.assertEqual(tools.mock_calls, [])
        with self.assertRaises(RuntimeError):
            loop.step('task')
        backend.generate.assert_called_once()

        # A failed feedback handoff retries the same dispatched instruction and
        # system evidence; it must never ask either policy to repeat the command.
        actor = Mock(act=Mock(side_effect=AssertionError('Unexpected Actor action')))
        exploration = ExplorationLoop(DefaultExplorationHarness(actor, Mock(), experience), EchoActionSpace())
        state = CountingState()
        exploration.harness.state = state
        baseline = state.instruction_writes
        exploration.accept(result.directive)
        feedback = exploration.advance(result.directive.directive_id).feedback
        original_record = experience.record
        captured = []

        def lost_feedback_response(record):
            captured.append(deepcopy(record))
            original_record(record)
            if len(captured) == 1:
                raise OSError('feedback commit response lost')

        experience.record = lost_feedback_response
        with self.assertRaises(OSError):
            loop.receive_feedback(result.directive, feedback)
        self.assertEqual(loop.pending(), result.directive)
        retried = exploration.accept(result.directive).feedback
        self.assertEqual(retried, feedback)
        loop.receive_feedback(result.directive, retried)
        exploration.acknowledge(result.directive.directive_id)
        self.assertIsNone(loop.pending())
        self.assertEqual(state.instruction_writes, baseline + 1)
        self.assertEqual(loop.harness.state.completed, 1)
        self.assertEqual(captured[0], captured[1])
        self.assertEqual(captured[0].source_refs, feedback.evidence_refs)
        self.assertEqual(len(experience.retrieve()), 2)
        backend.generate.assert_called_once()
        actor.act.assert_not_called()

    def test_all_modes_are_limited_to_context_including_default_execute(self):
        for mode in ('execute', 'sft', 'rl', 'evaluate', 'configure_prompt', 'unknown'):
            with self.subTest(mode=mode), self.assertRaisesRegex(ValueError, 'Disallowed'):
                self.parse({'kind': 'explore', 'mode': mode, 'strategy': 'test'}, modes=())
        with self.assertRaisesRegex(ValueError, 'Disallowed'):
            self.parse({'kind': 'explore', 'strategy': 'test'}, modes=('configure_prompt',))
        with self.assertRaisesRegex(ValueError, 'Unsupported'):
            self.parse({'kind': 'explore', 'mode': 'unknown', 'strategy': 'test'}, modes=('unknown',))

    def test_prompt_json_requires_string_and_accepts_empty_clear(self):
        payload = {'kind': 'explore', 'mode': 'configure_prompt', 'strategy': 'test'}
        for value in (None, 1, False, [], {}):
            with self.subTest(value=value), self.assertRaises(ValueError):
                self.parse(dict(payload, instructions=value))
        with self.assertRaises((KeyError, ValueError)):
            self.parse(payload)
        self.assertEqual(self.parse(dict(payload, instructions='')).directive.payload, PromptSpec(''))

    def test_existing_modes_remain_parseable_when_explicitly_enabled(self):
        candidate = Candidate('candidate', 'v1', {}, ('actor',))
        feedback = Feedback('earlier', ExecutionStatus.COMPLETED, 'trained', False, candidates=(candidate,))
        for mode, fields, expected in (
            ('execute', {}, None),
            ('sft', {'recipe_ref': 'recipe'}, SFTSpec('recipe')),
            ('rl', {'recipe_ref': 'recipe', 'updates': 2}, RLSpec('recipe', 2)),
            ('evaluate', {'candidate_id': 'candidate', 'protocol_ref': 'protocol'},
             EvaluationSpec(candidate, 'protocol')),
        ):
            with self.subTest(mode=mode):
                decision = self.parse(dict(kind='explore', strategy='test', mode=mode, **fields),
                                      modes=(mode,), feedback=feedback)
                self.assertEqual(decision.directive.payload, expected)


if __name__ == '__main__':
    unittest.main()
