"""Accepted candidates remain available across training and evaluation rounds."""
from tests import ROOT as REPO_ROOT

from copy import deepcopy
from dataclasses import asdict, replace
import json
from pathlib import Path
import unittest
from unittest.mock import Mock, patch

from rsisys.config import DemoConfig, build_demo
from rsisys.experience.retrieval import BasicExperienceSystem
from rsisys.experience.storage import InMemoryExperienceStore
from rsisys.loops.contracts.artifacts import ArtifactRef
from rsisys.loops.contracts.candidate import Candidate
from rsisys.loops.contracts.directive import Directive
from rsisys.loops.contracts.feedback import Feedback
from rsisys.loops.contracts.lifecycle import ExecutionStatus
from rsisys.loops.contracts.model_service import GenerationResponse, ModelRef
from rsisys.loops.contracts.requests import EvaluationSpec
from rsisys.loops.decision_making.action_space import DemoDecisionActionSpace, OperationResult
from rsisys.loops.decision_making.action_space.directives import (
    DEFAULT_DIRECTIVE_MODES, DirectiveDefinitions, apply_candidate_directive,
    builtin_directives, training_directive,
)
from rsisys.loops.decision_making.harness.decision import Decision
from rsisys.loops.decision_making.harness.engine import DefaultDecisionHarness
from rsisys.loops.decision_making.harness.policy import ModelDecisionPolicy
from rsisys.loops.decision_making.harness.state import DecisionContext, DecisionState


MODEL = ModelRef('test', 'decision', 'v1')


def candidate(ident='first'):
    return Candidate(ident, 'base', {
        'model': ArtifactRef(f'{ident}-model', 'model', f'memory://{ident}'),
    }, ('actor',))


class CandidateContextTests(unittest.TestCase):
    def setUp(self):
        self.store = InMemoryExperienceStore()
        self.policy = Mock()
        self.harness = DefaultDecisionHarness(
            self.policy, Mock(), BasicExperienceSystem(self.store),
            experience_limit=1, memory_limit=1,
        )
        self.space = DemoDecisionActionSpace()

    def dispatch(self, *candidates):
        directive = Directive(f'round-{self.harness.state.issued}', 'task', 'test')
        self.policy.decide.return_value = Decision('explore', directive=directive)
        self.assertEqual(self.harness.step('task', self.space).directive, directive)
        feedback = Feedback(directive.directive_id, ExecutionStatus.COMPLETED,
                            'done', True, candidates=candidates)
        return directive, feedback

    def test_context_keeps_old_positional_arguments_and_empty_default(self):
        feedback = Feedback('old', ExecutionStatus.COMPLETED, 'done', True)
        context = DecisionContext('task', (), (), (), (), feedback, (), ('evaluate',))
        self.assertEqual(context.candidates, ())
        self.assertIs(context.last_feedback, feedback)
        self.assertEqual(context.directive_modes, ('evaluate',))
        first, second = DecisionState(), DecisionState()
        first.candidates['first'] = candidate()
        self.assertEqual(second.candidates, {})

    def test_accumulation_deduplication_and_empty_feedback_survive_context_limits(self):
        first, second = candidate(), candidate('second')
        for values in ((first, first), (second, deepcopy(first)), ()):
            directive, feedback = self.dispatch(*values)
            self.harness.receive_feedback(directive, feedback)
            self.harness.receive_feedback(deepcopy(directive), deepcopy(feedback))
        self.assertEqual(self.harness.state.completed, 3)
        self.assertEqual(len(self.harness.memory.history(kind='feedback')), 3)
        self.assertEqual(len(self.store.records()), 3)
        self.assertEqual(self.harness.state.candidates, {'first': first, 'second': second})
        self.policy.decide.return_value = Decision('finish')
        self.harness.step('task', self.space)
        context = self.policy.decide.call_args.args[0]
        self.assertEqual(context.candidates, (first, second))
        self.assertEqual(context.last_feedback.candidates, ())
        self.assertEqual(len(context.experience), 1)
        self.assertGreater(len(context.history), 1)  # Complete small history stays below the token trigger.

    def test_candidate_copies_isolate_feedback_and_policy_mutations(self):
        original = candidate()
        expected = deepcopy(original)
        directive, feedback = self.dispatch(original)
        self.harness.receive_feedback(directive, feedback)
        original.changes.clear()
        self.assertEqual(self.harness.state.candidates['first'], expected)
        self.assertEqual(self.harness.state.last_feedback.candidates, (expected,))
        self.policy.decide.return_value = Decision('finish')
        self.harness.step('task', self.space)
        context = self.policy.decide.call_args.args[0]
        context.candidates[0].changes.clear()
        self.assertEqual(context.last_feedback.candidates, (expected,))
        context.last_feedback.candidates[0].changes.clear()
        self.assertEqual(self.harness.state.candidates['first'], expected)
        self.assertEqual(self.harness.state.last_feedback.candidates, (expected,))

    def test_conflicts_reject_whole_batch_before_writes_or_state_changes(self):
        first = candidate()
        for existing in (False, True):
            for conflicting in (replace(first, base_version='other'),
                                replace(first, changes={}),
                                replace(first, affected_roles=('decision_maker',))):
                with self.subTest(existing=existing, conflicting=conflicting):
                    self.setUp()
                    if existing:
                        self.harness.receive_feedback(*self.dispatch(first))
                    directive, feedback = self.dispatch(candidate('new'), first, conflicting)
                    self.harness.state.operation_results.append(OperationResult('prepared', (), True))
                    before = deepcopy(self.harness.state)
                    with patch.object(self.harness.memory, 'append') as append, \
                         patch.object(self.harness.experience, 'record') as record:
                        with self.assertRaisesRegex(ValueError, 'Candidate ID.*different content'):
                            self.harness.receive_feedback(directive, feedback)
                        append.assert_not_called()
                        record.assert_not_called()
                    self.assertEqual(self.harness.state, before)
                    # Rejection leaves the pending delivery usable with valid content.
                    self.harness.receive_feedback(directive, replace(feedback, candidates=(first,)))
                    self.assertEqual(self.harness.state.candidates, {'first': first})
                    self.assertEqual(self.harness.state.completed, int(existing) + 1)

    def test_feedback_write_failures_retry_without_partial_candidate_publication(self):
        for target in ('memory', 'experience'):
            for after_commit in (False, True):
                with self.subTest(target=target, after_commit=after_commit):
                    self.setUp()
                    first, second = candidate(), candidate('second')
                    self.harness.receive_feedback(*self.dispatch(first))
                    directive, feedback = self.dispatch(second)
                    self.harness.state.operation_results.append(OperationResult('prepared', (), True))
                    before = deepcopy(self.harness.state)
                    owner = self.harness.memory if target == 'memory' else self.harness.experience
                    method = 'append' if target == 'memory' else 'record'
                    commit = getattr(owner, method)

                    def fail(value):
                        if after_commit:
                            commit(value)
                        raise OSError('feedback write failed')

                    with patch.object(owner, method, side_effect=fail):
                        with self.assertRaisesRegex(OSError, 'feedback write failed'):
                            self.harness.receive_feedback(directive, feedback)
                    self.assertEqual(self.harness.state, before)
                    self.harness.receive_feedback(directive, feedback)
                    self.harness.receive_feedback(deepcopy(directive), deepcopy(feedback))
                    self.assertEqual(self.harness.state.candidates, {'first': first, 'second': second})
                    self.assertEqual(self.harness.state.completed, 2)
                    self.assertIsNone(self.harness.pending())
                    self.assertEqual(self.harness.state.operation_results, [])
                    self.assertEqual(len(self.harness.memory.history(kind='feedback')), 2)
                    self.assertEqual(len(self.store.records()), 2)


class CandidatePolicyTests(unittest.TestCase):
    def parse(self, context, ident):
        payload = {'kind': 'explore', 'mode': 'evaluate', 'strategy': 'measure',
                   'arguments': {'candidate_id': ident, 'protocol_ref': 'protocol'}}
        backend = Mock(generate=Mock(return_value=GenerationResponse(json.dumps(payload), MODEL, True)))
        definitions = DirectiveDefinitions()
        definitions.parse = Mock(wraps=definitions.parse)
        policy = ModelDecisionPolicy(backend, MODEL, directives=definitions)
        decision = policy.decide(context)
        return decision, json.loads(backend.generate.call_args.args[0].prompt), definitions.parse

    def test_evaluate_uses_context_and_legacy_feedback_union_without_duplicates(self):
        first, second = candidate(), candidate('second')
        feedback = Feedback('old', ExecutionStatus.COMPLETED, 'trained', True,
                            candidates=(deepcopy(first), second))
        context = DecisionContext('task', (), (), (), last_feedback=feedback,
                                  directive_modes=('evaluate',), candidates=(first,))
        for expected in (first, second):
            with self.subTest(candidate=expected.candidate_id):
                decision, prompt, parse = self.parse(context, expected.candidate_id)
                self.assertEqual(decision.directive.payload, EvaluationSpec(expected, 'protocol'))
                self.assertEqual(parse.call_args.kwargs['candidates'], (first, second))
                self.assertEqual(prompt['candidates'], json.loads(json.dumps([asdict(first)])))
                self.assertNotIn('latest feedback', prompt['directives'][0]['description'])

    def test_context_candidates_work_without_feedback_and_legacy_context_still_works(self):
        first = candidate()
        feedback = Feedback('old', ExecutionStatus.COMPLETED, 'trained', True, candidates=(first,))
        for context in (
            DecisionContext('task', (), (), (), directive_modes=('evaluate',), candidates=(first,)),
            DecisionContext('task', (), (), (), (), feedback, (), ('evaluate',)),
        ):
            with self.subTest(context=context):
                self.assertEqual(self.parse(context, 'first')[0].directive.payload.candidate, first)
                with self.assertRaisesRegex(ValueError, 'available candidate'):
                    self.parse(context, 'unknown')


class ApplyCandidateDirectiveTests(unittest.TestCase):
    def test_opt_in_definition_does_not_change_default_modes(self):
        self.assertNotIn('apply_candidate', DEFAULT_DIRECTIVE_MODES)
        self.assertNotIn('apply_candidate', DirectiveDefinitions().modes())
        definition = apply_candidate_directive()
        self.assertEqual(definition.spec.mode, 'apply_candidate')
        self.assertEqual(definition.spec.input_schema['required'], ['candidate_id'])
        self.assertEqual(set(definition.spec.input_schema['properties']), {'candidate_id'})
        self.assertFalse(definition.spec.input_schema['additionalProperties'])

    def test_parser_rejects_missing_unknown_invalid_and_extra_arguments(self):
        catalog = DirectiveDefinitions((apply_candidate_directive(),))
        for arguments in ({}, {'candidate_id': 'unknown'}, {'candidate_id': ''},
                          {'candidate_id': '  '}, {'candidate_id': None},
                          {'candidate_id': 1}, {'candidate_id': []}, {'candidate_id': {}},
                          {'candidate_id': 'first', 'job_id': 'job'},
                          {'candidate_id': 'first', 'protocol_ref': 'protocol'}):
            with self.subTest(arguments=arguments), self.assertRaises(ValueError):
                catalog.parse('apply_candidate', arguments, candidates=(candidate(),))
        with self.assertRaisesRegex(ValueError, 'available candidate'):
            catalog.parse('apply_candidate', {'candidate_id': 'first'})

    def test_model_policy_parses_remembered_and_legacy_candidates_without_jobs(self):
        from rsisys.loops.contracts.requests import ApplyCandidateSpec

        builtin = {definition.spec.mode: definition for definition in builtin_directives()}
        catalog = DirectiveDefinitions((training_directive(), builtin['evaluate'],
                                        apply_candidate_directive(), builtin['execute']))
        modes = ('train', 'evaluate', 'apply_candidate', 'execute')
        self.assertEqual(catalog.modes(), modes)
        first = candidate()
        evaluated = Feedback('evaluation', ExecutionStatus.COMPLETED, 'measured', False)
        legacy = replace(evaluated, candidates=(first,))
        for feedback, remembered in ((evaluated, (first,)), (legacy, ())):
            for nested in (False, True):
                with self.subTest(remembered=remembered, nested=nested):
                    payload = {'kind': 'explore', 'mode': 'apply_candidate', 'strategy': 'select'}
                    fields = {'candidate_id': first.candidate_id}
                    payload.update({'arguments': fields} if nested else fields)
                    backend = Mock(generate=Mock(return_value=GenerationResponse(
                        json.dumps(payload), MODEL, False)))
                    policy = ModelDecisionPolicy(backend, MODEL, directives=catalog)
                    context = DecisionContext('task', (), (), (), last_feedback=feedback,
                                              directive_modes=modes, candidates=remembered)
                    decision = policy.decide(context)
                    self.assertEqual(decision.directive.payload, ApplyCandidateSpec('first'))
                    self.assertEqual(asdict(decision.directive.payload), {'candidate_id': 'first'})
                    prompt = json.loads(backend.generate.call_args.args[0].prompt)
                    self.assertEqual([spec['mode'] for spec in prompt['directives']], list(modes))
                    with self.assertRaisesRegex(ValueError, 'Disallowed'):
                        policy.decide(replace(context, directive_modes=('execute',)))


class CandidateRoundTripTests(unittest.TestCase):
    def test_two_training_rounds_then_repeated_evaluation_of_earlier_candidate(self):
        config = REPO_ROOT / 'configs/demo.json'
        runner, store = build_demo(DemoConfig.load(config), with_learning=True)
        contexts = []

        def generate(request):
            context = json.loads(request.prompt)
            contexts.append(context)
            step = len(contexts)
            if step in (1, 3):
                payload = {'kind': 'invoke', 'operation': 'prepare_data', 'arguments': {'text': 'data'}}
            elif step in (2, 4):
                payload = {'kind': 'explore', 'mode': 'sft', 'strategy': 'train', 'recipe_ref': 'recipe'}
            elif step in (5, 6, 7):
                selected = context['candidates'][1 if step == 7 else 0]['candidate_id']
                payload = {'kind': 'explore', 'mode': 'evaluate', 'strategy': 'measure',
                           'candidate_id': selected, 'protocol_ref': 'protocol'}
            else:
                payload = {'kind': 'finish'}
            return GenerationResponse(json.dumps(payload), request.model, True)

        harness = runner.decision_making.harness
        harness.policy.backend = Mock(generate=Mock(side_effect=generate))
        base = runner.exploration.harness.policy.model
        for _ in range(80):
            if runner.run('train and evaluate', 1).status == 'finished':
                break
        else:
            self.fail('Training/evaluation sequence did not finish')
        self.assertEqual(harness.state.completed, 5)
        self.assertEqual(len(harness.state.candidates), 2)
        self.assertEqual(len(contexts), 8)
        first, second = harness.state.candidates.values()
        evaluations = [r for r in store.records() if r.kind == 'evaluation']
        self.assertEqual([r.details['request']['candidate']['candidate_id'] for r in evaluations],
                         [first.candidate_id, first.candidate_id, second.candidate_id])
        for context in contexts[5:]:
            self.assertEqual(context['last_feedback']['candidates'], [])
            self.assertEqual([c['candidate_id'] for c in context['candidates']],
                             [first.candidate_id, second.candidate_id])
        self.assertEqual(runner.exploration.harness.policy.model, base)


if __name__ == '__main__':
    unittest.main()
