"""Research queries are available during a running experiment, with bounded execution."""
import json
import unittest
from unittest.mock import Mock

from rsisys.experience.retrieval import BasicExperienceSystem
from rsisys.experience.storage import InMemoryExperienceStore
from rsisys.loops.contracts.coordination import ExplorationProgress, ReviewDecision
from rsisys.loops.contracts.directive import Directive
from rsisys.loops.contracts.feedback import Feedback
from rsisys.loops.contracts.lifecycle import ExecutionStatus
from rsisys.loops.contracts.model_service import GenerationResponse, ModelRef, TokenUsage
from rsisys.loops.decision_making.action_space import (
    ModularDecisionActionSpace, OperationCall, OperationResult, OperationSpec,
)
from rsisys.loops.decision_making.harness.decision import Decision
from rsisys.loops.decision_making.harness.engine import DefaultDecisionHarness
from rsisys.loops.decision_making.harness.policy import ModelDecisionPolicy
from rsisys.loops.decision_making.harness.tools import BoundDecisionTools
from rsisys.loops.decision_making.loop import DecisionMakingLoop


class Definition:
    def __init__(self, name, read_only):
        self.spec = OperationSpec(name, name, {'type': 'object'}, read_only=read_only)

    def validate(self, arguments):
        if arguments:
            raise ValueError('No arguments expected')
        return OperationCall(self.spec.name, {})


class ResearcherReviewTests(unittest.TestCase):
    def setUp(self):
        self.backend = Mock()
        self.model = ModelRef('test', 'researcher', 'v1')
        self.policy = ModelDecisionPolicy(self.backend, self.model)
        self.query = Mock(return_value=OperationResult('loss is falling', (), False))
        self.write = Mock(return_value=OperationResult('changed', (), False))
        self.store = InMemoryExperienceStore()
        self.harness = DefaultDecisionHarness(self.policy,
            BoundDecisionTools({'inspect': self.query, 'modify': self.write}),
            BasicExperienceSystem(self.store), review_tool_limit=2)
        self.loop = DecisionMakingLoop(self.harness, ModularDecisionActionSpace(
            (Definition('inspect', True), Definition('modify', False))))
        self.directive = Directive('active', 'task', 'experiment', simulated=False)
        self.harness.state.pending_directive_id = 'active'
        self.harness.state.pending_directive = self.directive
        self.harness.state.feedback_record_id = 'receipt-active'
        self.progress = ExplorationProgress('active', 'running', self.directive)

    def responses(self, *payloads):
        self.backend.generate.side_effect = [GenerationResponse(json.dumps(p), self.model, False,
            TokenUsage(3, 2)) for p in payloads]

    def test_model_queries_then_decides_with_evidence_and_usage(self):
        self.responses({'kind': 'invoke', 'operation': 'inspect', 'arguments': {}},
                       {'action': 'continue', 'next_review_seconds': 5, 'reason': 'loss is falling'})
        result = self.loop.review('task', self.progress)
        self.assertEqual(result.next_review_seconds, 5)
        contexts = [json.loads(c.args[0].prompt) for c in self.backend.generate.call_args_list]
        self.assertEqual([o['name'] for o in contexts[0]['operations']], ['inspect'])
        self.assertEqual(contexts[0]['directive_modes'], [])
        self.assertEqual(contexts[1]['operation_results'][-1]['summary'], 'loss is falling')
        self.assertEqual(self.harness.state.measured_input_tokens, 6)
        self.assertEqual(self.harness.state.measured_output_tokens, 4)
        self.query.assert_called_once_with({})
        self.write.assert_not_called()
        self.assertEqual(self.harness.pending(), self.directive)
        entries = self.harness.memory.history(task_id='active')
        query = next(e for e in entries if e.details.get('event') == 'review_query')
        final = next(e for e in entries if e.details.get('event') == 'review')
        self.assertEqual(final.details['query_refs'], [f'history:decision_maker:{query.record_id}'])
        self.assertFalse(any(r.details.get('event') == 'review_query' for r in self.store.records()))

    def test_last_call_requires_final_decision_when_query_budget_exhausted(self):
        ask = {'kind': 'invoke', 'operation': 'inspect', 'arguments': {}}
        self.responses(ask, ask, {'action': 'continue'})
        self.loop.review('task', self.progress)
        context = json.loads(self.backend.generate.call_args.args[0].prompt)
        self.assertEqual(context['operations'], [])
        self.assertEqual(self.query.call_count, 2)

    def test_custom_policy_cannot_bypass_read_only_or_budget(self):
        for name, limit in [('modify', 2), ('inspect', 0)]:
            with self.subTest(name=name, limit=limit):
                self.setUp()
                self.harness.review_tool_limit = limit
                self.policy.review = Mock(return_value=Decision('invoke', operation=OperationCall(name, {})))
                with self.assertRaisesRegex(ValueError, 'available'):
                    self.loop.review('task', self.progress)
        self.write.assert_not_called()
        self.query.assert_not_called()

    def test_query_failure_is_evidence_and_does_not_fail_the_experiment(self):
        self.query.side_effect = OSError('log service unavailable')
        self.responses({'kind': 'invoke', 'operation': 'inspect', 'arguments': {}},
                       {'action': 'continue', 'next_review_seconds': 10})
        self.assertEqual(self.loop.review('task', self.progress).next_review_seconds, 10)
        context = json.loads(self.backend.generate.call_args.args[0].prompt)
        self.assertEqual(context['operation_results'][-1]['status'], 'failed')
        self.assertIn('unavailable', context['operation_results'][-1]['error'])
        self.assertEqual(self.harness.pending(), self.directive)

    def test_final_record_retry_does_not_repeat_queries_or_model_calls(self):
        self.responses({'kind': 'invoke', 'operation': 'inspect', 'arguments': {}}, {'action': 'cancel'})
        original = self.harness.experience.record
        self.harness.experience.record = Mock(side_effect=OSError('temporary recording failure'))
        with self.assertRaises(OSError):
            self.loop.review('task', self.progress)
        self.harness.experience.record = original
        self.assertEqual(self.loop.review('task', self.progress).action, 'cancel')
        self.assertEqual(self.backend.generate.call_count, 2)
        self.query.assert_called_once()
        self.assertEqual(self.harness.state.measured_input_tokens, 6)

    def test_invalid_query_arguments_can_be_corrected_within_budget(self):
        self.responses({'kind': 'invoke', 'operation': 'inspect', 'arguments': {'bad': 1}},
                       {'kind': 'invoke', 'operation': 'inspect', 'arguments': {}},
                       {'action': 'continue'})
        self.loop.review('task', self.progress)
        context = json.loads(self.backend.generate.call_args_list[1].args[0].prompt)
        self.assertEqual(context['operation_results'][-1]['status'], 'failed')
        self.query.assert_called_once_with({})

    def test_simulated_query_marks_final_review_simulated(self):
        self.query.return_value = OperationResult('mock data', (), True)
        self.responses({'kind': 'invoke', 'operation': 'inspect', 'arguments': {}}, {'action': 'continue'})
        self.assertTrue(self.loop.review('task', self.progress).simulated)

    def test_legacy_external_harness_review_still_works(self):
        class ExternalHarness:
            def review(self, objective, progress):
                return ReviewDecision('continue', 7)
        external = DecisionMakingLoop(ExternalHarness(), self.loop.action_space)
        self.assertEqual(external.review('task', self.progress).next_review_seconds, 7)

    def test_legacy_subclass_override_is_not_bypassed(self):
        class CustomHarness(DefaultDecisionHarness):
            def review(self, objective, progress):
                return ReviewDecision('cancel', reason='custom implementation')
        custom = CustomHarness(self.policy, self.harness.tools, self.harness.experience)
        self.assertEqual(DecisionMakingLoop(custom, self.loop.action_space).review(
            'task', self.progress).action, 'cancel')
        self.backend.generate.assert_not_called()

    def test_query_record_retry_preserves_result_id_budget_and_usage(self):
        for after_commit in (False, True):
            with self.subTest(after_commit=after_commit):
                self.setUp()
                self.harness.review_tool_limit = 1
                self.responses({'kind': 'invoke', 'operation': 'inspect', 'arguments': {}},
                               {'action': 'continue'})
                append = self.harness.memory.append
                def broken(entry):
                    if after_commit:
                        append(entry)
                    raise OSError('record response lost')
                self.harness.memory.append = broken
                with self.assertRaises(OSError):
                    self.loop.review('task', self.progress)
                self.harness.memory.append = append
                self.loop.review('task', self.progress)
                self.query.assert_called_once()
                self.assertEqual(self.backend.generate.call_count, 2)
                last_context = json.loads(self.backend.generate.call_args.args[0].prompt)
                self.assertEqual(last_context['operations'], [])
                self.assertEqual(len(last_context['operation_results']), 1)
                queries = [row for row in self.harness.memory.history()
                           if row.details.get('event') == 'review_query']
                self.assertEqual(len(queries), 1)
                self.assertIsNone(self.harness.state.review_session)
                self.assertEqual(self.harness.state.measured_input_tokens, 6)

    def test_model_cannot_invoke_write_or_forge_directive_during_review(self):
        for payload in ({'kind': 'invoke', 'operation': 'modify', 'arguments': {}},
                        {'kind': 'explore', 'mode': 'sft'}):
            with self.subTest(payload=payload):
                self.responses(payload)
                with self.assertRaises(ValueError):
                    self.loop.review('task', self.progress)
        self.write.assert_not_called()
        self.assertEqual(self.harness.state.measured_input_tokens, 6)

    def test_completion_can_overtake_query_record_failure(self):
        self.responses({'kind': 'invoke', 'operation': 'inspect', 'arguments': {}})
        append = self.harness.memory.append
        self.harness.memory.append = Mock(side_effect=OSError('temporary'))
        with self.assertRaises(OSError):
            self.loop.review('task', self.progress)
        self.harness.memory.append = append
        self.harness.receive_feedback(self.directive, Feedback('active', ExecutionStatus.COMPLETED, 'done', False))
        self.assertIsNone(self.harness.state.review_session)
        self.assertIsNone(self.harness.pending())
        self.query.assert_called_once()
        self.backend.generate.assert_called_once()
        self.assertEqual(len([row for row in self.harness.memory.history()
                              if row.details.get('event') == 'review_query']), 1)


if __name__ == '__main__':
    unittest.main()
