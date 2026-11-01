"""Public decision rationales are optional metadata, separate from action arguments."""

from dataclasses import replace
import json
import unittest
from unittest.mock import Mock

from rsisys.loops.contracts.jobs import JobRef
from rsisys.loops.contracts.model_service import GenerationResponse, ModelRef, TokenUsage
from rsisys.loops.contracts.requests import PromptSpec
from rsisys.loops.decision_making.action_space import OperationSpec
from rsisys.loops.decision_making.harness.policy import ModelDecisionPolicy
from rsisys.loops.decision_making.harness.state import DecisionContext


class DecisionRationaleTests(unittest.TestCase):
    def setUp(self):
        self.model = ModelRef('test', 'decision', 'v1')
        self.usage = TokenUsage(12, 7)
        self.backend = Mock()
        self.policy = ModelDecisionPolicy(self.backend, self.model)
        self.context = DecisionContext('improve answers', (), (
            OperationSpec('inspect', 'Read current metrics', {'type': 'object'}, read_only=True),
        ), (), directive_modes=('execute', 'configure_prompt'))

    def respond(self, payload):
        self.backend.generate.return_value = GenerationResponse(
            json.dumps(payload), self.model, True, self.usage)

    def decisions(self):
        return (
            {'kind': 'invoke', 'operation': 'inspect', 'arguments': {'metric': 'loss'}},
            {'kind': 'explore', 'mode': 'configure_prompt', 'strategy': 'concise answers',
             'arguments': {'instructions': 'Be concise.'}},
            {'kind': 'wait'},
            {'kind': 'wait', 'job': {'job_id': 'running', 'backend': 'test'}},
            {'kind': 'finish'},
        )

    def test_all_kinds_preserve_reason_and_legacy_defaults(self):
        for payload in self.decisions():
            # None here means the field is absent; explicit JSON null is invalid.
            for reason in (None, '', '回答过长；调整提示，检查答案长度。'):
                with self.subTest(payload=payload, reason=reason):
                    output = dict(payload)
                    if reason is not None:
                        output['reason'] = reason
                    self.respond(output)
                    decision = self.policy.decide(self.context)
                    self.assertEqual(decision.kind, payload['kind'])
                    self.assertEqual(decision.reason, reason or '')
                    self.assertTrue(decision.simulated)
                    self.assertEqual(decision.usage, self.usage)
                    if decision.operation:
                        self.assertEqual(decision.operation.arguments, payload['arguments'])
                    if decision.directive:
                        self.assertEqual(decision.directive.payload, PromptSpec('Be concise.'))
                    if decision.job:
                        self.assertEqual(decision.job, JobRef('running', 'test'))

    def test_all_decisions_and_reviews_reject_non_string_reason(self):
        cases = [(self.policy.decide, p) for p in self.decisions()]
        cases += [(self.policy.review, p) for p in (
            self.decisions()[0], {'action': 'continue'}, {'action': 'cancel'},
        )]
        for method, payload in cases:
            for reason in (None, True, False, 0, 1.5, [], {}):
                with self.subTest(method=method.__name__, payload=payload, reason=reason):
                    self.respond({**payload, 'reason': reason})
                    with self.assertRaisesRegex(ValueError, 'reason must be a string'):
                        method(self.context)

    def test_explore_reason_is_not_a_legacy_mode_parameter(self):
        for parameters in ({}, {'arguments': {}}):
            with self.subTest(parameters=parameters):
                self.respond({'kind': 'explore', 'strategy': 'check baseline',
                              'reason': 'No baseline; run an episode and check its score.', **parameters})
                decision = self.policy.decide(self.context)
                self.assertEqual(decision.directive.mode, 'execute')
                self.assertIsNone(decision.directive.payload)
                self.assertIn('baseline', decision.reason)
        for reason in (None, 'Answers are verbose; shorten the prompt and check length.'):
            payload = {'kind': 'explore', 'mode': 'configure_prompt',
                       'strategy': 'concise answers', 'instructions': 'Be concise.'}
            if reason is not None:
                payload['reason'] = reason
            self.respond(payload)
            decision = self.policy.decide(self.context)
            self.assertEqual(decision.directive.payload, PromptSpec('Be concise.'))
            self.assertEqual(decision.reason, reason or '')

    def test_reason_does_not_relax_mixed_parameter_validation(self):
        self.respond({'kind': 'explore', 'mode': 'configure_prompt', 'strategy': 'concise answers',
                      'arguments': {'instructions': 'Be concise.'}, 'instructions': 'Other prompt.',
                      'reason': 'Compare lengths.'})
        with self.assertRaisesRegex(ValueError, 'Use either arguments or top-level'):
            self.policy.decide(self.context)

    def test_review_invoke_and_final_actions_preserve_optional_reason(self):
        for payload in (self.decisions()[0], {'action': 'continue'}, {'action': 'cancel'}):
            for reason in (None, '', 'Loss rose; inspect metrics and check for regression.'):
                with self.subTest(payload=payload, reason=reason):
                    output = dict(payload)
                    if reason is not None:
                        output['reason'] = reason
                    self.respond(output)
                    decision = self.policy.review(self.context)
                    self.assertEqual(decision.reason, reason or '')
                    self.assertTrue(decision.simulated)
                    self.assertEqual(decision.usage, self.usage)
                    if 'kind' in payload:
                        self.assertEqual(decision.kind, 'invoke')
                        self.assertEqual(decision.operation.arguments, payload['arguments'])
                    else:
                        self.assertEqual(decision.action, payload['action'])

    def test_invoke_nested_reason_remains_an_operation_argument(self):
        for method in (self.policy.decide, self.policy.review):
            with self.subTest(method=method.__name__):
                self.respond({'kind': 'invoke', 'operation': 'inspect',
                              'arguments': {'reason': {'filter': 'tool-specific'}},
                              'reason': 'Inspect loss before deciding.'})
                decision = method(self.context)
                self.assertEqual(decision.reason, 'Inspect loss before deciding.')
                self.assertEqual(decision.operation.arguments, {'reason': {'filter': 'tool-specific'}})

    def test_review_reason_preserves_operation_and_field_restrictions(self):
        valid = {'kind': 'invoke', 'operation': 'inspect', 'arguments': {}, 'reason': 'Check loss.'}
        for payload in ({**valid, 'extra': True}, {**valid, 'operation': 'unknown'},
                        {k: v for k, v in valid.items() if k != 'arguments'}):
            with self.subTest(payload=payload):
                self.respond(payload)
                with self.assertRaises(ValueError):
                    self.policy.review(self.context)
        self.respond(valid)
        with self.assertRaisesRegex(ValueError, 'available operation'):
            self.policy.review(replace(self.context, operations=()))

    def test_invalid_reason_uses_existing_bounded_validation_retry(self):
        policy = ModelDecisionPolicy(self.backend, self.model, max_validation_attempts=2)
        self.backend.generate.side_effect = [
            GenerationResponse('{"kind":"wait","reason":null}', self.model, False, self.usage),
            GenerationResponse('{"kind":"wait","reason":"Check pending results next turn."}',
                               self.model, False, self.usage),
        ]
        recorded = []
        with policy.track_usage(recorded.append):
            decision = policy.decide(self.context)
        self.assertEqual(decision.reason, 'Check pending results next turn.')
        self.assertEqual(recorded, [self.usage, self.usage])
        self.assertEqual(self.backend.generate.call_count, 2)
        retry_context = json.loads(self.backend.generate.call_args.args[0].prompt)
        self.assertIn('reason must be a string', retry_context['objective'])

    def test_prompts_request_public_rationale_without_reading_private_reasoning(self):
        class PrivateResponse(GenerationResponse):
            @property
            def reasoning_content(self):
                raise AssertionError('Private provider reasoning must not be read or logged')

        for method in (self.policy.decide, self.policy.review):
            with self.subTest(method=method.__name__):
                payload = {'kind': 'wait'} if method == self.policy.decide else {'action': 'continue'}
                self.backend.generate.return_value = PrivateResponse(
                    json.dumps({**payload, 'reason': 'Check pending metrics.'}), self.model, False, self.usage)
                recorded = []
                with self.policy.track_usage(recorded.append):
                    decision = method(self.context)
                self.assertEqual(decision.reason, 'Check pending metrics.')
                self.assertEqual(recorded, [self.usage])
                system = self.backend.generate.call_args.args[0].system
                for instruction in ('brief public rationale', 'optional top-level reason string',
                                    'observed issue', 'why this action', 'expect to check',
                                    'separate from arguments'):
                    self.assertIn(instruction, system)
                self.assertIn('Do not provide hidden chain-of-thought or raw reasoning_content', system)


if __name__ == '__main__':
    unittest.main()
