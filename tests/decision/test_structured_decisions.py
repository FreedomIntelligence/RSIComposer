"""Per-turn output schemas constrain syntax; execution still owns validation."""

from copy import deepcopy
from dataclasses import replace
import json
import unittest
from unittest.mock import Mock

from rsisys.loops.contracts.artifacts import ArtifactRef
from rsisys.loops.contracts.jobs import JobRef
from rsisys.loops.contracts.model_service import GenerationResponse, ModelRef, TokenUsage
from rsisys.loops.decision_making.action_space import OperationResult, OperationSpec
from rsisys.loops.decision_making.action_space.directives import (
    DirectiveDefinition, DirectiveDefinitions, DirectiveSpec,
)
from rsisys.loops.decision_making.harness.policy import ModelDecisionPolicy
from rsisys.loops.decision_making.harness.state import DecisionContext


MODEL = ModelRef('fixture', 'researcher', 'v1')


class Backend:
    def __init__(self, *responses, supported=True):
        self.supports_response_schema = supported
        self.responses = iter(responses or ({'kind': 'finish'},))
        self.requests = []

    def generate(self, request):
        self.requests.append(request)
        value = next(self.responses)
        return GenerationResponse(value if isinstance(value, str) else json.dumps(value),
                                  request.model, False, TokenUsage(5, 3))


class StructuredDecisionTests(unittest.TestCase):
    def setUp(self):
        self.parameters = {
            'type': 'object',
            'properties': {
                'kind': {'enum': ['agenda', 'todo']},
                'options': {'type': 'object', 'properties': {
                    'limit': {'type': 'integer', 'minimum': 1, 'maximum': 32},
                }, 'required': ['limit'], 'additionalProperties': False},
            },
            'required': ['kind'], 'additionalProperties': False,
            'description': 'Arbitrary custom parameters, including a nested kind.',
        }
        def parse(arguments, candidates):
            if arguments.get('kind') not in ('agenda', 'todo'):
                raise ValueError('Choose a declared note kind')
            return arguments
        self.catalog = DirectiveDefinitions((
            DirectiveDefinition(DirectiveSpec('custom_notes', 'Custom domain mode', self.parameters), parse),
            DirectiveDefinition(DirectiveSpec('unavailable', 'Hidden this turn', {}), parse),
            DirectiveDefinition(DirectiveSpec('no_inputs', 'No artifact inputs', {},
                                              accepts_input_refs=False), parse),
        ))
        self.context = DecisionContext('Choose your own next step', (), (
            OperationSpec('research_read', 'Read notes', deepcopy(self.parameters), read_only=True),
        ), (), directive_modes=('custom_notes',), review_actions=('continue',))
        self.backend = Backend()
        self.policy = ModelDecisionPolicy(self.backend, MODEL, directives=self.catalog)

    def test_schemas_include_only_current_actions_with_closed_const_envelopes(self):
        schema = self.policy.decision_schema(self.context)
        branches = schema['oneOf']
        self.assertEqual([branch['properties']['kind']['const'] for branch in branches],
                         ['invoke', 'explore', 'wait', 'finish'])
        operation, directive = branches[:2]
        self.assertEqual(operation['properties']['operation'], {'const': 'research_read'})
        self.assertEqual(directive['properties']['mode'], {'const': 'custom_notes'})
        for branch in branches:
            self.assertEqual(branch['type'], 'object')
            self.assertIs(branch['additionalProperties'], False)
            self.assertEqual(branch['properties']['reason'], {'type': 'string'})
            self.assertNotIn('reason', branch['required'])
        for branch in (operation, directive):
            self.assertEqual(branch['properties']['arguments'], self.parameters)
            self.assertNotIn('options', branch['properties'])
            self.assertIn('arguments', branch['required'])
        self.assertEqual(directive['required'], ['kind', 'mode', 'strategy', 'arguments'])
        empty = self.policy.decision_schema(replace(self.context, operations=(), directive_modes=()))
        self.assertEqual([b['properties']['kind']['const'] for b in empty['oneOf']], ['wait', 'finish'])
        self.assertNotIn('unavailable', json.dumps(schema))

    def test_schema_copies_preserve_custom_constraints_and_do_not_mutate_catalog(self):
        schema = self.policy.decision_schema(self.context)
        schema['oneOf'][0]['properties']['arguments']['properties']['kind']['enum'].append('changed')
        schema['oneOf'][1]['properties']['arguments']['properties']['options']['required'].clear()
        again = self.policy.decision_schema(self.context)
        for branch in again['oneOf'][:2]:
            self.assertEqual(branch['properties']['arguments'], self.parameters)
        self.assertEqual(self.context.operations[0].input_schema, self.parameters)
        self.assertEqual(self.catalog.describe(('custom_notes',))[0].input_schema, self.parameters)

    def test_auto_on_off_and_explicit_supported_attach_schemas_to_both_requests(self):
        for supported, setting, enabled in ((True, None, True), (False, None, False),
                                             (True, False, False), (False, False, False),
                                             (True, True, True)):
            with self.subTest(supported=supported, setting=setting):
                backend = Backend({'kind': 'finish'}, {'action': 'continue'}, supported=supported)
                policy = ModelDecisionPolicy(backend, MODEL, directives=self.catalog,
                                             structured_output=setting)
                self.assertIs(policy.structured_output, enabled)
                self.assertEqual(policy.decide(self.context).kind, 'finish')
                self.assertEqual(policy.review(self.context).action, 'continue')
                expected = (policy.decision_schema(self.context), policy.review_schema(self.context))
                for request, schema in zip(backend.requests, expected):
                    self.assertEqual(request.response_schema, schema if enabled else None)

    def test_auto_requires_an_explicit_true_capability_and_unsupported_true_fails_early(self):
        for backend in (Mock(), Backend(supported=False), Backend(supported='yes')):
            with self.subTest(backend=type(backend).__name__):
                self.assertFalse(ModelDecisionPolicy(backend, MODEL).structured_output)
                with self.assertRaisesRegex(ValueError, 'supports_response_schema=True'):
                    ModelDecisionPolicy(backend, MODEL, structured_output=True)
        for value in (0, 1, 'auto', []):
            with self.subTest(value=value), self.assertRaisesRegex(TypeError, 'bool or None'):
                ModelDecisionPolicy(self.backend, MODEL, structured_output=value)

    def test_custom_mode_nested_kind_inputs_timing_and_rationale_survive(self):
        ref = ArtifactRef('selected', 'custom_data', 'memory://selected')
        context = replace(self.context, operation_results=(OperationResult('Prepared', (ref,), False),))
        value = {'kind': 'explore', 'mode': 'custom_notes', 'strategy': 'Read selected notes',
                 'arguments': {'kind': 'agenda', 'options': {'limit': 2}}, 'input_ids': ['selected'],
                 'review_after_seconds': 2.5, 'reason': 'Compare available evidence'}
        backend = Backend(value)
        policy = ModelDecisionPolicy(backend, MODEL, directives=self.catalog)
        decision = policy.decide(context)
        self.assertEqual(decision.reason, value['reason'])
        self.assertEqual(decision.directive.payload, value['arguments'])
        self.assertEqual(decision.directive.strategy, value['strategy'])
        self.assertEqual(decision.directive.input_refs, (ref,))
        self.assertEqual(decision.directive.review_after_seconds, 2.5)
        branch = backend.requests[0].response_schema['oneOf'][1]
        self.assertEqual(branch['properties']['input_ids'], {'type': 'array', 'items': {'type': 'string'}})
        self.assertEqual(branch['properties']['review_after_seconds'],
                         {'type': ['number', 'null'], 'exclusiveMinimum': 0})

    def test_non_input_mode_schema_allows_only_empty_input_ids(self):
        context = replace(self.context, directive_modes=('no_inputs',))
        branch = self.policy.decision_schema(context)['oneOf'][1]
        self.assertEqual(branch['properties']['input_ids']['maxItems'], 0)
        self.assertNotIn('input_ids', branch['required'])

    def test_finish_and_wait_keep_optional_reason_job_and_schedule_yield(self):
        for value in ({'kind': 'finish'}, {'kind': 'wait'}, {'kind': 'wait', 'job': None},
                      {'kind': 'wait', 'job': {'job_id': 'running', 'backend': 'fixture'}, 'reason': 'Yield'}):
            with self.subTest(value=value):
                backend = Backend(value)
                decision = ModelDecisionPolicy(backend, MODEL, directives=self.catalog).decide(self.context)
                self.assertEqual(decision.reason, value.get('reason', ''))
                self.assertEqual(decision.job, JobRef(**value['job']) if value.get('job') else None)
                branch = backend.requests[0].response_schema['oneOf'][-2 if value['kind'] == 'wait' else -1]
                self.assertEqual(branch['required'], ['kind'])

    def test_review_schema_has_only_available_queries_and_controls(self):
        schema = self.policy.review_schema(self.context)
        invocation, control = schema['oneOf']
        self.assertEqual(invocation['properties']['operation'], {'const': 'research_read'})
        self.assertEqual(invocation['properties']['arguments'], self.parameters)
        self.assertEqual(control['properties']['action'], {'const': 'continue'})
        self.assertEqual(control['required'], ['action'])
        self.assertEqual(control['properties']['next_review_seconds'],
                         {'type': ['number', 'null'], 'exclusiveMinimum': 0})
        self.assertIs(control['additionalProperties'], False)
        controls_only = self.policy.review_schema(replace(self.context, operations=()))
        self.assertEqual(controls_only, {'oneOf': [control]})
        no_actions = replace(self.context, operations=(), review_actions=())
        self.assertEqual(self.policy.review_schema(no_actions), {'not': {}})
        self.assertNotIn('explore', json.dumps(schema))
        self.assertNotIn('cancel', json.dumps(schema))

    def test_review_nested_operation_and_final_control_keep_reason_and_timing(self):
        backend = Backend({'kind': 'invoke', 'operation': 'research_read',
                           'arguments': {'kind': 'agenda'}, 'reason': 'Inspect notes'},
                          {'action': 'continue', 'next_review_seconds': 3, 'reason': 'Keep measuring'})
        policy = ModelDecisionPolicy(backend, MODEL, directives=self.catalog)
        query = policy.review(self.context)
        self.assertEqual(query.operation.arguments, {'kind': 'agenda'})
        self.assertEqual(query.reason, 'Inspect notes')
        review = policy.review(self.context)
        self.assertEqual((review.action, review.next_review_seconds, review.reason),
                         ('continue', 3, 'Keep measuring'))

    def test_advertised_support_does_not_bypass_existing_parser_checks(self):
        for method, value, error in (
            ('decide', {'kind': 'explore', 'mode': 'unavailable', 'strategy': 'invalid', 'arguments': {}},
             'Disallowed exploration mode'),
            ('decide', {'kind': 'explore', 'mode': 'custom_notes', 'strategy': 'invalid',
                        'arguments': {'kind': 'invalid'}}, 'declared note kind'),
            ('review', {'action': 'cancel'}, 'not currently permitted'),
            ('review', {'action': 'continue', 'next_review_seconds': 0}, 'finite and positive'),
        ):
            with self.subTest(method=method, value=value):
                policy = ModelDecisionPolicy(Backend(value), MODEL, directives=self.catalog)
                with self.assertRaisesRegex(ValueError, error):
                    getattr(policy, method)(self.context)

    def test_both_parsers_reject_duplicate_keys_at_every_depth_even_when_disabled(self):
        invalid = (
            '{"kind":"invoke","kind":"finish","operation":"research_read","arguments":{}}',
            '{"kind":"invoke","operation":"research_read","arguments":{"kind":"agenda","kind":"todo"}}',
            '{"kind":"invoke","operation":"research_read","arguments":{"items":[{"id":1,"id":1}]}}',
        )
        for setting in (None, False):
            for method in ('decide', 'review'):
                for value in invalid:
                    with self.subTest(setting=setting, method=method, value=value):
                        backend = Backend(value)
                        policy = ModelDecisionPolicy(backend, MODEL, directives=self.catalog,
                                                     structured_output=setting)
                        with self.assertRaisesRegex(ValueError, 'Duplicate JSON key'):
                            getattr(policy, method)(self.context)
                        self.assertEqual(len(backend.requests), 1)
        policy = ModelDecisionPolicy(Backend('{"action":"continue","action":"cancel"}'), MODEL)
        with self.assertRaisesRegex(ValueError, "Duplicate JSON key 'action'"):
            policy.review(self.context)

    def test_duplicate_rejection_uses_existing_bounded_retry_and_usage_tracking(self):
        backend = Backend('{"kind":"wait","kind":"finish"}', {'kind': 'finish', 'reason': 'Done'})
        policy = ModelDecisionPolicy(backend, MODEL, directives=self.catalog, max_validation_attempts=2)
        usage = []
        with policy.track_usage(usage.append):
            self.assertEqual(policy.decide(self.context).reason, 'Done')
        self.assertEqual(usage, [TokenUsage(5, 3), TokenUsage(5, 3)])
        self.assertIn('Duplicate JSON key', backend.requests[1].prompt)
        self.assertEqual(backend.requests[0].response_schema, backend.requests[1].response_schema)

    def test_local_references_fail_clearly_before_generation_for_both_catalogs(self):
        for ref in ('#', '#/$defs/value', '#named', ''):
            for keyword in ('$ref', '$dynamicRef', '$recursiveRef'):
                with self.subTest(ref=ref, keyword=keyword):
                    schema = {'type': 'object', '$defs': {'value': {'type': 'string'}},
                              'properties': {'value': {keyword: ref}}}
                    operation = replace(self.context.operations[0], input_schema=schema)
                    context = replace(self.context, operations=(operation,))
                    backend = Backend()
                    policy = ModelDecisionPolicy(backend, MODEL, directives=self.catalog)
                    for method in (policy.decide, policy.review):
                        with self.assertRaisesRegex(ValueError, 'cannot embed local'):
                            method(context)
                    self.assertFalse(backend.requests)
        catalog = DirectiveDefinitions((DirectiveDefinition(
            DirectiveSpec('local', 'Local schema', {'$ref': '#/definitions/x'}), lambda args, _: args),))
        policy = ModelDecisionPolicy(self.backend, MODEL, directives=catalog)
        with self.assertRaisesRegex(ValueError, "directive 'local'"):
            policy.decision_schema(replace(self.context, directive_modes=('local',)))

    def test_literal_reference_data_is_preserved_and_off_mode_works_with_local_refs(self):
        schema = {'type': 'object', 'properties': {'kind': {'const': {'$ref': '#literal'}}},
                  'examples': [{'$ref': '#example'}], 'default': {'$ref': '#default'}}
        context = replace(self.context, operations=(replace(self.context.operations[0], input_schema=schema),))
        embedded = self.policy.decision_schema(context)['oneOf'][0]['properties']['arguments']
        self.assertEqual(embedded, schema)
        context = replace(context, operations=(replace(context.operations[0], input_schema={'$ref': '#'}),))
        backend = Backend()
        self.assertEqual(ModelDecisionPolicy(backend, MODEL, structured_output=False).decide(
            replace(context, directive_modes=())).kind, 'finish')
        self.assertIsNone(backend.requests[0].response_schema)

    def test_full_medical_catalog_request_uses_schema_without_executing_an_action(self):
        if __package__:
            from tests.experiments.medical.test_medical_experiment import FINISH, FixtureResearcher, MedicalExperimentIntegrationTests
        else:
            from tests.experiments.medical.test_medical_experiment import FINISH, FixtureResearcher, MedicalExperimentIntegrationTests

        fixture = MedicalExperimentIntegrationTests()
        fixture.setUp()
        self.addCleanup(fixture.doCleanups)
        backend = FixtureResearcher(MODEL, fixture.process, [FINISH])
        backend.supports_response_schema = True
        result = fixture.execute([], decision_model=MODEL, decision_backend=backend)
        self.assertEqual(result['status'], 'completed', result)
        self.assertEqual(len(backend.requests), 1)
        self.assertFalse(fixture.process.requests)
        self.assertEqual(result['trainings_reserved'], 0)
        request = backend.requests[0]
        context = json.loads(request.prompt)
        branches = request.response_schema['oneOf']
        operations = {branch['properties']['operation']['const']: branch for branch in branches
                      if 'operation' in branch['properties']}
        modes = {branch['properties']['mode']['const'] for branch in branches
                 if 'mode' in branch['properties']}
        self.assertEqual(set(operations), {spec['name'] for spec in context['operations']})
        self.assertEqual(modes, set(context['directive_modes']))
        self.assertIn('train', modes)
        self.assertNotIn('evaluate', modes)
        arguments = operations['research_read']['properties']['arguments']
        self.assertIn('kind', arguments['properties'])
        self.assertEqual(operations['research_read']['properties']['kind'], {'const': 'invoke'})


if __name__ == '__main__':
    unittest.main()
