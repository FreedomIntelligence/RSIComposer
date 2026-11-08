"""Control metadata and validators agree without submitting or advancing jobs."""
from copy import deepcopy
from types import SimpleNamespace
import re
import unittest
from unittest.mock import Mock

from rsisys.loops.contracts.backend import BackendCapabilities
from rsisys.loops.contracts.candidate import Candidate, SystemVersion
from rsisys.loops.contracts.coordination import ExplorationProgress, UpdateReceipt
from rsisys.loops.contracts.directive import Directive
from rsisys.loops.decision_making.action_space import ModularDecisionActionSpace, OperationCall
from rsisys.loops.decision_making.action_space.access import authorized_space
from rsisys.loops.exploration.control import ExperimentControls
from rsisys.runtime.systems import ReferenceSystemActivator, SystemRegistry, system_adoption_capability


class ExperimentControlContractTests(unittest.TestCase):
    def setUp(self):
        self.progress = ExplorationProgress('job', 'waiting', Directive('job', 'task', '', simulated=True))
        self.exploration = SimpleNamespace(
            inspect=Mock(return_value=self.progress), cancel=Mock(return_value=self.progress),
            update=Mock(side_effect=lambda request: UpdateReceipt(request, request.expected_version + 1)))
        self.decision = SimpleNamespace(compare_candidates=Mock(return_value=()))
        self.controls = ExperimentControls(self.exploration, self.decision)
        self.capabilities = {item.definition.spec.name: item for item in self.controls.describe()}
        self.update = {'directive_id': 'job', 'expected_version': 0, 'idempotency_key': 'key',
                       'changes': {'remaining_polls': 1}}
        self.compare = {'candidate_ids': ['a', 'b'], 'metric': 'quality', 'higher_is_better': True}

    def assert_rejected_without_execution(self, capability, arguments, error=ValueError):
        with self.assertRaises(error):
            capability.definition.validate(arguments)
        with self.assertRaises(error):
            capability.handler(arguments)
        self.exploration.inspect.assert_not_called()
        self.exploration.cancel.assert_not_called()
        self.exploration.update.assert_not_called()
        self.decision.compare_candidates.assert_not_called()

    def test_exact_arguments_are_enforced_by_validator_and_handler(self):
        examples = {'inspect': {'directive_id': 'job'}, 'cancel': {'directive_id': 'job'},
                    'update': self.update, 'compare': self.compare}
        for kind, arguments in examples.items():
            capability = self.capabilities[f'experiment_{kind}']
            schema = capability.definition.spec.input_schema
            self.assertEqual(set(schema['required']), set(arguments))
            self.assertEqual(set(schema['properties']), set(arguments))
            self.assertIs(schema['additionalProperties'], False)
            invalid = [None, {**arguments, 'extra': 1}]
            invalid.extend({key: value for key, value in arguments.items() if key != missing}
                           for missing in arguments)
            for value in invalid:
                with self.subTest(kind=kind, arguments=value):
                    self.assert_rejected_without_execution(capability, value)

    def test_inspect_and_cancel_require_nonempty_string_ids(self):
        for name in ('experiment_inspect', 'experiment_cancel'):
            capability = self.capabilities[name]
            schema = capability.definition.spec.input_schema['properties']['directive_id']
            self.assertEqual(schema, {'type': 'string', 'minLength': 1})
            for ident in ('', None, True, 1, []):
                with self.subTest(name=name, ident=ident):
                    self.assert_rejected_without_execution(capability, {'directive_id': ident})

    def test_compare_schema_declares_distinct_nonempty_ids_and_boolean_direction(self):
        capability = self.capabilities['experiment_compare']
        fields = capability.definition.spec.input_schema['properties']
        self.assertEqual(fields['candidate_ids'], {
            'type': 'array', 'items': {'type': 'string', 'minLength': 1},
            'minItems': 2, 'uniqueItems': True})
        self.assertEqual(fields['higher_is_better'], {'type': 'boolean'})
        invalid = [{'candidate_ids': ids} for ids in ('ab', [], ['a'], ['a', 'a'], ['a', ''], ['a', 1])]
        invalid += [{'metric': metric} for metric in ('', None, 1)]
        invalid += [{'higher_is_better': value} for value in (0, 1, 'true', None)]
        for changes in invalid:
            with self.subTest(changes=changes):
                self.assert_rejected_without_execution(capability, {**self.compare, **changes})
        normalized = capability.definition.validate({**self.compare, 'candidate_ids': ('a', 'b')})
        self.assertEqual(normalized.arguments['candidate_ids'], ['a', 'b'])

    def test_update_schema_and_validator_require_version_key_and_nonempty_changes(self):
        capability = self.capabilities['experiment_update']
        fields = capability.definition.spec.input_schema['properties']
        self.assertEqual(fields['expected_version'], {'type': 'integer', 'minimum': 0})
        self.assertEqual(fields['idempotency_key'], {'type': 'string', 'minLength': 1})
        self.assertEqual(fields['changes'], {'type': 'object', 'minProperties': 1, 'additionalProperties': True})
        invalid = [{'expected_version': value} for value in (-1, True, 0.5, '0')]
        invalid += [{'idempotency_key': value} for value in ('', None, 1)]
        invalid += [{'changes': value} for value in ({}, [], None, {1: 'invalid'}, {'x': float('nan')})]
        for changes in invalid:
            with self.subTest(changes=changes):
                self.assert_rejected_without_execution(capability, {**self.update, **changes})
        normalized = capability.definition.validate(self.update)
        normalized.arguments['changes']['remaining_polls'] = 99
        self.assertEqual(self.update['changes'], {'remaining_polls': 1})

    def test_explicit_backend_fields_bound_schema_validator_and_direct_handler(self):
        controls = ExperimentControls(self.exploration, backend_capabilities=BackendCapabilities(
            update_fields=('remaining_polls',)))
        capability = next(item for item in controls.describe() if item.definition.spec.name == 'experiment_update')
        schema = capability.definition.spec.input_schema['properties']['changes']
        self.assertEqual(schema, {'type': 'object', 'minProperties': 1,
                                 'properties': {'remaining_polls': {}}, 'additionalProperties': False})
        for changes in ({'learning_rate': 0.1}, {'remaining_polls': 1, 'learning_rate': 0.1}):
            self.assert_rejected_without_execution(capability, {**self.update, 'changes': changes},
                                                   NotImplementedError)
        original = deepcopy(self.update)
        capability.definition.validate(self.update)
        self.exploration.update.assert_not_called()
        result = capability.handler(self.update)
        self.assertEqual(result.data['version'], 1)
        self.assertTrue(result.simulated)
        self.assertEqual(self.update, original)
        self.assertEqual(len(self.exploration.update.call_args_list), 1)

    def test_empty_update_fields_never_advertise_updates_even_with_other_flags(self):
        for checkpoint in (False, True):
            for reattach in (False, True):
                controls = ExperimentControls(self.exploration, backend_capabilities=BackendCapabilities(
                    supports_checkpoint=checkpoint, supports_reattach=reattach))
                self.assertEqual(set(controls.bindings), {'experiment_inspect', 'experiment_cancel'})
                space = ModularDecisionActionSpace(controls.definitions)
                with self.assertRaisesRegex(ValueError, 'Unknown decision operation'):
                    space.validate(OperationCall('experiment_update', self.update))

    def test_backend_declaration_is_explicit_and_legacy_controls_keep_backend_validation(self):
        self.assertIn('experiment_update', self.controls.bindings)
        self.assertNotIn('experiment_compare', ExperimentControls(self.exploration).bindings)
        self.exploration.update.side_effect = NotImplementedError('Unsupported live updates')
        capability = self.capabilities['experiment_update']
        capability.definition.validate(self.update)
        with self.assertRaisesRegex(NotImplementedError, 'Unsupported live updates'):
            capability.handler(self.update)
        self.exploration.inspect.assert_not_called()
        for invalid in ({}, (), False):
            with self.assertRaises(TypeError):
                ExperimentControls(self.exploration, backend_capabilities=invalid)

    def test_review_exposes_only_inspect_and_compare(self):
        space = authorized_space(ModularDecisionActionSpace(self.controls.definitions), 'review')
        self.assertEqual({spec.name for spec in space.describe()}, {'experiment_inspect', 'experiment_compare'})
        for name, arguments in (('experiment_cancel', {'directive_id': 'job'}),
                                ('experiment_update', self.update)):
            with self.assertRaises(PermissionError):
                space.validate(OperationCall(name, arguments))
        self.exploration.cancel.assert_not_called()
        self.exploration.update.assert_not_called()


class SystemAdoptionSchemaTests(unittest.TestCase):
    def test_schema_rejects_the_same_blank_identities_as_validator_before_mutation(self):
        initial = SystemVersion('initial', {})
        registry = SystemRegistry(initial, ReferenceSystemActivator())
        candidate = Candidate('candidate', 'base', {}, ('actor',), 'initial')
        capability = system_adoption_capability(registry, lambda: (candidate,))
        fields = capability.definition.spec.input_schema['properties']
        arguments = {'candidate_ids': ['candidate'], 'version': 'combined', 'idempotency_key': 'apply'}
        for name, schema in (('candidate_ids', fields['candidate_ids']['items']),
                             ('version', fields['version']), ('idempotency_key', fields['idempotency_key'])):
            self.assertEqual(schema['type'], 'string')
            self.assertEqual(schema['minLength'], 1)
            for blank in ('', ' ', '\t\n', '\u3000'):
                with self.subTest(name=name, blank=blank):
                    self.assertIsNone(re.search(schema['pattern'], blank))
                    invalid = {**arguments, name: [blank] if name == 'candidate_ids' else blank}
                    with self.assertRaises(ValueError):
                        capability.definition.validate(invalid)
                    with self.assertRaises(ValueError):
                        capability.handler(invalid)
            self.assertIsNotNone(re.search(schema['pattern'], ' value '))
        self.assertEqual(registry.current, initial)
        for version in ('candidate', 'combined'):
            with self.assertRaises(KeyError):
                registry.get(version)
        normalized = capability.definition.validate(arguments)
        self.assertEqual(normalized.arguments, arguments)
        self.assertEqual(registry.current, initial)


if __name__ == '__main__':
    unittest.main()
