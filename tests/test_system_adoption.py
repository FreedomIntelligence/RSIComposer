"""Multi-candidate adoption uses one confirmed activation and preserves provenance."""
from copy import deepcopy
from dataclasses import asdict, replace
import unittest

from rsisys.loops.contracts.artifacts import ArtifactRef
from rsisys.loops.contracts.candidate import Candidate, SystemVersion
from rsisys.loops.contracts.lifecycle import ExecutionStatus
from rsisys.loops.contracts.model_service import ModelRef
from rsisys.loops.decision_making.action_space import ModularDecisionActionSpace, OperationCall
from rsisys.loops.decision_making.action_space.access import ActionPermissions, authorized_space
from rsisys.loops.decision_making.action_space.capabilities import Capabilities
from rsisys.loops.decision_making.harness.tools import BoundDecisionTools, execute_operation
from rsisys.runtime.systems import (
    ReferenceSystemActivator, SystemActivationRejected, SystemRegistry, system_adoption_capability,
)


MODEL = ModelRef('service', 'model', 'base')
PROMPT = ArtifactRef('prompt', 'prompt', 'file:///operator/prompt.txt')
DATA = ArtifactRef('data', 'training_data', 's3://operator/data', 'v2', 'digest')
UNCHANGED = ArtifactRef('policy', 'policy', 'memory://policy')


class RecordingActivator:
    def __init__(self):
        self.calls = []

    def activate(self, system, *, idempotency_key):
        self.calls.append((deepcopy(system), idempotency_key))
        return deepcopy(system)


class SystemAdoptionTests(unittest.TestCase):
    def setUp(self):
        self.initial = SystemVersion('initial', {'policy': UNCHANGED}, MODEL)
        self.activator = RecordingActivator()
        self.registry = SystemRegistry(self.initial, self.activator)
        self.prompt = self.candidate('prompt', {'prompt': PROMPT})
        self.data = self.candidate('data', {'data': DATA})
        for candidate in (self.prompt, self.data):
            self.registry.register(candidate)

    def candidate(self, ident, changes, *, parent='initial', base='base'):
        return Candidate(ident, base, changes, ('actor',), parent)

    def adopt(self, ids=('prompt', 'data'), *, version='combined', key='adopt'):
        return self.registry.adopt(ids, version=version, idempotency_key=key)

    def test_compose_unions_changes_and_retains_parent_without_activation(self):
        expected = SystemVersion('combined', {'policy': UNCHANGED, 'data': DATA, 'prompt': PROMPT},
                                 MODEL, 'initial')
        self.assertEqual(self.registry.compose(iter(['prompt', 'data']), version='combined'), expected)
        self.assertEqual(self.registry.get('combined'), expected)
        self.assertEqual(self.registry.current, self.initial)
        self.assertEqual(self.activator.calls, [])
        self.assertEqual(self.registry.compose(['data', 'prompt'], version='combined'), expected)

    def test_single_candidate_uses_the_same_composition_path(self):
        result = self.adopt(('prompt',))
        self.assertEqual(result.components, {'policy': UNCHANGED, 'prompt': PROMPT})
        self.assertEqual(len(self.activator.calls), 1)

    def test_nonempty_unique_registered_source_ids_and_version_required(self):
        for ids in ([], ['prompt', 'prompt'], [''], [1], 'prompt'):
            with self.subTest(ids=ids), self.assertRaises(ValueError):
                self.registry.compose(ids, version='combined')
        for ids in (['missing'], ['initial']):
            with self.subTest(ids=ids), self.assertRaises(KeyError):
                self.registry.compose(ids, version='combined')
        for version in ('', ' ', 'initial'):
            with self.subTest(version=version), self.assertRaises(ValueError):
                self.registry.compose(['prompt'], version=version)
        self.assertEqual(self.activator.calls, [])

    def test_overlapping_equal_values_are_accepted_and_different_values_rejected(self):
        self.registry.register(self.candidate('equal', {'prompt': PROMPT, 'data': DATA}))
        self.assertEqual(self.registry.compose(['prompt', 'equal'], version='equal-union').components,
                         {'policy': UNCHANGED, 'prompt': PROMPT, 'data': DATA})
        self.registry.register(self.candidate('conflict', {'prompt': replace(PROMPT, digest='different')}))
        for ids in (['prompt', 'conflict'], ['conflict', 'prompt']):
            with self.subTest(ids=ids), self.assertRaisesRegex(ValueError, 'Conflicting component'):
                self.registry.compose(ids, version='bad')
        with self.assertRaises(KeyError):
            self.registry.get('bad')
        self.assertEqual(self.registry.current, self.initial)

    def test_only_changes_are_merged_so_parent_values_do_not_override_siblings(self):
        parent = SystemVersion('initial', {'prompt': replace(PROMPT, artifact_id='old')}, MODEL)
        registry = SystemRegistry(parent, self.activator)
        registry.register(self.prompt)
        registry.register(self.data)
        actual = registry.compose(['data', 'prompt'], version='combined')
        self.assertEqual(actual.components, {'prompt': PROMPT, 'data': DATA})

    def test_parent_must_be_explicit_known_and_common(self):
        with self.assertRaisesRegex(ValueError, 'parent_system'):
            self.registry.register(replace(self.prompt, candidate_id='missing-parent', parent_system=None))
        with self.assertRaises(KeyError):
            self.registry.register(replace(self.prompt, candidate_id='unknown-parent', parent_system='unknown'))
        self.registry.register(self.candidate('other-parent', {}, parent='prompt'))
        with self.assertRaisesRegex(ValueError, 'same explicit parent'):
            self.registry.compose(['data', 'other-parent'], version='bad')

    def test_model_base_must_match_parent_revision_not_parent_version(self):
        for base in ('initial', 'another-model-revision'):
            with self.subTest(base=base), self.assertRaisesRegex(ValueError, 'model base revision'):
                self.registry.register(replace(self.prompt, candidate_id='bad-base', base_version=base))
        self.assertEqual(self.registry.compose(['prompt', 'data'], version='valid').model, MODEL)

    def test_unknown_parent_model_still_requires_candidates_to_share_base(self):
        registry = SystemRegistry(SystemVersion('initial', {}), ReferenceSystemActivator())
        registry.register(self.prompt)
        registry.register(replace(self.data, base_version='other'))
        with self.assertRaisesRegex(ValueError, 'same model base'):
            registry.compose(['prompt', 'data'], version='bad')

    def test_explicit_model_change_can_combine_with_unchanged_parent_model(self):
        new_model = replace(MODEL, revision='trained')
        self.registry.register(self.candidate('trained', {'model': DATA}), model=new_model)
        for ids in (['trained', 'prompt'], ['prompt', 'trained']):
            with self.subTest(ids=ids):
                actual = self.registry.compose(ids, version='new-model')
                self.assertEqual(actual.model, new_model)
                self.assertEqual(actual.components['prompt'], PROMPT)
        self.registry.register(self.candidate('same-model', {}), model=new_model)
        self.assertEqual(self.registry.compose(['trained', 'same-model'], version='same').model, new_model)

    def test_distinct_explicit_models_are_rejected_including_full_identity(self):
        self.registry.register(self.candidate('model-a', {}), model=replace(MODEL, revision='new'))
        self.registry.register(self.candidate('model-b', {}),
                               model=ModelRef('other-service', 'other-model', 'new'))
        with self.assertRaisesRegex(ValueError, 'Conflicting model'):
            self.registry.compose(['model-a', 'model-b'], version='bad')

    def test_version_identity_includes_sources_and_cannot_impersonate_candidates(self):
        self.registry.register(self.candidate('equivalent-prompt', {'prompt': PROMPT}))
        self.registry.compose(['prompt', 'data'], version='combined')
        for ids, version in ((['equivalent-prompt', 'data'], 'combined'), (['prompt'], 'prompt')):
            with self.subTest(ids=ids), self.assertRaisesRegex(ValueError, 'identity conflicts'):
                self.registry.compose(ids, version=version)
        combined_candidate = self.candidate('combined', {'data': DATA, 'prompt': PROMPT})
        with self.assertRaisesRegex(ValueError, 'identity conflicts'):
            self.registry.register(combined_candidate)
        with self.assertRaises(KeyError):
            self.registry.compose(['combined'], version='nested')

    def test_adopt_activates_exact_union_once_and_returns_receipt_after_current_advances(self):
        first = self.adopt()
        self.assertEqual(first, self.registry.current)
        self.assertEqual(self.activator.calls, [(first, 'adopt')])
        self.registry.register(self.candidate('later', {}, parent='combined'))
        latest = self.registry.adopt(['later'], version='latest', idempotency_key='next')
        self.assertEqual(self.adopt(('data', 'prompt')), first)
        self.assertEqual(self.registry.current, latest)
        self.assertEqual(len(self.activator.calls), 2)

    def test_old_parent_can_be_composed_but_new_activation_is_stale(self):
        self.adopt()
        old = self.registry.compose(['prompt'], version='old-parent')
        self.assertEqual(old.parent_version, 'initial')
        with self.assertRaisesRegex(ValueError, 'stale'):
            self.registry.activate('old-parent', idempotency_key='stale')
        with self.assertRaisesRegex(ValueError, 'stale'):
            self.adopt(version='other', key='other')
        self.assertEqual(len(self.activator.calls), 1)

    def test_application_key_cannot_change_version_or_source_set(self):
        self.adopt()
        for ids, version in ((['prompt', 'data'], 'different'), (['prompt'], 'combined')):
            with self.subTest(ids=ids, version=version), self.assertRaisesRegex(ValueError, 'reused|conflicts'):
                self.adopt(ids, version=version)
        with self.assertRaisesRegex(ValueError, 'reused'):
            self.registry.activate('prompt', idempotency_key='adopt')
        self.assertEqual(len(self.activator.calls), 1)

    def test_blank_application_keys_do_not_compose_or_activate(self):
        for key in ('', ' ', None, 1):
            with self.subTest(key=key), self.assertRaisesRegex(ValueError, 'idempotency key'):
                self.adopt(key=key)
        with self.assertRaises(KeyError):
            self.registry.get('combined')

    def test_uncertain_response_preserves_pending_and_replays_activator_receipt(self):
        class LostResponse(RecordingActivator):
            def __init__(self):
                super().__init__()
                self.receipts = {}
                self.applied = 0

            def activate(self, system, *, idempotency_key):
                self.calls.append((deepcopy(system), idempotency_key))
                if idempotency_key in self.receipts:
                    return self.receipts[idempotency_key]
                self.receipts[idempotency_key] = deepcopy(system)
                self.applied += 1
                raise TimeoutError('Response lost after applying')

        activator = LostResponse()
        self.registry.activator = activator
        with self.assertRaises(TimeoutError):
            self.adopt()
        self.assertEqual(self.registry.current, self.initial)
        for ids, version, key, error in (
            (['prompt'], 'combined', 'adopt', ValueError),
            (['prompt', 'data'], 'different', 'adopt', ValueError),
            (['prompt', 'data'], 'combined', 'other-key', RuntimeError),
            (['prompt'], 'another-version', 'other-key', RuntimeError),
        ):
            with self.subTest(version=version, key=key), self.assertRaises(error):
                self.adopt(ids, version=version, key=key)
        with self.assertRaisesRegex(RuntimeError, 'unconfirmed'):
            self.registry.activate('prompt', idempotency_key='direct')
        actual = self.adopt(('data', 'prompt'))
        self.assertEqual(self.registry.current, actual)
        self.assertEqual(activator.applied, 1)
        self.assertEqual(len(activator.calls), 2)
        self.assertEqual(self.adopt(), actual)
        self.assertEqual(len(activator.calls), 2)

    def test_confirmed_rejection_retains_failure_receipt_and_releases_pending(self):
        class Reject(RecordingActivator):
            def activate(self, system, *, idempotency_key):
                self.calls.append((deepcopy(system), idempotency_key))
                raise SystemActivationRejected('No components applied')

        reject = Reject()
        self.registry.activator = reject
        for _ in range(2):
            with self.assertRaisesRegex(SystemActivationRejected, 'No components applied'):
                self.adopt()
        self.assertEqual(len(reject.calls), 1)
        self.assertEqual(self.registry.current, self.initial)
        with self.assertRaises(ValueError):
            self.adopt(version='different')
        self.registry.activator = self.activator
        self.assertEqual(self.adopt(key='fresh').version, 'combined')

    def test_mismatched_activator_manifest_never_publishes_success(self):
        class WrongManifest:
            def activate(self, system, *, idempotency_key):
                return replace(system, components={})

        self.registry.activator = WrongManifest()
        with self.assertRaisesRegex(ValueError, 'different system manifest'):
            self.adopt()
        self.assertEqual(self.registry.current, self.initial)
        with self.assertRaisesRegex(RuntimeError, 'unconfirmed'):
            self.adopt(key='other')
        self.registry.activator = self.activator
        self.adopt()

    def test_reference_activator_applies_references_but_rejects_model_deployment(self):
        self.registry.activator = ReferenceSystemActivator(MODEL)
        actual = self.adopt()
        self.assertEqual(actual.components['data'], DATA)
        self.assertEqual(actual.model, MODEL)
        self.registry.register(self.candidate('trained', {}, parent='combined'),
                               model=replace(MODEL, revision='trained'))
        with self.assertRaisesRegex(SystemActivationRejected, 'deployment'):
            self.adopt(['trained'], version='deployment', key='deployment')
        self.assertEqual(self.registry.current, actual)

    def test_inputs_and_exported_manifests_are_defensive_copies(self):
        ids = ['prompt', 'data']
        before = deepcopy((ids, self.prompt, self.data, self.initial))
        composed = self.registry.compose(ids, version='combined')
        actual = self.adopt(ids)
        self.assertEqual((ids, self.prompt, self.data, self.initial), before)
        composed.components.clear()
        actual.components.clear()
        self.registry.current.components.clear()
        self.registry.get('combined').components.clear()
        ids.clear()
        self.prompt.changes.clear()
        self.initial.components.clear()
        receipt = self.adopt()
        self.assertEqual(receipt.components, {'policy': UNCHANGED, 'prompt': PROMPT, 'data': DATA})
        self.assertEqual(self.registry.current, receipt)


class SystemAdoptionCapabilityTests(unittest.TestCase):
    def setUp(self):
        self.activator = RecordingActivator()
        self.initial = SystemVersion('initial', {}, MODEL)
        self.registry = SystemRegistry(self.initial, self.activator)
        self.known = [Candidate('prompt', 'base', {'prompt': PROMPT}, ('actor',), 'initial'),
                      Candidate('data', 'base', {'data': DATA}, ('actor',), 'initial')]
        self.callback_calls = 0

        def candidates():
            self.callback_calls += 1
            return iter(self.known)

        self.capability = system_adoption_capability(self.registry, candidates)
        self.capabilities = Capabilities((self.capability,))
        self.space = ModularDecisionActionSpace(self.capabilities.definitions, directive_modes=())
        self.tools = BoundDecisionTools(self.capabilities.bindings)
        self.arguments = {'candidate_ids': ['prompt', 'data'], 'version': 'combined',
                          'idempotency_key': 'adopt'}

    def call(self, **arguments):
        return OperationCall('system_adopt', {**self.arguments, **arguments})

    def test_opt_in_assembly_and_validation_are_read_only(self):
        self.assertEqual(self.callback_calls, 0)
        self.assertEqual(ModularDecisionActionSpace(()).describe(), ())
        self.assertEqual(self.space.describe()[0].name, 'system_adopt')
        self.assertFalse(self.space.describe()[0].read_only)
        before = deepcopy((self.arguments, self.known))
        validated = self.space.validate(self.call())
        self.assertEqual((self.arguments, self.known), before)
        validated.arguments['candidate_ids'].clear()
        self.assertEqual(self.arguments['candidate_ids'], ['prompt', 'data'])
        for ident in ('prompt', 'data', 'combined'):
            with self.assertRaises(KeyError):
                self.registry.get(ident)
        self.assertEqual(self.registry.current, self.initial)
        self.assertEqual(self.activator.calls, [])

    def test_success_exposes_actual_manifest_and_adopted_ids_once(self):
        before = deepcopy((self.arguments, self.known))
        result = self.tools.execute(self.space.validate(self.call()))
        self.assertEqual(result.status, ExecutionStatus.COMPLETED)
        self.assertFalse(result.simulated)
        self.assertEqual(result.adopted_candidate_ids, ('prompt', 'data'))
        self.assertEqual(result.data, {'system': asdict(self.registry.current)})
        self.assertEqual(result.data['system']['components']['data'], asdict(DATA))
        self.assertEqual((self.arguments, self.known), before)
        self.assertEqual(self.tools.execute(self.space.validate(self.call())), result)
        self.assertEqual(len(self.activator.calls), 1)
        result.data['system']['components'].clear()
        self.assertEqual(self.registry.current.components, {'prompt': PROMPT, 'data': DATA})

    def test_validates_exact_arguments_and_live_known_selection(self):
        invalid = ({}, {**self.arguments, 'extra': True},
                   {**self.arguments, 'candidate_ids': []},
                   {**self.arguments, 'candidate_ids': ['prompt', 'prompt']},
                   {**self.arguments, 'candidate_ids': 'prompt'},
                   {**self.arguments, 'candidate_ids': ['unknown']},
                   {**self.arguments, 'version': ' '},
                   {**self.arguments, 'idempotency_key': ''})
        for arguments in invalid:
            with self.subTest(arguments=arguments), self.assertRaises(ValueError):
                self.space.validate(OperationCall('system_adopt', arguments))
        validated = self.space.validate(self.call())
        self.known.pop()
        with self.assertRaisesRegex(ValueError, 'known candidates'):
            self.tools.execute(validated)
        self.assertEqual(self.activator.calls, [])

    def test_callback_identity_conflicts_do_not_replace_registered_model(self):
        self.registry.register(self.known[0])
        self.known[0] = replace(self.known[0], changes={'prompt': replace(PROMPT, digest='changed')})
        with self.assertRaisesRegex(ValueError, 'identity conflicts'):
            self.space.validate(self.call())
        self.assertEqual(self.activator.calls, [])

    def test_callback_duplicate_conflicts_and_non_candidates_are_rejected(self):
        self.known.append(replace(self.known[0], changes={}))
        with self.assertRaisesRegex(ValueError, 'conflicting identities'):
            self.space.validate(self.call())
        self.known[:] = [object()]
        with self.assertRaisesRegex(TypeError, 'Candidate instances'):
            self.space.validate(self.call())

    def test_pre_registered_explicit_model_is_preserved(self):
        trained = replace(MODEL, revision='trained')
        self.registry.register(self.known[0], model=trained)
        result = self.tools.execute(self.space.validate(self.call()))
        self.assertEqual(result.data['system']['model'], asdict(trained))
        self.assertEqual(self.registry.current.model, trained)

    def test_model_artifact_is_never_interpreted_as_a_model_ref(self):
        self.known[0] = replace(self.known[0], changes={'weights': DATA})
        self.tools.execute(self.space.validate(self.call()))
        self.assertEqual(self.registry.current.model, MODEL)
        self.assertEqual(self.registry.current.components['weights'], DATA)

    def test_simulation_is_explicit(self):
        capability = system_adoption_capability(self.registry, lambda: self.known, simulated=True)
        result = capability.handler(self.arguments)
        self.assertTrue(result.simulated)
        self.assertEqual(result.adopted_candidate_ids, ('prompt', 'data'))
        for simulated in (None, 0, 'false'):
            with self.subTest(simulated=simulated), self.assertRaises(TypeError):
                system_adoption_capability(self.registry, lambda: self.known, simulated=simulated)

    def test_existing_permissions_deny_write_and_review_without_execution(self):
        self.space.permissions = ActionPermissions(operations=frozenset())
        execution = execute_operation(self.tools, authorized_space(self.space), self.call())
        self.assertIsInstance(execution.error, PermissionError)
        self.assertEqual(execution.outcome.adopted_candidate_ids, ())
        self.space.permissions = ActionPermissions()
        self.assertEqual(authorized_space(self.space, 'review').describe(), ())
        with self.assertRaises(PermissionError):
            authorized_space(self.space, 'review').validate(self.call())
        self.assertEqual(self.activator.calls, [])
        self.assertEqual(self.registry.current, self.initial)

    def test_uncertain_activation_has_no_successful_adoption_result(self):
        class LostResponse(RecordingActivator):
            def activate(self, system, *, idempotency_key):
                self.calls.append((deepcopy(system), idempotency_key))
                if len(self.calls) == 1:
                    raise TimeoutError('Unconfirmed application')
                return system

        delayed = LostResponse()
        self.registry.activator = delayed
        failed = execute_operation(self.tools, self.space, self.call()).outcome
        self.assertEqual(failed.status, ExecutionStatus.FAILED)
        self.assertEqual(failed.adopted_candidate_ids, ())
        self.assertEqual(failed.data, {})
        self.assertEqual(self.registry.current, self.initial)
        blocked = execute_operation(self.tools, self.space, self.call(idempotency_key='other')).outcome
        self.assertEqual(blocked.status, ExecutionStatus.FAILED)
        self.assertEqual(blocked.adopted_candidate_ids, ())
        retried = execute_operation(self.tools, self.space, self.call()).outcome
        self.assertEqual(retried.status, ExecutionStatus.COMPLETED)
        self.assertEqual(retried.adopted_candidate_ids, ('prompt', 'data'))
        self.assertEqual(len(delayed.calls), 2)


if __name__ == '__main__':
    unittest.main()
