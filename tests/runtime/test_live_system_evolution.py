from copy import deepcopy
from dataclasses import asdict, replace
import json
from pathlib import Path
from tempfile import TemporaryDirectory
from types import SimpleNamespace
import unittest

from rsisys.loops.contracts.candidate import Candidate, SystemVersion
from rsisys.loops.contracts.decision import OperationCall
from rsisys.loops.contracts.execution import LabToolError
from rsisys.loops.contracts.model_service import ModelRef
from rsisys.adapters.local_lab_feedback import RoutedLab
from rsisys.loops.decision_making.action_space.access import ActionPermissions, AccessRequest, SensitivePolicy
from rsisys.runtime.artifacts import FileArtifactStore
from rsisys.runtime.accounting import HostAdmissionStop
from rsisys.runtime.evolution import ActivationPort, BindingSystemActivator, EvolutionLab, EvolutionState
from rsisys.runtime.systems import SystemRegistry, SystemActivationRejected
from rsisys.snapshot.codec import decode, encode


class LiveSystemEvolutionTests(unittest.TestCase):
    def setUp(self):
        temp = TemporaryDirectory()
        self.addCleanup(temp.cleanup)
        self.root = Path(temp.name)
        self.artifacts = FileArtifactStore(self.root / 'artifacts')
        self.workspace = self.root / 'workspace'
        self.workspace.mkdir()
        self.live = {'meta.harness': {'strategy': 'original'}, 'lab.harness': {'strategy': 'original'}}
        self.tasks, self.memory, self.calls = ['pinned-debug'], ['raw failure'], [17]
        def port(name):
            return ActivationPort('fixture/v1', lambda value: deepcopy(value),
                lambda: deepcopy(self.live[name]), lambda value: self.live.__setitem__(name, deepcopy(value)))
        self.ports = {k: port(k) for k in self.live}
        self.initial = SystemVersion('initial', {})
        self.activator = BindingSystemActivator(self.ports, self.artifacts, path=self.root / 'activator.json')
        self.registry = SystemRegistry(self.initial, self.activator, path=self.root / 'registry.json')

    def candidate(self, ident, changes, role='decision_maker'):
        refs = {k: self.artifacts.put(json.dumps(v).encode(), kind='system_component') for k, v in changes.items()}
        candidate = Candidate(ident, 'base', refs, (role,), 'initial')
        self.registry.register(candidate)
        return candidate

    def test_joint_binding_activation_preserves_tasks_memory_and_call_account(self):
        self.candidate('meta', {'meta.harness': {'strategy': 'reflected'}})
        self.candidate('lab', {'lab.harness': {'strategy': 'evolved'}}, 'actor')
        version = self.registry.adopt(('meta', 'lab'), version='joint', idempotency_key='apply')
        self.assertEqual(self.live, {'meta.harness': {'strategy': 'reflected'}, 'lab.harness': {'strategy': 'evolved'}})
        self.assertEqual((self.tasks, self.memory, self.calls), (['pinned-debug'], ['raw failure'], [17]))
        self.assertEqual(self.registry.activate('joint', idempotency_key='apply'), version)
        saved = EvolutionState(self.registry).capture()
        self.live.update({name: {'strategy': 'original'} for name in self.live})
        replacement = BindingSystemActivator(self.ports, self.artifacts, path=self.root / 'activator.json')
        registry = SystemRegistry(self.initial, replacement, path=self.root / 'registry.json')
        EvolutionState(registry).restore(decode(encode(saved)))
        self.assertEqual(self.live['meta.harness'], {'strategy': 'reflected'})
        self.assertEqual(registry.current, version)

    def test_unversioned_live_mutation_cannot_be_silently_checkpointed(self):
        self.live['meta.harness'] = {'strategy': 'untracked'}
        with self.assertRaisesRegex(TypeError, 'versioned'):
            EvolutionState(self.registry).capture()

    def test_failed_joint_apply_verifies_rollback_and_does_not_claim_adoption(self):
        original = deepcopy(self.live)
        def apply(value):
            self.live['lab.harness'] = deepcopy(value)
            if value['strategy'] == 'fail':
                raise ValueError('synthetic target readiness failure')
        self.activator.ports['lab.harness'] = replace(self.ports['lab.harness'], apply=apply)
        self.candidate('joint', {'meta.harness': {'strategy': 'next'}, 'lab.harness': {'strategy': 'fail'}})
        with self.assertRaises(SystemActivationRejected):
            self.registry.activate('joint', idempotency_key='apply')
        self.assertEqual(self.registry.current, self.initial)
        self.assertEqual(self.live, original)
        self.assertIsNone(self.activator.state['pending'])

    def test_unknown_apply_is_durably_blocked_and_never_automatically_replayed(self):
        def fail(_):
            raise RuntimeError('cannot confirm apply or rollback')
        self.activator.ports['meta.harness'] = replace(self.ports['meta.harness'], apply=fail)
        self.candidate('uncertain', {'meta.harness': {'strategy': 'next'}})
        with self.assertRaises(RuntimeError):
            self.registry.activate('uncertain', idempotency_key='apply')
        self.assertIsNotNone(json.loads((self.root / 'activator.json').read_text())['pending'])
        replacement = BindingSystemActivator(self.ports, self.artifacts, path=self.root / 'activator.json')
        registry = SystemRegistry(self.initial, replacement, path=self.root / 'registry.json')
        with self.assertRaisesRegex(RuntimeError, 'reconciliation'):
            registry.activate('uncertain', idempotency_key='apply')
        self.assertEqual(self.live['meta.harness'], {'strategy': 'original'})

    def test_meta_builds_candidate_and_registration_is_idempotent_no_activation(self):
        lab = EvolutionLab(RoutedLab(SimpleNamespace(workspace=self.workspace, require_admission=lambda: None)),
                           self.registry, self.artifacts)
        (self.workspace / 'harness.json').write_text('{"strategy":"meta-built"}')
        spec = {'candidate_id': 'from-meta', 'parent_system': 'initial', 'base_version': 'base',
            'affected_roles': ['decision_maker'], 'components': {'meta.harness': 'harness.json'}}
        (self.workspace / 'candidate.json').write_text(json.dumps(spec))
        call = OperationCall('system_register', {'manifest_path': 'candidate.json'})
        result = lab.execute(call)
        self.assertEqual(lab.execute(call), result)
        self.assertEqual(self.registry.current, self.initial)
        (self.workspace / 'harness.json').write_text('{"strategy":"later-edit"}')
        with self.assertRaises(LabToolError):
            lab.execute(call)
        lab.execute(OperationCall('system_activate', {'candidate_ids': ['from-meta'],
            'version': 'active', 'idempotency_key': 'apply'}))
        self.assertEqual(self.live['meta.harness'], {'strategy': 'meta-built'})

    def test_malformed_manifests_return_correctable_errors_without_registry_mutation(self):
        lab = EvolutionLab(SimpleNamespace(workspace=self.workspace, require_admission=lambda: None),
                           self.registry, self.artifacts)
        spec = {'candidate_id': 'from-meta', 'parent_system': 'initial', 'base_version': 'base',
            'affected_roles': ['decision_maker'], 'components': {}}
        malformed = [[], None, {**spec, 'components': []}, {**spec, 'affected_roles': 'decision_maker'},
            {**spec, 'affected_roles': ['actor', 'actor']}, {**spec, 'candidate_id': []},
            {**spec, 'models': []}, {**spec, 'model_bases': {'actor': 'base'}},
            {**spec, 'models': {'actor': {'backend_id': 'x', 'model_id': 'x', 'revision': 'base'}}}]
        before = self.registry.capture_state()
        for value in malformed:
            with self.subTest(value=value):
                (self.workspace / 'candidate.json').write_text(json.dumps(value))
                with self.assertRaises(LabToolError):
                    lab.execute(OperationCall('system_register', {'manifest_path': 'candidate.json'}))
                self.assertEqual(self.registry.capture_state(), before)

    def test_ready_model_ports_are_available_to_decision_manifest_and_joint_activation(self):
        base = {name: ModelRef('fixture', name, 'base') for name in ('decision_maker', 'actor')}
        live = {name: asdict(model) for name, model in base.items()}
        def port(name):
            return ActivationPort('fixture-ready/v1', lambda raw: asdict(ModelRef(**raw)),
                lambda: deepcopy(live[name]), lambda raw: live.__setitem__(name, deepcopy(raw)))
        activator = BindingSystemActivator({}, self.artifacts, models={name: port(name) for name in base})
        registry = SystemRegistry(SystemVersion('initial', {}, models=base), activator)
        lab = EvolutionLab(SimpleNamespace(workspace=self.workspace, require_admission=lambda: None),
                           registry, self.artifacts)
        for name in base:
            spec = {'candidate_id': name, 'parent_system': 'initial', 'base_version': 'base',
                'affected_roles': [name], 'components': {},
                'models': {name: asdict(replace(base[name], revision='ready-next'))},
                'model_bases': {name: 'base'}}
            (self.workspace / 'candidate.json').write_text(json.dumps(spec))
            lab.execute(OperationCall('system_register', {'manifest_path': 'candidate.json'}))
        self.assertEqual(live, {name: asdict(model) for name, model in base.items()})
        lab.execute(OperationCall('system_activate', {'candidate_ids': list(base),
            'version': 'joint', 'idempotency_key': 'apply'}))
        self.assertEqual(live, {name: asdict(replace(model, revision='ready-next')) for name, model in base.items()})
        self.assertEqual(registry.current.models, {name: replace(model, revision='ready-next') for name, model in base.items()})

    def test_closed_admission_prevents_register_and_activation_but_status_remains_readable(self):
        def closed():
            raise HostAdmissionStop('synthetic absolute deadline')
        lab = EvolutionLab(SimpleNamespace(workspace=self.workspace, require_admission=closed),
                           self.registry, self.artifacts)
        before = self.registry.capture_state()
        for name in ('system_register', 'system_activate'):
            with self.assertRaises(HostAdmissionStop):
                lab.execute(OperationCall(name, {}))
        self.assertEqual(self.registry.capture_state(), before)
        self.assertEqual(lab.execute(OperationCall('system_status', {})).data['system']['version'], 'initial')

    def test_two_target_models_compose_but_same_target_conflicts(self):
        base = ModelRef('fixture', 'initial', 'base')
        registry = SystemRegistry(SystemVersion('initial', {}, base),
            SimpleNamespace(activate=lambda system, **kw: system))
        refs = {}
        for ident, role in [('meta', 'decision_maker'), ('actor', 'actor'), ('actor-conflict', 'actor')]:
            refs[ident] = ModelRef('fixture', ident, 'v1')
            registry.register(Candidate(ident, 'base', {}, (role,), 'initial'), model=refs[ident])
        joint = registry.adopt(('meta', 'actor'), version='joint', idempotency_key='apply')
        self.assertEqual(joint.model_for('decision_maker'), refs['meta'])
        self.assertEqual(joint.model_for('actor'), refs['actor'])
        with self.assertRaisesRegex(ValueError, 'Conflicting model'):
            registry.compose(('actor', 'actor-conflict'), version='conflict')
        legacy = encode(SystemVersion('legacy', {}, base))
        del legacy['fields']['models']
        self.assertEqual(decode(legacy).model, base)

    def test_serializable_sensitive_rules_restore_without_closing_ordinary_research(self):
        policy = ActionPermissions(check=SensitivePolicy('operator-boundaries', 'v2', (
            {'name': 'change_permissions'}, {'name': 'read_file', 'argument': 'path', 'path_prefix': '/operator'},)))
        restored = ActionPermissions.from_dict(policy.to_dict())
        for permissions in (policy, restored):
            self.assertTrue(permissions.allows(AccessRequest('decision', 'operation', 'train_arbitrary_algorithm')))
            self.assertFalse(permissions.allows(AccessRequest('decision', 'operation', 'change_permissions')))
            self.assertFalse(permissions.allows(AccessRequest('decision', 'operation', 'read_file', action=
                OperationCall('read_file', {'path': '/operator/scoring.json'}))))
        self.assertEqual(restored.to_dict(), policy.to_dict())
        with self.assertRaises(TypeError):
            ActionPermissions(check=lambda req: True).to_dict()


if __name__ == '__main__':
    unittest.main()
