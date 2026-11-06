"""Whole-method admission and recovery; no fixed algorithm phases or real ML."""
from dataclasses import replace
import json
from pathlib import Path
import tempfile
import unittest

from examples.rsi_methods import catalog as demo_catalog
from rsisys.application import Application, ApplicationConfig
from rsisys.assembly import ModuleAssembly
from rsisys.config import DemoConfig, build_demo
from rsisys.loops.decision_making.action_space import ModularDecisionActionSpace
from rsisys.loops.decision_making.action_space.access import ActionPermissions
from rsisys.loops.decision_making.loop import DecisionMakingLoop
from rsisys.methods import MethodCatalog, MethodDefinition, MethodIdentity
from rsisys.snapshot.local import LocalSnapshotCapture, restore_snapshot
from tests.test_module_assembly_state import Counter, NativeLab, Port


ROOT = Path(__file__).resolve().parents[1]


class MethodSelectionTests(unittest.TestCase):
    def test_invalid_selections_are_rejected_before_builder(self):
        calls = []
        definitions = MethodCatalog({'valid': MethodDefinition(lambda config: calls.append(config), '1'),
            'foreign': MethodDefinition(lambda config: calls.append(config), '1', protocol='foreign/v2')})
        for selection in ({}, {'method': 'missing'}, {'method': 'valid', 'unexpected': True},
                {'method': 'foreign'}, {'method': 'valid', 'config': []},
                {'method': 'valid', 'config': {'not_json': (1, 2)}},
                {'method': 'valid', 'config': {'nan': float('nan')}}):
            with self.subTest(selection=selection), self.assertRaises((TypeError, ValueError)):
                definitions.select(selection)
        self.assertEqual(calls, [])

    def test_config_is_copied_for_validator_builder_and_saved_identity(self):
        calls = []
        def validate(config):
            if config['nested']['count'] > 2:
                raise ValueError('host config bound')
            config['nested']['count'] = 99
        def build(config):
            calls.append(config['nested']['count'])
            config['nested']['count'] = 77
            return ModuleAssembly()
        raw = {'method': 'whole', 'config': {'nested': {'count': 2}}}
        definitions = {'whole': MethodDefinition(build, '1', validate_config=validate)}
        catalog = MethodCatalog(definitions)
        definitions.clear()
        modules = catalog.select(raw)
        raw['config']['nested']['count'] = 66
        manifest = modules.method_identity.configuration()
        manifest['config']['nested']['count'] = 55
        self.assertEqual(calls, [2])
        self.assertEqual(modules.configuration()['method']['config'], {'nested': {'count': 2}})
        with self.assertRaisesRegex(ValueError, 'config bound'):
            catalog.select(raw)
        self.assertEqual(calls, [2])

    def test_wrong_builder_result_and_identity_override_are_rejected(self):
        for result in (object(), ModuleAssembly(method_identity=MethodIdentity('replacement', '2', {}))):
            with self.subTest(result=result), self.assertRaises((TypeError, ValueError)):
                MethodCatalog({'whole': MethodDefinition(lambda config: result, '1')}).select({'method': 'whole'})
        with self.assertRaises(TypeError):
            ModuleAssembly(method_identity={})

    def test_whole_and_composed_methods_share_launcher_and_close_once(self):
        for name in ('whole', 'composed'):
            with self.subTest(method=name):
                modules = demo_catalog().select({'method': name, 'config': {'attempts': 2}})
                runner, _ = build_demo(DemoConfig.load(ROOT / 'configs/demo.json'), modules=modules)
                with runner:
                    lab = runner.assembly.lab_execution
                    self.assertEqual(lab.results, [])
                    self.assertEqual(runner.run('synthetic integration', 1, max_decision_steps=8).status, 'finished')
                    self.assertEqual(len(lab.results), 2)
                    self.assertEqual(lab.results[-1]['weights'], [2.0, 1.0])
                    self.assertEqual(lab.results[-1]['predictions'], [1.5, 2.5])
                    self.assertEqual(runner.assembly.method_identity.configuration()['method'], name)
                    self.assertEqual(runner.assembly.components is None, name == 'whole')
                self.assertEqual(lab.closed, 1)

    def test_both_methods_keep_host_operation_permissions(self):
        for name in ('whole', 'composed'):
            with self.subTest(method=name):
                runner, _ = build_demo(DemoConfig.load(ROOT / 'configs/demo.json'),
                    modules=demo_catalog().select({'method': name}),
                    action_permissions=ActionPermissions(operations=set()))
                with runner:
                    with self.assertRaises(PermissionError):
                        runner.step('denied operation')
                    self.assertEqual(runner.assembly.lab_execution.results, [])

    def test_incomplete_method_ports_fail_before_step_and_release_owned_resources(self):
        harness, lab = Counter(), NativeLab()
        harness.pending = None
        modules = MethodCatalog({'bad': MethodDefinition(lambda config: ModuleAssembly(
            decision_factory=lambda context, exploration:
                DecisionMakingLoop(harness, ModularDecisionActionSpace(())),
            lab_factory=lambda context: lab), '1')}).select({'method': 'bad'})
        with self.assertRaisesRegex(TypeError, 'decision harness.*pending'):
            build_demo(DemoConfig.load(ROOT / 'configs/demo.json'), modules=modules)
        self.assertEqual(harness.count, 0)
        self.assertEqual(harness.closed, 1)
        self.assertEqual(lab.closed, 1)


class MethodRecoveryTests(unittest.TestCase):
    def setUp(self):
        temporary = tempfile.TemporaryDirectory()
        self.addCleanup(temporary.cleanup)
        self.root = Path(temporary.name)
        self.config = replace(ApplicationConfig.load(ROOT / 'configs/local.json'), data_dir=self.root / 'run')
        self.created = []

    def method(self, *, name='whole', version='1', count=0, states=True, reserved=False):
        def build(config):
            def decision(context, exploration):
                harness = Counter()
                harness.count = config['count']
                self.created.append(harness)
                return DecisionMakingLoop(harness, ModularDecisionActionSpace((), permissions=context.action_permissions))
            def state_factory(context, decision, exploration):
                ports = {'decision': Port('counter/v1', lambda: decision.harness.count,
                            lambda value: setattr(decision.harness, 'count', value)),
                         'exploration': Port('empty/v1'), 'lab_execution': Port('empty/v1')}
                if reserved:
                    ports['method_manifest'] = Port('fake/v1')
                return ports
            return ModuleAssembly(decision_factory=decision, lab_factory=lambda context: NativeLab(),
                                  state_factory=state_factory if states else None)
        return MethodCatalog({name: MethodDefinition(build, version)}).select(
            {'method': name, 'config': {'count': count}})

    def test_resume_keeps_method_state_without_adding_internal_slots(self):
        modules = self.method()
        self.assertEqual(self.created, [])
        with Application(self.config, modules=modules, durable=True) as app:
            app.runner.step('one')
            app.runner.step('two')
            manifest = json.loads((self.config.data_dir / 'module_assembly.json').read_text())
            self.assertEqual(manifest['method'], modules.method_identity.configuration())
            self.assertIn('method_manifest', app.assembly.state_adapters)
            self.assertIsNone(app.assembly.components)
        self.assertEqual(self.created[-1].closed, 1)
        with Application.resume(self.config.data_dir, modules=modules) as app:
            self.assertEqual(app.runner.decision_making.harness.count, 2)
            self.assertEqual(app.assembly.lab_execution.submissions, 0)

    def test_changed_or_dropped_method_is_rejected_before_resource_factories(self):
        original = self.method()
        with Application(self.config, modules=original, durable=True):
            pass
        count = len(self.created)
        for modules in (self.method(name='other'), self.method(version='2'), self.method(count=7),
                        replace(original, method_identity=None)):
            with self.subTest(manifest=modules.configuration()), self.assertRaisesRegex(TypeError, 'original module'):
                Application.resume(self.config.data_dir, modules=modules)
            self.assertEqual(len(self.created), count)

    def test_snapshot_keeps_manifest_and_rejects_changed_method_before_construction(self):
        modules = self.method()
        with Application(self.config, modules=modules, durable=True) as app:
            app.runner.step('one')
            snapshot = LocalSnapshotCapture(app).capture('whole method')
            directory = app.config.data_dir / 'snapshots' / snapshot.snapshot_id
        self.assertIn('module_assembly.json', snapshot.file_digests)
        count = len(self.created)
        for changed in (self.method(name='other'), self.method(version='2'), self.method(count=9)):
            with self.assertRaisesRegex(TypeError, 'original modules'):
                restore_snapshot(directory, self.root / 'invalid', modules=changed)
            self.assertEqual(len(self.created), count)
            self.assertFalse((self.root / 'invalid').exists())
        with restore_snapshot(directory, self.root / 'portable', modules=modules) as app:
            self.assertEqual(app.runner.decision_making.harness.count, 1)
            self.assertEqual(app.assembly.lab_execution.submissions, 0)
            self.assertEqual(json.loads((app.config.data_dir / 'module_assembly.json').read_text()), modules.configuration())

    def test_state_ports_remain_explicit_and_cannot_replace_method_manifest(self):
        with self.assertRaisesRegex(TypeError, 'explicit state_factory'):
            Application(self.config, modules=self.method(states=False), durable=True)
        self.assertEqual(self.created, [])
        with self.assertRaisesRegex(ValueError, 'cannot replace'):
            Application(self.config, modules=self.method(reserved=True), durable=True)
        self.assertEqual(self.created[-1].closed, 1)

    def test_method_rebinding_is_rejected_before_next_durable_step(self):
        with Application(self.config, modules=self.method(), durable=True) as app:
            harness = app.runner.decision_making.harness
            app.assembly.method_identity = MethodIdentity('replacement', '1', {})
            with self.assertRaisesRegex(TypeError, 'components changed'):
                app.runner.step('must not execute')
            self.assertEqual(harness.count, 0)


if __name__ == '__main__':
    unittest.main()
