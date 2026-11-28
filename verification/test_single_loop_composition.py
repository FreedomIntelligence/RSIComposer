"""Integration checks for the single_loop Controller/Lab binding."""
from dataclasses import replace
import json
from pathlib import Path
from tempfile import TemporaryDirectory
import unittest
from unittest.mock import patch

from rsicomposer.api import (ActionPermissions, Application, ApplicationConfig, AssemblyContext,
                             ControllerDefinition, ControllerVersion, Decision, LocalSnapshotCapture,
                             MethodDefinition, OperationCall, OperationResult, OperationSpec, Registry, restore_snapshot,
                             save_controller_version)
from rsicomposer.nested_worker import _build_modules
from rsicomposer.runtime.evidence.retrieval import BasicEvidenceRepository
from rsicomposer.runtime.evidence.storage import InMemoryExperienceStore
from rsicomposer.runtime.execution.sandbox import MockSandboxBackend
from rsicomposer.runtime.storage.artifacts import InMemoryArtifactStore


class BareLab:
    def describe(self):
        return (OperationSpec('demo.measure', 'Measure', {}), OperationSpec('inspect', 'Inspect', {}))

    def execute(self, call):
        raise AssertionError('Constructing or adopting a Controller should not dispatch Lab work')


class CatalogPolicy:
    def __init__(self, context, config):
        self.catalog = context.lab_catalog

    def decide(self, window):
        return Decision('finish')

    def state(self):
        return {}

    def restore(self, state):
        assert state == {}


class TrialPolicy(CatalogPolicy):
    def decide(self, window):
        if not window.feedbacks:
            return Decision.invoke(OperationCall('trial.measure', {}))
        return Decision('finish')


class MarkedLab:
    def __init__(self, context, config):
        self.label = config.get('label', 'private default')
        self.calls = 0

    def describe(self):
        return (OperationSpec('measure', self.label, {}),)

    def execute(self, call):
        self.calls += 1
        return OperationResult(self.label, (), True)

    def state(self):
        return {'calls': self.calls, 'label': self.label}

    def restore(self, state):
        if state['label'] != self.label:
            raise ValueError('restored a different Lab')
        self.calls = state['calls']


class SingleLoopCompositionTest(unittest.TestCase):
    def launch_inputs(self, data_dir, operation, calls):
        class FixtureLab:
            def describe(self):
                return (OperationSpec('evaluate', 'Fixture evaluation', {'type': 'object'}),)

            def execute(self, call):
                calls.append((call.name, call.arguments))
                return OperationResult('fixture evaluated', (), True)

        registry = Registry()
        registry.register_lab('fixture', lambda context, config: FixtureLab(), version='1')
        modules = registry.build({
            'controller': {'name': 'single_loop', 'config': {
                'operation': operation, 'iterations': 2, 'arguments': {'prompt': 'baseline'},
            }},
            'labs': {'math': {'name': 'fixture'}},
        })
        application = ApplicationConfig('Check composed operations', data_dir, {}, {}, max_decision_steps=8)
        return application, modules

    def test_supported_operation_runs_without_changing_dispatch(self):
        with TemporaryDirectory() as temp_dir:
            calls = []
            config, modules = self.launch_inputs(Path(temp_dir), 'math.evaluate', calls)
            with Application(config, modules=modules) as application:
                result = application.run(wait_for_jobs=True)
            self.assertEqual(result.status, 'finished')
            self.assertEqual(calls, [('evaluate', {'prompt': 'baseline'})] * 2)

    def test_wrong_alias_or_operation_is_rejected_during_startup(self):
        for operation in ('other.evaluate', 'math.unknown'):
            with self.subTest(operation=operation), TemporaryDirectory() as temp_dir:
                calls = []
                config, modules = self.launch_inputs(Path(temp_dir), operation, calls)
                with self.assertRaises(ValueError) as caught:
                    with Application(config, modules=modules):
                        pass
                self.assertEqual(str(caught.exception), f'Controller requires missing Lab operations: {operation}')
                self.assertEqual(calls, [])

    def test_bare_lab_catalog_is_available_on_initial_build_and_adoption(self):
        registry = Registry(builtins=False)
        registry.register_controller('catalog', CatalogPolicy, version='1')
        registry.register_lab('fixture', lambda *_: BareLab(), version='1')
        modules = registry.build({'controller': {'name': 'catalog'}, 'labs': {'test': {'name': 'fixture'}}})
        modules = replace(modules, lab_factory=lambda ctx: BareLab(), state_factory=None)
        with TemporaryDirectory() as directory:
            with Application(ApplicationConfig('catalog', Path(directory), {}, {}), modules=modules) as app:
                host = app.assembly.controller_host
                initial = host.active.policy.catalog
                version = ControllerVersion('new-catalog', {'name': 'catalog'})
                host.request_version(save_controller_version(app.artifacts, version))
                host.advance()
                self.assertEqual(host.version_id, version.version_id)
                for catalog in (initial, host.active.policy.catalog):
                    self.assertEqual(catalog.aliases, ('demo',))
                    self.assertEqual([spec.name for spec in catalog.describe()], ['demo.measure', 'inspect'])
                    with self.assertRaisesRegex(KeyError, 'declares no primary metric'):
                        catalog.primary_metric('demo')

    def test_direct_nested_lab_requires_explicit_parent_configuration(self):
        modules = Registry().build({
            'controller': {'name': 'single_loop', 'config': {'operation': 'reflect.inspect'}},
            'labs': {'reflect': {'name': 'self_reflection'}},
        })
        with TemporaryDirectory() as directory:
            for configured in (False, True):
                with self.subTest(configured=configured):
                    root = Path(directory) / str(configured)
                    config = ApplicationConfig('nested', root, {}, {})
                    parent = {'application': config.to_dict(), 'registry_source': modules.registry_source}
                    context = AssemblyContext(BasicEvidenceRepository(InMemoryExperienceStore()),
                        InMemoryArtifactStore(), {}, MockSandboxBackend(), {}, data_dir=root,
                        host_config=parent if configured else None)
                    try:
                        if configured:
                            with modules.build(context):
                                self.assertIn('reflect.inspect', {s.name for s in context.lab_execution.describe()})
                                manifest = context.nested_manifest({'name': 'single_loop',
                                    'config': {'operation': 'reflect.inspect'}}, None)
                                expected = {**modules.registration.configuration(), 'controller': {
                                    'name': 'single_loop', 'config': {'operation': 'reflect.inspect'}}}
                                self.assertEqual(manifest['selection'], expected)
                                self.assertEqual(manifest['plugins'], list(modules.plugin_paths))
                        else:
                            with self.assertRaisesRegex(ValueError, 'explicit host_config'):
                                modules.build(context)
                    finally:
                        context.close()

    def private_parent(self, root, *, builtins=True, only_current=False, permissions=None, named_trial=False):
        # This method's default Lab and Controller closure have no registered names.
        plugin = root / 'registration.py'
        plugin.write_text(f'''from rsicomposer.api import MethodDefinition
from {__name__} import CatalogPolicy, MarkedLab, TrialPolicy

def register(registry):
    def private_config(options):
        if options:
            raise ValueError('private Controller expects empty options')
    def lab_config(options):
        if set(options) != {{'label'}}:
            raise ValueError('missing Lab label')
    if {named_trial!r}:
        registry.register_controller('named_trial', TrialPolicy, version='1')
    registry.register_lab('marked', MarkedLab, version='1', validate_config=lab_config)
    registry.register_method('private', MethodDefinition.from_factories(
        controller=lambda ctx, cfg: CatalogPolicy(ctx, cfg), labs={{'original': MarkedLab}},
        version='1', validate_controller=private_config))
''')
        registry = Registry(builtins=builtins)
        registry.load_plugin(plugin)
        modules = registry.build({'method': {'name': 'private'}})
        if only_current:
            modules = replace(modules, controller_catalog={'current': modules.controller_catalog['current']},
                              controller_validators={'current': modules.controller_validators['current']})
        return Application(ApplicationConfig('parent', root / 'parent', {}, {}), modules=modules,
                           action_permissions=permissions)

    @staticmethod
    def nested_app(parent, directory, controller, labs=None):
        manifest = parent.assembly.nested_manifest(controller, labs)
        modules, version = _build_modules(manifest)
        config = ApplicationConfig('nested', directory, {}, {}, parent_run_id=parent.config.run_id)
        return Application(config, modules=modules, _initial_controller_version=version)

    def test_grandchild_runs_parent_lab_override_under_parent_current(self):
        with TemporaryDirectory() as temp:
            root = Path(temp)
            options = {'operation': 'trial.measure'}
            selection = {'name': 'single_loop', 'config': options}
            override = {'trial': {'name': 'marked', 'config': {'label': 'child selection'}}}
            with self.private_parent(root) as parent:
                with self.assertRaisesRegex(ValueError, 'missing Lab label'):
                    parent.assembly.nested_manifest(selection, {'trial': {'name': 'marked'}})
                with self.nested_app(parent, root / 'child', selection, override) as child:
                    with self.assertRaisesRegex(ValueError, 'single_loop requires'):
                        child.assembly.nested_manifest({'name': 'current'}, None)
                    inherited = {'name': 'current', 'config': options}
                    with self.nested_app(child, root / 'grandchild', inherited) as grandchild:
                        for app in (child, grandchild):
                            self.assertEqual(set(app.modules.controller_catalog), {'current', 'single_loop'})
                            self.assertEqual(app.assembly.lab_execution.aliases, ('trial',))
                            self.assertEqual(app.run(wait_for_jobs=True).status, 'finished')
                            self.assertEqual(app.assembly.host_execution.state.operation_results[-1].summary,
                                             'child selection')
                        # All inherited slots stay flat; only the explicitly selected alias changes.
                        self.assertEqual(grandchild.modules.registration.configuration(),
                            {**child.modules.registration.configuration(), 'controller': inherited})
                        self.assertEqual(grandchild.modules.plugin_paths, child.modules.plugin_paths)
                    # Admission in the next generation still checks the original plugin digest.
                    with (root / 'registration.py').open('a') as plugin:
                        plugin.write('\n# changed\n')
                    with self.assertRaisesRegex(ValueError, 'plugin source changed'):
                        child.assembly.nested_manifest(inherited, None)

    def test_private_current_survives_inherited_and_replaced_labs(self):
        current = {'name': 'current'}
        override = {'trial': {'name': 'marked', 'config': {'label': 'child selection'}}}
        for labs in (None, override):
            with self.subTest(labs=labs), TemporaryDirectory() as temp:
                root = Path(temp)
                with self.private_parent(root) as parent, self.nested_app(parent, root / 'child', current, labs) as child:
                    with self.assertRaisesRegex(ValueError, 'private Controller expects'):
                        child.assembly.nested_manifest({'name': 'current', 'config': {'bad': True}}, None)
                    with self.nested_app(child, root / 'grandchild', current) as grandchild:
                        policy = grandchild.assembly.controller_host.active.policy
                        self.assertIsInstance(policy, CatalogPolicy)
                        self.assertEqual(policy.catalog.aliases, ('original' if labs is None else 'trial',))
                        self.assertEqual(grandchild.run(wait_for_jobs=True).status, 'finished')
                        visible = set(parent.modules.controller_catalog) if labs is None else {'current'}
                        self.assertEqual(set(grandchild.modules.controller_catalog), visible)

    def test_python_controller_and_labs_inherit_without_a_registered_root(self):
        with TemporaryDirectory() as temp:
            root = Path(temp)
            modules = MethodDefinition.from_factories(controller=CatalogPolicy,
                labs={'original': MarkedLab}, version='1').factory({})
            with Application(ApplicationConfig('parent', root / 'parent', {}, {}), modules=modules) as parent:
                override = {'tools': {'name': 'self_reflection'}}
                with self.nested_app(parent, root / 'child', ControllerDefinition(CatalogPolicy), override) as child:
                    with self.nested_app(child, root / 'grandchild', {'name': 'nested_python'}) as grandchild:
                        policy = grandchild.assembly.controller_host.active.policy
                        self.assertIsInstance(policy, CatalogPolicy)
                        self.assertEqual(policy.catalog.aliases, ('tools',))
                        self.assertEqual(grandchild.run(wait_for_jobs=True).status, 'finished')
                        self.assertEqual(set(grandchild.modules.controller_catalog), {'current', 'nested_python'})

    def test_inheritance_does_not_widen_host_catalog_or_enable_disabled_builtins(self):
        current = {'name': 'current'}
        for builtins in (False, True):
            with self.subTest(builtins=builtins), TemporaryDirectory() as temp:
                root = Path(temp)
                with self.private_parent(root, builtins=builtins, only_current=builtins) as parent:
                    with self.nested_app(parent, root / 'child', current) as child:
                        with self.nested_app(child, root / 'grandchild', current) as grandchild:
                            for app in (parent, child, grandchild):
                                self.assertEqual(set(app.modules.controller_catalog), {'current'})
                                with self.assertRaisesRegex(ValueError, 'unknown Controller'):
                                    app.assembly.nested_manifest({'name': 'single_loop'}, None)
                                if not builtins:
                                    with self.assertRaisesRegex(ValueError, 'Unknown Lab: self_reflection'):
                                        app.assembly.nested_manifest(current, {'trial': {'name': 'self_reflection'}})

    @staticmethod
    def snapshot(app):
        capture = LocalSnapshotCapture(app)
        return capture.store.directory(capture.capture('selection').snapshot_id)

    def test_portable_registered_method_requires_matching_caller_plugin_not_original_path(self):
        with TemporaryDirectory() as temp:
            root = Path(temp)
            with self.private_parent(root, builtins=False, permissions=ActionPermissions([])) as original:
                snapshot = self.snapshot(original)
                selection = original.modules.registration.configuration()
                self.assertFalse(selection['builtins'])
                self.assertEqual(selection['catalog'], {'current': 'current'})
                self.assertNotIn(str(root), (snapshot / 'module_assembly.json').read_text())
                run_id = original.config.run_id

            with self.assertRaisesRegex(ValueError, 'original registration plugins'):
                restore_snapshot(snapshot, root / 'missing')
            relocated = root / 'other' / 'registration.py'
            relocated.parent.mkdir()
            original_source = (root / 'registration.py').read_bytes()
            tripwire = root / 'unverified-registration-ran'
            relocated.write_text(f'from pathlib import Path\nPath({str(tripwire)!r}).touch()\n')
            with self.assertRaisesRegex(ValueError, 'plugin source changed'):
                restore_snapshot(snapshot, root / 'changed', plugins=[relocated])
            self.assertFalse(tripwire.exists())
            self.assertFalse((root / 'changed').exists())

            relocated.write_bytes(original_source)
            with restore_snapshot(snapshot, root / 'restored', plugins=[relocated]) as restored:
                self.assertEqual(restored.config.run_id, run_id)
                self.assertEqual(restored.modules.registration.configuration(), selection)
                self.assertEqual(restored.modules.plugin_paths, (str(relocated.resolve()),))
                self.assertEqual(restored.assembly.lab_execution.aliases, ('original',))
                self.assertEqual(restored.modules.controller_catalog.keys(), {'current'})
                self.assertEqual(restored.assembly.action_permissions, ActionPermissions([]))
                nested = restored.assembly.nested_manifest({'name': 'current'}, None)
                self.assertEqual(nested['action_permissions'], {'operations': []})

    def test_snapshot_rebuilds_effective_nested_controller_labs_and_catalog(self):
        with TemporaryDirectory() as temp:
            root = Path(temp)
            chosen = {'name': 'named_trial'}
            override = {'trial': {'name': 'marked', 'config': {'label': 'not the parent Lab'}}}
            with self.private_parent(root, named_trial=True) as parent, self.nested_app(
                    parent, root / 'child', chosen, override) as child:
                self.assertEqual(child.run(wait_for_jobs=True).status, 'finished')
                snapshot = self.snapshot(child)
                initial_id = child.assembly.controller_host.version_id
                selection = child.modules.registration.configuration()
            with restore_snapshot(snapshot, root / 'restored', plugins=[root / 'registration.py']) as restored:
                self.assertEqual(restored.modules.registration.configuration(), selection)
                self.assertEqual(restored.assembly.lab_execution.aliases, ('trial',))
                self.assertEqual(restored.assembly.controller_host.version_id, initial_id)
                self.assertIsInstance(restored.assembly.controller_host.active.policy, TrialPolicy)
                self.assertEqual(restored.modules.controller_catalog.keys(), {'current', 'named_trial'})
                self.assertEqual(restored.assembly.host_execution.state.operation_results[-1].summary, 'not the parent Lab')
                with self.nested_app(restored, root / 'grandchild', chosen) as fresh:
                    self.assertFalse(fresh.assembly.host_execution.state.operation_results)
                    self.assertEqual(fresh.run(wait_for_jobs=True).status, 'finished')

    def test_portable_snapshot_never_imports_embedded_python_definitions(self):
        with TemporaryDirectory() as temp:
            root = Path(temp)
            with self.private_parent(root) as parent, self.nested_app(parent, root / 'child',
                    ControllerDefinition(CatalogPolicy), {'trial': {'name': 'marked', 'config': {'label': 'test'}}}) as child:
                modules = child.modules
                snapshot = self.snapshot(child)
            with patch('rsicomposer.registry.load_factory', side_effect=AssertionError('untrusted factory import')):
                with self.assertRaisesRegex(TypeError, 'original explicit ModuleAssembly'):
                    restore_snapshot(snapshot, root / 'unsafe', plugins=[root / 'registration.py'])
                self.assertFalse((root / 'unsafe').exists())
                with restore_snapshot(snapshot, root / 'explicit', modules=modules) as restored:
                    self.assertIsInstance(restored.assembly.controller_host.active.policy, CatalogPolicy)

    def test_registered_pair_and_fixed_method_use_same_selection_resolver(self):
        with TemporaryDirectory() as temp:
            root = Path(temp)
            plugin = root / 'fixed.py'
            plugin.write_text('''from rsicomposer.api import ModuleAssembly, MethodDefinition
def register(registry):
    def build(config):
        def decision(ctx, lab):
            raise AssertionError('selecting a method must not launch its Controller')
        return ModuleAssembly(decision_factory=decision)
    registry.register_method('fixed', MethodDefinition(build, 'v1'))
''')
            registry = Registry()
            registry.load_plugin(plugin)
            for raw in ({'method': {'name': 'fixed'}}, {'controller': {'name': 'single_loop', 'config': {
                    'operation': 'r.inspect'}}, 'labs': {'r': {'name': 'self_reflection'}}}):
                with self.subTest(selection=raw):
                    modules = registry.build(raw)
                    selection = modules.configuration()['registry_selection']
                    self.assertEqual(Registry.reconstruct(selection, plugins=[plugin]).configuration(), modules.configuration())
                    self.assertEqual(registry.build(selection['framework']).configuration(), modules.configuration())
                    self.assertNotIn(str(root), json.dumps(selection))

    def test_python_replacement_cannot_be_replayed_as_original_registry_selection(self):
        with TemporaryDirectory() as temp:
            root = Path(temp)
            with self.private_parent(root) as parent:
                described = parent.modules.registration
                described.catalog.clear()
                self.assertIn('current', parent.modules.registration.catalog)
                original = parent.modules.lab_factory

                def custom(context):
                    return original(context)

                changed = replace(parent.modules, lab_factory=custom)
                self.assertIsNone(changed.registry_source)
                self.assertNotIn('registry_selection', changed.configuration())
                with Application(ApplicationConfig('python', root / 'python', {}, {}), modules=changed) as app:
                    snapshot = self.snapshot(app)
            with self.assertRaisesRegex(TypeError, 'original explicit ModuleAssembly'):
                restore_snapshot(snapshot, root / 'wrong', plugins=[root / 'registration.py'])
            with restore_snapshot(snapshot, root / 'right', modules=changed) as restored:
                self.assertEqual(restored.assembly.lab_execution.aliases, ('original',))

    def test_nested_admission_refuses_nonportable_permission_callbacks(self):
        with TemporaryDirectory() as temp, self.private_parent(Path(temp), permissions=ActionPermissions(
                check=lambda request: False)) as parent:
            with self.assertRaisesRegex(TypeError, 'serializable ActionPermissions'):
                parent.assembly.nested_manifest({'name': 'current'}, None)


class MethodBindingTest(unittest.TestCase):
    data = Path(__file__).resolve().parents[1] / 'src/rsicomposer/algorithms/meta_rsi/configs/cpu/data'

    def options(self, *, overlap):
        roles = {'train': 'train', 'search': 'search', 'validation': 'validation'}
        parent = {'manifest': str(self.data / 'manifest.json'), 'split_roles': {**roles, 'test': 'test'}}
        child = ({'manifest': parent['manifest'], 'split_roles': roles} if overlap else
                 {'manifest': str(self.data / 'meta-manifest.json'),
                  'split_roles': {**roles, 'search': 'meta_search', 'validation': 'meta_validation'}})
        return {'meta_targets': ['schedule'], 'trial_lab': child, 'trial_budget': {'seconds': 24}}, parent

    @staticmethod
    def modules(style, controller, math, name='meta_rsi'):
        if style == 'pair':
            selection = {'controller': {'name': name, 'config': controller}, 'labs': {
                'math': {'name': 'meta_rsi', 'config': math}, 'reflect': {'name': 'self_reflection'}}}
        else:
            selection = {'method': {'name': name, 'config': {'controller': controller,
                         'labs': {'math': math, 'reflect': {}}}}}
        return Registry().build(selection)

    def test_independent_validation_is_required_for_both_compositions(self):
        for style in ('method', 'pair'):
            with self.subTest(style=style), TemporaryDirectory() as temp:
                cfg = ApplicationConfig('matched trials', Path(temp) / 'invalid', {}, {})
                with self.assertRaisesRegex(ValueError, 'must not overlap'):
                    with Application(cfg, modules=self.modules(style, *self.options(overlap=True))):
                        self.fail('A trial sharing its parent measurements was mounted')
                with Application(replace(cfg, data_dir=Path(temp) / 'valid'),
                                 modules=self.modules(style, *self.options(overlap=False))) as app:
                    self.assertEqual(app.assembly.lab_execution.aliases, ('math', 'reflect'))

    def test_adoption_rechecks_against_the_live_labs(self):
        bad, math = self.options(overlap=True)
        with TemporaryDirectory() as temp, Application(ApplicationConfig('replace', Path(temp), {}, {}),
                modules=self.modules('pair', {}, math)) as app:
            host = app.assembly.controller_host
            old = host.active
            version = ControllerVersion('overlapping', {'name': 'meta_rsi', 'config': bad})
            host.request_version(save_controller_version(app.artifacts, version))
            with self.assertRaisesRegex(ValueError, 'must not overlap'):
                host.advance()
            self.assertIs(host.active, old)

    def test_model_paired_release_must_respect_actual_controller_margin(self):
        _, math = self.options(overlap=False)
        math['split_roles'].pop('test')
        math.update(model_driven=True, aime={}, paper_kernel={'min_gain': 0.1})
        controller = {'campaign': {'track': 'adaptive'}, 'paper_kernel': True,
                      'warmup': 0, 'steps': 2, 'min_delta': 0.2}
        for style in ('method', 'pair'):
            with self.subTest(style=style), TemporaryDirectory() as temp:
                cfg = ApplicationConfig('release gate', Path(temp) / 'invalid', {}, {})
                with self.assertRaisesRegex(ValueError, 'weaker than the controller margin'):
                    with Application(cfg, modules=self.modules(style, controller, math, 'meta_rsi_model')):
                        self.fail('Weak release bound')
                with Application(replace(cfg, data_dir=Path(temp) / 'valid'), modules=self.modules(
                        style, {**controller, 'min_delta': 0.1}, math, 'meta_rsi_model')) as app:
                    self.assertIsNotNone(app.assembly.controller_host)
