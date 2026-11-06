"""Independent composition and real host boundaries; synthetic CPU checks only."""
from copy import deepcopy
from dataclasses import replace
from itertools import product
import json
from pathlib import Path
import sys
import tempfile
import time
import unittest

from examples.modular_research import catalog, modules as demo_modules, TracingHarness
from rsisys.application import Application, ApplicationConfig
from rsisys.adapters.training import ProcessTrainingBackend, TrainingCommand
from rsisys.assembly import AssemblyContext, ModuleAssembly
from rsisys.components import ComponentDefinition, ComponentGraph, ComponentInstances
from rsisys.config import DemoConfig, build_demo
from rsisys.loops.decision_making.action_space import ModularDecisionActionSpace
from rsisys.loops.decision_making.action_space.access import ActionPermissions
from rsisys.loops.decision_making.loop import DecisionMakingLoop
from rsisys.loops.contracts.jobs import JobStatus
from rsisys.loops.contracts.model_service import ModelRef
from rsisys.loops.exploration.harness.learner import ActorTrainingRequest
from rsisys.snapshot.local import LocalSnapshotCapture, restore_snapshot
from tests.test_module_assembly_state import Counter, NativeLab, Port
from tests.test_framework_adapters import FIXTURE

ROOT = Path(__file__).resolve().parents[1]


class ComponentGraphTests(unittest.TestCase):
    def setUp(self):
        self.calls = []

    def definition(self, provides=('value/v1',), requires=None, **options):
        def create(context, dependencies, config):
            self.calls.append((dependencies, config))
            return {'config': config, 'dependencies': dependencies}
        return ComponentDefinition(create, '1', provides, requires=requires or {}, **options)

    def context(self):
        return AssemblyContext(None, None, {}, None, {})

    def test_invalid_graphs_fail_before_constructing_any_resource(self):
        definitions = {'value': self.definition(),
            'dependent': self.definition(requires={'input': 'value/v1'})}
        cases = [
            {'a': {'component': 'not_installed'}},
            {'a': {'component': 'value', 'unexpected': True}},
            {'a': {'component': 'dependent'}},
            {'a': {'component': 'dependent', 'dependencies': {'input': 'missing'}}},
            {'a': {'component': 'value', 'dependencies': {'input': 'a'}}},
            {'a': {'component': 'dependent', 'dependencies': {'input': 'a'}}},
            {'a': {'component': 'dependent', 'dependencies': {'input': 'b'}},
             'b': {'component': 'dependent', 'dependencies': {'input': 'a'}}},
            {'a': {'component': 'value', 'config': {'bad': float('nan')}}},
            {'a': {'component': 'value', 'config': {'not_json': (1, 2)}}},
        ]
        for selections in cases:
            with self.subTest(selections=selections), self.assertRaises((ValueError, TypeError)):
                ComponentGraph(definitions, selections)
        incompatible = {'foreign': self.definition(provides=('other/v1',)), **definitions}
        with self.assertRaisesRegex(ValueError, 'Incompatible'):
            ComponentGraph(incompatible, {'a': {'component': 'foreign'},
                'b': {'component': 'dependent', 'dependencies': {'input': 'a'}}})
        self.assertEqual(self.calls, [])

    def test_native_config_validation_is_preflight_and_cannot_rewrite_selection(self):
        def check(config):
            if config.get('steps', 1) > 10:
                raise ValueError('steps outside host limit')
            config['mutated'] = True
        definition = self.definition(validate_config=check)
        with self.assertRaisesRegex(ValueError, 'host limit'):
            ComponentGraph({'impl': definition}, {'slot': {'component': 'impl', 'config': {'steps': 11}}})
        self.assertEqual(self.calls, [])
        graph = ComponentGraph({'impl': definition}, {'slot': {'component': 'impl', 'config': {'steps': 2}}})
        self.assertEqual(graph.build(self.context())['slot']['config'], {'steps': 2})

    def test_bindings_are_ordered_per_run_and_configuration_is_copied(self):
        raw = {'a': {'component': 'dependent', 'dependencies': {'input': 'z'}},
               'z': {'component': 'value', 'config': {'items': [1]}}}
        definitions = {'value': self.definition(stateless=True),
            'dependent': self.definition(requires={'input': 'value/v1'}, stateless=True)}
        graph = ComponentGraph(definitions, raw)
        raw['z']['config']['items'].append(2)
        definitions.clear()
        one, two = graph.build(self.context()), graph.build(self.context())
        self.assertIs(one['a']['dependencies']['input'], one['z'])
        self.assertIsNot(one['z'], two['z'])
        one['z']['config']['items'].append(3)
        self.assertEqual(two['z']['config']['items'], [1])
        self.assertEqual(graph.configuration()['components']['z']['config'], {'items': [1]})
        with self.assertRaises(TypeError):
            one['replacement'] = object()

    def test_failure_cleans_owned_dependencies_in_reverse_order_with_retry_barrier(self):
        closed = []
        class Resource:
            def __init__(self, name):
                self.name, self.fail = name, name == 'child'
            def close(self):
                closed.append(self.name)
                if self.fail:
                    self.fail = False
                    raise RuntimeError('close still pending')
        def resource(name, requires=None):
            return ComponentDefinition(lambda c, d, p: Resource(name), '1', ('resource/v1',),
                requires=requires or {}, owned=True)
        def fail(context, dependencies, config):
            raise RuntimeError('construction failed')
        graph = ComponentGraph({'root': resource('root'), 'child': resource('child', {'parent': 'resource/v1'}),
            'fail': ComponentDefinition(fail, '1', ('unused/v1',), {'parent': 'resource/v1'})}, {
                'root': {'component': 'root'},
                'child': {'component': 'child', 'dependencies': {'parent': 'root'}},
                'last': {'component': 'fail', 'dependencies': {'parent': 'child'}}})
        context = self.context()
        with self.assertRaisesRegex(RuntimeError, 'construction failed'):
            graph.build(context)
        with self.assertRaisesRegex(RuntimeError, 'close still pending'):
            context.close()
        self.assertEqual(closed, ['child'])
        context.close()
        self.assertEqual(closed, ['child', 'child', 'root'])
        context.close()
        self.assertEqual(len(closed), 3)

    def test_instance_validation_failure_retains_its_owned_cleanup(self):
        resource = Counter()
        def invalid(value):
            raise ValueError('wrong actual interface')
        context = self.context()
        graph = ComponentGraph({'bad': ComponentDefinition(lambda c, d, p: resource,
            '1', ('counter/v1',), validate=invalid, owned=True)}, {'slot': {'component': 'bad'}})
        with self.assertRaisesRegex(ValueError, 'actual interface'):
            graph.build(context)
        context.close()
        self.assertEqual(resource.closed, 1)

    def test_closed_context_cannot_acquire_components(self):
        context = self.context()
        context.close()
        graph = ComponentGraph({'value': self.definition()}, {'slot': {'component': 'value'}})
        with self.assertRaisesRegex(RuntimeError, 'Assembly is closed'):
            graph.build(context)
        self.assertEqual(self.calls, [])

    def test_duplicate_owned_resource_is_closed_once(self):
        context, resource = self.context(), Counter()
        graph = ComponentGraph({'shared': ComponentDefinition(lambda c, d, p: resource,
            '1', ('counter/v1',), owned=True)}, {
                'first': {'component': 'shared'}, 'second': {'component': 'shared'}})
        with self.assertRaisesRegex(ValueError, 'multiple owners'):
            graph.build(context)
        context.close()
        self.assertEqual(resource.closed, 1)


class IndependentCompositionTests(unittest.TestCase):
    def setUp(self):
        self.selections = json.loads((ROOT / 'configs/modular-demo.json').read_text())
        self.demo = DemoConfig.load(ROOT / 'configs/demo.json')

    def test_32_combinations_change_only_the_selected_behavior_without_launcher_changes(self):
        for training, inference, rsi, harness, backend in product(
                ('gradient', 'analytic'), ('direct', 'rounded'), ('single_pass', 'repeat'),
                ('default_harness', 'tracing_harness'), ('python_vectors', 'precise_vectors')):
            names = (training, inference, rsi, harness, backend)
            with self.subTest(names=names):
                selection = deepcopy(self.selections)
                for slot, name in zip(('training', 'inference', 'rsi', 'harness', 'infer_engine'), names):
                    selection[slot]['component'] = name
                runner, _ = build_demo(self.demo, modules=demo_modules(selection))
                with runner:
                    lab = runner.assembly.lab_execution
                    self.assertEqual(lab.results, [])
                    result = runner.run('synthetic composition check', 1, max_decision_steps=8)
                    self.assertEqual(result.status, 'finished')
                    self.assertEqual(len(lab.results), 1 if rsi == 'single_pass' else 2)
                    row = lab.results[-1]
                    if training == 'analytic':
                        self.assertEqual(row['weights'], [2.0, 1.0])
                    else:
                        self.assertAlmostEqual(row['weights'][0], 1.993466983941497)
                    raw = [row['weights'][0] * x + row['weights'][1] for x in (0.25, 0.75)]
                    self.assertEqual(row['predictions'], [float(round(x)) for x in raw]
                        if inference == 'rounded' else raw)
                    self.assertEqual(runner.assembly.components['infer_engine'].precise, backend == 'precise_vectors')
                    self.assertFalse(runner.assembly.components['train_engine'].precise)
                    if harness == 'tracing_harness':
                        self.assertIsInstance(runner.decision_making.harness, TracingHarness)
                        self.assertGreaterEqual(runner.decision_making.harness.steps, 2)
                self.assertEqual(lab.closed, 1)

    def test_swapping_rsi_policy_does_not_bypass_host_permission_checks(self):
        runner, _ = build_demo(self.demo, modules=demo_modules(self.selections),
            action_permissions=ActionPermissions(operations=set()))
        with runner:
            with self.assertRaises(PermissionError):
                runner.step('disallowed synthetic operation')
            self.assertEqual(runner.assembly.lab_execution.results, [])

    def test_host_binding_requires_actual_lab_interface(self):
        definitions = {'decision': ComponentDefinition(lambda c, d, p: lambda c, e: None,
            '1', ('decision-factory/v1',)),
            'bad_lab': ComponentDefinition(lambda c, d, p: object(), '1', ('lab-execution/v1',))}
        graph = ComponentGraph(definitions, {'decision': {'component': 'decision'}, 'lab': {'component': 'bad_lab'}})
        with self.assertRaisesRegex(TypeError, 'assembly binding'):
            build_demo(self.demo, modules=ModuleAssembly.from_components(graph, decision='decision', lab='lab'))


class NativeAdapterCompositionTests(unittest.TestCase):
    def test_strategy_changes_without_changing_process_engine_or_launching_during_build(self):
        # A real subprocess validates the native bridge, not real ML training.
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            script = root / 'fixture.py'
            script.write_text(FIXTURE)
            def strategy(name):
                return ComponentDefinition(lambda c, d, p: lambda request, directory:
                    TrainingCommand((sys.executable, str(script), name), str(directory), timeout_seconds=5),
                    '1', ('training-command/v1',), stateless=True)
            definitions = {'first': strategy('first'), 'second': strategy('second'),
                'process': ComponentDefinition(lambda c, d, p:
                    ProcessTrainingBackend(c.data_dir / 'jobs', command=d['command']),
                    '1', ('training-backend/v1',), requires={'command': 'training-command/v1'})}
            for name in ('first', 'second'):
                with self.subTest(strategy=name):
                    context = AssemblyContext(None, None, {}, None, {}, data_dir=root / name)
                    components = ComponentGraph(definitions, {
                        'strategy': {'component': name},
                        'engine': {'component': 'process', 'dependencies': {'command': 'strategy'}}}).build(context)
                    backend, job = components['engine'], None
                    try:
                        self.assertEqual(list(backend.root.iterdir()), [])
                        request = ActorTrainingRequest(ModelRef('fixture', 'base', 'r0'), (), 'native.json')
                        job = backend.submit(request, idempotency_key='composition')
                        deadline = time.monotonic() + 8
                        while time.monotonic() < deadline:
                            outcome = backend.poll(job)
                            if outcome.terminal or outcome.status == JobStatus.UNKNOWN:
                                break
                            time.sleep(.025)
                        self.assertEqual(outcome.status, JobStatus.SUCCEEDED)
                        self.assertEqual(backend.submit(request, idempotency_key='composition'), job)
                        directory = backend.root / job.job_id
                        self.assertEqual(json.loads((directory / 'argv.json').read_text()), [name])
                        self.assertEqual((directory / 'starts').read_text(), 'start\n')
                        self.assertEqual(backend.result(job).model.revision, 'r1')
                    finally:
                        # This backend has job-level cancellation, not close().
                        # Its owning Lab must manage jobs in a production graph.
                        if job is not None:
                            backend.cancel(job)
                        context.close()


class ComponentRecoveryTests(unittest.TestCase):
    def setUp(self):
        temporary = tempfile.TemporaryDirectory()
        self.addCleanup(temporary.cleanup)
        self.root = Path(temporary.name)
        self.config = replace(ApplicationConfig.load(ROOT / 'configs/local.json'), data_dir=self.root / 'run')
        self.created = []

    def modules(self, *, initial=0, version='1', decision='decision', missing=False):
        def counter(context, dependencies, config):
            value = Counter()
            value.count = config.get('initial', 0)
            self.created.append(value)
            return value
        def decision_factory(context, dependencies, config):
            return lambda c, e: DecisionMakingLoop(dependencies['policy'],
                ModularDecisionActionSpace((), permissions=c.action_permissions))
        graph = ComponentGraph({
            'counter': ComponentDefinition(counter, version, ('rsi-fixture/v1',), owned=True),
            'decision': ComponentDefinition(decision_factory, '1', ('decision-factory/v1',),
                {'policy': 'rsi-fixture/v1'}, stateless=True),
            'lab': ComponentDefinition(lambda c, d, p: NativeLab(), '1', ('lab-execution/v1',),
                owned=True, stateless=True)}, {
            'counter': {'component': 'counter', 'config': {'initial': initial}},
            'decision': {'component': 'decision', 'dependencies': {'policy': 'counter'}},
            'other': {'component': 'decision', 'dependencies': {'policy': 'counter'}},
            'lab': {'component': 'lab'}})
        def states(context, decision_loop, exploration):
            instance = context.components['counter']
            ports = {'decision': Port('top-level/v1'), 'exploration': Port('top-level/v1'),
                     'lab_execution': Port('top-level/v1')}
            if not missing:
                ports['component:counter'] = Port('counter/v1', lambda: instance.count,
                    lambda count: setattr(instance, 'count', count))
            return ports
        return ModuleAssembly.from_components(graph, decision=decision, lab='lab', state_factory=states)

    def test_restore_and_portable_snapshot_preserve_state_without_executing(self):
        modules = self.modules()
        with Application(self.config, modules=modules, durable=True) as app:
            counter = app.assembly.components['counter']
            app.runner.step('one')
            app.runner.step('two')
            snapshot = LocalSnapshotCapture(app).capture('composed modules')
            directory = app.config.data_dir / 'snapshots' / snapshot.snapshot_id
        self.assertEqual(counter.closed, 1)
        with Application.resume(self.config.data_dir, modules=modules) as app:
            self.assertEqual(app.assembly.components['counter'].count, 2)
            self.assertEqual(app.assembly.lab_execution.submissions, 0)
            app.runner.step('three')
            self.assertEqual(app.assembly.components['counter'].count, 3)
        with restore_snapshot(directory, self.root / 'portable', modules=modules) as app:
            self.assertEqual(app.assembly.components['counter'].count, 2)
            self.assertEqual(app.assembly.lab_execution.submissions, 0)

    def test_changed_config_version_or_root_binding_is_rejected_before_any_factory(self):
        with Application(self.config, modules=self.modules(), durable=True) as app:
            app.runner.step('one')
        count = len(self.created)
        for modules in (self.modules(initial=7), self.modules(version='2'), self.modules(decision='other')):
            with self.subTest(configuration=modules.component_configuration()), self.assertRaisesRegex(
                    TypeError, 'original module assembly'):
                Application.resume(self.config.data_dir, modules=modules)
            self.assertEqual(len(self.created), count)

    def test_portable_snapshot_checks_graph_identity_before_resource_factories(self):
        modules = self.modules()
        with Application(self.config, modules=modules, durable=True) as app:
            snapshot = LocalSnapshotCapture(app).capture('graph identity')
            directory = app.config.data_dir / 'snapshots' / snapshot.snapshot_id
        count = len(self.created)
        for changed in (self.modules(initial=7), self.modules(version='2'), self.modules(decision='other')):
            with self.assertRaisesRegex(TypeError, 'original modules'):
                restore_snapshot(directory, self.root / 'invalid', modules=changed)
            self.assertEqual(len(self.created), count)
            self.assertFalse((self.root / 'invalid').exists())

    def test_stateful_component_cannot_silently_restore_with_no_port(self):
        with self.assertRaisesRegex(TypeError, 'component:counter'):
            Application(self.config, modules=self.modules(missing=True), durable=True)
        self.assertEqual(self.created[-1].closed, 1)

    def test_live_component_rebinding_is_blocked_before_another_step(self):
        with Application(self.config, modules=self.modules(), durable=True) as app:
            original = app.assembly.components['counter']
            app.assembly.components = ComponentInstances({}, {'components': {}})
            with self.assertRaisesRegex(TypeError, 'components changed'):
                app.runner.step('must not execute')
            self.assertEqual(original.count, 0)


if __name__ == '__main__':
    unittest.main()
