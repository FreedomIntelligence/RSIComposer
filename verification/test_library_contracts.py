"""Public assembly and scientific evidence contracts, exercised without a model backend."""
from concurrent.futures import ThreadPoolExecutor
from dataclasses import replace
import gc
from hashlib import sha256
import json
import os
from pathlib import Path
import subprocess
import sys
from tempfile import TemporaryDirectory
from threading import Barrier, Event, Thread
import unittest
from unittest.mock import patch
import weakref

from rsicomposer.api import (
    Application, ApplicationConfig, BackendBinding, ComponentDefinition, ComponentGraph, ControllerDefinition,
    Decision, GoalRunPolicy, JobOutcome, JobProgress, JobRef, JobStatus, LocalSnapshotCapture, MetricGoal,
    MetricSpec, ModelRef, ModuleAssembly, NullHarness, OperationCall, OperationProgress, OperationResult,
    OperationSpec, Registry, RunDecision, RunResult, TimeRunPolicy, WorkerLostError, build_default_decision,
    build_harness_decision, restore_snapshot,
)
from rsicomposer.loops.contracts.profiles import SEALED, Invariant
from rsicomposer.runtime.orchestration.lab_group import LabGroup
from rsicomposer.runtime.orchestration.run_policy import EvaluationOutcome
from rsicomposer.runtime.orchestration.runner import Runner


class ScriptedPolicy:
    def __init__(self, actions):
        self.actions = iter(actions)

    def decide(self, _):
        call = next(self.actions, None)
        return Decision('invoke', operations=(call,)) if call else Decision('finish')


class LibraryContractTests(unittest.TestCase):
    def setUp(self):
        temp = TemporaryDirectory()
        self.addCleanup(temp.cleanup)
        self.root = Path(temp.name)

    def config(self, name):
        return ApplicationConfig('Check library contracts', self.root / name, {}, {})

    def test_builtin_factory_identity_validates_in_a_fresh_interpreter(self):
        # The TTRL factory contains a generator expression: its code object has
        # a different address on every process restart. No provider is launched.
        script = '''\
import json, sys
from pathlib import Path
from rsicomposer.api import Registry
root = Path(sys.argv[1])
selection = {'method': {'name': 'ttrl', 'config': {'controller': {
    'initial_model': 'base', 'train_data': 'train', 'eval_data': 'dev'}}}}
modules = Registry().build(selection)
if sys.argv[2] == 'save':
    (root / 'module_assembly.json').write_text(json.dumps(modules.configuration()))
else:
    modules.validate_saved(root)
'''
        for mode, seed in (('save', '1'), ('restore', '2')):
            subprocess.run([sys.executable, '-B', '-c', script, str(self.root), mode], check=True,
                           env={**os.environ, 'PYTHONHASHSEED': seed}, timeout=30)

    def test_one_registered_description_can_build_overlapping_applications(self):
        class EmptyLab:
            def describe(self):
                return ()

            def execute(self, call):
                raise AssertionError('No work expected')

        modules = registry_modules(EmptyLab, ())
        original = modules.controller_factory
        ready = Barrier(2)

        def synchronize(context, lab):
            port = original(context, lab)
            ready.wait(timeout=5)  # Both instances exist before either is activated.
            return port

        shared = replace(modules, controller_factory=synchronize)

        def launch(name):
            config = self.config(name)
            with Application(config, modules=shared) as app:
                instance = app.assembly.controller_host.active
                self.assertEqual(instance.context.data_dir, config.data_dir.resolve())
                self.assertIs(instance.endpoint, app.assembly.host_execution.endpoint)
                self.assertEqual(app.run().stop_reason, 'researcher_finished')
                return weakref.ref(app.assembly)

        with ThreadPoolExecutor(max_workers=2) as pool:
            owners = list(pool.map(launch, ('left', 'right')))
        gc.collect()
        self.assertTrue(all(ref() is None for ref in owners), 'A descriptor kept a closed run alive')

    def test_public_lab_definition_shared_by_pair_and_method(self):
        from rsicomposer.api import MethodDefinition
        from rsicomposer.algorithms import _template as method
        from rsicomposer.labs import _template as public

        registry = Registry(builtins=False)
        method.register(registry)  # No Lab plugin yet: its Controller may still pair with another Lab.
        whole = {'method': {'name': 'template_prompt_search'}}
        pair = {'controller': {'name': 'template_prompt_search'}, 'labs': {'math': {'name': 'template_math'}}}
        with self.assertRaisesRegex(ValueError, 'Unknown Lab: template_math'):
            registry.build(whole)
        registry.register_lab('other', lambda *_: None, version='2')
        registry.build({**pair, 'labs': {'math': {'name': 'other'}}})
        public.register(registry)
        definition = registry.lab('template_math')
        self.assertIs(definition.factory, public.ArithmeticLab)
        self.assertIs(definition.validate_config, public.validate_config)
        self.assertEqual(registry.describe()['labs'], {'template_math': {'version': definition.version},
                                                      'other': {'version': '2'}})
        self.assertNotIn('lab', registry.describe()['components'])
        for kind in ('lab', 'scene'):
            with self.assertRaisesRegex(ValueError, 'Unknown component role'):
                registry.register_component(kind, 'template_math', lambda *_: None, version='other')
            with self.assertRaisesRegex(ValueError, 'Unknown .* component'):
                registry.resolve_component(kind, 'template_math', None)
        with self.assertRaisesRegex(ValueError, 'Duplicate registration'):
            registry.register_lab('template_math', lambda *_: None, version='other')
        for selection in (whole, pair):
            with self.assertRaisesRegex(ValueError, 'datasets must map train and dev'):
                registry.build(selection)
        options = {'datasets': {'train': 'unopened-train', 'dev': 'unopened-dev'}, 'max_concurrent': 2}
        whole['method']['config'] = {'labs': {'math': options}}
        pair['labs']['math']['config'] = options
        for selection in (whole, pair):
            registry.build(selection)  # Only validates; neither path needs a GPU or data files yet.
        with self.assertRaisesRegex(ValueError, 'Registered Lab already owns'):
            MethodDefinition.from_factories(controller=method.CONTROLLER, labs={'math': definition},
                version='1', validate_labs={'math': lambda _: None})
        registry.register_method('private', MethodDefinition.from_factories(controller=method.CONTROLLER,
            labs={'internal': lambda *_: None}, version='1'))
        registry.build({'method': {'name': 'private'}})
        with self.assertRaisesRegex(ValueError, 'Unknown Lab: internal'):
            registry.build({**pair, 'labs': {'math': {'name': 'internal'}}})

    def test_bound_graph_lab_has_one_port_in_portable_and_durable_inventory(self):
        events, omit = [], set()

        class Port:
            state_id = 'fixture/v1'

            def capture(self):
                return {}

            def restore(self, state):
                pass

            def validate_snapshot(self):
                pass

        class Lab(Port):
            value = 0

            def describe(self):
                return ()

            def execute(self, call):
                raise AssertionError('No work expected')

            def capture(self):
                events.append('capture')
                return self.value

            def restore(self, state):
                events.append('restore')
                self.value = state

            def export_snapshot_files(self, directory, state):
                events.append('export')
                (directory / 'fixture').write_text(str(state))

            def import_snapshot_files(self, directory, state):
                events.append('import')
                if (directory / 'fixture').read_text() != str(state):
                    raise ValueError('Unexpected portable Lab state')

            def validate_snapshot(self):
                events.append('validate')

        def decision(ctx, lab):
            return build_default_decision(ctx, lab, policy=ScriptedPolicy(()))

        def ports(ctx, endpoint):
            result = {'lab_execution': ctx.lab_execution, 'decision': Port(), 'execution': Port(),
                      'component:policy': Port(), 'component:data': ctx.components['data']}
            return {key: value for key, value in result.items() if key not in omit}

        graph = ComponentGraph({
            'factory': ComponentDefinition(lambda *_: decision, '1', ('decision-factory/v1',)),
            'data': ComponentDefinition(lambda *_: Port(), '2', ('data/v2',)),
            'spec': ComponentDefinition(lambda *_: object(), '1', ('spec/v1',), stateless=True),
            'lab': ComponentDefinition(lambda *_: Lab(), '3', ('lab-execution/v1',), requires={'input': 'data/v2'}),
        }, {'policy': {'component': 'factory'}, 'data': {'component': 'data'}, 'spec': {'component': 'spec'},
            'work': {'component': 'lab', 'dependencies': {'input': 'data'}}})
        modules = ModuleAssembly.from_components(graph, decision='policy', lab='work', state_factory=ports)
        with Application(self.config('graph-source'), modules=modules) as source:
            adapters = source.assembly.state_adapters
            self.assertEqual(adapters.keys(), {'lab_execution', 'decision', 'execution', 'component:policy',
                                               'component:data', 'component:spec', 'component_manifest'})
            manifest = adapters['component_manifest'].configuration()
            self.assertEqual(manifest['bindings'], {'decision': 'policy', 'lab': 'work'})
            self.assertEqual(manifest['components']['work']['version'], '3')
            self.assertEqual(manifest['components']['work']['dependencies'], {'input': 'data'})
            self.assertIs(source.assembly.components['work'], source.assembly.lab_execution)
            source.assembly.lab_execution.value = 42
            capture = LocalSnapshotCapture(source)
            directory = capture.store.directory(capture.capture('one-owner').snapshot_id)
        with restore_snapshot(directory, self.root / 'graph-restored', modules=modules) as restored:
            self.assertEqual(restored.assembly.lab_execution.value, 42)
        self.assertEqual(events, ['validate', 'capture', 'export', 'import', 'restore', 'validate'])

        def duplicate(ctx, endpoint):
            return {**ports(ctx, endpoint), 'component:work': ctx.lab_execution}

        with self.assertRaisesRegex(ValueError, 'supply only lab_execution'):
            Application(self.config('double'), modules=replace(modules, state_factory=duplicate))
        for name in ('component:policy', 'component:data', 'lab_execution'):
            omit.add(name)
            with self.subTest(name=name), self.assertRaisesRegex(TypeError, 'Missing component state adapters: ' + name):
                Application(self.config('missing-' + name.replace(':', '-')), modules=modules, durable=True)
            omit.clear()

    def test_direct_and_graph_decision_factories_are_activated(self):
        def decision(context, lab):
            return build_default_decision(context, lab, policy=ScriptedPolicy(()))

        graph = ComponentGraph({
            'decide': ComponentDefinition(lambda ctx, deps, cfg: decision, '1', ('decision-factory/v1',)),
            'lab': ComponentDefinition(lambda ctx, deps, cfg: LabGroup(), '1', ('lab-execution/v1',)),
        }, {'policy': {'component': 'decide'}, 'lab': {'component': 'lab'}})
        for name, modules in (
            ('direct', ModuleAssembly(decision_factory=decision, lab_factory=lambda _: LabGroup())),
            ('graph', ModuleAssembly.from_components(graph, decision='policy', lab='lab')),
        ):
            with self.subTest(name=name), Application(self.config(name), modules=modules) as app:
                result = app.run(wait_for_jobs=True)
                self.assertIsInstance(result, RunResult)
                self.assertEqual(result.stop_reason, 'researcher_finished')
                self.assertEqual(result.decision_steps, 1)

    def test_controller_operation_requirements_are_checked_in_all_assemblies(self):
        class Policy(ScriptedPolicy):
            def required_operations(self):
                return ('fixture.missing',)

        class Lab:
            def describe(self):
                return (OperationSpec('available', 'Fixture', {'type': 'object'}),)

            def execute(self, call):
                raise AssertionError('Fixture must fail before dispatch')

        def group_factory(_):
            group = LabGroup()
            group.add('fixture', Lab())
            return group

        def decision(context, lab):
            return build_harness_decision(context, lab, policy=Policy(()), harness=NullHarness())

        graph = ComponentGraph({
            'decision': ComponentDefinition(lambda *_: decision, '1', ('decision-factory/v1',)),
            'lab': ComponentDefinition(lambda ctx, *_: group_factory(ctx), '1', ('lab-execution/v1',)),
        }, {'decision': {'component': 'decision'}, 'lab': {'component': 'lab'}})
        registry = Registry(builtins=False)
        registry.register_controller('fixture', lambda *_: Policy(()), version='1')
        registry.register_lab('fixture', lambda *_: Lab(), version='1')
        for name, modules in (
            ('direct', ModuleAssembly(decision_factory=decision, lab_factory=group_factory)),
            ('graph', ModuleAssembly.from_components(graph, decision='decision', lab='lab')),
            ('registry', registry.build({'controller': {'name': 'fixture'}, 'labs': {'fixture': {'name': 'fixture'}}})),
        ):
            with self.subTest(name=name), self.assertRaisesRegex(ValueError, 'requires missing Lab operations'):
                Application(self.config(name), modules=modules)

    def test_direct_and_graph_close_join_workers_before_releasing_lab_dependencies(self):
        for entry in ('direct', 'graph'):
            with self.subTest(entry=entry):
                entered, joining, release, worker_exited = (Event() for _ in range(4))
                order, errors = [], []

                class Lab:
                    def describe(self):
                        return (OperationSpec('fixture.work', 'Fixture', {'type': 'object'}),)

                    def execute(self, call):
                        entered.set()
                        if not release.wait(5):
                            raise RuntimeError('test did not release the worker')
                        order.append('worker exited')
                        worker_exited.set()
                        return OperationResult('finished', (), False)

                    def close(self):
                        order.append('Lab closed')

                class JoiningExecutor(ThreadPoolExecutor):
                    def shutdown(self, *args, **kwargs):
                        joining.set()
                        return super().shutdown(*args, **kwargs)

                def decision(context, lab):
                    harness = NullHarness()
                    harness.close = lambda: order.append('Harness closed')
                    return build_harness_decision(context, lab, harness=harness,
                                                  policy=ScriptedPolicy((OperationCall('fixture.work', {}),)))

                if entry == 'direct':
                    modules = ModuleAssembly(decision_factory=decision, lab_factory=lambda _: Lab())
                else:
                    graph = ComponentGraph({
                        'decision': ComponentDefinition(lambda *_: decision, '1', ('decision-factory/v1',)),
                        'lab': ComponentDefinition(lambda *_: Lab(), '1', ('lab-execution/v1',), owned=True),
                    }, {'decision': {'component': 'decision'}, 'lab': {'component': 'lab'}})
                    modules = ModuleAssembly.from_components(graph, decision='decision', lab='lab')
                app = Application(self.config(entry), modules=modules)

                def close():
                    try:
                        app.close()
                    except Exception as error:
                        errors.append(error)

                closer = Thread(target=close)
                with patch('rsicomposer.host.execution.ThreadPoolExecutor', JoiningExecutor):
                    try:
                        app.runner.step(app.config.objective)
                        self.assertTrue(entered.wait(2))
                        closer.start()
                        self.assertTrue(joining.wait(2))
                        self.assertEqual(order, [])
                    finally:
                        release.set()
                        if closer.ident is not None:
                            closer.join(5)
                        else:
                            app.close()
                self.assertFalse(closer.is_alive())
                self.assertEqual(errors, [])
                self.assertTrue(worker_exited.is_set())
                self.assertEqual(order, ['worker exited', 'Harness closed', 'Lab closed'])

    def test_registered_lab_snapshot_veto_is_portable_only(self):
        veto = False
        calls = []

        class Policy(ScriptedPolicy):
            def state(self):
                return {}

            def restore(self, state):
                pass

        class Lab(Policy):
            def describe(self):
                return ()

            def execute(self, call):
                raise AssertionError('No fixture operations')

            def validate_snapshot(self):
                calls.append('portable validation')
                if veto:
                    raise RuntimeError('Lab still owns an external job')

        registry = Registry(builtins=False)
        registry.register_controller('fixture', lambda *_: Policy(()), version='1')
        registry.register_lab('fixture', lambda *_: Lab(()), version='1')
        modules = registry.build({'controller': {'name': 'fixture'}, 'labs': {'fixture': {'name': 'fixture'}}})
        cfg = self.config('durable')
        with Application(cfg, modules=modules, durable=True) as app:
            self.assertEqual(calls, [])
            snapshot = LocalSnapshotCapture(app)
            manifest = snapshot.capture('portable')
            source = cfg.data_dir / 'snapshots' / manifest.snapshot_id
            self.assertEqual(calls, ['portable validation'])
            veto = True
            with self.assertRaisesRegex(RuntimeError, 'still owns an external job'):
                snapshot.capture('blocked')
            self.assertEqual(calls, ['portable validation'] * 2)
        with self.assertRaisesRegex(RuntimeError, 'still owns an external job'):
            restore_snapshot(source, self.root / 'restored', modules=modules)
        self.assertEqual(calls, ['portable validation'] * 3)
        with Application.resume(cfg.data_dir, modules=modules):
            pass
        self.assertEqual(calls, ['portable validation'] * 3)

    def test_raw_lab_cannot_advertise_an_unenforced_sealed_split(self):
        class Lab:
            def describe(self):
                return (OperationSpec('fixture.evaluate', 'Fixture', {'type': 'object'}),)

            def execute(self, call):
                raise AssertionError('No work at assembly time')

            def split_roles(self):
                return {'heldout': SEALED}

        modules = ModuleAssembly(decision_factory=lambda ctx, lab: build_default_decision(ctx, lab,
            policy=ScriptedPolicy(())), lab_factory=lambda _: Lab())
        with self.assertRaisesRegex(ValueError, 'Sealed splits require a Lab composed through LabGroup or the Registry'):
            Application(self.config('sealed'), modules=modules)

    def test_direct_lab_cannot_advertise_unenforceable_candidate_history(self):
        class Lab:
            def describe(self):
                return (OperationSpec('measure', 'Measure', {}), OperationSpec('select', 'Select', {}))

            def execute(self, call):
                raise AssertionError('must reject before dispatch')

            def invariants(self):
                return (rule,)

        for rule in (Invariant.min_candidates(2), Invariant.measured_selection()):
            modules = ModuleAssembly(decision_factory=lambda ctx, lab: build_default_decision(ctx, lab,
                policy=ScriptedPolicy(())), lab_factory=lambda _: Lab())
            with self.subTest(rule=rule), self.assertRaisesRegex(ValueError, 'Candidate history invariants require'):
                Application(self.config(rule.kind), modules=modules)

    def test_direct_lab_evaluation_declaration_supports_metric_goal(self):
        args = {'candidate_id': 'one', 'protocol_ref': 'fixture/v1', 'input_refs': ['dataset:a']}
        call = OperationCall('direct.measure', args, call_id='measure')

        class LabExecution:
            def describe(self):
                return (OperationSpec('direct.measure', 'Fixture', {'type': 'object', 'properties': {
                    'candidate_id': {'type': 'string'}, 'protocol_ref': {'type': 'string'},
                    'input_refs': {'type': 'array', 'items': {'type': 'string'}},
                }}),)

            def evaluation_operations(self):
                return ('direct.measure',)

            def candidate_identity_fields(self):
                return {'direct.measure': ('candidate_id',)}

            def execute(self, received):
                return OperationResult('evaluation', (), False, data={**received.arguments,
                                       'quality_metrics': {'accuracy': .91}})

        def decision(context, lab):
            return build_default_decision(context, lab, policy=ScriptedPolicy((call,)))

        modules = ModuleAssembly(decision_factory=decision, lab_factory=lambda _: LabExecution())
        goal = MetricGoal('accuracy', .9, 'fixture/v1', input_refs=('dataset:a',))
        with Application(self.config('direct-goal'), modules=modules, run_policy=GoalRunPolicy(goal)) as app:
            result = app.run(max_decision_steps=2, wait_for_jobs=True)
        self.assertEqual(result.stop_reason, 'goal_reached')
        self.assertEqual(result.decision_steps, 1)

        class UndeclaredLab(LabExecution):
            def evaluation_operations(self):
                return ()

            def is_evaluation(self, name):
                return True

        raw = ModuleAssembly(decision_factory=decision, lab_factory=lambda _: UndeclaredLab())
        with Application(self.config('undeclared-goal'), modules=raw, run_policy=GoalRunPolicy(goal)) as app:
            result = app.run(max_decision_steps=1, wait_for_jobs=True)
        self.assertNotEqual(result.stop_reason, 'goal_reached')

    def test_inferred_cohorts_never_credit_a_subset_as_the_whole_split(self):
        class Measure:
            def describe(self):
                return (OperationSpec('measure', 'Score', {'type': 'object'}),)

            def evaluation_operations(self):
                return ('measure',)

            def candidate_identity_fields(self):
                return {'measure': ('candidate_id',)}

            def primary_metric(self):
                return MetricSpec('accuracy', 'maximize', 'fixture/v1')

            def execute(self, call):
                extra = {'input_refs': ['split:dev']} if call.call_id == 'partial' else {}
                return OperationResult('measured', (), False, data={'quality_metrics': {'accuracy': 1}, **extra})

        args = {'candidate_id': 'one', 'split': 'dev'}
        subsets = {'one': {'task_ids': ['q1']}, 'two': {'task_ids': ['q1', 'q2']},
                   'indices': {'indices': [1, 0], 'dataset': 'v2'}, 'empty': {'task_ids': []},
                   'bool': {'indices': [True]}, 'duplicate': {'indices': [2, 2]},
                   'string': {'indices': ['1']}, 'mixed': {'indices': [1, '0']}, 'negative': {'indices': [-1]},
                   'self_labeled': {'task_ids': ['q1'], 'input_refs': ['split:dev']},
                   'partial': {'task_ids': ['q1']}, 'full': {'indices': None}}
        calls = tuple(OperationCall('test.measure', {**args, **selectors}, call_id=ident)
                      for ident, selectors in subsets.items())
        goal = MetricGoal('accuracy', .9, 'fixture/v1', input_refs=('split:dev',))
        with Application(self.config('subset'), modules=registry_modules(Measure, calls),
                         run_policy=GoalRunPolicy(goal)) as app:
            run = app.run(max_decision_steps=len(calls), wait_for_jobs=True)
            receipts = app.assembly.host_execution.state.operation_results
            self.assertEqual([r.call_id for r in receipts], list(subsets))
            outcomes = app.runner._outcomes
            self.assertEqual(outcomes['one'].input_refs, ('split:dev', 'task_ids:q1'))
            self.assertEqual(outcomes['two'].input_refs, ('split:dev', 'task_ids:q1', 'task_ids:q2'))
            self.assertEqual(outcomes['indices'].input_refs, ('split:dev', 'indices:1', 'indices:0', 'dataset:v2'))
            for ident in ('empty', 'bool', 'duplicate', 'string', 'mixed', 'negative', 'self_labeled', 'partial'):
                self.assertNotIsInstance(outcomes[ident], EvaluationOutcome)
            self.assertNotIn('input_refs', outcomes['self_labeled'].receipt.data)
            self.assertEqual(outcomes['partial'].receipt.data['input_refs'], ['split:dev'])
            self.assertEqual(run.stop_reason, 'goal_reached')
            self.assertEqual(run.decision_steps, len(calls))

    def test_evidence_respects_explicit_cohorts_and_rejects_contradictions(self):
        args = {'candidate_id': 'one', 'split': 'dev', 'indices': [1, 0]}
        identity = {'candidate_id': 'one', 'protocol_ref': 'fixture/v1', 'input_refs': ['dataset:hash']}
        for request, reported, eligible in (
                (args, identity, True),  # The Lab can name this exact cohort using its own IDs.
                ({**args, 'input_refs': ['cohort:A']}, {**identity, 'input_refs': ['cohort:A']}, True),
                ({**args, 'input_refs': ['cohort:B']}, identity, False),
                ({**args, 'input_refs': []}, identity, False),
                ({**args, 'input_refs': ['split:dev']}, {**identity, 'input_refs': ['split:dev']}, False),
                ({**args, 'indices': {'range': 'last page'}, 'input_refs': ['cohort:A']},
                 {**identity, 'input_refs': ['cohort:A']}, True),
                (args, {**identity, 'split': 'test'}, False),
                (args, {**identity, 'indices': [0, 1]}, False),
                (args, {**identity, 'input_refs': ['split:dev']}, False),
                (args, {**identity, 'input_refs': ['split:dev', 'indices:0', 'indices:1']}, False)):
            with self.subTest(request=request, reported=reported):
                call = OperationCall('test.measure', request)
                result = OperationResult('measured', (), False, data={**reported, 'quality_metrics': {'accuracy': 1}})
                self.assertEqual(EvaluationOutcome.from_call(call, result, candidate_fields=('candidate_id',)) is not None, eligible)

    def test_only_attested_or_native_cohorts_qualify_even_on_journal_replay(self):
        gold = ['cohort:full-development']
        train = {'split': 'train', 'task_ids': ['t1']}
        native = ['split:train', 'task_ids:t1']
        measured = []

        class Measure:
            def describe(self):
                return (OperationSpec('score', 'Score', {'type': 'object'}),)

            def evaluation_operations(self):
                return ('score',)

            def candidate_identity_fields(self):
                return {'score': ('candidate_id',)}

            def primary_metric(self):
                return MetricSpec('accuracy', 'maximize', 'fixture/v1')

            def execute(self, call):
                measured.append(call.call_id)
                attestation = {'input_refs': gold} if call.arguments.get('split') == 'dev' else {}
                return OperationResult('measured', (), False, data={'quality_metrics': {'accuracy': 1}, **attestation})

        calls = tuple(OperationCall('test.score', {'candidate_id': 'one', **args}, call_id=name) for name, args in (
            ('journal', {**train, 'input_refs': gold}), ('request_only', {'input_refs': gold}),
            ('native', {**train, 'input_refs': native}), ('attested', {'split': 'dev', 'input_refs': gold})))
        goal = MetricGoal('accuracy', .9, 'fixture/v1', input_refs=tuple(gold))
        with Application(self.config('attested-cohorts'), modules=registry_modules(Measure, calls),
                         run_policy=GoalRunPolicy(goal)) as app:
            group, host = app.assembly.lab_execution, app.assembly.host_execution
            group.execute(calls[0])  # Replay a native, score-only terminal; never relabel it as the requested cohort.
            result = app.run(max_decision_steps=len(calls), wait_for_jobs=True)
            self.assertEqual((result.stop_reason, result.decision_steps), ('goal_reached', len(calls)))
            self.assertEqual(measured, [c.call_id for c in calls])
            outcomes = app.runner._outcomes
            for call in calls[:2]:
                self.assertNotIn('input_refs', group.stored_receipt(call).data)
                self.assertNotIn('input_refs', outcomes[call.call_id].receipt.data)
                self.assertIsNone(host.evaluation_outcome(call, outcomes[call.call_id].receipt))
                self.assertNotIsInstance(outcomes[call.call_id], EvaluationOutcome)
            self.assertNotIn('input_refs', group.stored_receipt(calls[2]).data)
            self.assertEqual(outcomes['native'].input_refs, tuple(native))
            self.assertEqual(group.stored_receipt(calls[3]).data['input_refs'], gold)
            self.assertEqual(outcomes['attested'].input_refs, tuple(gold))

    def test_score_only_receipts_match_across_sync_poll_replay_and_retry(self):
        call = OperationCall('test.measure', {'candidate_id': 'one', 'split': 'dev', 'task_ids': ['q2']},
                             call_id='measurement')
        job = JobRef('measurement-job', 'fixture')
        raw = OperationResult('measured', (), False, data={'quality_metrics': {'accuracy': .95}})
        refs = ('split:dev', 'task_ids:q2')
        goal = MetricGoal('accuracy', .9, 'fixture/v1', input_refs=refs)
        for mode in ('sync', 'poll', 'replay', 'retry'):
            with self.subTest(mode=mode):
                executions, polls, delivered = [], [], []

                class Measure:
                    def describe(self):
                        return (OperationSpec('measure', 'Score', {'type': 'object'}, idempotent=True),)

                    def evaluation_operations(self):
                        return ('measure',)

                    def candidate_identity_fields(self):
                        return {'measure': ('candidate_id',)}

                    def primary_metric(self):
                        return MetricSpec('accuracy', 'maximize', 'fixture/v1')

                    def execute(self, received):
                        executions.append(received.call_id)
                        if mode == 'retry' and len(executions) == 1:
                            raise WorkerLostError('uncertain first dispatch')
                        return job if mode in ('poll', 'replay') else raw

                    def poll_job(self, observed):
                        polls.append(observed)
                        return JobOutcome(observed, JobStatus.SUCCEEDED, raw)

                n = 2 if mode == 'retry' else 1
                with Application(self.config(mode), modules=registry_modules(Measure, (call,) * n),
                                 run_policy=GoalRunPolicy(goal)) as app:
                    group, host = app.assembly.lab_execution, app.assembly.host_execution
                    if mode == 'replay':
                        handle = group.execute(call)
                        group.poll_job(call, handle, decision_id='before-interruption', after=0)
                    host.add_receipt_listener(delivered.append, owner=host.harness)
                    run = app.run(max_decision_steps=n, wait_for_jobs=True)
                    self.assertEqual(run.stop_reason, 'goal_reached')
                    self.assertEqual(run.decision_steps, n)
                    native = group.stored_receipt(call)
                    self.assertEqual(native, replace(raw, call_id=call.call_id))
                    receipt = replace(native, data={**native.data, 'candidate_id': 'one',
                                      'protocol_ref': 'fixture/v1', 'input_refs': refs})
                    self.assertEqual(host.state.operation_results[-1], receipt)
                    self.assertEqual(app.runner._outcomes[call.call_id].receipt, receipt)
                    self.assertEqual(delivered[-1].result, receipt.payload())
                    self.assertEqual(executions, [call.call_id] * n)
                    self.assertEqual(polls, [job] if mode in ('poll', 'replay') else [])

    def test_external_completion_uses_the_installed_direct_lab_declaration(self):
        job = JobRef('external-job', 'fixture')
        call = OperationCall('direct.measure', {'candidate_id': 'one', 'input_refs': ['cohort:A']})
        goal = MetricGoal('accuracy', .9, 'fixture/v1', input_refs=('cohort:A',))
        for declared, attested in ((False, True), (True, False), (True, True)):
            with self.subTest(declared=declared, attested=attested):
                raw = OperationResult('measured', (), False, data={'quality_metrics': {'accuracy': .95},
                    **({'input_refs': ['cohort:A']} if attested else {})})
                effects = []
                class Direct:
                    def describe(self):
                        return (OperationSpec('direct.measure', 'Score', {'type': 'object'}),)

                    def evaluation_operations(self):
                        return ('direct.measure',) if declared else ()

                    def candidate_identity_fields(self):
                        return {'direct.measure': ('candidate_id',)}

                    def is_evaluation(self, name):
                        return not declared  # An arbitrary classifier is not a declaration.

                    def primary_metric(self):
                        if effects:
                            raise AssertionError('Host called the Lab declaration after dispatch')
                        effects.append('declare')
                        return MetricSpec('accuracy', 'maximize', 'fixture/v1')

                    def execute(self, _):
                        effects.append('execute')
                        return job

                def decision(ctx, lab):
                    return build_harness_decision(ctx, lab, policy=ScriptedPolicy((call,)), harness=NullHarness())

                modules = ModuleAssembly(decision_factory=decision, lab_factory=lambda _: Direct())
                with Application(self.config(f'external-{declared}-{attested}'), modules=modules) as app:
                    host = app.assembly.host_execution
                    app.run(max_decision_steps=1, wait_for_jobs=True)
                    self.assertEqual(host.in_flight_operations()[0]['job'], job)
                    delivered = host.operation_scheduler.complete(call.call_id, raw)
                    app.runner._loop_result(delivered)
                    self.assertEqual(host.state.operation_results[-1], delivered.operation_result)
                    self.assertEqual(goal(app.runner._run_context(app.config.objective)), declared and attested)
                    if not attested:
                        self.assertNotIn('input_refs', delivered.operation_result.data)
                    self.assertEqual(effects, ['declare', 'execute'] if declared else ['execute'])
                    if not declared:
                        self.assertEqual(delivered.operation_result.data, raw.data)

    def test_resume_replays_only_missing_terminal_progress_before_goal_delivery(self):
        from verification.test_run_control import ImmediateExecutor, Policy

        job = JobRef('finished-job', 'retired-backend')
        call = OperationCall('test.measure', {'candidate_id': 'one', 'indices': [2]}, call_id='managed')
        raw = OperationResult('measured', (), False, data={'quality_metrics': {'accuracy': 1}})
        effects = []
        phase = 'running'

        class Measure:
            def describe(self):
                return (OperationSpec('measure', 'Score', {'type': 'object'}),)

            def evaluation_operations(self):
                return ('measure',)

            def candidate_identity_fields(self):
                return {'measure': ('candidate_id',)}

            def primary_metric(self):
                return MetricSpec('accuracy', 'maximize', 'fixture/v1')

            def state(self):
                return {}

            def restore(self, state):
                pass

            def execute(self, _):
                effects.append('submit')
                return job

            def poll_job(self, observed):
                effects.append('poll:' + phase)
                if phase == 'retired':
                    raise AssertionError('Completed backend job no longer exists')
                return (JobOutcome(observed, JobStatus.RUNNING) if phase == 'running' else
                        JobOutcome(observed, JobStatus.SUCCEEDED, raw))

            def job_events(self, observed, *, after):
                end = 1 if phase == 'running' else 2
                return tuple(OperationProgress(call.call_id, observed, n, 'progress', JobProgress(step=n))
                             for n in range(after + 1, end + 1))

            def reattach_job(self, _):
                raise AssertionError('Terminal job must never be reattached')

        modules = Registry(builtins=False)
        modules.register_lab('test', lambda *_: Measure(), version='1')
        modules.register_controller('test', lambda *_: Policy((call,)), version='1')
        modules = modules.build({'controller': {'name': 'test'}, 'labs': {'test': {'name': 'test'}}})
        config = self.config('terminal-before-host')
        goal = MetricGoal('accuracy', .9, 'fixture/v1', input_refs=('indices:2',))
        with Application(config, modules=modules, durable=True, run_policy=GoalRunPolicy(goal)) as app:
            with patch('rsicomposer.host.execution.ThreadPoolExecutor', ImmediateExecutor):
                app.runner.step(config.objective)
            host = app.assembly.host_execution
            self.assertEqual([e.sequence for e in host.state.operation_progress], [1])
            phase = 'finished'
            # The post-progress Host checkpoint is durable. Only the Lab gets to publish its terminal.
            app.assembly.lab_execution.poll_job(call, job, decision_id=host.in_flight_operations()[0]['decision_id'],
                                                after=1)
        phase = 'retired'
        with Application.resume(config.data_dir, modules=modules) as app:
            host = app.assembly.host_execution
            self.assertIsNone(app.recovery.intervention)
            self.assertEqual(host.in_flight_operations()[0]['job'], job)
            observed = []
            host.add_receipt_listener(lambda _: observed.append([e.sequence for e in host.state.operation_progress]),
                                      owner=host.harness)
            result = app.run(max_decision_steps=1, wait_for_jobs=True)
            self.assertEqual(result.stop_reason, 'goal_reached')
            self.assertEqual(observed, [[1, 2]])
            self.assertEqual(host.state.operation_results[0].data['input_refs'], ('indices:2',))
        self.assertEqual(effects, ['submit', 'poll:running', 'poll:finished'])

    def test_builtin_math_benchmark_receipt_is_not_implicitly_metric_goal_evidence(self):
        data = json.dumps({'id': 'q', 'prompt': 'What is one plus one?', 'answer': '2'}).encode() + b'\n'
        (self.root / 'tasks.jsonl').write_bytes(data)
        manifest = self.root / 'datasets.json'
        manifest.write_text(json.dumps({'schema_version': 1, 'splits': {'dev': {
            'file': 'tasks.jsonl', 'sha256': sha256(data).hexdigest(), 'count': 1,
        }}}), encoding='utf-8')
        registry = Registry()
        call = OperationCall('math.evaluate', {'split': 'dev', 'instruction': 'Return a marked answer.'})
        registry.register_controller('fixture', lambda *_: ScriptedPolicy((call,)), version='1')
        modules = registry.build({'controller': {'name': 'fixture'}, 'labs': {'math': {'name': 'math', 'config': {
            'manifest': str(manifest), 'split_roles': {'dev': 'dev'}, 'max_output_tokens': 64,
        }}}})
        config = ApplicationConfig('Built-in Math evidence', self.root / 'math',
                                   {'mock': BackendBinding('mock', 'mock')},
                                   {'actor': ModelRef('mock', 'fixture', 'v1')})
        with Application(config, modules=modules) as app:
            app.run(max_decision_steps=1, wait_for_jobs=True)
            [receipt] = app.assembly.host_execution.state.operation_results
            self.assertIn('accuracy', receipt.data['quality_metrics'])
            protocol = receipt.data['benchmarks']['math']
            goal = MetricGoal('accuracy', 0, protocol, allow_simulated=True)
            self.assertIsNone(EvaluationOutcome.from_call(call, receipt))
            self.assertFalse(goal(app.runner._run_context(config.objective)))

    def test_direct_lab_evaluation_identity_is_checked_on_durable_resume(self):
        class Lab:
            def __init__(self, declared, fields):
                self.declared, self.fields = declared, fields

            def describe(self):
                return (OperationSpec('example.measure', 'Measure', {'type': 'object'}),)

            def evaluation_operations(self):
                return self.declared

            def candidate_identity_fields(self):
                return {name: self.fields for name in self.declared}

            def execute(self, _):
                raise AssertionError('restore should not execute Lab work')

        class Contract:
            contract_id = 'fixture-no-work/v1'

            def capture(self, app):
                return {}

            def restore(self, app, state):
                self.restored = state

        def decision(context, lab):
            return build_default_decision(context, lab, policy=ScriptedPolicy(()))

        def modules(declared, fields=('encoder', 'index.sha256')):
            return ModuleAssembly(decision_factory=decision, lab_factory=lambda _: Lab(declared, fields))

        cfg = self.config('direct-goal-durable')
        with Application(cfg, modules=modules(('example.measure',)), durable=True, recovery_contract=Contract()):
            pass
        with Application.resume(cfg.data_dir, modules=modules(('example.measure',)), recovery_contract=Contract()):
            pass
        with self.assertRaisesRegex(ValueError, 'differ from saved method identity'):
            Application.resume(cfg.data_dir, modules=modules(()), recovery_contract=Contract())
        with self.assertRaisesRegex(ValueError, 'differ from saved method identity'):
            Application.resume(cfg.data_dir, modules=modules(('example.measure',), ('model',)),
                               recovery_contract=Contract())

    def test_rejecting_a_foreign_host_does_not_close_its_harness(self):
        closed = []

        def decision(context, lab):
            endpoint = build_default_decision(context, lab, policy=ScriptedPolicy(()))
            endpoint.harness.close = lambda: closed.append('owner')
            return endpoint

        owner = ModuleAssembly(decision_factory=decision, lab_factory=lambda _: LabGroup())
        with Application(self.config('owner'), modules=owner) as app:
            foreign = app.assembly.host_execution.endpoint
            borrower = ModuleAssembly(decision_factory=lambda ctx, lab: foreign, lab_factory=lambda _: LabGroup())
            with self.assertRaisesRegex(ValueError, 'belongs to another run execution'):
                Application(self.config('borrower'), modules=borrower)

            def wrong_executor(ctx, lab):
                endpoint = decision(ctx, lab)
                endpoint.host_execution = object()
                return endpoint

            with self.assertRaisesRegex(ValueError, 'belongs to another run execution'):
                Application(self.config('wrong-executor'), modules=replace(owner, decision_factory=wrong_executor))
            self.assertEqual(closed, [])
            initial_version = foreign.controller_version
            registry = Registry(builtins=False)
            registry.register_controller('fixture', ControllerDefinition(lambda *_: foreign, endpoint=True), version='1')
            class BorrowerLab:
                def describe(self):
                    return ()

                def execute(self, call):
                    raise AssertionError('No fixture operations')

            registry.register_lab('fixture', lambda *_: BorrowerLab(), version='1')
            registered = registry.build({'controller': {'name': 'fixture'}, 'labs': {'fixture': {'name': 'fixture'}}})
            with self.assertRaisesRegex(ValueError, 'belongs to another run execution'):
                Application(self.config('registered-borrower'), modules=registered)
            self.assertEqual(closed, [])
            self.assertEqual(foreign.controller_version, initial_version)
            self.assertEqual(app.run().stop_reason, 'researcher_finished')
        self.assertEqual(closed, ['owner'])

    def test_role_declaration_failure_closes_new_lab(self):
        closed = []

        class Lab:
            def resource_roles(self):
                raise ValueError('invalid roles')

            def describe(self):
                return ()

            def execute(self, _):
                raise AssertionError('should never dispatch')

            def close(self):
                closed.append(True)

        with self.assertRaisesRegex(ValueError, 'invalid roles'):
            Application(self.config('roles'), modules=registry_modules(Lab, ()))
        self.assertEqual(closed, [True])

    def test_unsupported_dotted_local_names_fail_during_assembly(self):
        class Lab:
            def describe(self):
                return (OperationSpec('task.inspect', 'Fixture', {'type': 'object'}),)

            def execute(self, _):
                raise AssertionError('should never dispatch')

        with self.assertRaisesRegex(ValueError, 'local operation names cannot contain dots'):
            Application(self.config('names'), modules=registry_modules(Lab, ()))

    def test_standalone_lab_constraints_contribute_to_identity(self):
        class Lab:
            def describe(self):
                return ()

            def execute(self, _):
                raise AssertionError('should never dispatch')

            def split_roles(self):
                return {'reserved': SEALED}

            def invariants(self):
                return (Invariant.sealed_after_finish(),)

        group = LabGroup()
        group.add('test', Lab())
        identity = group.profile_identity()['test']
        self.assertIn('reserved', identity['split_roles'])
        self.assertEqual(identity['invariants'][0]['kind'], 'sealed_after_finish')

    def test_registry_build_does_not_call_controller_memory_factory(self):
        made = []

        def memory(config):
            made.append(config)
            return None

        class Lab:
            def describe(self):
                return ()

            def execute(self, _):
                raise AssertionError('should never dispatch')

        modules = registry_modules(Lab, (), memory=memory)
        self.assertEqual(made, [])
        with Application(self.config('memory'), modules=modules):
            self.assertEqual(made, [{}])

    def test_stop_timeout_does_not_forget_a_previously_settled_lab(self):
        elapsed = [0.0]
        shutdown = []

        class Waiting:
            def pending_research_job(self):
                return JobRef('fixture-job', 'fixture')

        def stop_lab(mode):
            shutdown.append(mode)
            return True

        runner = Runner(Waiting(), clock=lambda: elapsed[0], cleanup_timeout_seconds=2,
                        execution_shutdown=stop_lab)
        runner._begin_stop(RunDecision('stop', 'fixture', 'cancel'))
        self.assertEqual(runner.step('fixture').kind, 'waiting')
        elapsed[0] = 3.0
        stop = runner.step('fixture').event['run_stop']
        self.assertEqual(shutdown, ['cancel'])
        self.assertTrue(stop['cleanup_timed_out'])
        self.assertTrue(stop['unresolved_research_job'])
        self.assertFalse(stop['unresolved_execution'])

    def test_retry_evaluation_is_considered_the_latest_delivery(self):
        from rsicomposer.loops.contracts.lifecycle import ExecutionStatus
        args = {'candidate_id': 'one', 'protocol_ref': 'fixture/v1', 'input_refs': ['dataset:a']}

        class EvaluationRunner(Runner):
            def _classify_outcome(self, receipt, decision_id):
                return EvaluationOutcome.from_call(OperationCall('test.measure', args), receipt, candidate_fields=('candidate_id',))

        runner = EvaluationRunner(object(), clock=lambda: 0.0)
        runner._run_started = 0.0
        goal = MetricGoal('accuracy', .9, 'fixture/v1')
        runner._record_outcome(OperationResult('uncertain', (), False, ExecutionStatus.FAILED, 'unknown',
                               data={**args, 'effects_uncertain': True, 'retryable': True}, call_id='a'), None)
        runner._record_outcome(OperationResult('missed', (), False,
                               data={**args, 'quality_metrics': {'accuracy': .4}}, call_id='b'), None)
        self.assertFalse(goal(runner._run_context('fixture')))
        runner._record_outcome(OperationResult('succeeded on retry', (), False,
                               data={**args, 'quality_metrics': {'accuracy': .95}}, call_id='a'), None)
        self.assertTrue(goal(runner._run_context('fixture')))
        runner._record_outcome(OperationResult('not measured', (), False,
                               data={**args, 'quality_metrics': {'accuracy': 1.0}, 'scored': False}, call_id='c'), None)
        self.assertFalse(goal(runner._run_context('fixture')))

    def test_hard_stop_reports_receipt_recovered_from_pending_observation(self):
        attempts = []
        calls = []

        class Lab:
            def describe(self):
                return (OperationSpec('inspect', 'Fixture', {'type': 'object'}),)

            def execute(self, _):
                calls.append('inspect')
                return OperationResult('Recorded once', (), False)

        call = OperationCall('test.inspect', {}, call_id='inspection')
        with Application(self.config('terminal-receipt'), modules=registry_modules(Lab, (call,)),
                         run_policy=TimeRunPolicy(1)) as app:
            time = [10.0]
            app.runner._clock = lambda: time[0]
            harness = app.assembly.host_execution.harness
            original_observe = harness.observe

            def observe(event):
                if event.task_id == call.call_id:
                    attempts.append(event.event_id)
                    if len(attempts) == 1:
                        raise ValueError('temporary receipt observer failure')
                return original_observe(event)

            harness.observe = observe
            with self.assertRaisesRegex(ValueError, 'temporary receipt observer failure'):
                app.run(max_decision_steps=1, wait_for_jobs=True)
            time[0] = 12.0
            result = app.run(wait_for_jobs=True)
            self.assertTrue(result.shutdown_complete)
            delivered, = (event['operation_result'] for event in result.events if event.get('operation_result'))
            self.assertEqual(delivered['call_id'], call.call_id)
            self.assertIn(call.call_id, app.runner._outcomes)
            self.assertFalse(any(event.get('operation_result') for event in app.run().events))
        self.assertEqual(calls, ['inspect'])
        self.assertEqual(len(attempts), 2)
        self.assertEqual(*attempts)

    def test_goal_ignores_training_unmatched_inputs_and_unverified_identity(self):
        calls = []

        class Lab:
            def describe(self):
                schema = {'type': 'object', 'properties': {
                    'candidate_id': {'type': 'string'}, 'protocol_ref': {'type': 'string'},
                    'input_refs': {'type': 'array', 'items': {'type': 'string'}},
                }}
                return (OperationSpec('train', 'Train', schema),
                        OperationSpec('measure', 'Measure', schema))

            def evaluation_operations(self):
                return ('measure',)

            def candidate_identity_fields(self):
                return {'measure': ('candidate_id',)}

            def execute(self, call):
                calls.append(call.call_id)
                data = {**call.arguments, 'quality_metrics': {'accuracy': .99}}
                if call.call_id == 'mismatch':
                    data['candidate_id'] = 'not-the-requested-model'
                return OperationResult(call.name, (), call.call_id == 'simulated', data=data)

        matched = {'candidate_id': 'model', 'protocol_ref': 'evaluator/v2', 'input_refs': ['cohort:valid']}
        plan = tuple(OperationCall(f'test.{name}', args, call_id=ident) for ident, name, args in (
            ('train', 'train', matched), ('wrong_inputs', 'measure', {**matched, 'input_refs': ['cohort:train']}),
            ('mismatch', 'measure', matched), ('simulated', 'measure', matched), ('real', 'measure', matched)))
        modules = registry_modules(Lab, plan)
        goal = MetricGoal('accuracy', .95, 'evaluator/v2', input_refs=('cohort:valid',))
        with Application(self.config('goal'), modules=modules, run_policy=GoalRunPolicy(goal)) as app:
            result = app.run(max_decision_steps=6, wait_for_jobs=True)
        self.assertEqual(calls, [call.call_id for call in plan])
        self.assertEqual(result.stop_reason, 'goal_reached')
        self.assertEqual(result.decision_steps, 5)


class MemoryAssemblyContractTests(unittest.TestCase):
    def setUp(self):
        root = TemporaryDirectory()
        self.addCleanup(root.cleanup)
        self.root = Path(root.name)

    def config(self, name):
        return ApplicationConfig('Check Memory composition', self.root / name,
            {'mock': BackendBinding('mock', 'mock')}, {'decision_maker': ModelRef('mock', 'model', 'v1')})

    def test_template_policy_receives_only_search_options(self):
        from rsicomposer.algorithms import _template as template
        options = {'max_evaluations': 1, 'harness': 'scoreboard', 'scoreboard_rows': 3, 'memory': 'best_score'}
        template.validate_config(options)
        with patch.object(template, 'PromptSearchController') as policy:
            template.policy_factory('context', options)
            policy.assert_called_once_with('context', {'max_evaluations': 1})
        self.assertEqual(template.harness_factory(None, options).limit, 3)
        for name, invalid in [('harness', 'unknown'), ('memory', 'unknown'), ('scoreboard_rows', 0),
                              ('max_evaluations', 100)]:
            with self.subTest(name=name), self.assertRaisesRegex(ValueError, name):
                template.validate_config({**options, name: invalid})

    def test_optional_memory_tools_and_context_work_with_either_harness(self):
        from types import SimpleNamespace
        from rsicomposer.api import Capability, MemoryAssembly, ResearchAssembly
        from rsicomposer.harnesses.decision.feedback.service import FeedbackLearningService

        class Memory:
            def context(self, query):
                queries.append(query.objective)
                return 'window-memory-marker'

            def observe(self, event):
                if event.kind == 'receipt':
                    receipts.append(event)

            def capabilities(self):
                name = 'memory_echo'
                return (Capability.from_spec(OperationSpec(name, 'Echo a note', {'type': 'object'}),
                    lambda args: OperationCall(name, {'text': args['text'].strip()}),
                    lambda args: OperationResult('Memory handler', (), False, data=args)),)

        class ForeignPolicy(ScriptedPolicy):
            def decide(self, window):
                windows.append(window)
                return super().decide(window)

        close = FeedbackLearningService.close
        for custom in (False, True):
            queries, receipts, windows, closed = [], [], [], []
            memory = Memory()
            marker = object()
            policy = ForeignPolicy((OperationCall('memory_echo', {'text': '  saved  '}),
                                    OperationCall('feedback_learning_read', {})))
            policy.implementation = SimpleNamespace(parser=SimpleNamespace(space=marker))
            harness = NullHarness()

            def endpoint(ctx, lab):
                if custom:
                    return build_harness_decision(ctx, lab, policy=policy, harness=harness)
                return build_default_decision(ctx, lab, policy=policy)

            modules = ModuleAssembly(decision_factory=endpoint, lab_factory=lambda _: LabGroup(),
                memory=MemoryAssembly(ResearchAssembly(context_memory=memory, trajectory_archive=None)))
            with self.subTest(custom=custom), patch.object(FeedbackLearningService, 'close',
                    lambda service: (closed.append(service), close(service))[-1]):
                with Application(self.config(str(custom)), modules=modules) as app:
                    loop = app.runner.decision_making
                    self.assertIs(policy.implementation.parser.space, marker)
                    if custom:
                        self.assertIs(loop.harness, harness)
                    self.assertEqual(len(loop.host_execution._receipt_listeners), 1)
                    app.run(max_decision_steps=3, wait_for_jobs=True)
                    self.assertEqual(len(queries), len(windows))
                    self.assertEqual(len(windows), 3)
                    for window in windows:
                        self.assertEqual(window.context.count('window-memory-marker'), 1)
                        names = [row['name'] for row in window.capabilities]
                        self.assertEqual(names.count('memory_echo'), 1)
                        self.assertEqual(names.count('feedback_learning_read'), 1)
                    self.assertEqual(windows[1].feedbacks[0].data, {'text': 'saved'})
                    self.assertEqual(windows[2].feedbacks[0].summary, 'Feedback learning proposals; validation still required')
                    self.assertEqual(sum(event.request.get('name') == 'memory_echo' for event in receipts), 1)
                    self.assertEqual(len({event.event_id for event in receipts}), len(receipts))
                self.assertEqual(closed, [loop.integration])
                self.assertEqual(loop.host_execution._receipt_listeners, [])

    def test_four_port_harness_can_use_builtin_agent_memory(self):
        from rsicomposer.api import MemoryAssembly, ResearchAssembly
        windows = []

        def policy(window):
            windows.append(window)
            if len(windows) == 1:
                return Decision.invoke(OperationCall('memory_search', {}))
            return Decision('finish')

        modules = ModuleAssembly(decision_factory=lambda ctx, lab: build_harness_decision(
            ctx, lab, policy=policy, harness=NullHarness()), lab_factory=lambda _: LabGroup(),
            memory=MemoryAssembly(ResearchAssembly.single_meta()))
        with Application(self.config('four-ports'), modules=modules) as app:
            app.run(max_decision_steps=2, wait_for_jobs=True)
            self.assertEqual(json.loads(windows[0].context)['provider'], 'agent_managed')
            self.assertEqual(windows[1].feedbacks[0].summary, 'Meta-agent memory/task operation completed')

    def test_failed_install_preserves_existing_bindings_and_subscriptions(self):
        from rsicomposer.api import ActionPermissions, ResearchAssembly
        from rsicomposer.research import assemble_research

        class ConflictingHarness(NullHarness):
            def tools(self):
                return (OperationSpec('memory_search', 'Private Harness tool', {'type': 'object'}),)

        for conflict in (True, False):
            def endpoint(ctx, lab):
                if conflict:
                    return build_harness_decision(ctx, lab, policy=ScriptedPolicy(()), harness=ConflictingHarness())
                return build_default_decision(ctx, lab, policy=ScriptedPolicy(()))

            modules = ModuleAssembly(decision_factory=endpoint, lab_factory=lambda _: LabGroup())
            with self.subTest(conflict=conflict), Application(self.config('rollback-' + str(conflict)),
                    modules=modules, action_permissions=ActionPermissions([])) as app:
                loop = app.runner.decision_making
                space, bindings = loop.action_space, loop.operation_router.host_bindings
                self.assertEqual(space.describe(), ())
                if conflict:
                    with self.assertRaisesRegex(ValueError, 'unique'):
                        assemble_research(loop, assembly=ResearchAssembly.single_meta())
                else:
                    # Failure after the service, its projected window and all tools were attached.
                    with patch('rsicomposer.research.atomic_write', side_effect=OSError('manifest write failed')), \
                            self.assertRaisesRegex(OSError, 'manifest write failed'):
                        assemble_research(loop, assembly=ResearchAssembly.single_meta(),
                                          path=self.root / 'feedback.json')
                    self.assertIsNone(loop.harness.feedback_service)
                    self.assertEqual(loop.harness._context_processors, [])
                self.assertIsNone(loop.integration)
                self.assertIs(loop.action_space, space)
                self.assertIs(loop.operation_router.host_bindings, bindings)
                self.assertEqual(loop.host_execution._receipt_listeners, [])

    def test_native_parser_and_agent_memory_share_authorized_catalog(self):
        from rsicomposer.api import ActionPermissions, MemoryAssembly, ResearchAssembly
        from rsicomposer.backends.serving.chat_completions import MockModelBackend
        from rsicomposer.loops.contracts.model_service import GenerationResponse
        requests = []

        def generate(backend, request):
            requests.append(request)
            payload = ({'kind': 'invoke', 'operation': 'memory_search', 'arguments': {}} if len(requests) == 1
                       else {'kind': 'finish'})
            return GenerationResponse(json.dumps(payload), request.model, simulated=True)

        modules = ModuleAssembly(decision_factory=build_default_decision, lab_factory=lambda _: LabGroup(),
            memory=MemoryAssembly(ResearchAssembly.single_meta()))
        with patch.object(MockModelBackend, 'supports_response_schema', True, create=True), \
                patch.object(MockModelBackend, 'generate', generate), Application(self.config('native'),
                modules=modules, action_permissions=ActionPermissions(['memory_search'])) as app:
            app.run(max_decision_steps=2, wait_for_jobs=True)
            self.assertEqual(len(requests), 2)
            for request in requests:
                context = json.loads(request.prompt)
                self.assertEqual([row['name'] for row in context['operations']], ['memory_search'])
                self.assertEqual([row['properties']['operation']['const'] for row in request.response_schema['oneOf']
                                  if 'operation' in row['properties']], ['memory_search'])
                self.assertEqual(json.loads(context['memory_context'])['provider'], 'agent_managed')
            receipt, = app.runner.decision_making.host_execution.state.operation_results
            self.assertEqual(receipt.data['revision'], 0)
            self.assertEqual(receipt.summary, 'Meta-agent memory/task operation completed')
            with self.assertRaises(PermissionError):
                app.runner.decision_making.action_space.validate(OperationCall('feedback_learning_read', {}))


def registry_modules(lab, actions, *, memory=None):
    registry = Registry()
    registry.register_controller('test', ControllerDefinition(lambda ctx, cfg: ScriptedPolicy(actions),
                                                              memory=memory), version='1')
    registry.register_lab('test', lambda ctx, cfg: lab(), version='1')
    return registry.build({'controller': {'name': 'test'}, 'labs': {'test': {'name': 'test'}}})
