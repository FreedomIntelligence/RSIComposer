"""External registration and real host boundaries; no LLM/GPU training."""
from tests import ROOT as REPO_ROOT
from dataclasses import replace
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
import tempfile
import unittest
from unittest.mock import patch

from rsisys.api import (Application, ApplicationConfig, Decision, OperationCall,
                        OperationResult, OperationSpec, ExecutionStatus, Registry,
                        MethodDefinition, ModuleAssembly)
from rsisys.loops.contracts.model_service import BackendBinding, ModelRef, GenerationResponse
from rsisys.loops.decision_making.action_space.access import ActionPermissions


ROOT = REPO_ROOT


class CounterLab:
    def __init__(self, *, fail=False):
        self.calls, self.closed, self.fail = 0, 0, fail
    def describe(self):
        return (OperationSpec('step', 'CPU interface fixture', {
            'type': 'object', 'properties': {}, 'additionalProperties': False}),)
    def execute(self, call):
        if call != OperationCall('step', {}):
            raise ValueError('Incorrectly routed operation')
        self.calls += 1
        if self.fail:
            return OperationResult('failed', (), False, ExecutionStatus.FAILED, 'fixture failure')
        return OperationResult('done', (), False, data={'count': self.calls})
    def close(self):
        self.closed += 1


class RegistryTests(unittest.TestCase):
    def setUp(self):
        temporary = tempfile.TemporaryDirectory()
        self.addCleanup(temporary.cleanup)
        self.root = Path(temporary.name)
        self.config = ApplicationConfig('CPU fixture', self.root / 'run', {}, {}, max_decision_steps=8)
        self.registry = Registry()
        self.lab = CounterLab()
        self.registry.register_lab('counter', lambda context, config: self.lab, version='1')

    def selection(self, name='single_loop', config=None, labs=None):
        return {'controller': {'name': name, 'config': config or {'operation': 'a.step', 'iterations': 2}},
                'labs': labs or {'a': {'name': 'counter'}}}

    def test_single_loop_requires_no_models_and_observes_each_result(self):
        modules = self.registry.build(self.selection())
        with patch('rsisys.runtime.model.MockModelBackend.generate', side_effect=AssertionError('model called')):
            with Application(self.config, modules=modules) as app:
                self.assertEqual(app.backends, {})
                self.assertIsNone(app.runner.decision_making.harness.current_model())
                self.assertEqual(app.run().status, 'finished')
                self.assertEqual(self.lab.calls, 2)
                self.assertEqual([r.data['count'] for r in app.runner.decision_making.harness.state.operation_results], [1, 2])
                manifest = json.loads((self.config.data_dir / 'module_assembly.json').read_text())
                self.assertEqual(manifest['method']['config']['controller']['version'], '1')
        self.assertEqual(self.lab.closed, 1)

    def test_external_policy_gets_detached_feedback_and_routes_multiple_labs(self):
        other, windows = CounterLab(), []
        self.registry.register_lab('other', lambda context, config: other, version='2')
        def choose(window):
            windows.append(window)
            if window.feedback is None:
                return Decision('invoke', operation=OperationCall('a.step', {}))
            if window.last_operation.name == 'a.step':
                self.assertEqual(window.feedback.data['count'], 1)
                window.feedback.data['count'] = -99
                return Decision('invoke', operation=OperationCall('b.step', {}))
            return Decision('finish')
        self.registry.register_controller('custom', lambda context, config: choose, version='3')
        selection = self.selection('custom', {'custom_option': 4},
            {'a': {'name': 'counter'}, 'b': {'name': 'other'}})
        with Application(self.config, modules=self.registry.build(selection)) as app:
            self.assertEqual(app.run().status, 'finished')
            self.assertEqual(app.runner.decision_making.harness.state.operation_results[0].data['count'], 1)
        self.assertEqual((self.lab.calls, other.calls, len(windows)), (1, 1, 3))
        self.assertEqual((self.lab.closed, other.closed), (1, 1))

    def test_failed_iteration_stops_and_keeps_failure_evidence(self):
        self.lab.fail = True
        with Application(self.config, modules=self.registry.build(self.selection())) as app:
            self.assertEqual(app.run().status, 'finished')
            self.assertEqual(self.lab.calls, 1)
            self.assertEqual(app.runner.decision_making.harness.state.operation_results[0].status, ExecutionStatus.FAILED)

    def test_host_permissions_apply_to_qualified_lab_names(self):
        with Application(self.config, modules=self.registry.build(self.selection()),
                         action_permissions=ActionPermissions(operations=set())) as app:
            with self.assertRaises(PermissionError):
                app.run()
            self.assertEqual(self.lab.calls, 0)
        self.assertEqual(self.lab.closed, 1)

    def test_meta_agent_uses_the_same_labs_and_fixed_host_model(self):
        model = ModelRef('llm', 'fixture-model', 'v1')
        config = replace(self.config, backends={'llm': BackendBinding('llm', 'mock')},
                         roles={'decision_maker': model})
        requests = []
        def generate(request):
            requests.append(request)
            payload = ({'kind': 'invoke', 'operation': 'a.step', 'arguments': {}}
                       if len(requests) == 1 else {'kind': 'finish'})
            return GenerationResponse(json.dumps(payload), request.model, simulated=True)
        selection = {'controller': {'name': 'meta_agent'}, 'labs': {'a': {'name': 'counter'}}}
        with patch('rsisys.runtime.model.MockModelBackend.generate', side_effect=generate):
            with Application(config, modules=self.registry.build(selection)) as app:
                self.assertTrue(hasattr(app.runner.decision_making.harness, 'agent_memory'))
                self.assertEqual(app.run().status, 'finished')
        self.assertEqual(self.lab.calls, 1)
        self.assertEqual(len(requests), 2)
        self.assertTrue(all(request.model == model for request in requests))

    def test_missing_meta_model_fails_before_execution_and_cleans_lab(self):
        selection = {'controller': {'name': 'meta_agent'}, 'labs': {'a': {'name': 'counter'}}}
        with self.assertRaisesRegex(ValueError, 'requires a decision_maker model'):
            Application(self.config, modules=self.registry.build(selection))
        self.assertEqual((self.lab.calls, self.lab.closed), (0, 1))

    def test_invalid_selection_and_duplicate_names_fail_before_factories(self):
        for selection in ({}, {'method': {}, 'labs': {}}, self.selection('unknown'),
                self.selection(config={'operation': 'a.step', 'iterations': 0}),
                self.selection(labs={'a.b': {'name': 'counter'}}),
                self.selection(labs={'a': {'name': 'unknown'}})):
            with self.subTest(selection=selection), self.assertRaises((ValueError, TypeError)):
                self.registry.build(selection)
        with self.assertRaisesRegex(ValueError, 'Duplicate'):
            self.registry.register_lab('counter', lambda c, p: None, version='2')
        self.assertEqual((self.lab.calls, self.lab.closed), (0, 0))

    def test_partial_lab_construction_failure_releases_previous_resources(self):
        def broken(context, config):
            raise RuntimeError('construction failed')
        self.registry.register_lab('broken', broken, version='1')
        selection = self.selection(labs={'a': {'name': 'counter'}, 'b': {'name': 'broken'}})
        with self.assertRaisesRegex(RuntimeError, 'construction failed'):
            Application(self.config, modules=self.registry.build(selection))
        self.assertEqual(self.lab.closed, 1)

    def test_cleanup_failure_retains_owner_and_retries_only_failed_lab(self):
        class RetryLab(CounterLab):
            def close(self):
                self.closed += 1
                if self.closed == 1:
                    raise RuntimeError('retry cleanup')
        retry = RetryLab()
        self.registry.register_lab('retry', lambda context, config: retry, version='1')
        app = Application(self.config, modules=self.registry.build(self.selection(
            labs={'a': {'name': 'counter'}, 'b': {'name': 'retry'}})))
        self.addCleanup(app.close)
        with self.assertRaises(ExceptionGroup):
            app.close()
        self.assertFalse(app._lock.closed)
        self.assertEqual((self.lab.closed, retry.closed), (1, 1))
        app.close()
        self.assertTrue(app._lock.closed)
        self.assertEqual((self.lab.closed, retry.closed), (1, 2))

    def test_builtin_local_process_lab_requires_explicit_host_and_releases_session(self):
        selection = self.selection(config={'operation': 'a.lab_status', 'iterations': 1},
                                   labs={'a': {'name': 'local_process'}})
        with self.assertRaisesRegex(ValueError, 'execution=local_process'):
            Application(self.config, modules=self.registry.build(selection))
        config = replace(self.config, data_dir=self.root / 'local', execution='local_process')
        with Application(config, modules=self.registry.build(selection)) as app:
            backend = app.sandbox
            self.assertEqual(app.run().status, 'finished')
        self.assertEqual(backend.active_count, 0)

    def test_invalid_controller_cannot_silently_become_meta_agent(self):
        self.registry.register_controller('invalid', lambda context, config: None, version='1')
        with self.assertRaisesRegex(TypeError, 'Controller factory'):
            Application(self.config, modules=self.registry.build(self.selection('invalid')))
        self.assertEqual(self.lab.closed, 1)

    def test_custom_state_requires_explicit_recovery_integration(self):
        with self.assertRaisesRegex(TypeError, 'state_factory'):
            Application(self.config, modules=self.registry.build(self.selection()), durable=True)
        self.assertFalse(self.config.data_dir.exists())

    def test_whole_method_registration_keeps_its_version_and_config(self):
        self.registry.register_method('whole', MethodDefinition(lambda config: ModuleAssembly(), '7'))
        modules = self.registry.build({'method': {'name': 'whole', 'config': {'native': 3}}})
        self.assertEqual(modules.method_identity.configuration()['config']['method'], {
            'protocol': 'rsisys-method/v1', 'method': 'whole', 'version': '7', 'config': {'native': 3}})

    def test_failed_plugin_registration_rolls_back_catalog(self):
        plugin = self.root / 'bad.py'
        plugin.write_text("def register(r):\n    r.register_lab('partial', lambda c, p: None, version='1')\n    raise ValueError('invalid plugin')\n")
        with self.assertRaisesRegex(ValueError, 'invalid plugin'):
            self.registry.load_plugin(plugin)
        with self.assertRaisesRegex(ValueError, 'Unknown Lab'):
            self.registry.build(self.selection(labs={'a': {'name': 'partial'}}))

    def test_external_cli_runs_from_an_unrelated_directory_with_public_imports(self):
        project = self.root / 'external-project'
        shutil.copytree(ROOT / 'examples/registered', project)
        environment = dict(os.environ, PYTHONPATH=str(ROOT / 'src'))
        for config in ('single-loop.json', 'custom-controller.json'):
            completed = subprocess.run([sys.executable, '-m', 'rsisys', 'run',
                '--config', str(project / config), '--plugin', str(project / 'plugin.py')],
                cwd=self.root, env=environment, capture_output=True, text=True, timeout=30)
            self.assertEqual(completed.returncode, 0, completed.stderr)
            output = json.loads(completed.stdout)
            self.assertEqual(output['status'], 'finished')
            self.assertFalse(output['snapshot_supported'])
            self.assertIsNone(output['training_performed'])
            self.assertEqual(output['framework']['method']['config']['plugins'][0]['name'], 'plugin.py')
        self.assertTrue((project / 'runs/single-loop/result.json').is_file())
        saved = json.loads((project / 'runs/single-loop/config.json').read_text())
        self.assertEqual(saved['roles'], {})
        self.assertEqual(saved['backends'], {})


if __name__ == '__main__':
    unittest.main()
