"""Two module boundaries, reusable internal components and the host runtime."""
from dataclasses import replace
import json
from pathlib import Path
import tempfile
import time
import unittest
from unittest.mock import patch

from rsisys.api import (Application, ApplicationConfig, DecisionPolicy, RSIController,
    Lab, LabExecution, LabRequest, LabFeedback, OperationCall, OperationResult, Registry)
from tests.support.math import make_math_selection
from rsisys.labs.math import MathLab
from rsisys.labs.math_engines import FixtureMathEngine
from rsisys.labs.math_verifiers import StrictNumericVerifier
from rsisys.loops.contracts.model_service import BackendBinding, ModelRef



class ModuleBoundaryTests(unittest.TestCase):
    def setUp(self):
        tmp = tempfile.TemporaryDirectory()
        self.addCleanup(tmp.cleanup)
        self.root = Path(tmp.name)
        self.config = ApplicationConfig('test modules', self.root / 'run', {}, {}, max_decision_steps=12)
        self.data = self.root / 'data'
        self.selection = make_math_selection(self.data)

    def test_public_names_share_existing_runtime_contracts(self):
        self.assertIs(RSIController, DecisionPolicy)
        self.assertIs(Lab, LabExecution)
        self.assertIs(LabRequest, OperationCall)
        self.assertIs(LabFeedback, OperationResult)

    def test_builtin_rest_em_and_math_need_no_plugin(self):
        with Application(self.config, modules=Registry().build(self.selection)) as app:
            self.assertEqual(app.run().status, 'finished')
            receipts = app.runner.decision_making.harness.state.operation_results
            self.assertEqual(len(receipts), 9)
            self.assertEqual(receipts[3].data['manifest']['engine'], 'fixture-v1')
            self.assertEqual(receipts[7].data['request']['initial_model'], 'fixture-initial')
            self.assertEqual(app.modules.method_identity.configuration()['config']['plugins'], [])

    def test_same_lab_with_another_controller_uses_same_runtime(self):
        self.selection['controller'] = {'name': 'single_loop', 'config': {
            'operation': 'math.evaluate', 'iterations': 1,
            'arguments': {'round': 0, 'model': 'fixture-initial', 'dataset': 'eval.jsonl'}}}
        with Application(self.config, modules=Registry().build(self.selection)) as app:
            self.assertEqual(app.run().status, 'finished')
            self.assertEqual(len(app.runner.decision_making.harness.state.operation_results), 1)

    def test_replace_internal_engine_and_verifier_without_changing_controller(self):
        trained = []
        class RecordingEngine(FixtureMathEngine):
            name = 'recording-fixture-v1'
            def train(self, ref, rows, output, round_number):
                trained.append((ref, rows, round_number))
        class SelectiveVerifier(StrictNumericVerifier):
            name = 'only-eight-fixture-v1'
            def is_correct(self, completion, answer):
                return self.parse(answer) == 8 and super().is_correct(completion, answer)
        registry = Registry()
        registry.register_lab('custom_math', lambda context, config: MathLab(
            context, config, engine=RecordingEngine(), verifier=SelectiveVerifier()), version='1')
        self.selection['labs']['math']['name'] = 'custom_math'
        with Application(self.config, modules=registry.build(self.selection)) as app:
            self.assertEqual(app.run().status, 'finished')
            receipts = app.runner.decision_making.harness.state.operation_results
            self.assertEqual([receipts[i].data['count'] for i in (2, 6)], [1, 1])
            self.assertEqual(receipts[3].data['manifest']['verifier'], 'only-eight-fixture-v1')
        self.assertEqual([r[0] for r in trained], ['fixture-initial', 'fixture-initial'])
        self.assertTrue(all('answer' not in row for _, rows, _ in trained for row in rows))

    def test_math_lab_uses_requested_initial_model_not_its_own_algorithm(self):
        lab = MathLab(type('Context', (), {'data_dir': self.root})(), {'engine': 'fixture'})
        lab.initial = 'evaluated-baseline'
        accepted = lab._write(lab.root / 'round-1/accepted.jsonl',
            [{'id': '1', 'prompt': 'test', 'completion': '#### 1'}])
        result = lab._train({'round': 1, 'initial_model': 'controller-selected-base', 'dataset': accepted['dataset']})
        self.assertEqual(result['manifest']['initial_model'], 'controller-selected-base')

    def test_meta_agent_selects_existing_dsh_adapter_internally(self):
        config = replace(self.config, backends={'llm': BackendBinding('llm', 'mock')},
                         roles={'decision_maker': ModelRef('llm', 'test-model', 'v1')})
        native = {'home': str(self.root / 'dsh-home'), 'workspace': str(self.root / 'workspace'),
                  'model': 'test-model', 'session_id': 'test-session'}
        self.selection['controller'] = {'name': 'meta_agent', 'config': {
            'harness': 'dsh', 'memory': True, 'dsh': native}}
        from tests.runtime.test_dsh_harness import Runtime
        runtime = Runtime([
            json.dumps({'kind': 'invoke', 'operation': 'math.evaluate', 'arguments': {
                'round': 0, 'model': 'fixture-initial', 'dataset': 'eval.jsonl'}}),
            '{"kind":"finish"}'])
        with patch('rsisys.loops.decision_making.harness.dsh._sdk_runtime', return_value=runtime):
            with Application(config, modules=Registry().build(self.selection)) as app:
                deadline = time.monotonic() + 3
                result = app.run()
                while result.status != 'finished' and time.monotonic() < deadline:
                    time.sleep(.002)
                    result = app.run()
                self.assertEqual(result.status, 'finished')
                self.assertEqual(len(app.runner.decision_making.harness.state.operation_results), 1)
                self.assertIn('quality_metrics', runtime.prompts[-1])
        self.assertEqual(runtime.opened, [('start', 'test-session')])
        self.assertTrue(runtime.closed)

    def test_multiple_math_labs_own_separate_output_directories(self):
        self.selection['labs']['other'] = {'name': 'math', 'config': {
            'engine': 'fixture', 'data_root': str(self.data), 'output_subdir': 'other-math'}}
        with Application(self.config, modules=Registry().build(self.selection)) as app:
            self.assertEqual(app.run().status, 'finished')
            names = {spec.name for spec in app.assembly.lab_execution.describe()}
            self.assertIn('math.train', names)
            self.assertIn('other.train', names)
        self.assertTrue((self.config.data_dir / 'other-math').is_dir())

    def test_invalid_components_and_harness_config_fail_early(self):
        for config in ({'harness': 'unknown'}, {'harness': 'dsh'}, {'dsh': {}}, {'memory': 'yes'}):
            self.selection['controller'] = {'name': 'meta_agent', 'config': config}
            with self.subTest(config=config), self.assertRaises((ValueError, TypeError)):
                Registry().build(self.selection)
        with self.assertRaises(TypeError):
            MathLab(type('Context', (), {'data_dir': self.root})(), {}, engine=object())
        self.assertFalse((self.root / 'math-lab').exists())


if __name__ == '__main__':
    unittest.main()
