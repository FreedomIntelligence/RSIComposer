"""Integration checks for the single_loop Controller/Lab binding."""
from pathlib import Path
from tempfile import TemporaryDirectory
import unittest

from rsicomposer.api import Application, ApplicationConfig, OperationResult, OperationSpec, Registry


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
