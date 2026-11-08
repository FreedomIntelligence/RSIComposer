"""Registered weight-evolution methods execute through the actual host loop."""
import unittest
from pathlib import Path
from tempfile import TemporaryDirectory

from rsisys.api import (
    Application,
    ApplicationConfig,
    ExecutionStatus,
    OperationResult,
    OperationSpec,
    Registry,
    SEALController,
    TTRLController,
)


class RegisteredWeightTests(unittest.TestCase):
    def run_method(self, cls, fail=False):
        calls = []
        class Lab:
            def describe(self):
                return tuple(OperationSpec(name, name, {'type': 'object'}) for name in ('evaluate', *cls.phases))

            def execute(self, call):
                calls.append(call)
                args = call.arguments
                if fail:
                    return OperationResult('failed', (), True, ExecutionStatus.FAILED, 'known worker failure')
                data = {'request': args}
                if call.name == 'evaluate':
                    data['quality_metrics'] = {'score': 0.1}
                elif call.name == 'train':
                    data['model'] = f'model-{args["round"]}'
                else:
                    data.update(dataset=f'{call.name}-{args["round"]}', count=2, sha256='a' * 64)
                return OperationResult('simulated', (), True, data=data)
        registry = Registry()
        registry.register_lab('fixture_weights', lambda context, config: Lab(), version='test')
        modules = registry.build({'controller': {'name': cls.method, 'config': {
            'initial_model': 'base', 'train_data': 'train', 'eval_data': 'heldout', 'iterations': 2}},
            'labs': {cls.method: {'name': 'fixture_weights'}}})
        with TemporaryDirectory() as tmp:
            config = ApplicationConfig('registered methods', Path(tmp) / 'run', {}, {}, max_decision_steps=16)
            with Application(config, modules=modules) as app:
                result = app.run()
                report = app.controller_report()
                self.assertEqual(result.status, 'finished')
                self.assertFalse(report['training_performed'])
                if fail:
                    self.assertEqual(report['status'], 'failed')
                    self.assertEqual(report['rounds_completed'], 0)
                else:
                    self.assertEqual(report['status'], 'completed')
                    self.assertEqual(report['rounds_completed'], 2)
                    self.assertEqual(report['current_model'], 'model-2')
                    self.assertEqual(len(calls), 2 + 2 * len(cls.phases))

    def test_both_methods_follow_registered_host_execution(self):
        for cls in (SEALController, TTRLController):
            with self.subTest(method=cls.method):
                self.run_method(cls)

    def test_finished_transport_does_not_hide_algorithm_failure(self):
        self.run_method(TTRLController, fail=True)
