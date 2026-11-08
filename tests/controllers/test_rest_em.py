"""ReST^EM controller contracts, without ML dependencies."""
from copy import deepcopy
import subprocess
import sys
import unittest

from rsisys.api import (DecisionInput, ExecutionStatus,
                        OperationCall, OperationResult, Registry, RestEMController)


class RestEMTests(unittest.TestCase):
    def controller(self, **overrides):
        return RestEMController(**{'initial_model': 'base', 'train_data': 'train', 'eval_data': 'heldout', **overrides})

    def receipt(self, controller, **extra):
        pending = deepcopy(controller.pending)
        data = {'request': pending.arguments, **extra}
        return DecisionInput('test', feedback=OperationResult('done', (), True, data=data), last_operation=pending)

    def baseline(self, controller):
        self.assertEqual(controller.decide(DecisionInput('test')).operation.name, 'math.evaluate')
        return controller.decide(self.receipt(controller, quality_metrics={'accuracy': 0.5}))

    def test_two_rounds_current_generation_fresh_initial_training_no_accumulation(self):
        controller = self.controller()
        action = self.baseline(controller)
        for round_number in (1, 2):
            self.assertEqual(action.operation.arguments['model'], 'base' if round_number == 1 else 'model-1')
            self.assertEqual(action.operation.arguments['dataset'], 'train')
            self.assertEqual(action.operation.arguments['round'], round_number)
            action = controller.decide(self.receipt(controller, dataset=f'generated-{round_number}', count=8))
            self.assertEqual(action.operation.name, 'math.filter')
            action = controller.decide(self.receipt(controller, dataset=f'accepted-{round_number}', count=2))
            self.assertEqual(action.operation.arguments, {'round': round_number,
                'initial_model': 'base', 'dataset': f'accepted-{round_number}'})
            action = controller.decide(self.receipt(controller, model=f'model-{round_number}'))
            self.assertEqual(action.operation.arguments['model'], f'model-{round_number}')
            self.assertEqual(action.operation.arguments['dataset'], 'heldout')
            # Regression is measured, not silently turned into best-model selection.
            action = controller.decide(self.receipt(controller, quality_metrics={'accuracy': 0.5 / round_number}))
        self.assertEqual(action.kind, 'finish')
        self.assertEqual(controller.current_model, 'model-2')
        self.assertEqual(len(controller.history), 3)
        self.assertEqual(controller.decide(DecisionInput('test')).kind, 'finish')

    def test_missing_stale_or_forged_receipt_cannot_advance(self):
        for mode in ('missing', 'operation', 'request'):
            with self.subTest(mode=mode):
                controller = self.controller()
                self.baseline(controller)
                window = self.receipt(controller, dataset='samples', count=8)
                if mode == 'missing':
                    window = DecisionInput('test')
                elif mode == 'operation':
                    window = DecisionInput('test', feedback=window.feedback, last_operation=OperationCall('math.train', {}))
                else:
                    window.feedback.data['request']['round'] = 99
                with self.assertRaises((ValueError, RuntimeError)):
                    controller.decide(window)
                self.assertEqual(controller.stage, 'generate')

    def test_empty_acceptance_stops_before_training(self):
        controller = self.controller()
        self.baseline(controller)
        controller.decide(self.receipt(controller, dataset='samples', count=4))
        result = controller.decide(self.receipt(controller, dataset='empty', count=0))
        self.assertEqual(result.kind, 'finish')
        self.assertIn('No accepted samples', result.reason)
        self.assertEqual(controller.current_model, 'base')

    def test_failure_or_cancel_stops_at_every_stage(self):
        for stage in ('evaluate', 'generate', 'filter', 'train', 'candidate_evaluate'):
            for status in (ExecutionStatus.FAILED, ExecutionStatus.CANCELLED):
                with self.subTest(stage=stage, status=status):
                    controller = self.controller()
                    controller.decide(DecisionInput('test'))
                    if stage != 'evaluate':
                        controller.decide(self.receipt(controller, quality_metrics={'accuracy': 0.0}))
                    if stage in ('filter', 'train', 'candidate_evaluate'):
                        controller.decide(self.receipt(controller, dataset='samples', count=4))
                    if stage in ('train', 'candidate_evaluate'):
                        controller.decide(self.receipt(controller, dataset='accepted', count=2))
                    if stage == 'candidate_evaluate':
                        controller.decide(self.receipt(controller, model='candidate'))
                    result = controller.decide(DecisionInput('test', last_operation=controller.pending,
                        feedback=OperationResult('failed', (), True, status, 'failure' if status == ExecutionStatus.FAILED else None)))
                    self.assertEqual(result.kind, 'finish')
                    self.assertEqual(controller.current_model, 'base')

    def test_bad_configs_rejected_at_registration(self):
        registry = Registry()
        for override in ({'iterations': True}, {'samples_per_prompt': 0}, {'lab': 'a.b'},
                         {'initial_model': ''}, {'train_data': 'heldout'}, {'surprise': 1}):
            with self.subTest(override=override), self.assertRaises(ValueError):
                registry.build({'controller': {'name': 'rest_em', 'config': {
                    'initial_model': 'base', 'train_data': 'train', 'eval_data': 'heldout', **override}},
                    'labs': {'math': {'name': 'local_process'}}})

    def test_invalid_counts_metrics_and_model_identity(self):
        controller = self.controller()
        controller.decide(DecisionInput('test'))
        with self.assertRaises(ValueError):
            controller.decide(self.receipt(controller, quality_metrics={}))
        controller.decide(self.receipt(controller, quality_metrics={'accuracy': 0.0}))
        with self.assertRaises(ValueError):
            controller.decide(self.receipt(controller, dataset='samples', count=True))
        controller.decide(self.receipt(controller, dataset='samples', count=2))
        with self.assertRaises(ValueError):
            controller.decide(self.receipt(controller, dataset='accepted', count=3))
        controller.decide(self.receipt(controller, dataset='accepted', count=1))
        with self.assertRaises(ValueError):
            controller.decide(self.receipt(controller, model='base'))

    def test_library_import_does_not_import_training_dependencies(self):
        process = subprocess.run([sys.executable, '-c',
            'import sys; from rsisys.api import RestEMController; '
            'assert "torch" not in sys.modules; assert "transformers" not in sys.modules'],
            capture_output=True, text=True)
        self.assertEqual(process.returncode, 0, process.stderr)
