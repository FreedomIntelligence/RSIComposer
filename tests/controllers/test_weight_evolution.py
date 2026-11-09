"""Decision invariants shared by registered SEAL and TTRL implementations."""
import unittest
from copy import deepcopy
from types import SimpleNamespace

from rsisys.api import (
    DecisionInput,
    ExecutionStatus,
    OperationResult,
    Registry,
    SEALController,
    TTRLController,
)


class WeightControllerTests(unittest.TestCase):
    def make(self, cls):
        return cls(initial_model='base', train_data='train', eval_data='heldout')

    def feedback(self, c, **values):
        pending = deepcopy(c.pending)
        return DecisionInput('learn', last_operation=pending,
            feedback=OperationResult('done', (), False, data={'request': pending.arguments, **values}))

    def test_two_rounds_continue_current_weights_and_evaluate_only_before_and_after(self):
        for cls in (SEALController, TTRLController):
            c = cls(initial_model='base', train_data=['train-1', 'train-2'], eval_data='heldout')
            action = c.decide(DecisionInput('learn'))
            self.assertEqual(action.operation.name, c.method + '.evaluate')
            action = c.decide(self.feedback(c, quality_metrics={'score': 0.1}))
            for round_number in (1, 2):
                for stage in c.phases:
                    self.assertEqual(action.operation.name, c.method + '.' + stage)
                    if stage == c.phases[0]:
                        self.assertEqual(action.operation.arguments['dataset'], f'train-{round_number}')
                    self.assertEqual(action.operation.arguments['model'], 'base' if round_number == 1 else 'model-1')
                    if stage == 'train':
                        action = c.decide(self.feedback(c, model=f'model-{round_number}', training_performed=True))
                    else:
                        action = c.decide(self.feedback(c, dataset=f'{stage}-{round_number}', count=4, sha256='a' * 64))
            self.assertEqual(action.operation.arguments['dataset'], 'heldout')
            self.assertEqual(action.operation.arguments['model'], 'model-2')
            action = c.decide(self.feedback(c, quality_metrics={'score': 0.0}))
            self.assertEqual(action.kind, 'finish')
            self.assertEqual(c.current_model, 'model-2')
            self.assertEqual(len(c.history), 2)

    def test_forged_or_missing_receipt_and_false_training_claim_rejected(self):
        c = self.make(SEALController)
        c.decide(DecisionInput('learn'))
        with self.assertRaises(RuntimeError):
            c.decide(DecisionInput('learn'))
        bad = self.feedback(c, quality_metrics={'score': 0.0})
        bad.feedback.data['request']['round'] = 999
        with self.assertRaises(ValueError):
            c.decide(bad)
        c.decide(self.feedback(c, quality_metrics={'score': 0.0}))
        for _ in range(3):
            c.decide(self.feedback(c, dataset='stage', count=2, sha256='a' * 64))
        with self.assertRaises(ValueError):
            c.decide(self.feedback(c, model='new', training_performed=False))
        self.assertEqual(c.current_model, 'base')

    def test_empty_artifact_or_failed_worker_stops_without_update(self):
        for cls in (SEALController, TTRLController):
            c = self.make(cls)
            c.decide(DecisionInput('learn'))
            c.decide(self.feedback(c, quality_metrics={'score': 0.0}))
            action = c.decide(self.feedback(c, dataset='empty', count=0, sha256='0' * 64))
            self.assertEqual(action.kind, 'finish')
            c = self.make(cls)
            op = c.decide(DecisionInput('learn')).operation
            action = c.decide(DecisionInput('learn', last_operation=op,
                feedback=OperationResult('fail', (), False, ExecutionStatus.FAILED, 'worker failed')))
            self.assertEqual(action.kind, 'finish')
            self.assertEqual(c.current_model, 'base')

    def test_registered_config_rejects_overlap_invalid_width_and_unknown_options(self):
        for cls in (SEALController, TTRLController):
            for override in ({'train_data': 'heldout'}, {'iterations': True}, {cls.option: 0}, {'unknown': 1}):
                with self.assertRaises(ValueError):
                    Registry().build({'controller': {'name': cls.method, 'config': {
                        'initial_model': 'base', 'train_data': 'train', 'eval_data': 'heldout', **override}},
                        'labs': {'seal': {'name': 'seal'}}})

    def test_incompatible_lab_fails_at_assembly_before_any_execution(self):
        modules = Registry().build({'controller': {'name': 'ttrl', 'config': {
            'initial_model': 'base', 'train_data': 'train', 'eval_data': 'evaluation'}},
            'labs': {'ttrl': {'name': 'math'}}})
        context = SimpleNamespace(lab_execution=SimpleNamespace(describe=lambda: ()), add_owned_cleanup=lambda x: None)
        with self.assertRaisesRegex(ValueError, 'requires missing Lab operations'):
            modules.decision_factory(context, None)
