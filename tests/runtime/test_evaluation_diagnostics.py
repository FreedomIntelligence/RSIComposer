from copy import deepcopy
import unittest

from rsisys.runtime.evaluation_diagnostics import prediction_diagnostics, compare_evaluations


class DiagnosticsTests(unittest.TestCase):
    def test_collapsed_answers_are_visible_without_changing_accuracy(self):
        scored = [{'benchmark_id': 'medqa', 'id': str(i), 'text': 'C' if i < 485 else '',
                   'extracted_answer': 'C' if i < 485 else None, 'correct': i < 129} for i in range(500)]
        before = deepcopy(scored)
        result = prediction_diagnostics(scored)
        group = result['benchmarks']['medqa']
        self.assertEqual(group['invalid'], 15)
        self.assertEqual(group['answer_counts'], {'C': 485})
        self.assertIn('answer_distribution_collapsed', group['warnings'])
        self.assertFalse(result['affects_score'])
        self.assertEqual(scored, before)

    def test_training_attribution_requires_matching_inference_and_protocol(self):
        original = {'status': 'succeeded', 'protocol_sha256': 'p', 'dataset_sha256': {'test': 'd'},
            'runtime_sha256': 'r', 'execution_budget': {'seconds': 10}, 'inference_sha256': 'i',
            'weight_sha256': 'w', 'metrics': {'test': {'accuracy': .6, 'n': 500}}}
        new = {**original, 'weight_sha256': 'w2', 'metrics': {'test': {'accuracy': .62, 'n': 500}}}
        comparison = compare_evaluations(original, new)
        self.assertTrue(comparison['weight_effect_isolated'])
        self.assertAlmostEqual(comparison['accuracy_delta']['test'], .02)
        self.assertFalse(compare_evaluations(original, {**new, 'inference_sha256': 'i2'})['weight_effect_isolated'])
        self.assertFalse(compare_evaluations(original, {**new, 'dataset_sha256': {'test': 'other'}})['comparable'])
        legacy = deepcopy(original)
        del legacy['runtime_sha256']
        self.assertFalse(compare_evaluations(legacy, new)['comparable'])


if __name__ == '__main__':
    unittest.main()
