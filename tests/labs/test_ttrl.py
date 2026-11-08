"""No gold in TTRL rewards; stable votes, variance, and data split protection."""
import json
import unittest
from pathlib import Path
from tempfile import TemporaryDirectory
from types import SimpleNamespace

from rsisys.api import ExecutionStatus, OperationCall
from rsisys.labs.math_verifiers import StrictNumericVerifier
from rsisys.labs.ttrl import TTRLLab, majority_rewards


class TTRLTests(unittest.TestCase):
    def engine(self):
        return SimpleNamespace(name='test-v1', simulated=True, generate=lambda *a, **k: None,
                               train=lambda *a, **k: None)

    def test_majority_validity_ties_and_group_standardization(self):
        v = StrictNumericVerifier()
        result = majority_rewards([r'\boxed{2}', 'bad', r'\boxed{2}', r'\boxed{3}'], v)
        self.assertEqual(result['rewards'], [1, 0, 1, 0])
        self.assertEqual(result['majority_ratio'], 0.5)
        self.assertAlmostEqual(sum(result['advantages']), 0)
        self.assertGreater(result['advantages'][0], 0)
        self.assertLess(result['advantages'][1], 0)
        self.assertEqual(majority_rewards(['bad', 'bad'], v)['advantages'], [0, 0])
        self.assertEqual(majority_rewards([r'\boxed{3}', r'\boxed{2}'], v)['pseudo_answer'], r'\boxed{3}')

    def test_gold_columns_are_rejected_before_inference(self):
        with TemporaryDirectory() as tmp:
            base = Path(tmp, 'base')
            base.mkdir()
            data = Path(tmp, 'leaking.json')
            data.write_text(json.dumps([{'id': '1', 'prompt': 'question', 'answer': 'SECRET'}]))
            lab = TTRLLab(SimpleNamespace(data_dir=tmp), {}, engine=self.engine(), verifier=StrictNumericVerifier())
            lab.initial_model = str(base.resolve())
            result = lab.execute(OperationCall('rollout', {'round': 1, 'model': str(base),
                                                         'dataset': str(data), 'samples_per_prompt': 4}))
            self.assertEqual(result.status, ExecutionStatus.FAILED)
            self.assertIn('gold labels are forbidden', result.error)

    def test_reward_preserves_sampled_token_ids_and_ignores_heldout_answers(self):
        with TemporaryDirectory() as tmp:
            lab = TTRLLab(SimpleNamespace(data_dir=tmp), {}, engine=self.engine(), verifier=StrictNumericVerifier())
            artifact = lab.write('rollout', 1, [{'id': 'q', 'prompt': 'p', 'responses': [
                {'completion': r'\boxed{2}', 'response_ids': [101, 102]},
                {'completion': r'\boxed{3}', 'response_ids': [201, 202]}]}])
            # A reward operation has no access path for a labeled dataset.
            output = lab._reward({'round': 1, 'dataset': artifact['dataset']})
            rows = json.loads(Path(output['dataset']).read_text())
            self.assertEqual(rows[0]['response_ids'], [101, 102])
            self.assertEqual(rows[1]['response_ids'], [201, 202])
            self.assertEqual([r['reward'] for r in rows], [1, 0])
