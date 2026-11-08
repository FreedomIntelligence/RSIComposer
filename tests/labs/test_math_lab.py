from pathlib import Path
from types import SimpleNamespace
import tempfile
import unittest
from rsisys.api import OperationCall, ExecutionStatus
from rsisys.labs.math import MathLab
from tests.support.math import make_math_selection

class RestEMTests(unittest.TestCase):
    def test_overlap_rejected_before_generation_and_failure_preserved(self):
        with tempfile.TemporaryDirectory() as directory:
            make_math_selection(Path(directory) / 'data')
            lab = MathLab(SimpleNamespace(data_dir=directory),
                          {'engine': 'fixture', 'data_root': str(Path(directory) / 'data')})
            call = OperationCall('evaluate', {'round': 0, 'model': 'base', 'dataset': 'train.jsonl'})
            self.assertEqual(lab.execute(call).status, ExecutionStatus.COMPLETED)
            failure = lab.execute(OperationCall('generate', {'round': 1, 'model': 'base',
                'dataset': 'train.jsonl', 'samples_per_prompt': 4}))
            self.assertEqual(failure.status, ExecutionStatus.FAILED)
            self.assertIn('overlap', failure.error)

    def test_training_cannot_substitute_gold_data_or_mutate_accepted_artifact(self):
        with tempfile.TemporaryDirectory() as directory:
            make_math_selection(Path(directory) / 'data')
            lab = MathLab(SimpleNamespace(data_dir=directory),
                          {'engine': 'fixture', 'data_root': str(Path(directory) / 'data')})
            lab.initial = 'base'
            with self.assertRaisesRegex(ValueError, 'preceding stage'):
                lab._train({'round': 1, 'initial_model': 'base', 'dataset': 'train.jsonl'})
            artifact = lab._write(lab.root / 'round-1/accepted.jsonl', [
                {'id': 'one', 'prompt': 'test', 'completion': '#### 1'}])
            Path(artifact['dataset']).write_text('{}\n')
            with self.assertRaisesRegex(ValueError, 'changed after publication'):
                lab._train({'round': 1, 'initial_model': 'base', 'dataset': artifact['dataset']})
