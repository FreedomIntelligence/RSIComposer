"""SEAL domain leakage, stable selection, and immutable artifact checks."""
import json
import unittest
from pathlib import Path
from tempfile import TemporaryDirectory
from types import SimpleNamespace

from rsisys.api import ExecutionStatus, OperationCall
from rsisys.labs.seal import SEALLab, squad_scores, train_sequences


class SealTests(unittest.TestCase):
    def engine(self):
        return SimpleNamespace(name='test-v1', simulated=True, generate=lambda *a, **k: None,
                               adapt=lambda *a, **k: None, train=lambda *a, **k: None)

    def test_squad_scoring_and_context_sequence(self):
        self.assertEqual(squad_scores('The Denver Broncos!', 'Denver Broncos')['f1'], 1)
        self.assertEqual(squad_scores('Paris', 'London')['f1'], 0)
        self.assertEqual(train_sequences('intro\nImplications: fact1 --- fact2', 'context', 'Title'),
                         ['Title\nfact1', 'Title\nfact2', 'Title\ncontext'])

    def test_top_one_uses_measured_reward_and_never_injects_qa_answers(self):
        with TemporaryDirectory() as tmp:
            lab = SEALLab(SimpleNamespace(data_dir=tmp), {}, engine=self.engine())
            rows = [{'doc_index': 0, 'candidate': k, 'prompt': 'article only', 'completion': f'edit-{k}',
                     'adapter_mean': score, 'questions': [{'answer': 'PRIVATE GOLD'}]}
                    for k, score in enumerate([0.1, 0.9, 0.9])]
            artifact = lab.write('adapt', 1, rows)
            selected = lab._select({'round': 1, 'dataset': artifact['dataset']})
            accepted = json.loads(Path(selected['dataset']).read_text())
            self.assertEqual(accepted[0]['candidate'], 1)
            self.assertNotIn('PRIVATE GOLD', json.dumps(accepted))
            Path(artifact['dataset']).write_text('[]')
            with self.assertRaises(ValueError):
                lab._select({'round': 1, 'dataset': artifact['dataset']})

    def test_engine_failure_is_terminal_and_receipt_retained(self):
        with TemporaryDirectory() as tmp:
            Path(tmp, 'base').mkdir()
            Path(tmp, 'heldout.json').write_text('[]')
            lab = SEALLab(SimpleNamespace(data_dir=tmp), {'data_root': tmp}, engine=self.engine())
            result = lab.execute(OperationCall('evaluate', {'round': 0, 'model': 'base', 'dataset': 'heldout.json'}))
            self.assertEqual(result.status, ExecutionStatus.FAILED)
            self.assertTrue((lab.root / 'round-0/evaluate-receipt.json').is_file())

    def test_invalid_engine_or_unversioned_scorer_rejected_before_workspace_creation(self):
        with TemporaryDirectory() as tmp:
            context = SimpleNamespace(data_dir=tmp)
            with self.assertRaises(TypeError):
                SEALLab(context, {}, engine=object())
            with self.assertRaises(TypeError):
                SEALLab(context, {}, scorer=lambda p, g: {})
            self.assertFalse((Path(tmp) / 'seal').exists())
