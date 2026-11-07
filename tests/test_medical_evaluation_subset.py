"""Subset identities are fixed independently of scores and preserve inputs."""
import hashlib
import json
from pathlib import Path
from tempfile import TemporaryDirectory
import unittest

from examples.medical.evaluation_subset import freeze_evaluation_subset
from examples.medical.official_protocol import official_inputs


class MedicalEvaluationSubsetTests(unittest.TestCase):
    def parent(self, root, name='parent', rows=30):
        path = root / (name + '.jsonl')
        lines = [json.dumps({'id': str(i),
            'question': 'Q' + str(i) + '\nAnswer Choices: (A) a (B) b (C) c (D) d',
            'options': dict(zip('ABCD', 'abcd')), 'gold': 'ABCD'[i % 4]},
            separators=(',', ':')).encode() + b'\n' for i in range(rows)]
        raw = b''.join(lines)
        path.write_bytes(raw)
        return {'benchmark_id': 'medqa', 'path': str(path),
                'sha256': hashlib.sha256(raw).hexdigest()}, lines

    def test_deterministic_membership_preserves_original_bytes_order_and_parent(self):
        with TemporaryDirectory() as directory:
            root = Path(directory)
            group, lines = self.parent(root)
            before = Path(group['path']).read_bytes()
            first, manifest = freeze_evaluation_subset([group], root / 'first', count=12)
            second, repeated = freeze_evaluation_subset([group], root / 'second', count=12)
            self.assertEqual(manifest, repeated)
            indices = manifest['groups'][0]['selected_indices']
            self.assertEqual(indices, sorted(indices))
            self.assertEqual(len(set(indices)), 12)
            self.assertEqual(Path(first[0]['path']).read_bytes(), b''.join(lines[i] for i in indices))
            self.assertEqual(first[0]['sha256'], second[0]['sha256'])
            self.assertEqual(Path(group['path']).read_bytes(), before)
            self.assertEqual(len(official_inputs(first)[1]), 12)
            with self.assertRaises(FileExistsError):
                freeze_evaluation_subset([group], root / 'first', count=12)

    def test_membership_does_not_depend_on_answers(self):
        with TemporaryDirectory() as directory:
            root = Path(directory)
            original, lines = self.parent(root)
            altered = root / 'altered.jsonl'
            altered.write_bytes(b''.join(json.dumps({**json.loads(line), 'gold': 'D'}).encode()
                                         + b'\n' for line in lines))
            changed = {**original, 'path': str(altered),
                       'sha256': hashlib.sha256(altered.read_bytes()).hexdigest()}
            _, first = freeze_evaluation_subset([original], root / 'first', count=10)
            _, second = freeze_evaluation_subset([changed], root / 'second', count=10)
            self.assertEqual(first['groups'][0]['selected_ids'], second['groups'][0]['selected_ids'])

    def test_invalid_counts_and_parent_identity_fail_without_creating_subset(self):
        with TemporaryDirectory() as directory:
            root = Path(directory)
            group, _ = self.parent(root, rows=5)
            for count in (0, True, 6):
                with self.subTest(count=count), self.assertRaises(ValueError):
                    freeze_evaluation_subset([group], root / 'subset', count=count)
                self.assertFalse((root / 'subset').exists())
            with self.assertRaises(ValueError):
                freeze_evaluation_subset([{**group, 'sha256': 'wrong'}], root / 'subset', count=3)
            self.assertFalse((root / 'subset').exists())
