"""Deterministic training/development split, without model/data downloads."""
import unittest

from experiments.rest_em import prepare_gsm8k as prepare
from experiments.rest_em.prepare_math import partition


class GSM8KPreparationTests(unittest.TestCase):
    def test_partition_is_repeatable_disjoint_and_retains_official_identity(self):
        rows = [{'question': f'question {i}', 'answer': f'#### {i}'} for i in range(20)]
        train, dev = prepare.split_train(rows, 5, 7, 73)
        self.assertEqual((train, dev), prepare.split_train(rows, 5, 7, 73))
        self.assertEqual((len(train), len(dev)), (5, 7))
        self.assertFalse({r['id'] for r in train} & {r['id'] for r in dev})
        self.assertNotEqual((train, dev), prepare.split_train(rows, 5, 7, 74))
        for row in train + dev:
            index = int(row['id'].rsplit('/', 1)[1])
            self.assertEqual(row['prompt'], rows[index]['question'])
            self.assertEqual(row['answer'], rows[index]['answer'])
            self.assertIn('/train/', row['id'])

    def test_invalid_sizes_or_duplicate_questions_rejected(self):
        rows = [{'question': 'same question', 'answer': '#### 1'}] * 5
        with self.assertRaisesRegex(ValueError, 'Duplicate questions'):
            prepare.split_train(rows, 2, 2, 73)
        for train, dev in ((0, 1), (1, 0), (4, 4)):
            with self.assertRaises(ValueError):
                prepare.split_train(rows, train, dev, 73)

    def test_few_shot_excludes_all_target_questions_and_duplicates(self):
        raw = [{'question': f'q{i}', 'answer': f'#### {i}'} for i in range(12)]
        excluded = [{'prompt': 'q0'}, {'prompt': 'q2'}, {'prompt': 'q5'}]
        shots = prepare.few_shot_rows(raw + raw, excluded, 8)
        self.assertEqual(len(shots), 8)
        self.assertFalse({r['prompt'] for r in shots} & {r['prompt'] for r in excluded})
        self.assertEqual(shots[0]['id'], 'gsm8k/main/train/1')
        with self.assertRaisesRegex(ValueError, 'Not enough'):
            prepare.few_shot_rows(raw, excluded, 10)


class MathPreparationTests(unittest.TestCase):
    def test_train_dev_demonstrations_and_test_are_disjoint(self):
        rows = [{'id': str(i), 'prompt': f'problem {i}', 'answer': 'solution'} for i in range(20)]
        test = [rows[0], rows[4]]
        a = partition(rows + rows, test, 5, 5, 4)
        self.assertEqual(a, partition(rows + rows, test, 5, 5, 4))
        self.assertEqual(list(map(len, a)), [5, 5, 4])
        all_prompts = [r['prompt'] for group in a for r in group]
        self.assertEqual(len(all_prompts), len(set(all_prompts)))
        self.assertFalse(set(all_prompts) & {r['prompt'] for r in test})
        for bad in (0, -1, True):
            with self.assertRaises(ValueError):
                partition(rows, test, bad, 1, 1)
