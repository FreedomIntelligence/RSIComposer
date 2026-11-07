import unittest
from examples.medical.training_support import token_ids, sft_example, padded_examples


class Tokenizer:
    def apply_chat_template(self, messages, **kwargs):
        result = ''.join(m['role'] + ':' + m['content'] + '\n' for m in messages)
        return result + ('assistant:' if kwargs['add_generation_prompt'] else '')

    def __call__(self, text, **kwargs):
        return {'input_ids': [ord(ch) for ch in text], 'attention_mask': [1] * len(text)}


class TrainingSupportTests(unittest.TestCase):
    def test_tokenizer_mapping_and_single_batch_are_normalized_without_string_dimensions(self):
        self.assertEqual(token_ids({'input_ids': [[1, 2, 3]]}), [1, 2, 3])
        for bad in ({'attention_mask': [1]}, ['input_ids'], ['123'], [[1], [2]], [True], [-1]):
            with self.subTest(bad=bad), self.assertRaises(ValueError):
                token_ids(bad)

    def test_assistant_only_loss_mask_and_padding_have_equal_lengths(self):
        first = sft_example(Tokenizer(), [{'role': 'user', 'content': 'Question?'}], 'A')
        second = sft_example(Tokenizer(), [{'role': 'user', 'content': 'Longer question?'}], 'B')
        batch = padded_examples([first, second], 0)
        self.assertEqual(len(batch['input_ids'][0]), len(batch['labels'][0]))
        self.assertEqual(batch['labels'][0][-1], -100)
        self.assertEqual(batch['attention_mask'][0][-1], 0)
        self.assertEqual([v for v in first['labels'] if v != -100], [ord('A'), ord('\n')])
        with self.assertRaisesRegex(ValueError, 'supervised response'):
            sft_example(Tokenizer(), [{'role': 'user', 'content': 'Question?'}], 'A', max_tokens=2)


if __name__ == '__main__':
    unittest.main()
