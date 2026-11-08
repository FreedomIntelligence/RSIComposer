"""Base-model prompts must not silently require a chat tokenizer or leak targets."""
import json
from pathlib import Path
import tempfile
import unittest
from rsisys.labs.math import validate_config
from rsisys.labs.math_engines import TransformersMathEngine


class MathPromptTests(unittest.TestCase):
    def test_completion_prompt_and_stop_delimiter(self):
        with tempfile.TemporaryDirectory() as temporary:
            path = Path(temporary) / 'shots.jsonl'
            path.write_text(json.dumps({'prompt': 'A demonstration', 'answer': 'reasoning\n#### 2'}) + '\n')
            engine = TransformersMathEngine({'prompt_format': 'completion', 'few_shot_data': str(path)})
            # An object without a chat template must work for base-model prompting.
            prefix = engine._prefix(object(), 'The target problem')
            self.assertIn('Question: A demonstration\nAnswer: reasoning\n#### 2', prefix)
            self.assertTrue(prefix.endswith('Question: The target problem\nAnswer:'))
            with self.assertRaisesRegex(ValueError, 'overlaps'):
                engine._prefix(object(), ' A demonstration ')
            class Tokenizer:
                def decode(self, tokens, **kwargs):
                    return 'my solution\n#### 42\n\nQuestion: unrelated problem'
            self.assertEqual(engine._decode(Tokenizer(), []), 'my solution\n#### 42')

    def test_malformed_demonstrations_and_configuration_rejected(self):
        with tempfile.TemporaryDirectory() as temporary:
            path = Path(temporary) / 'bad.jsonl'
            for contents in ('', '{}', '{"prompt":"x","answer":"unparseable"}'):
                path.write_text(contents)
                with self.assertRaisesRegex(ValueError, 'Few-shot data'):
                    TransformersMathEngine({'prompt_format': 'completion', 'few_shot_data': str(path)})
        for config in ({'prompt_format': 'invalid'}, {'few_shot_data': 'ignored.jsonl'},
                       {'prompt_format': 'completion', 'few_shot_data': ''}):
            with self.assertRaises(ValueError):
                validate_config(config)
