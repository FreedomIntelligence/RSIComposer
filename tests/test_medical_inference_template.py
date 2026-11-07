from contextlib import nullcontext, redirect_stdout, redirect_stderr
import io
import json
from types import SimpleNamespace
import unittest
from unittest.mock import patch

from examples.medical.templates import infer


class InferenceTemplateTests(unittest.TestCase):
    def test_raw_response_reaches_host_without_first_character_extraction(self):
        calls = {}
        class Encoded(dict):
            def to(self, device):
                return self
        class Output:
            def __getitem__(self, index):
                self.index = index
                return [1, 2, 3]
        class Tokenizer:
            eos_token_id = 0
            def apply_chat_template(self, messages, **kwargs):
                calls['template'] = kwargs
                return 'complete question and options'
            def __call__(self, prompt, **kwargs):
                return Encoded(input_ids=SimpleNamespace(shape=(1, 10)))
            def decode(self, tokens, **kwargs):
                return 'Among A through D, the answer is B'
        class Model:
            device = 'cpu'
            def eval(self):
                pass
            def generate(self, **kwargs):
                calls['generation'] = kwargs
                return Output()
        transformers = SimpleNamespace(
            AutoTokenizer=SimpleNamespace(from_pretrained=lambda *args, **kwargs: Tokenizer()),
            AutoModelForCausalLM=SimpleNamespace(from_pretrained=lambda *args, **kwargs: Model()))
        payload = {'protocol': 'synthetic', 'items': [{'benchmark_id': 'fixture', 'id': '1',
            'messages': [{'role': 'user', 'content': 'complete question and options'}]}]}
        out, err = io.StringIO(), io.StringIO()
        with patch.dict('sys.modules', {'torch': SimpleNamespace(bfloat16='bf16', inference_mode=nullcontext),
                                      'transformers': transformers}), \
             patch.object(infer.Path, 'read_text', return_value=json.dumps({'base_model_path': '/synthetic/base'})), \
             patch('sys.stdin', io.StringIO(json.dumps(payload))), redirect_stdout(out), redirect_stderr(err):
            infer.main()
        self.assertEqual(json.loads(out.getvalue())['predictions'][0]['text'], 'Among A through D, the answer is B')
        self.assertFalse(calls['template']['enable_thinking'])
        self.assertEqual(calls['generation']['max_new_tokens'], 64)
        self.assertTrue(json.loads(err.getvalue())['candidate_telemetry'])


if __name__ == '__main__':
    unittest.main()
