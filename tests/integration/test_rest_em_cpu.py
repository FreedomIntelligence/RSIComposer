"""Opt-in real tensor smoke test; tiny random local model, CPU only, no download."""
import json
import os
from pathlib import Path
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch

from rsisys.labs.math import MathLab


@unittest.skipUnless(os.environ.get('RSISYS_REST_EM_CPU_INTEGRATION') == '1',
                     'opt-in tiny CPU Transformers/PEFT test')
class RestEMCPUTests(unittest.TestCase):
    def test_fresh_base_sft_save_reload_and_generate(self):
        import torch
        from tokenizers import Tokenizer
        from tokenizers.models import WordLevel
        from tokenizers.pre_tokenizers import Whitespace
        from safetensors.torch import load_file
        from transformers import LlamaConfig, LlamaForCausalLM, PreTrainedTokenizerFast
        self.assertEqual(os.environ.get('CUDA_VISIBLE_DEVICES'), '', 'Run this test with CUDA_VISIBLE_DEVICES empty')
        torch.set_num_threads(1)
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            base = root / 'base'
            tokenizer = Tokenizer(WordLevel({'<pad>': 0, '<eos>': 1, '<unk>': 2,
                'user': 3, 'assistant': 4, '2': 5, '+': 6, '=': 7, '4': 8, '####': 9}, unk_token='<unk>'))
            tokenizer.pre_tokenizer = Whitespace()
            fast = PreTrainedTokenizerFast(tokenizer_object=tokenizer, pad_token='<pad>', eos_token='<eos>', unk_token='<unk>')
            fast.chat_template = "{% for message in messages %}{{ message['role'] + ' ' + message['content'] + ' ' }}{% endfor %}{% if add_generation_prompt %}assistant {% endif %}"
            fast.save_pretrained(base)
            initial = LlamaForCausalLM(LlamaConfig(vocab_size=10, hidden_size=16, intermediate_size=32,
                num_hidden_layers=1, num_attention_heads=2, num_key_value_heads=2,
                max_position_embeddings=128, eos_token_id=1, pad_token_id=0))
            initial.save_pretrained(base)
            lab = MathLab(SimpleNamespace(data_dir=root / 'run'), {
                'mode': 'transformers', 'device': 'cpu', 'epochs': 1,
                'max_length': 128, 'max_new_tokens': 2, 'learning_rate': 0.01,
                'generation_batch_size': 2, 'train_batch_size': 2, 'gradient_accumulation_steps': 2})
            lab.initial = str(base)
            calls = []
            original = lab.engine._load
            def record(ref, **kwargs):
                model, tok = original(ref, **kwargs)
                if kwargs.get('train'):
                    calls.append(ref)
                    for name, value in model.state_dict().items():
                        torch.testing.assert_close(value, initial.state_dict()[name])
                return model, tok
            models = []
            with patch.object(lab.engine, '_load', side_effect=record):
                for round_number in (1, 2):
                    data = lab._write(lab.root / f'round-{round_number}' / 'accepted.jsonl',
                        [{'id': f'train-{i}', 'prompt': '2 + 2' if i % 2 else '2',
                          'completion': '2 + 2 = 4 #### 4'} for i in range(5)])
                    trained = lab._train({'round': round_number, 'initial_model': str(base), 'dataset': data['dataset']})
                    models.append(trained['model'])
                    self.assertTrue(trained['manifest']['training_performed'])
                    self.assertTrue((Path(trained['model']) / 'adapter/adapter_model.safetensors').is_file())
                    weights = load_file(Path(trained['model']) / 'adapter/adapter_model.safetensors')
                    self.assertTrue(any(bool(torch.count_nonzero(value)) for name, value in weights.items()
                                        if 'lora_B' in name), 'An optimizer step must update the initially zero LoRA B weights')
                    stats = json.loads((Path(trained['model']) / 'training.json').read_text())
                    self.assertEqual(stats['steps'], 2)
                    self.assertEqual(stats['microsteps'], 3)
                    self.assertEqual(stats['samples'], 5)
                    # Reload each saved adapter through the real generation path.
                    replies = lab._completions(trained['model'], [{'prompt': '2 + 2'}, {'prompt': '2'}], 2, round_number, False)
                    self.assertEqual([len(r) for r in replies], [2, 2])
            self.assertEqual(calls, [str(base), str(base)])
            self.assertNotEqual(*models)
