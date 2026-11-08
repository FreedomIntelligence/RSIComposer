"""Fixed host inference regression with fake tensors, no model/API/accelerator.

The process/recovery/record checks use the actual host entry point. Fake runtime
modules only substitute for heavy model loading and generation in these tests.
"""
from tests import ROOT as REPO_ROOT
from contextlib import nullcontext
from copy import deepcopy
import hashlib
import json
import os
from pathlib import Path
import sys
from tempfile import TemporaryDirectory
from types import SimpleNamespace
import unittest
from unittest.mock import patch

from experiments.medical import checkpoint_evaluation as host
from experiments.medical.open_lab_experiment import RetainedAllocatedProcessBackend
from rsisys.adapters.local_lab_state import LocalLabState
from rsisys.loops.contracts.decision import OperationCall
from rsisys.loops.contracts.execution import LabToolError, LabInfrastructureError
from rsisys.runtime.artifacts import InMemoryArtifactStore


FAKE_TORCH = '''from contextlib import nullcontext
bfloat16 = 'bfloat16'
inference_mode = nullcontext
'''

FAKE_TRANSFORMERS = '''from types import SimpleNamespace
import json, os, time
from pathlib import Path
class Tensor(list):
    def tolist(self): return list(self)
    def __getitem__(self, index):
        value = super().__getitem__(index)
        return Tensor(value) if isinstance(index, slice) else value
class Batch(dict):
    def to(self, device): return self
class Tokenizer:
    eos_token_id = 0
    pad_token_id = None
    def apply_chat_template(self, messages, **options):
        assert options == dict(tokenize=False, add_generation_prompt=True, enable_thinking=False)
        return json.dumps(messages)
    def __call__(self, prompt, **options):
        assert options == dict(return_tensors='pt', add_special_tokens=False, truncation=False)
        assert 'J. option J' in prompt and 'full user question' in prompt
        return Batch(input_ids=[Tensor([1,2,3])])
    def decode(self, ids, **options):
        assert options == dict(skip_special_tokens=True)
        return 'J'
class AutoTokenizer:
    @staticmethod
    def from_pretrained(path, **options):
        assert options == dict(local_files_only=True, trust_remote_code=False)
        assert Path(path).name == 'base'
        return Tokenizer()
class AutoConfig:
    @staticmethod
    def from_pretrained(path, **options):
        assert Path(path).name == 'base'
        return SimpleNamespace(model_type='qwen3_5')
class Model:
    def eval(self): return self
    def get_base_model(self): return self
    def get_input_embeddings(self): return SimpleNamespace(weight=SimpleNamespace(device='cpu'))
    def generate(self, **options):
        assert options['generation_config'].options['max_new_tokens'] == 32
        if os.environ.get('CHECKPOINT_FIXTURE_PAUSE'): time.sleep(1)
        return [Tensor([1,2,3,7])]
class AutoModelForImageTextToText:
    @staticmethod
    def from_pretrained(path, **options):
        assert options['config'].model_type == 'qwen3_5'
        assert options['use_safetensors'] and not options['trust_remote_code']
        assert options['attn_implementation'] == 'eager'
        return Model(), {}
class AutoModelForCausalLM:
    @staticmethod
    def from_pretrained(*args, **options): raise AssertionError('wrong Qwen3.5 loader')
class GenerationConfig:
    def __init__(self, **options): self.options = options
    def to_dict(self): return self.options
'''

FAKE_PEFT = '''class PeftModel:
    @staticmethod
    def from_pretrained(model, path, **options):
        assert options == dict(is_trainable=False, local_files_only=True, use_safetensors=True)
        return model
'''


class CheckpointEvaluationTests(unittest.TestCase):
    def setUp(self):
        self.directory = TemporaryDirectory()
        self.addCleanup(self.directory.cleanup)
        self.root = Path(self.directory.name).resolve()
        self.base = self.root / 'base'
        self.base.mkdir()
        (self.base / 'config.json').write_text('{"model_type":"qwen3_5"}')
        (self.base / 'tokenizer_config.json').write_text('{"chat_template":"fixture"}')
        (self.base / 'model.safetensors').write_bytes(b'fixture-weights-not-a-real-model')
        self.data = self.root / 'dev.jsonl'
        self.data.write_text(json.dumps({'id': 'one', 'question': 'full user question with H in its text',
            'options': {letter: 'option ' + letter for letter in 'ABCDEFGHIJ'}, 'gold': 'J'}) + '\n')
        self.options = {'evaluation_groups': [{'benchmark_id': 'fixture', 'path': str(self.data),
                'sha256': hashlib.sha256(self.data.read_bytes()).hexdigest()}],
            'evaluation_root': self.root / 'evaluations', 'python': sys.executable,
            'base_model_paths': [self.base], 'device_type': 'cpu', 'timeout_seconds': 15}
        fixtures = self.root / 'fake-runtime'
        fixtures.mkdir()
        for name, content in [('torch', FAKE_TORCH), ('transformers', FAKE_TRANSFORMERS), ('peft', FAKE_PEFT)]:
            (fixtures / (name + '.py')).write_text(content)
        repo = REPO_ROOT
        self.environment = {'CUDA_VISIBLE_DEVICES': '0',
            'PYTHONPATH': os.pathsep.join(map(str, (fixtures, repo / 'src', repo)))}
        self.backend = RetainedAllocatedProcessBackend(self.root / 'executor', self.environment)
        self.lab = host.CheckpointMedicalLab(self.backend, InMemoryArtifactStore(), **self.options)
        self.addCleanup(self.lab.close)
        self.args = {'key': 'fixed', 'candidate_id': 'baseline', 'base_model_path': str(self.base)}

    def submit(self, **changes):
        return self.lab.execute(OperationCall('medical_evaluate', {**self.args, **changes})).data['job_id']

    def complete(self, ident):
        status = self.backend.wait(self.lab._jobs[ident][0])
        self.assertEqual(status.status.value, 'succeeded', status.stderr)
        return self.lab.execute(OperationCall('lab_result', {'job_id': ident}))

    def test_one_formal_entry_rejects_all_candidate_evaluation_hooks(self):
        specs = {s.name: s for s in self.lab.describe()}
        self.assertNotIn('medical_diagnose', specs)
        self.assertEqual(set(specs['medical_evaluate'].input_schema['properties']),
            {'key', 'candidate_id', 'base_model_path', 'checkpoint_path', 'adapter_path'})
        for key, value in [('argv', ['python', 'mine.py']), ('endpoint', 'http://localhost:8001'),
                ('max_new_tokens', 1), ('enable_thinking', True), ('parser', 'mine'),
                ('tokenizer_path', str(self.base)), ('profile', {})]:
            with self.subTest(key=key), self.assertRaises(LabToolError):
                self.submit(**{key: value})
        with self.assertRaises(LabToolError):
            self.submit(checkpoint_path='.', adapter_path='.')
        with self.assertRaises(LabToolError):
            self.submit(checkpoint_path=str(self.base))  # candidate weights stay inside workspace
        with self.assertRaises(LabToolError):
            self.submit(base_model_path=str(self.root))
        self.assertEqual(self.lab._jobs, {})
        self.assertEqual(self.lab._keys, {})

    def test_raw_answers_are_never_scanned_for_a_letter(self):
        for text in ('Hmm, let me think through this carefully.', 'The correct answer is B.',
                     '<think>B</think>', 'A or B'):
            self.assertIsNone(host.answer_letter(text, list('ABCDEFGHIJ')))
        self.assertEqual(host.answer_letter('J.\n', list('ABCDEFGHIJ')), 'J')
        self.assertIsNone(host.answer_letter('J', list('ABCD')))
        metrics, _ = host.score_predictions([{'benchmark_id': 'f', 'id': '1', 'text': 'The correct answer is B.'}],
                                             {('f', '1'): 'H'})
        self.assertEqual(metrics['f']['accuracy'], 0)

    def test_real_host_job_records_weights_full_options_profile_and_raw_generations(self):
        ident = self.submit()
        submitted = self.lab._evaluations[ident]
        self.assertNotIn('candidate_argv', submitted['config'])
        result = self.complete(ident)
        fixed = result.data['fixed_evaluation']
        self.assertEqual(fixed['metrics']['fixture'], {'accuracy': 1.0, 'n': 1})
        self.assertEqual(fixed['diagnostics'], {'invalid_answers': 0, 'output_limit_hits': 0})
        record = json.loads(Path(submitted['config']['output']).read_text())
        self.assertEqual(record['traces'][0]['raw_text'], 'J')
        self.assertEqual(record['traces'][0]['generated_token_ids'], [7])
        self.assertEqual(record['runtime']['loader'], 'AutoModelForImageTextToText')
        self.assertEqual(record['weight_identity'], host.weight_identity(self.base))
        self.assertFalse(record['profile']['enable_thinking'])
        self.assertFalse(fixed['adopted'])
        self.assertEqual(result.feedback_delivery.event_id,
                         self.lab.execute(OperationCall('lab_result', {'job_id': ident})).feedback_delivery.event_id)
        with patch.object(self.backend, 'exec', side_effect=AssertionError('resubmitted')):
            self.assertTrue(self.lab.execute(OperationCall('medical_evaluate', self.args)).data['replayed'])
            with self.assertRaises(LabToolError):
                self.submit(candidate_id='different')

    def test_adapter_and_full_checkpoint_use_same_frozen_profile_as_base(self):
        baseline = self.complete(self.submit()).data['fixed_evaluation']
        checkpoint = self.lab.workspace / 'checkpoint'
        checkpoint.mkdir()
        (checkpoint / 'model.safetensors').write_bytes(b'trained-fixture')
        # Candidate configuration is ignored; model config and tokenizer come from the registered base.
        (checkpoint / 'config.json').write_text('{"model_type":"wrong"}')
        (checkpoint / 'generation_config.json').write_text('{"max_new_tokens":1,"do_sample":true}')
        full = self.complete(self.submit(key='full', checkpoint_path='checkpoint')).data['fixed_evaluation']
        adapter = self.lab.workspace / 'adapter'
        adapter.mkdir()
        (adapter / 'adapter_config.json').write_text('{"peft_type":"LORA"}')
        (adapter / 'adapter_model.safetensors').write_bytes(b'adapter-fixture')
        adapted = self.complete(self.submit(key='adapter', adapter_path='adapter')).data['fixed_evaluation']
        self.assertEqual(baseline['profile_sha256'], full['profile_sha256'])
        self.assertEqual(baseline['profile_sha256'], adapted['profile_sha256'])
        self.assertNotEqual(baseline['weight_identity'], full['weight_identity'])
        self.assertEqual(adapted['adapter_identity']['sha256'], host.weight_identity(adapter, adapter=True)['sha256'])

    def test_busy_research_must_release_devices_before_formal_evaluation(self):
        ident = self.lab.execute(OperationCall('lab_start', {'key': 'research',
            'argv': [sys.executable, '-c', 'import time;time.sleep(10)']})).data['job_id']
        with self.assertRaises(LabToolError) as error:
            self.submit()
        self.assertEqual(error.exception.data['reason'], 'evaluation_devices_busy')
        self.assertNotIn(self.args['key'], self.lab._keys)
        self.lab.execute(OperationCall('lab_cancel', {'job_id': ident}))

    def test_uncollected_checkpoint_evaluation_recovers_without_running_again(self):
        ident = self.submit()
        self.backend.wait(self.lab._jobs[ident][0])
        def port(lab):
            return LocalLabState(lab, capture_extra=lambda: deepcopy(lab._evaluations),
                restore_extra=lambda state: setattr(lab, '_evaluations', state),
                configuration=lab.formal_configuration)
        state = port(self.lab).capture()
        self.lab.close()
        backend = RetainedAllocatedProcessBackend(self.root / 'executor', self.environment)
        restored = host.CheckpointMedicalLab(backend, self.lab.artifacts, defer_session=True, **self.options)
        self.addCleanup(restored.close)
        with patch.object(backend, 'exec', side_effect=AssertionError('resubmitted after restart')):
            port(restored).restore(state)
            fixed = restored.execute(OperationCall('lab_result', {'job_id': ident})).data['fixed_evaluation']
            self.assertEqual(fixed['metrics']['fixture']['accuracy'], 1.0)
            self.assertTrue(restored.execute(OperationCall('medical_evaluate', self.args)).data['replayed'])

    def test_running_formal_evaluation_prevents_new_research_programs(self):
        self.backend.environment['CHECKPOINT_FIXTURE_PAUSE'] = '1'
        ident = self.submit()
        with self.assertRaises(LabToolError) as error:
            self.lab.execute(OperationCall('lab_start', {'key': 'training', 'argv': [sys.executable, '-c', 'pass']}))
        self.assertEqual(error.exception.data['reason'], 'evaluation_devices_busy')
        self.assertNotIn('training', self.lab._keys)
        self.complete(ident)

    def test_changed_dataset_or_forged_score_is_not_published(self):
        ident = self.submit()
        self.backend.wait(self.lab._jobs[ident][0])
        output = Path(self.lab._evaluations[ident]['config']['output'])
        record = json.loads(output.read_text())
        record['metrics']['fixture']['accuracy'] = .75
        output.write_text(json.dumps(record))
        with self.assertRaises(LabInfrastructureError) as error:
            self.lab.execute(OperationCall('lab_result', {'job_id': ident}))
        self.assertNotIn('fixed_evaluation', error.exception.result.data)

    def test_weight_mutation_fails_and_preserves_partial_raw_evidence(self):
        # No subprocess needed to inject the mutation precisely between hashing and final verification.
        with patch.object(self.lab, '_start', return_value=SimpleNamespace(data={'job_id': 'fixture'})):
            self.submit()
        config = self.lab._evaluations['fixture']['config']
        def mutate(*args):
            (self.base / 'model.safetensors').write_bytes(b'changed-during-generation')
            return {'benchmark_id': 'fixture', 'id': 'one', 'raw_text': 'J'}
        with patch.object(host, 'load_runtime', return_value=(None, None, None, None, {})), \
             patch.object(host, 'generate_one', side_effect=mutate), \
             self.assertRaisesRegex(ValueError, 'changed during inference'):
            host.evaluate(config)
        self.assertFalse(Path(config['output']).exists())
        partial = json.loads(Path(config['output'] + '.partial.json').read_text())
        self.assertEqual(partial['traces'][0]['raw_text'], 'J')
        self.assertNotIn('metrics', partial)

    def test_limit_truncated_thinking_is_saved_raw_and_scores_as_invalid(self):
        runtime = {}
        exec(FAKE_TRANSFORMERS, runtime)
        tokenizer = runtime['Tokenizer']()
        tokenizer.decode = lambda *args, **kwargs: 'Hmm, let me think through this carefully.'
        model = runtime['Model']()
        model.generate = lambda **kwargs: [runtime['Tensor']([1, 2, 3] + [8] * 32)]
        payload, _, _ = host.evaluation_inputs(self.options['evaluation_groups'])
        trace = host.generate_one(SimpleNamespace(inference_mode=nullcontext), tokenizer,
                                  model, None, payload['items'][0])
        self.assertEqual(trace['raw_text'], 'Hmm, let me think through this carefully.')
        self.assertIsNone(trace['answer_letter'])
        self.assertTrue(trace['output_limit_hit'])
        self.assertEqual(trace['output_tokens'], 32)

    def test_sharded_weights_are_content_bound_and_index_cannot_escape(self):
        weights = self.root / 'sharded'
        weights.mkdir()
        index = weights / 'model.safetensors.index.json'
        index.write_text(json.dumps({'weight_map': {'layer.a': 'one.safetensors', 'layer.b': 'two.safetensors'}}))
        (weights / 'one.safetensors').write_bytes(b'one')
        (weights / 'two.safetensors').write_bytes(b'two')
        before = host.weight_identity(weights)
        (weights / 'two.safetensors').write_bytes(b'changed')
        self.assertNotEqual(before['sha256'], host.weight_identity(weights)['sha256'])
        index.write_text(json.dumps({'weight_map': {'layer.a': '../outside.safetensors'}}))
        with self.assertRaisesRegex(ValueError, 'shard index'):
            host.weight_identity(weights)

    def test_changed_registered_tokenizer_fails_before_any_inference(self):
        with patch.object(self.lab, '_start', return_value=SimpleNamespace(data={'job_id': 'fixture'})):
            self.submit()
        config = self.lab._evaluations['fixture']['config']
        (self.base / 'tokenizer_config.json').write_text('{"chat_template":"changed"}')
        with patch.object(host, 'load_runtime', side_effect=AssertionError('loaded after identity change')), \
             self.assertRaisesRegex(ValueError, 'tokenizer/config changed'):
            host.evaluate(config)


if __name__ == '__main__':
    unittest.main()
