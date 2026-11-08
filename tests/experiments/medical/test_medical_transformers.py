"""Runtime routing, true-byte identity and native Auto-class tests; no torch required."""
from contextlib import contextmanager, nullcontext
from dataclasses import asdict, replace
import json
import os
from pathlib import Path
import subprocess
import sys
import tempfile
from types import SimpleNamespace, ModuleType
import unittest
from unittest.mock import patch

from experiments.medical.transformers_runtime import LocalTransformerServing, tokenize_factory, _DecisionJSONStop
from experiments.medical.task import Sample, encoded
from rsisys.adapters.training import _native_inputs, _transformers_peft
from rsisys.loops.contracts.candidate import SystemVersion
from rsisys.loops.contracts.model_service import GenerationRequest, ModelRef
from rsisys.runtime.artifacts import InMemoryArtifactStore

BASE = ModelRef('local', 'Qwen/Qwen3.5-9B', 'official-commit')
DECISION = replace(BASE, backend_id='controller')


class Tensor:
    def __init__(self, rows):
        self.rows = rows
        self.shape = (len(rows), len(rows[0]))

    def __getitem__(self, key):
        return self.rows[key[0]][key[1]] if isinstance(key, tuple) else self.rows[key]


class Inputs(dict):
    def to(self, device):
        self.device = device
        return self


class Tokenizer:
    chat_template = 'official-test-template'
    special_tokens_map = {'eos_token': '<end>'}
    model_max_length = 10000
    pad_token_id = 0
    eos_token_id = 3

    def get_vocab(self):
        return {'A': 65, '<end>': 3}

    def apply_chat_template(self, messages, *, tokenize, add_generation_prompt, enable_thinking):
        assert not tokenize and not enable_thinking
        rendered = ''.join(f'<{m["role"]}>{m["content"]}<end>\n' for m in messages)
        return rendered + ('<assistant>' if add_generation_prompt else '')

    def __call__(self, text, *, add_special_tokens, truncation, return_tensors=None):
        assert not add_special_tokens and not truncation
        ids = list(map(ord, text))
        if return_tensors:
            return Inputs(input_ids=Tensor([ids]), attention_mask=Tensor([[1] * len(ids)]))
        return {'input_ids': ids}

    def decode(self, tokens, **kwargs):
        return ''.join(map(chr, tokens))


class Model:
    def __init__(self, log):
        self.log = log
        self.config = SimpleNamespace(text_config=SimpleNamespace(max_position_embeddings=10000))
        self.active = None
        self.disabled = False
        self.adapters = {}

    def eval(self):
        return self

    def load_adapter(self, path, *, adapter_name, **kwargs):
        assert kwargs == {'is_trainable': False, 'local_files_only': True}
        self.adapters[adapter_name] = Path(path).name

    def set_adapter(self, name):
        self.active = name

    @contextmanager
    def disable_adapter(self):
        self.disabled = True
        try:
            yield
        finally:
            self.disabled = False

    def generate(self, *, input_ids, attention_mask, **kwargs):
        route = 'base' if self.disabled or self.active is None else self.adapters[self.active]
        self.log.append((route, kwargs))
        return Tensor([input_ids.rows[0] + [ord('A')]])


class Peft:
    @staticmethod
    def from_pretrained(model, path, **kwargs):
        model.load_adapter(path, **kwargs)
        return model


class RuntimeTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.root = Path(self.tmp.name)
        self.base = self.root / 'official'
        self.base.mkdir()
        (self.base / 'config.json').write_text('{}')
        (self.base / 'model.safetensors').write_bytes(b'official base weights')
        self.artifacts = InMemoryArtifactStore()
        self.log, self.loads, self.cuda_calls = [], [], []
        self.torch = SimpleNamespace(inference_mode=nullcontext, cuda=SimpleNamespace(
            synchronize=lambda device: self.cuda_calls.append(('sync', device)),
            device=lambda device: nullcontext(), empty_cache=lambda: self.cuda_calls.append('empty')))
        def loader(serving):
            self.loads.append(serving.device)
            return self.torch, Tokenizer(), Model(self.log), Peft
        self.loader = loader
        self.serving = self.runtime()

    def runtime(self, **kwargs):
        return LocalTransformerServing(self.base, actor_model=BASE, decision_model=DECISION,
            artifacts=self.artifacts, native_model_class='image_text_to_text',
            runtime_loader=self.loader, **kwargs)

    def candidate(self, name='candidate', revision='trained', **config):
        path = self.root / name
        path.mkdir()
        (path / 'adapter_config.json').write_text(json.dumps({
            'peft_type': 'LORA', 'base_model_name_or_path': BASE.model_id, **config}))
        (path / 'adapter_model.safetensors').write_bytes(name.encode())
        native = {'artifact_id': name, 'uri': path.as_uri(), 'kind': 'peft_adapter', 'version': revision}
        manifest = self.artifacts.put(encoded({'native_ref': native}), kind='native_model_manifest')
        return SystemVersion(name, {'model': manifest}, replace(BASE, revision=revision)), path

    def request(self, model=BASE, role='actor', **kwargs):
        return GenerationRequest(role, model, 'Choose one', max_output_tokens=7, **kwargs)

    def test_lazy_load_no_optional_framework_import(self):
        self.assertEqual(self.loads, [])
        code = 'import experiments.medical.transformers_runtime; import sys; assert not set(("torch","transformers","peft")) & sys.modules.keys()'
        subprocess.run([sys.executable, '-c', code], check=True)

    def test_researcher_thinking_returns_final_only_and_keeps_actor_greedy(self):
        flags = []
        class ThinkingTokenizer(Tokenizer):
            def apply_chat_template(self, messages, *, tokenize, add_generation_prompt, enable_thinking):
                flags.append(enable_thinking)
                return super().apply_chat_template(messages, tokenize=tokenize,
                    add_generation_prompt=add_generation_prompt, enable_thinking=False)
        log = self.log
        class ThinkingModel(Model):
            def generate(self, *, input_ids, attention_mask, **kwargs):
                log.append(('fixture', kwargs))
                value = 'fixture private text</think>\n{"kind":"finish","reason":"fixture"}' if flags[-1] else 'A'
                return Tensor([input_ids.rows[0] + list(map(ord, value))])
        serving = self.runtime(decision_thinking=True)
        serving.runtime_loader = lambda s: (self.torch, ThinkingTokenizer(), ThinkingModel(log), Peft)
        response = serving.generate(replace(self.request(DECISION, 'decision_maker'), max_output_tokens=8192))
        self.assertEqual(json.loads(response.text)['kind'], 'finish')
        self.assertNotIn('private text', response.text)
        self.assertGreater(response.usage.output_tokens, len(response.text))
        self.assertTrue(log[-1][1]['do_sample'])
        self.assertEqual(log[-1][1]['temperature'], .6)
        self.assertEqual(len(log[-1][1]['stopping_criteria']), 1)
        self.assertEqual(serving.generate(self.request()).text, 'A')
        self.assertFalse(log[-1][1]['do_sample'])
        self.assertNotIn('stopping_criteria', log[-1][1])
        self.assertEqual(flags, [True, False])

    def test_decision_stopping_frames_one_final_object_only(self):
        stop = _DecisionJSONStop(Tokenizer(), 3, thinking=True)
        def check(text):
            return stop(Tensor([list(map(ord, 'pre' + text))]))
        self.assertFalse(check('{"reasoning_example":{}}'))
        self.assertFalse(check('</think>{"arguments":{}'))
        self.assertFalse(check('</think>{broken}'))
        self.assertTrue(check('</think>{"arguments":{"text":"a } brace"}}'))
        self.assertFalse(check('</think>{"a":1}\n{"b":2}'))
        plain = _DecisionJSONStop(Tokenizer(), 0, thinking=False)
        self.assertTrue(plain(Tensor([list(map(ord, '{"kind":"finish"}'))])))

    def test_incomplete_thinking_never_becomes_an_executable_final(self):
        class ThinkingTokenizer(Tokenizer):
            def apply_chat_template(self, messages, **kwargs):
                return super().apply_chat_template(messages, **{**kwargs, 'enable_thinking': False})
        serving = self.runtime(decision_thinking=True)
        serving.runtime_loader = lambda s: (self.torch, ThinkingTokenizer(), Model(self.log), Peft)
        with self.assertRaisesRegex(ValueError, 'thinking did not complete'):
            serving.generate(self.request(DECISION, 'decision_maker'))

    def test_candidate_requires_preparation_before_any_load(self):
        system, _ = self.candidate()
        with self.assertRaisesRegex(ValueError, 'prepare_inference'):
            self.serving.generate(self.request(system.model))
        self.assertEqual(self.loads, [])

    def test_adapter_cannot_reuse_base_revision_and_be_mismeasured_as_base(self):
        system, _ = self.candidate(revision=BASE.revision)
        with self.assertRaisesRegex(ValueError, 'reuse the official base revision'):
            self.serving.prepare_inference(system)
        self.assertEqual(self.loads, [])

    def test_native_loader_selects_explicit_image_auto_and_pins_pretrained_options(self):
        calls = []
        transformers, peft, torch = ModuleType('transformers'), ModuleType('peft'), ModuleType('torch')
        transformers.AutoTokenizer = SimpleNamespace(from_pretrained=lambda path, **kw: Tokenizer())
        transformers.AutoModelForImageTextToText = SimpleNamespace(from_pretrained=lambda path, **kw:
            calls.append((path, kw)) or Model(self.log))
        peft.PeftModel = Peft
        torch.bfloat16 = 'bf16-sentinel'
        torch.inference_mode = nullcontext
        serving = self.runtime()
        serving.runtime_loader = None
        with patch.dict(sys.modules, {'transformers': transformers, 'peft': peft, 'torch': torch}):
            serving.generate(self.request())
        path, kwargs = calls[0]
        self.assertEqual(path, str(self.base.resolve()))
        self.assertEqual(kwargs, {'revision': BASE.revision, 'local_files_only': True,
            'trust_remote_code': False, 'dtype': 'bf16-sentinel', 'device_map': 'cuda:0',
            'attn_implementation': 'sdpa'})

    def test_actor_controller_shared_base_routes_and_sha_are_isolated(self):
        first, _ = self.candidate('one', 'r1')
        second, _ = self.candidate('two', 'r2')
        self.serving.prepare_inference(first)
        self.serving.prepare_inference(second)
        for model, role in [(first.model, 'actor'), (DECISION, 'decision_maker'),
                            (BASE, 'actor'), (second.model, 'actor'), (first.model, 'actor')]:
            response = self.serving(self.request(model, role))
            self.assertEqual(response.model, model)
            self.assertTrue(response.revision_verified)
            self.assertFalse(response.simulated)
            self.assertEqual(response.text, 'A')
            self.assertEqual(response.usage.output_tokens, 1)
            self.assertIn(self.serving.actual_weight_sha256[model], response.provider_model)
        self.assertEqual([row[0] for row in self.log], ['one', 'base', 'base', 'two', 'one'])
        self.assertEqual(len(self.loads), 1)
        self.assertEqual(self.serving.generation_count, 5)
        self.assertEqual(self.serving.actual_weight_sha256[BASE], self.serving.actual_weight_sha256[DECISION])
        self.assertNotEqual(self.serving.actual_weight_sha256[first.model], self.serving.actual_weight_sha256[second.model])
        self.assertTrue(all(row[1]['max_new_tokens'] == 7 and not row[1]['do_sample'] for row in self.log))

    def test_controller_cannot_request_actor_candidate_or_foreign_role(self):
        system, _ = self.candidate()
        self.serving.prepare_inference(system)
        with self.assertRaisesRegex(ValueError, 'Controller'):
            self.serving.generate(self.request(system.model, 'decision_maker'))
        with self.assertRaisesRegex(ValueError, 'Unknown generation role'):
            self.serving.generate(self.request(role='unknown'))

    def test_identical_actor_controller_refs_are_supported(self):
        self.serving.decision_model = BASE
        system, _ = self.candidate()
        self.serving.prepare_inference(system)
        self.serving.generate(self.request(system.model))
        self.serving.generate(self.request(BASE, 'decision_maker'))
        self.assertEqual([row[0] for row in self.log], ['candidate', 'base'])

    def test_release_reloads_base_and_prepared_candidate(self):
        system, _ = self.candidate()
        self.serving.prepare_inference(system)
        original = self.serving.actual_weight_sha256[system.model]
        self.serving.release_memory()
        self.assertIsNone(self.serving._model)
        self.assertEqual(self.cuda_calls, [('sync', 'cuda:0'), 'empty'])
        self.serving.generate(self.request(DECISION, 'decision_maker'))
        self.serving.generate(self.request(system.model))
        self.assertEqual([r[0] for r in self.log], ['base', 'candidate'])
        self.assertEqual(len(self.loads), 2)
        self.assertEqual(self.serving.actual_weight_sha256[system.model], original)

    def test_changed_base_rejected_after_release(self):
        self.serving.generate(self.request())
        self.serving.release_memory()
        (self.base / 'model.safetensors').write_bytes(b'other weights')
        with self.assertRaisesRegex(ValueError, 'base weights changed'):
            self.serving.generate(self.request())

    def test_changed_adapter_never_silently_uses_loaded_old_weights(self):
        system, path = self.candidate()
        self.serving.prepare_inference(system)
        (path / 'adapter_model.safetensors').write_bytes(b'changed')
        with self.assertRaisesRegex(ValueError, 'adapter weights changed'):
            self.serving.generate(self.request(system.model))
        self.assertEqual(self.log, [])

    def test_revision_cannot_be_rebound_to_another_manifest(self):
        one, _ = self.candidate('one')
        two, _ = self.candidate('two')
        self.serving.prepare_inference(one)
        with self.assertRaisesRegex(ValueError, 'already bound'):
            self.serving.prepare_inference(two)

    def test_manifest_requires_managed_artifact_revision_and_base(self):
        for i, config in enumerate([{'base_model_name_or_path': 'foreign'}, {'peft_type': 'IA3'}]):
            system, _ = self.candidate(str(i), **config)
            with self.assertRaises(ValueError):
                self.serving.prepare_inference(system)
        system, path = self.candidate('wrong-base-revision')
        config_path = path / 'adapter_config.json'
        config_path.write_text(json.dumps({**json.loads(config_path.read_text()), 'revision': 'other'}))
        with self.assertRaisesRegex(ValueError, 'base revision'):
            self.serving.prepare_inference(system)
        system, _ = self.candidate('revision')
        with self.assertRaisesRegex(ValueError, 'revision'):
            self.serving.prepare_inference(replace(system, model=replace(BASE, revision='other')))
        with self.assertRaisesRegex(ValueError, 'managed'):
            self.serving.prepare_inference(SystemVersion('absent', {}, system.model))

    def test_context_overflow_rejected_without_truncation_or_generate(self):
        serving = self.runtime(max_context_tokens=10)
        with self.assertRaisesRegex(ValueError, 'no prompt was truncated'):
            serving.generate(self.request())
        self.assertEqual(self.log, [])

    def test_declared_revision_must_match_hf_snapshot_commit(self):
        snapshot = self.root / 'snapshots' / 'another-commit'
        snapshot.mkdir(parents=True)
        (snapshot / 'config.json').write_text('{}')
        (snapshot / 'model.safetensors').write_bytes(b'weights')
        self.serving.model_name_or_path = str(snapshot)
        with self.assertRaisesRegex(ValueError, 'snapshot commit'):
            self.serving.generate(self.request())
        self.assertEqual(self.loads, [])

    def test_bias_training_cannot_mutate_shared_official_base(self):
        system, _ = self.candidate(bias='all')
        with self.assertRaisesRegex(ValueError, 'base isolation'):
            self.serving.prepare_inference(system)

    def test_activation_key_pins_whole_system_manifest(self):
        system, _ = self.candidate()
        self.assertEqual(self.serving.activate(system, idempotency_key='apply'), system)
        self.assertEqual(self.serving.activate(system, idempotency_key='apply'), system)
        with self.assertRaisesRegex(ValueError, 'different system'):
            self.serving.activate(replace(system, version='other'), idempotency_key='apply')
        with self.assertRaises(ValueError):
            self.serving.activate(system, idempotency_key='')

    def test_failed_adapter_load_discards_mutated_base_and_prepared_route(self):
        system, _ = self.candidate()
        with patch.object(Peft, 'from_pretrained', side_effect=RuntimeError('bad weights')):
            with self.assertRaisesRegex(RuntimeError, 'bad weights'):
                self.serving.prepare_inference(system)
        self.assertIsNone(self.serving._model)
        with self.assertRaisesRegex(ValueError, 'prepare_inference'):
            self.serving.generate(self.request(system.model))

    def test_explicit_auto_class_and_matching_base_required(self):
        with self.assertRaisesRegex(ValueError, 'native_model_class'):
            LocalTransformerServing(self.base, actor_model=BASE, decision_model=DECISION,
                artifacts=self.artifacts, native_model_class='guess')
        with self.assertRaisesRegex(ValueError, 'same official base'):
            LocalTransformerServing(self.base, actor_model=BASE, decision_model=replace(DECISION, revision='other'),
                artifacts=self.artifacts, native_model_class='image_text_to_text')


class TokenizeTests(unittest.TestCase):
    def setUp(self):
        self.row = Sample('one', 'medqa', 'Question', tuple((x, x) for x in 'ABCD'), 'B')

    def factory(self, **kwargs):
        return tokenize_factory('official', tokenizer=Tokenizer(), max_length=kwargs.pop('max_length', 256), **kwargs)

    def test_completion_only_chat_tokens_and_variable_length(self):
        converter = self.factory()
        short, long = converter((self.row, replace(self.row, question='longer question')))
        count = next(i for i, label in enumerate(short['labels']) if label != -100)
        self.assertTrue(all(label == -100 for label in short['labels'][:count]))
        self.assertEqual(''.join(map(chr, short['labels'][count:])), 'B<end>\n')
        self.assertEqual(short['labels'][count:], short['input_ids'][count:])
        self.assertLess(len(short['input_ids']), len(long['input_ids']))
        self.assertTrue(all(short['attention_mask']))

    def test_fixed_padding_masks_labels_and_attention(self):
        feature, = self.factory(padding=True)((self.row,))
        self.assertEqual(len(feature['input_ids']), 256)
        self.assertEqual(len(feature['labels']), 256)
        self.assertEqual(feature['labels'][-1], -100)
        self.assertEqual(feature['attention_mask'][-1], 0)
        self.assertEqual(feature['input_ids'][-1], 0)  # zero is a valid pad ID

    def test_qwen_empty_thinking_prefix_does_not_mask_the_answer(self):
        class QwenTokenizer(Tokenizer):
            def apply_chat_template(self, messages, **kwargs):
                prefix = ''.join(f'<{m["role"]}>{m["content"]}<end>\n' for m in messages
                                 if m['role'] != 'assistant') + '<assistant>\n<think>\n\n</think>\n\n'
                return prefix if kwargs['add_generation_prompt'] else prefix + messages[-1]['content'] + '<end>\n'
        converter = tokenize_factory('official', tokenizer=QwenTokenizer(), max_length=256)
        feature, = converter((self.row,))
        labels = feature['labels']
        boundary = labels.index(ord('B'))
        self.assertTrue(all(value == -100 for value in labels[:boundary]))
        self.assertEqual(''.join(map(chr, labels[boundary:])), 'B<end>\n')

    def test_preflight_excludes_and_records_whole_long_samples(self):
        converter = self.factory(max_length=160)
        long = replace(self.row, id='long', question='X' * 300)
        kept, excluded = converter.filter_samples((self.row, long))
        self.assertEqual(kept, (self.row,))
        self.assertEqual(excluded[0]['id'], 'long')
        self.assertGreater(excluded[0]['length'], excluded[0]['max_length'])
        self.assertEqual(excluded[0]['reason'], 'sequence_too_long')
        with self.assertRaisesRegex(ValueError, 'filter_samples'):
            converter((long,))

    def test_conversion_identity_tracks_template_vocab_padding_and_length(self):
        converter = self.factory()
        self.assertEqual(converter.conversion_id, self.factory().conversion_id)
        self.assertNotEqual(converter.conversion_id, self.factory(padding=True).conversion_id)
        self.assertNotEqual(converter.conversion_id, self.factory(max_length=257).conversion_id)
        converter.tokenizer.chat_template = 'changed'
        self.assertNotEqual(converter.conversion_id, self.factory().conversion_id)
        converter.tokenizer.get_vocab = lambda: {'changed': 1}
        self.assertNotEqual(converter.metadata['tokenizer_sha256'], self.factory().metadata['tokenizer_sha256'])

    def test_missing_template_and_unstable_boundary_fail_closed(self):
        converter = self.factory()
        converter.tokenizer.chat_template = None
        with self.assertRaisesRegex(ValueError, 'official chat_template'):
            converter((self.row,))
        converter = self.factory()
        converter.tokenizer.apply_chat_template = lambda messages, **kw: 'prefix' if kw['add_generation_prompt'] else 'mismatch'
        with self.assertRaisesRegex(ValueError, 'inference prefix'):
            converter((self.row,))


class NativeWorkerTests(unittest.TestCase):
    def test_recipe_auto_selection_is_validated_without_frameworks(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            data = root / 'data.jsonl'
            data.write_text('{"input_ids":[1,2],"labels":[-100,2]}\n')
            recipe = root / 'recipe.json'
            recipe.write_text(json.dumps({'initial_model': asdict(BASE), 'train_dataset': str(data),
                'model_auto_class': 'guess'}))
            with self.assertRaisesRegex(ValueError, 'model_auto_class'):
                _native_inputs({'base': asdict(BASE)}, recipe)

    def test_worker_selects_mature_image_auto_and_disables_text_cache(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            data, recipe, requestfile = root / 'data.jsonl', root / 'recipe.json', root / 'request.json'
            data.write_text('{"input_ids":[1,2],"labels":[-100,2]}\n')
            recipe.write_text(json.dumps({'initial_model': asdict(BASE), 'train_dataset': str(data),
                'model_name_or_path': 'official', 'model_auto_class': 'image_text_to_text', 'use_cache': False,
                'model_kwargs': {'local_files_only': True, 'dtype': 'bfloat16'}, 'lora_config': {'r': 8},
                'training_args': {'bf16': True, 'gradient_checkpointing': True, 'max_steps': 64,
                    'gradient_accumulation_steps': 2, 'per_device_train_batch_size': 1}}))
            requestfile.write_text(json.dumps({'request_type': 'ActorTrainingRequest',
                'request': {'base': asdict(BASE), 'origin': {'parent_system': 'parent'}}}))
            calls = []
            base = SimpleNamespace(config=SimpleNamespace(use_cache=True, text_config=SimpleNamespace(use_cache=True)))
            class ImageAuto:
                @staticmethod
                def from_pretrained(path, **kwargs):
                    calls.append((path, kwargs))
                    return base
            class Trainer:
                def __init__(self, **kwargs):
                    self.args = SimpleNamespace(**kwargs['args'], world_size=1)
                    self.state = SimpleNamespace(global_step=64)
                    calls.append(kwargs)
                def train(self):
                    return SimpleNamespace(metrics={'loss': .5})
                def save_model(self, path):
                    pass
            transformers = ModuleType('transformers')
            transformers.AutoConfig = object
            transformers.AutoModelForImageTextToText = ImageAuto
            transformers.Trainer = Trainer
            transformers.TrainerCallback = object
            transformers.TrainingArguments = lambda **kwargs: kwargs
            transformers.default_data_collator = object()
            # No AutoModelForCausalLM export: accidental selection fails the test.
            peft, torch = ModuleType('peft'), ModuleType('torch')
            peft.LoraConfig = lambda **kwargs: kwargs
            peft.get_peft_model = lambda model, config: model
            peft.PeftModel = object
            torch.manual_seed = lambda seed: None
            env = {'RSISYS_JOB_DIR': str(root), 'RSISYS_REQUEST_FILE': str(requestfile),
                   'RSISYS_RESULT_FILE': str(root / 'result.json')}
            with patch.dict(sys.modules, {'transformers': transformers, 'peft': peft, 'torch': torch}), patch.dict(os.environ, env):
                _transformers_peft(recipe)
            self.assertEqual(calls[0], ('official', {'local_files_only': True, 'dtype': 'bfloat16'}))
            self.assertFalse(base.config.use_cache)
            self.assertFalse(base.config.text_config.use_cache)
            self.assertTrue(calls[1]['args']['bf16'])
            self.assertTrue(calls[1]['args']['gradient_checkpointing'])
            result = json.loads((root / 'result.json').read_text())
            self.assertEqual(result['candidate']['parent_system'], 'parent')
            evidence = json.loads((root / 'training-evidence.json').read_text())
            self.assertEqual(evidence['model_auto_class'], 'image_text_to_text')
            self.assertFalse(evidence['use_cache'])


if __name__ == '__main__':
    unittest.main()
