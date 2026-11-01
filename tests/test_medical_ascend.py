"""Hybrid HTTP/NPU serving tests; no frameworks, service, or device required."""
from contextlib import nullcontext
from dataclasses import replace
import json
import os
from pathlib import Path
import subprocess
import sys
import tempfile
from types import ModuleType, SimpleNamespace
import unittest
from unittest.mock import MagicMock, Mock, patch

from examples.medical.ascend_runtime import AscendMedicalServing
from examples.medical.task import encoded
from rsisys.loops.contracts.candidate import SystemVersion
from rsisys.loops.contracts.model_service import GenerationRequest, ModelRef
from rsisys.runtime.artifacts import InMemoryArtifactStore


class Tensor:
    def __init__(self, rows):
        self.rows, self.shape = rows, (len(rows), len(rows[0]))

    def __getitem__(self, key):
        return self.rows[key[0]][key[1]] if isinstance(key, tuple) else self.rows[key]


class Inputs(dict):
    def to(self, device):
        assert device == 'npu:0'
        return self


class Tokenizer:
    chat_template, model_max_length, pad_token_id = 'official', 32768, 0

    def apply_chat_template(self, messages, **kwargs):
        assert not kwargs['enable_thinking']
        return str(messages)

    def __call__(self, text, **kwargs):
        assert not kwargs['truncation']
        return Inputs(input_ids=Tensor([list(map(ord, text))]))

    def decode(self, tokens, **kwargs):
        return ''.join(map(chr, tokens))


class Model:
    def __init__(self):
        self.config = SimpleNamespace(max_position_embeddings=32768)
        self.adapters = {}

    def to(self, device):
        assert device == 'npu:0'
        return self

    def eval(self):
        return self

    def load_adapter(self, path, *, adapter_name, **kwargs):
        self.adapters[adapter_name] = path

    def set_adapter(self, name):
        assert name in self.adapters

    def generate(self, *, input_ids, **kwargs):
        assert not kwargs['do_sample']
        return Tensor([input_ids.rows[0] + [65]])


class Peft:
    @staticmethod
    def from_pretrained(model, path, **kwargs):
        model.load_adapter(path, **kwargs)
        return model


class AscendTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.root = Path(self.tmp.name)
        self.base = self.root / '.models' / 'Qwen3.5-9B-c2022362'
        self.base.mkdir(parents=True)
        (self.base / 'config.json').write_text('{}')
        (self.base / 'model.safetensors').write_bytes(b'official')
        # model_identity commonly returns the full local path, not the server alias.
        self.actor = ModelRef('medical-local', str(self.base), 'official-commit')
        self.decision = replace(self.actor, backend_id='researcher')
        self.artifacts = InMemoryArtifactStore()
        self.events = []
        self.torch = SimpleNamespace(inference_mode=nullcontext, npu=SimpleNamespace(
            synchronize=lambda device: self.events.append(('sync', device)),
            device=lambda device: nullcontext(),
            empty_cache=lambda: self.events.append('empty')))
        self.loader = Mock(side_effect=lambda serving: (self.torch, Tokenizer(), Model(), Peft))
        self.serving = self.runtime()
        self.addCleanup(self.serving.release_memory)

    def runtime(self, **kwargs):
        return AscendMedicalServing(self.base, actor_model=self.actor, decision_model=self.decision,
            artifacts=self.artifacts, native_model_class='image_text_to_text',
            runtime_loader=kwargs.pop('runtime_loader', self.loader), **kwargs)

    def candidate(self, name='candidate', **config):
        path = self.root / name
        path.mkdir()
        (path / 'adapter_config.json').write_text(json.dumps(dict(
            peft_type='LORA', base_model_name_or_path=self.actor.model_id, **config)))
        (path / 'adapter_model.safetensors').write_bytes(name.encode())
        native = dict(artifact_id=name, uri=path.as_uri(), kind='peft_adapter', version=name)
        manifest = self.artifacts.put(encoded({'native_ref': native}), kind='native_model_manifest')
        return SystemVersion(name, {'model': manifest}, replace(self.actor, revision=name)), path

    def request(self, model=None, **kwargs):
        return GenerationRequest(kwargs.pop('role', 'actor'), model or self.actor,
            'Choose one', max_output_tokens=kwargs.pop('max_output_tokens', 7), **kwargs)

    def test_import_is_lazy(self):
        subprocess.run([sys.executable, '-c', 'import examples.medical.ascend_runtime; import sys; '
            'assert not set(("torch", "torch_npu", "transformers", "peft")) & sys.modules.keys()'], check=True)
        self.loader.assert_not_called()

    def test_base_prepare_call_alias_and_honest_http_metadata_without_local_files(self):
        self.serving.model_name_or_path = '/not/on/this/host'
        base = SystemVersion('base', {}, self.actor)
        self.assertEqual(self.serving.prepare_inference(base), base)
        self.serving.activate(base, idempotency_key='base')
        opener = MagicMock()
        response = opener.open.return_value.__enter__.return_value
        response.read.return_value = encoded(dict(model='Qwen3.5-9B', model_revision=self.actor.revision,
            choices=[dict(finish_reason='stop', message=dict(content='B'))],
            usage=dict(prompt_tokens=20, completion_tokens=1)))
        with patch('rsisys.adapters.serving.build_opener', return_value=opener):
            result = self.serving(self.request(system='Only the label'))
        request = opener.open.call_args.args[0]
        self.assertEqual(request.full_url, 'http://127.0.0.1:8001/v1/chat/completions')
        self.assertEqual(request.method, 'POST')
        payload = json.loads(request.data)
        self.assertEqual(payload['model'], 'Qwen3.5-9B')
        self.assertEqual(payload['messages'][0]['content'], 'Only the label')
        self.assertEqual(payload['chat_template_kwargs'], {'enable_thinking': False})
        self.assertFalse(result.revision_verified)
        self.assertEqual(result.model, self.actor)
        self.assertEqual(result.usage.input_tokens, 20)
        self.assertFalse(result.simulated)
        self.assertEqual(self.serving.generation_count, 1)
        self.assertEqual(self.serving.weight_metadata[self.actor]['source'], 'http://127.0.0.1:8001/v1')
        self.assertNotIn(self.actor, self.serving.actual_weight_sha256)
        self.loader.assert_not_called()

    def test_local_candidate_digest_metadata_survives_http_metadata_refresh_and_reload(self):
        system, _ = self.candidate()
        with self.assertRaisesRegex(ValueError, 'prepare_inference'):
            self.serving(self.request(system.model))
        self.serving.prepare_inference(system)
        first = self.serving(self.request(system.model))
        metadata = dict(self.serving.weight_metadata[system.model])
        self.assertTrue(first.revision_verified)
        self.assertEqual(first.model.backend_id, self.actor.backend_id)
        self.assertEqual(first.model.model_id, self.actor.model_id)
        self.assertIn(metadata['actual_weight_sha256'], first.provider_model)
        self.assertEqual(metadata['serving_backend'], 'transformers_npu')
        self.assertFalse(metadata['cross_runtime_numerical_equivalence_verified'])
        self.serving._http_metadata()
        self.assertEqual(self.serving.weight_metadata[system.model], metadata)
        self.assertNotIn(self.actor, self.serving.actual_weight_sha256)
        self.assertFalse(self.serving.weight_metadata[self.actor]['revision_verified'])
        self.serving.release_memory()
        self.assertEqual(self.events, [('sync', 'npu:0'), 'empty'])
        self.assertIsNone(self.serving._model)
        self.assertFalse(self.serving._loaded)
        self.assertEqual(self.serving(self.request(system.model)).provider_model, first.provider_model)
        self.assertEqual(self.loader.call_count, 2)

    def test_inherited_manifest_digest_and_activation_checks(self):
        system, path = self.candidate()
        self.serving.activate(system, idempotency_key='apply')
        with self.assertRaisesRegex(ValueError, 'different system'):
            self.serving.activate(replace(system, version='different'), idempotency_key='apply')
        (path / 'adapter_model.safetensors').write_bytes(b'changed')
        with self.assertRaisesRegex(ValueError, 'adapter weights changed'):
            self.serving(self.request(system.model))
        (path / 'adapter_model.safetensors').write_bytes(b'candidate')
        self.serving.release_memory()
        (self.base / 'model.safetensors').write_bytes(b'changed-base')
        with self.assertRaisesRegex(ValueError, 'base weights changed'):
            self.serving.prepare_inference(system)

    def test_invalid_candidates_never_load(self):
        for name, options in [('foreign', {'base_model_name_or_path': 'other'}),
                              ('bias', {'bias': 'all'}), ('not-lora', {'peft_type': 'IA3'})]:
            with self.subTest(name=name):
                path = self.root / name
                path.mkdir()
                config = dict(peft_type='LORA', base_model_name_or_path=self.actor.model_id)
                config.update(options)
                (path / 'adapter_config.json').write_text(json.dumps(config))
                (path / 'adapter_model.safetensors').write_bytes(b'weights')
                native = dict(artifact_id=name, uri=path.as_uri(), kind='peft_adapter', version=name)
                manifest = self.artifacts.put(encoded({'native_ref': native}), kind='native_model_manifest')
                with self.assertRaises(ValueError):
                    self.serving.prepare_inference(SystemVersion(name, {'model': manifest},
                        replace(self.actor, revision=name)))
        self.loader.assert_not_called()

    def test_bounded_resident_adapters_and_context(self):
        systems = [self.candidate(str(i))[0] for i in range(3)]
        for system in systems:
            self.serving.prepare_inference(system)
            self.assertLessEqual(len(self.serving._loaded), 2)
        self.assertEqual(self.loader.call_count, 2)
        self.assertEqual(self.serving(self.request(systems[0].model)).text, 'A')
        with self.assertRaisesRegex(ValueError, 'Context limit'):
            self.serving(self.request(max_output_tokens=8192))
        with self.assertRaisesRegex(ValueError, 'no prompt was truncated'):
            self.serving(self.request(systems[0].model, max_output_tokens=8192))
        self.assertEqual(self.runtime(max_context_tokens=32768).max_context_tokens, 8192)
        with self.assertRaisesRegex(ValueError, 'npu:0'):
            self.runtime(device='cuda:0')

    def test_native_loader_explicit_npu_eager_and_single_visible_device(self):
        system, _ = self.candidate()
        torch, npu, transformers, peft = (ModuleType(x) for x in ('torch', 'torch_npu', 'transformers', 'peft'))
        torch.bfloat16, torch.inference_mode = 'bf16', nullcontext
        torch.npu = Mock()
        torch.npu.device_count.return_value = 1
        torch.npu.device.return_value = nullcontext()
        model = Model()
        model.to = Mock(wraps=model.to)
        tokenizer_loader = Mock(return_value=Tokenizer())
        model_loader = Mock(return_value=model)
        transformers.AutoTokenizer = SimpleNamespace(from_pretrained=tokenizer_loader)
        transformers.AutoModelForImageTextToText = SimpleNamespace(from_pretrained=model_loader)
        peft.PeftModel = Peft
        serving = self.runtime(runtime_loader=None)
        modules = dict(torch=torch, torch_npu=npu, transformers=transformers, peft=peft)
        with patch.dict(sys.modules, modules), patch.dict(os.environ, {'ASCEND_RT_VISIBLE_DEVICES': '13'}):
            serving.prepare_inference(system)
            serving.release_memory()
        kwargs = model_loader.call_args.kwargs
        self.assertEqual(kwargs['device_map'], 'npu:0')
        self.assertEqual(kwargs['attn_implementation'], 'eager')
        self.assertEqual(kwargs['dtype'], 'bf16')
        self.assertTrue(kwargs['local_files_only'])
        self.assertFalse(kwargs['trust_remote_code'])
        model.to.assert_called_once_with('npu:0')
        torch.npu.set_device.assert_called_once_with('npu:0')
        torch.npu.synchronize.assert_called_once_with('npu:0')
        torch.npu.empty_cache.assert_called_once_with()
        for visible in ('', '13,14', '-1', ' 13'):
            with patch.dict(sys.modules, modules), patch.dict(os.environ, {'ASCEND_RT_VISIBLE_DEVICES': visible}):
                with self.assertRaisesRegex(ValueError, 'numeric device'):
                    self.runtime(runtime_loader=None).prepare_inference(system)
        torch.npu.device_count.return_value = 2
        with patch.dict(sys.modules, modules), patch.dict(os.environ, {'ASCEND_RT_VISIBLE_DEVICES': '7'}):
            with self.assertRaisesRegex(ValueError, 'exactly one'):
                self.runtime(runtime_loader=None).prepare_inference(system)
        self.assertEqual(model_loader.call_count, 1)


if __name__ == '__main__':
    unittest.main()
