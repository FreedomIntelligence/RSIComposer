"""Medical SGLang routing/integrity/ownership tests; no ML or GPU required."""
from dataclasses import replace
import json
import os
from pathlib import Path
import signal
import subprocess
import sys
import tempfile
from types import ModuleType, SimpleNamespace
import unittest
from unittest.mock import Mock, patch

from examples.medical.sglang_runtime import SGLangServing, _OwnedEngine, _engine_worker
from examples.medical.task import encoded
from rsisys.loops.contracts.candidate import SystemVersion
from rsisys.loops.contracts.model_service import GenerationRequest, ModelRef
from rsisys.runtime.artifacts import InMemoryArtifactStore


BASE = ModelRef('actor', 'Qwen/Qwen3.5-27B', 'official-commit')
RESEARCHER = replace(BASE, backend_id='researcher')


class Tokenizer:
    chat_template = 'official fixture'
    model_max_length = 32768

    def __init__(self):
        self.thinking = []

    def apply_chat_template(self, messages, *, tokenize, add_generation_prompt, enable_thinking):
        assert not tokenize and add_generation_prompt
        self.thinking.append(enable_thinking)
        return str(messages) + ('<think>' if enable_thinking else '</think>')

    def __call__(self, text, *, add_special_tokens, truncation):
        assert not add_special_tokens and not truncation
        return {'input_ids': list(map(ord, text))}


class Engine:
    def __init__(self):
        self.runtime_metadata = {'tp_size': 8, 'version': 'fixture'}
        self.calls, self.loaded = [], {}
        self.closed = False
        self.text = 'A'
        self.load_result = SimpleNamespace(success=True, error_message=None)
        self.unload_result = {'success': True}

    def load_lora_adapter(self, lora_name, lora_path):
        self.loaded[lora_name] = lora_path
        self.calls.append(('load', lora_name, lora_path))
        return self.load_result

    def unload_lora_adapter(self, lora_name):
        del self.loaded[lora_name]
        self.calls.append(('unload', lora_name))
        return self.unload_result

    def generate(self, **kwargs):
        assert not self.closed
        assert kwargs['lora_path'] is None or kwargs['lora_path'] in self.loaded
        self.calls.append(('generate', kwargs))
        return {'text': self.text, 'meta_info': {'prompt_tokens': len(kwargs['input_ids']),
            'completion_tokens': 97, 'reasoning_tokens': 80}}

    def shutdown(self):
        self.closed = True
        self.calls.append(('shutdown',))


class SGLangTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.root = Path(self.tmp.name).resolve()
        self.base = self.root / 'official'
        self.base.mkdir()
        (self.base / 'config.json').write_text(json.dumps({
            'text_config': {'max_position_embeddings': 32768}}))
        (self.base / 'model.safetensors').write_bytes(b'official-27B-fixture')
        self.artifacts = InMemoryArtifactStore()
        self.engines, self.tokenizers = [], []
        self.serving = self.runtime()
        self.addCleanup(self.serving.release_memory)

    def loader(self, serving):
        tokenizer, engine = Tokenizer(), Engine()
        self.engines.append(engine)
        self.tokenizers.append(tokenizer)
        return tokenizer, engine

    def runtime(self, **kwargs):
        return SGLangServing(self.base, actor_model=BASE, decision_model=RESEARCHER,
            artifacts=self.artifacts, native_model_class='image_text_to_text',
            runtime_loader=kwargs.pop('runtime_loader', self.loader),
            runtime_root=self.root / 'serving', **kwargs)

    def candidate(self, name='candidate', revision='trained', **config):
        path = self.root / name
        path.mkdir()
        (path / 'adapter_config.json').write_text(json.dumps({
            'peft_type': 'LORA', 'base_model_name_or_path': BASE.model_id,
            'bias': 'none', 'r': 8, 'lora_alpha': 16, 'target_modules': ['q_proj', 'v_proj'], **config}))
        (path / 'adapter_model.safetensors').write_bytes(name.encode())
        native = {'artifact_id': name, 'uri': path.as_uri(), 'kind': 'peft_adapter', 'version': revision}
        manifest = self.artifacts.put(encoded({'native_ref': native}), kind='native_model_manifest')
        return SystemVersion(name, {'model': manifest}, replace(BASE, revision=revision)), path

    def request(self, model=BASE, role='actor', **kwargs):
        return GenerationRequest(role, model, 'Choose one', max_output_tokens=kwargs.pop('max_output_tokens', 128), **kwargs)

    def test_lazy_constructor_and_import_without_optional_frameworks(self):
        self.assertEqual(self.engines, [])
        code = ('import examples.medical.sglang_runtime; import sys; '
                'assert not set(("torch", "transformers", "peft", "sglang", "psutil")) & sys.modules.keys()')
        subprocess.run([sys.executable, '-c', code], check=True)

    def test_native_factory_options_use_true_tp8_and_bounded_kv(self):
        transformers = ModuleType('transformers')
        tokenizer, engine = Tokenizer(), Engine()
        transformers.AutoTokenizer = SimpleNamespace(from_pretrained=Mock(return_value=tokenizer))
        serving = self.runtime(runtime_loader=None, tensor_parallel_size=8, max_context_tokens=4096)
        with patch.dict(sys.modules, {'transformers': transformers}), patch(
                'examples.medical.sglang_runtime._OwnedEngine', return_value=engine) as factory:
            serving(self.request())
        options, root = factory.call_args.args
        self.assertEqual(root, self.root / 'serving')
        self.assertEqual(options['tp_size'], 8)
        self.assertEqual(options['dp_size'], 1)
        self.assertEqual(options['nnodes'], 1)
        self.assertEqual(options['gpu_id_step'], 1)
        self.assertEqual(options['base_gpu_id'], 0)
        self.assertTrue(options['enable_lora'])
        self.assertTrue(options['lora_strict_loading'])
        self.assertGreaterEqual(options['max_lora_rank'], 8)
        self.assertTrue({'q_proj', 'v_proj'}.issubset(options['lora_target_modules']))
        self.assertEqual(options['context_length'], 4096)
        self.assertEqual(options['max_total_tokens'], 4096)
        self.assertEqual(options['max_running_requests'], 1)
        self.assertTrue(transformers.AutoTokenizer.from_pretrained.call_args.kwargs['local_files_only'])
        self.assertFalse(options['trust_remote_code'])
        serving.release_memory()

    def test_researcher_base_and_actor_candidates_never_share_adapter_route(self):
        one, path1 = self.candidate('one', 'r1')
        two, path2 = self.candidate('two', 'r2')
        self.serving.prepare_inference(one)
        self.serving.prepare_inference(two)
        for ref, role in ((one.model, 'actor'), (RESEARCHER, 'decision_maker'),
                          (BASE, 'actor'), (two.model, 'actor'), (one.model, 'actor')):
            response = self.serving(self.request(ref, role))
            self.assertEqual(response.model, ref)
            self.assertEqual(response.text, 'A')
            self.assertTrue(response.revision_verified)
            self.assertFalse(response.simulated)
            self.assertIn(self.serving.actual_weight_sha256[ref], response.provider_model)
            self.assertEqual(response.usage.output_tokens, 97)
        engine = self.engines[0]
        routes = [row[1]['lora_path'] for row in engine.calls if row[0] == 'generate']
        self.assertEqual(routes[1:3], [None, None])
        self.assertEqual(routes[0], routes[4])
        self.assertNotEqual(routes[0], routes[3])
        self.assertEqual(engine.loaded[routes[0]], str(path1))
        self.assertEqual(engine.loaded[routes[3]], str(path2))
        self.assertEqual(self.serving.generation_count, 5)
        self.assertEqual(self.serving.actual_weight_sha256[BASE], self.serving.actual_weight_sha256[RESEARCHER])
        self.assertNotEqual(self.serving.actual_weight_sha256[one.model], self.serving.actual_weight_sha256[two.model])
        metadata = self.serving.weight_metadata[one.model]
        self.assertEqual(metadata['tensor_parallel_size'], 8)
        self.assertEqual(metadata['adapter_path'], str(path1))
        self.assertIn('native_manifest', metadata)
        json.dumps(metadata)

    def test_reasoning_removed_from_final_but_included_once_in_usage(self):
        serving = self.runtime(decision_thinking=True)
        serving.prepare_inference(SystemVersion('initial', {}, BASE))
        self.engines[-1].text = 'private reasoning </think>\n{"kind":"finish","reason":"done"}'
        response = serving(self.request(RESEARCHER, 'decision_maker'))
        self.assertEqual(json.loads(response.text)['kind'], 'finish')
        self.assertNotIn('private', response.text)
        self.assertEqual(response.usage.output_tokens, 97)
        params = self.engines[-1].calls[-1][1]['sampling_params']
        self.assertEqual(params, {'max_new_tokens': 128, 'temperature': .6, 'top_p': .95, 'top_k': 20})
        self.engines[-1].text = 'A'
        serving(self.request())
        self.assertEqual(self.tokenizers[-1].thinking, [True, False])
        self.assertEqual(self.engines[-1].calls[-1][1]['sampling_params']['temperature'], 0.0)
        serving.release_memory()

    def test_incomplete_thinking_never_exposes_reasoning_as_final(self):
        serving = self.runtime(decision_thinking=True)
        with self.assertRaisesRegex(ValueError, 'thinking did not complete'):
            serving(self.request(RESEARCHER, 'decision_maker'))
        self.engines[-1].text = 'private</think>   '
        with self.assertRaisesRegex(ValueError, 'no final proposal'):
            serving(self.request(RESEARCHER, 'decision_maker'))
        serving.release_memory()

    def test_unprepared_and_foreign_routes_rejected_before_engine_creation(self):
        system, _ = self.candidate()
        for request, pattern in ((self.request(system.model), 'prepare_inference'),
                (self.request(system.model, 'decision_maker'), 'Controller'),
                (self.request(role='other'), 'Unknown')):
            with self.assertRaisesRegex(ValueError, pattern):
                self.serving(request)
        self.assertEqual(self.engines, [])

    def test_candidate_cannot_reuse_official_revision(self):
        system, _ = self.candidate(revision=BASE.revision)
        with self.assertRaisesRegex(ValueError, 'reuse the official base revision'):
            self.serving.prepare_inference(system)
        self.assertEqual(self.engines, [])

    def test_manifest_and_standard_lora_validation_reused(self):
        for i, config in enumerate(({'base_model_name_or_path': 'foreign'}, {'peft_type': 'IA3'},
                {'bias': 'all'}, {'lora_bias': True}, {'r': 128},
                {'target_modules': ['visual']}, {'target_modules': 'all-linear'},
                {'use_rslora': True}, {'use_dora': True}, {'modules_to_save': ['lm_head']},
                {'rank_pattern': {'q_proj': 4}}, {'fan_in_fan_out': True})):
            with self.subTest(config=config):
                system, _ = self.candidate(str(i), **config)
                with self.assertRaises(ValueError):
                    self.serving.prepare_inference(system)
        system, _ = self.candidate('absent')
        with self.assertRaisesRegex(ValueError, 'managed'):
            self.serving.prepare_inference(replace(system, components={}))
        with self.assertRaisesRegex(ValueError, 'revision'):
            self.serving.prepare_inference(replace(system, model=replace(system.model, revision='wrong')))
        with self.assertRaisesRegex(ValueError, 'base/backend'):
            self.serving.prepare_inference(replace(system, model=replace(system.model, backend_id='foreign')))
        system, path = self.candidate('base-revision')
        config = json.loads((path / 'adapter_config.json').read_text())
        (path / 'adapter_config.json').write_text(json.dumps({**config, 'revision': 'wrong'}))
        with self.assertRaisesRegex(ValueError, 'base revision'):
            self.serving.prepare_inference(system)
        self.assertEqual(self.engines, [])

    def test_changed_adapter_is_never_served_from_stale_loaded_copy(self):
        system, path = self.candidate()
        self.serving.prepare_inference(system)
        (path / 'adapter_model.safetensors').write_bytes(b'changed')
        with self.assertRaisesRegex(ValueError, 'adapter weights changed'):
            self.serving(self.request(system.model))
        self.assertFalse(any(c[0] == 'generate' for c in self.engines[-1].calls))

    def test_content_identity_tracks_config_and_revision_cannot_rebind(self):
        one, _ = self.candidate('one')
        two, _ = self.candidate('two')
        self.serving.prepare_inference(one)
        with self.assertRaisesRegex(ValueError, 'already bound'):
            self.serving.prepare_inference(two)

    def test_load_failure_false_or_missing_success_discards_engine_and_candidate(self):
        system, _ = self.candidate()
        for result in ({'success': False, 'error_message': 'rank 7 failed'}, None):
            with self.subTest(result=result):
                def loader(serving):
                    tokenizer, engine = self.loader(serving)
                    engine.load_result = result
                    return tokenizer, engine
                self.serving.runtime_loader = loader
                with self.assertRaisesRegex(RuntimeError, 'LoRA update failed'):
                    self.serving.prepare_inference(system)
                self.assertTrue(self.engines[-1].closed)
                self.assertNotIn(system.model, self.serving._prepared)
                with self.assertRaisesRegex(ValueError, 'prepare_inference'):
                    self.serving(self.request(system.model))

    def test_changed_weights_during_load_discards_engine(self):
        system, path = self.candidate()
        def loader(serving):
            tokenizer, engine = self.loader(serving)
            original = engine.load_lora_adapter
            def load(*args):
                result = original(*args)
                (path / 'adapter_model.safetensors').write_bytes(b'changed during load')
                return result
            engine.load_lora_adapter = load
            return tokenizer, engine
        self.serving.runtime_loader = loader
        with self.assertRaisesRegex(ValueError, 'during native load'):
            self.serving.prepare_inference(system)
        self.assertTrue(self.engines[-1].closed)
        self.assertNotIn(system.model, self.serving.actual_weight_sha256)

    def test_bounded_eviction_and_reload_never_falls_back_to_base(self):
        systems = [self.candidate(str(i), str(i))[0] for i in range(3)]
        for system in systems:
            self.serving.prepare_inference(system)
        self.assertEqual(len(self.engines[0].loaded), 2)
        self.assertNotIn(systems[0].model, self.serving._loaded)
        response = self.serving(self.request(systems[0].model))
        self.assertTrue(response.revision_verified)
        self.assertIsNotNone(self.engines[0].calls[-1][1]['lora_path'])
        self.assertEqual(len(self.engines[0].loaded), 2)

    def test_unload_failure_releases_whole_engine(self):
        systems = [self.candidate(str(i), str(i))[0] for i in range(3)]
        for system in systems[:2]:
            self.serving.prepare_inference(system)
        self.engines[-1].unload_result = {'success': False}
        with self.assertRaisesRegex(RuntimeError, 'LoRA update failed'):
            self.serving.prepare_inference(systems[2])
        self.assertTrue(self.engines[-1].closed)
        self.assertEqual(self.serving._loaded, {})

    def test_release_is_synchronous_idempotent_and_recreates_prepared_adapter(self):
        system, _ = self.candidate()
        self.serving.activate(system, idempotency_key='apply')
        digest = self.serving.actual_weight_sha256[system.model]
        self.serving.release_memory()
        self.serving.release_memory()
        self.assertTrue(self.engines[0].closed)
        self.assertIsNone(self.serving._model)
        self.assertIsNone(self.serving._tokenizer)
        self.assertEqual(self.engines[0].calls.count(('shutdown',)), 1)
        self.serving(self.request(RESEARCHER, 'decision_maker'))
        self.assertIsNone(self.engines[1].calls[-1][1]['lora_path'])
        self.serving(self.request(system.model))
        self.assertEqual(self.serving.actual_weight_sha256[system.model], digest)
        self.assertIsNotNone(self.engines[1].calls[-1][1]['lora_path'])
        self.assertEqual(len(self.engines), 2)

    def test_release_error_propagates_and_cannot_claim_training_readiness(self):
        self.serving(self.request())
        engine = self.engines[0]
        with patch.object(engine, 'shutdown', side_effect=RuntimeError('worker alive')):
            with self.assertRaisesRegex(RuntimeError, 'worker alive'):
                self.serving.release_memory()
            self.assertIs(self.serving._model, engine)

    def test_deadline_baseexception_during_generation_releases_owned_engine(self):
        class Deadline(BaseException):
            pass
        self.serving(self.request())
        with patch.object(self.engines[0], 'generate', side_effect=Deadline):
            with self.assertRaises(Deadline):
                self.serving(self.request())
        self.assertTrue(self.engines[0].closed)
        self.assertIsNone(self.serving._model)

    def test_base_content_and_snapshot_pin_rechecked_after_release(self):
        self.serving(self.request())
        self.serving.release_memory()
        (self.base / 'model.safetensors').write_bytes(b'changed')
        with self.assertRaisesRegex(ValueError, 'base weights changed'):
            self.serving(self.request())
        snapshot = self.root / 'snapshots' / 'wrong-commit'
        snapshot.mkdir(parents=True)
        (snapshot / 'config.json').write_text('{}')
        (snapshot / 'model.safetensors').write_bytes(b'base')
        other = self.runtime()
        other.model_name_or_path = str(snapshot)
        with self.assertRaisesRegex(ValueError, 'snapshot commit'):
            other(self.request())

    def test_activation_idempotency_and_failed_activation_not_committed(self):
        system, _ = self.candidate()
        self.assertEqual(self.serving.activate(system, idempotency_key='apply'), system)
        self.serving.activate(system, idempotency_key='apply')
        self.assertEqual(sum(c[0] == 'load' for c in self.engines[0].calls), 1)
        with self.assertRaisesRegex(ValueError, 'different system'):
            self.serving.activate(replace(system, version='other'), idempotency_key='apply')
        with self.assertRaises(ValueError):
            self.serving.activate(system, idempotency_key='')
        failed, _ = self.candidate('failed', 'r2')
        self.engines[-1].load_result = {'success': False}
        with self.assertRaises(RuntimeError):
            self.serving.activate(failed, idempotency_key='failed')
        self.assertNotIn('failed', self.serving._activations)

    def test_context_budget_enforced_before_native_generate(self):
        serving = self.runtime(max_context_tokens=16)
        with self.assertRaisesRegex(ValueError, 'no prompt was truncated'):
            serving(self.request())
        self.assertEqual(self.engines[-1].calls, [])
        serving.release_memory()

    def test_constructor_validates_tp_and_preserves_existing_validation(self):
        for size in (0, -1, True, 1.5):
            with self.assertRaisesRegex(ValueError, 'tensor_parallel_size'):
                self.runtime(tensor_parallel_size=size)
        with self.assertRaisesRegex(ValueError, 'CUDA_VISIBLE_DEVICES'):
            self.runtime(device='cuda:7')
        with self.assertRaisesRegex(ValueError, 'native_model_class'):
            SGLangServing(self.base, actor_model=BASE, decision_model=RESEARCHER,
                artifacts=self.artifacts, native_model_class='guess')


class OwnedEngineTests(unittest.TestCase):
    def fake_proxy(self, *, running=True):
        proxy = object.__new__(_OwnedEngine)
        proxy._process = Mock(pid=99999)
        proxy._process.poll.return_value = None if running else 1
        proxy._identity = Mock()
        proxy._identity.children.return_value = []
        proxy._children = []
        proxy._connection = Mock()
        proxy._release_error = None
        proxy._call = Mock(return_value=None)
        return proxy

    def psutil(self):
        module = ModuleType('psutil')
        module.NoSuchProcess = ProcessLookupError
        module.STATUS_ZOMBIE = 'zombie'
        module.wait_procs = Mock(return_value=([], []))
        module.process_iter = Mock(return_value=[])
        return module

    def test_interrupted_rpc_reply_cannot_acknowledge_shutdown(self):
        proxy = object.__new__(_OwnedEngine)
        proxy._sequence = 1
        proxy.root = Path('/unused')
        proxy._connection = Mock()
        proxy._connection.poll.return_value = True
        proxy._connection.recv_bytes.side_effect = [
            encoded({'id': 1, 'ok': True, 'value': {'text': 'late generation'}}),
            encoded({'id': 2, 'ok': True, 'value': None})]
        self.assertIsNone(proxy._call('shutdown', {}, timeout=1))
        self.assertEqual(proxy._connection.recv_bytes.call_count, 2)
        self.assertEqual(json.loads(proxy._connection.send_bytes.call_args.args[0])['id'], 2)

    def test_real_private_process_ipc_shutdown_leaves_unrelated_process_alive(self):
        # The parent-side process inventory is mocked because these tests do
        # not require psutil installed. The worker/IPC/session and unrelated
        # subprocess are real, with a framework-free Engine fixture.
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            marker = root / 'released'
            (root / 'sglang.py').write_text('''
import os
from pathlib import Path
from types import SimpleNamespace
class Engine:
    def __init__(self, **options):
        assert os.environ['HF_HUB_OFFLINE'] == '1'
        self.options = options
    def get_server_info(self):
        return dict(tp_size=self.options['tp_size'], dp_size=1, version='fixture')
    def get_all_child_pids(self):
        return [os.getpid()]
    def generate(self, **kwargs):
        return dict(text='A', meta_info=dict(prompt_tokens=2, completion_tokens=1))
    def load_lora_adapter(self, **kwargs):
        return SimpleNamespace(success=True, error_message=None)
    def shutdown(self):
        Path(os.environ['FIXTURE_RELEASED']).write_text('released')
''')
            psutil = self.psutil()
            psutil.Process = Mock(return_value=SimpleNamespace(children=lambda **kw: []))
            unrelated = subprocess.Popen([sys.executable, '-c', 'import time; time.sleep(30)'])
            proxy = None
            try:
                env = {'PYTHONPATH': os.pathsep.join((str(root), os.environ.get('PYTHONPATH', ''))),
                       'FIXTURE_RELEASED': str(marker)}
                with patch.dict(sys.modules, {'psutil': psutil}), patch.dict(os.environ, env):
                    proxy = _OwnedEngine({'tp_size': 1}, root / 'runtime')
                    process = proxy._process
                    self.assertEqual(os.getpgid(process.pid), process.pid)
                    self.assertEqual(proxy.generate(input_ids=[1, 2])['text'], 'A')
                    self.assertTrue(proxy.load_lora_adapter('candidate', '/produced')['success'])
                    evidence = json.loads((proxy.root / 'engine.json').read_text())
                    self.assertEqual(evidence['tp_size'], 1)
                    proxy.shutdown()
                    self.assertIsNotNone(process.poll())
                    self.assertEqual(marker.read_text(), 'released')
                    self.assertIsNone(unrelated.poll())
                    proxy.shutdown()
            finally:
                if proxy is not None and proxy._process is not None:
                    with patch.dict(sys.modules, {'psutil': psutil}):
                        proxy.shutdown()
                unrelated.terminate()
                unrelated.wait(timeout=5)

    def test_successful_shutdown_never_signals_old_reaped_pgid(self):
        proxy = self.fake_proxy()
        proxy._process.poll.side_effect = [None, 0]
        with patch.dict(sys.modules, {'psutil': self.psutil()}), patch('os.killpg') as kill:
            proxy.shutdown()
        kill.assert_not_called()
        proxy._call.assert_called_once_with('shutdown', {}, timeout=15)
        self.assertIsNone(proxy._process)

    def test_unresponsive_root_group_killed_before_wait_and_only_owned_handles(self):
        proxy = self.fake_proxy()
        proxy._call.side_effect = TimeoutError('deadline')
        child = Mock()
        proxy._identity.children.return_value = [child]
        order = []
        proxy._process.wait.side_effect = lambda **kw: order.append('wait')
        with patch.dict(sys.modules, {'psutil': self.psutil()}), patch(
                'os.killpg', side_effect=lambda pid, sig: order.append((pid, sig))):
            proxy.shutdown()
        self.assertEqual(order, [(99999, signal.SIGKILL), 'wait'])
        child.kill.assert_called_once()

    def test_dead_root_cannot_claim_cleanup_and_never_signals_old_pgid(self):
        proxy = self.fake_proxy(running=False)
        child = Mock()
        proxy._children = [child]
        with patch.dict(sys.modules, {'psutil': self.psutil()}), patch('os.killpg') as kill:
            with self.assertRaisesRegex(RuntimeError, 'cannot confirm descendant cleanup'):
                proxy.shutdown()
            with self.assertRaisesRegex(RuntimeError, 'cannot confirm descendant cleanup'):
                proxy.shutdown()
        child.kill.assert_called_once()
        kill.assert_not_called()

    def test_worker_receives_only_native_calls_and_normalizes_msgspec_result(self):
        from multiprocessing import Pipe
        from threading import Thread
        parent, child = Pipe()
        engine = Engine()
        engine.get_server_info = lambda: {'tp_size': 8, 'dp_size': 1, 'version': 'fixture'}
        engine.get_all_child_pids = lambda: list(range(8))
        module = ModuleType('sglang')
        module.Engine = Mock(return_value=engine)
        # Duplicate the descriptor: worker owns its Connection independently.
        with patch.dict(sys.modules, {'sglang': module}):
            thread = Thread(target=_engine_worker, args=(os.dup(child.fileno()),), daemon=True)
            thread.start()
            for method, kwargs in (('initialize', {'tp_size': 8}),
                    ('load_lora_adapter', {'lora_name': 'candidate', 'lora_path': '/produced/peft'}),
                    ('generate', {'input_ids': [1, 2], 'lora_path': 'candidate'}), ('shutdown', {})):
                parent.send_bytes(encoded({'id': method, 'method': method, 'kwargs': kwargs}))
                self.assertTrue(parent.poll(5))
                response = json.loads(parent.recv_bytes())
                self.assertTrue(response['ok'], response)
                if method == 'load_lora_adapter':
                    self.assertEqual(response['value'], {'success': True, 'error_message': None})
            thread.join(5)
            self.assertFalse(thread.is_alive())
        parent.close()
        child.close()
        self.assertTrue(engine.closed)
        module.Engine.assert_called_once_with(tp_size=8)


if __name__ == '__main__':
    unittest.main()
