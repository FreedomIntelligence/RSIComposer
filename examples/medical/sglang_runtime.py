"""Optional, text-only SGLang serving with an owned native Engine process.

Verified against lion's SGLang 0.5.20: Engine.generate(input_ids=...,
lora_path=...), load_lora_adapter(name, path), unload_lora_adapter(name),
get_server_info(), get_all_child_pids(), and synchronous shutdown(). LoRA
updates report success only when all ranks succeed. Engine.shutdown kills all
descendants of its caller, so it MUST run in our dedicated process, never in
the controller. No HTTP server or other serving process is contacted.

The caller serializes training against inference and calls release_memory()
before DDP. The next request lazily recreates the TP engine. runtime_loader(s)
may return (tokenizer, engine) for tests; that engine must implement the native
methods above and synchronous shutdown. No ML framework is imported on import.
"""
from dataclasses import asdict
import json
import os
from pathlib import Path
import signal
import subprocess
import sys
import tempfile
import time

from rsisys.loops.contracts.model_service import GenerationResponse, TokenUsage
from .task import encoded
from .transformers_runtime import (
    LocalTransformerServing, _files_identity, _generation_messages, _render, _sha,
)


_TARGETS = ('q_proj', 'k_proj', 'v_proj', 'o_proj', 'gate_proj', 'up_proj', 'down_proj')
_MAX_RANK = 64


def _lora_success(result):
    # Native APIs return msgspec LoRAUpdateOutput, not a truthy success flag.
    success = result.get('success') if isinstance(result, dict) else getattr(result, 'success', None)
    error = result.get('error_message') if isinstance(result, dict) else getattr(result, 'error_message', None)
    if success is not True:
        raise RuntimeError(f'SGLang LoRA update failed: {error or result!s}')


class _OwnedEngine:
    """Private IPC, separate session, bounded waits; never kill controller children."""
    def __init__(self, options, runtime_root):
        from multiprocessing import Pipe
        import psutil

        if runtime_root is not None:
            Path(runtime_root).mkdir(parents=True, exist_ok=True)
        self.root = Path(tempfile.mkdtemp(prefix='sglang-', dir=runtime_root))
        self.runtime_metadata = {'engine_options': options, 'log_path': str(self.root / 'engine.log')}
        self._connection, child = Pipe()
        self._children = []
        self._release_error = None
        self._sequence = 0
        env = dict(os.environ, HF_HUB_OFFLINE='1', TRANSFORMERS_OFFLINE='1')
        try:
            with (self.root / 'engine.log').open('ab', buffering=0) as log:
                self._process = subprocess.Popen(
                    [sys.executable, '-m', 'examples.medical.sglang_runtime', '--worker', str(child.fileno())],
                    pass_fds=(child.fileno(),), start_new_session=True,
                    stdin=subprocess.DEVNULL, stdout=log, stderr=subprocess.STDOUT, env=env)
        except BaseException:
            self._connection.close()
            raise
        finally:
            child.close()
        try:
            self._identity = psutil.Process(self._process.pid)
            self.runtime_metadata.update(self._call('initialize', options))
            self._children = self._identity.children(recursive=True)
            if self.runtime_metadata['tp_size'] != options['tp_size']:
                raise RuntimeError('SGLang resolved a different tensor parallel size')
            if len(set(self.runtime_metadata['child_pids'])) < options['tp_size']:
                raise RuntimeError('SGLang did not report all tensor parallel workers')
            (self.root / 'engine.json').write_bytes(encoded(self.runtime_metadata))
        except BaseException:
            self.shutdown()
            raise

    def _call(self, method, kwargs, timeout=600):
        self._sequence += 1
        sequence = self._sequence
        self._connection.send_bytes(encoded({'id': sequence, 'method': method, 'kwargs': kwargs}))
        deadline = time.monotonic() + timeout
        while True:
            if not self._connection.poll(max(0, deadline - time.monotonic())):
                raise TimeoutError(f'SGLang {method} timed out; see {self.root / "engine.log"}')
            result = json.loads(self._connection.recv_bytes())
            if result['id'] == sequence:
                break  # A deadline may have interrupted an earlier RPC.
        if not result['ok']:
            raise RuntimeError(f'SGLang {method}: {result["error"]}; see {self.root / "engine.log"}')
        return result['value']

    def generate(self, **kwargs):
        return self._call('generate', kwargs)

    def load_lora_adapter(self, lora_name, lora_path):
        return self._call('load_lora_adapter', {'lora_name': lora_name, 'lora_path': lora_path})

    def unload_lora_adapter(self, lora_name):
        return self._call('unload_lora_adapter', {'lora_name': lora_name})

    def shutdown(self):
        # At most 40s of waits, within the supervisor's 60s cleanup grace.
        # A retry shares the original 45s deadline rather than restarting it.
        # psutil is an installed SGLang dependency; import it only on use.
        import psutil

        process = getattr(self, '_process', None)
        if process is None:
            if self._release_error is not None:
                raise RuntimeError(self._release_error)
            self._connection.close()
            return
        if not hasattr(self, '_shutdown_deadline'):
            self._shutdown_deadline = time.monotonic() + 45
        def remaining(cap):
            return max(0, min(cap, self._shutdown_deadline - time.monotonic()))
        children = list(self._children)
        acknowledged = False
        unexpected_exit = False
        try:
            if process.poll() is None:
                try:
                    children.extend(self._identity.children(recursive=True))
                except psutil.NoSuchProcess:
                    unexpected_exit = True
                try:
                    self._call('shutdown', {}, timeout=remaining(15))
                    acknowledged = True
                    process.wait(timeout=remaining(5))
                except (OSError, EOFError, RuntimeError, TimeoutError, subprocess.TimeoutExpired):
                    pass  # Cleanup below must also cover failed/partial Engine startup.
            else:
                unexpected_exit = True
        finally:
            # poll() returning None guarantees the root PID remains ours, even
            # if it exits immediately afterward (it is then an unreaped zombie).
            # Never signal a PGID after wait()/poll() has reaped that root.
            if process.poll() is None:
                # Keep identity-bound handles for late-starting workers too.
                # The root has not been reaped, so this private PGID cannot yet
                # have been reused. Enumerate before reaping it, not afterward.
                for member in psutil.process_iter():
                    try:
                        if os.getpgid(member.pid) == process.pid:
                            children.append(member)
                    except (ProcessLookupError, PermissionError):
                        pass
                try:
                    os.killpg(process.pid, signal.SIGKILL)
                except ProcessLookupError:
                    pass
            elif not acknowledged:
                unexpected_exit = True
            children = list(set(children))
            for child in children:
                try:
                    child.kill()
                except psutil.NoSuchProcess:
                    pass
            process.wait(timeout=remaining(10))
            _, alive = psutil.wait_procs(children, timeout=remaining(10))
            alive = [p for p in alive if p.is_running() and p.status() != psutil.STATUS_ZOMBIE]
            if alive:
                raise RuntimeError(f'Owned SGLang workers still alive: {[p.pid for p in alive]}')
            self._connection.close()
            self._process = None
            if unexpected_exit:
                self._release_error = 'Owned SGLang root exited unexpectedly; cannot confirm descendant cleanup'
                raise RuntimeError(self._release_error)


def _engine_worker(fd):
    """Only this process imports SGLang or owns inference CUDA allocations."""
    from multiprocessing.connection import Connection
    import traceback

    engine = None
    connection = Connection(fd)
    try:
        while True:
            request = json.loads(connection.recv_bytes())
            method, kwargs = request['method'], request['kwargs']
            try:
                if method == 'initialize':
                    from sglang import Engine
                    engine = Engine(**kwargs)
                    info = engine.get_server_info()
                    value = {k: info[k] for k in ('tp_size', 'dp_size', 'version')}
                    value['child_pids'] = engine.get_all_child_pids()
                elif method == 'shutdown':
                    if engine is not None:
                        engine.shutdown()
                        engine = None
                    connection.send_bytes(encoded({'id': request['id'], 'ok': True, 'value': None}))
                    return
                elif method in ('generate', 'load_lora_adapter', 'unload_lora_adapter'):
                    value = getattr(engine, method)(**kwargs)
                    if method != 'generate':
                        value = {'success': value.success, 'error_message': value.error_message}
                else:
                    raise ValueError(f'Unsupported engine operation: {method}')
                connection.send_bytes(encoded({'id': request['id'], 'ok': True, 'value': value}))
            except Exception as error:
                traceback.print_exc()
                connection.send_bytes(encoded({'id': request['id'], 'ok': False, 'error': str(error)}))
    except (EOFError, BrokenPipeError):
        pass
    finally:
        if engine is not None:
            engine.shutdown()
        connection.close()


class SGLangServing(LocalTransformerServing):
    """Drop-in medical serving adapter; TP shards computation across all ranks.

    Supports standard, unmerged LoRA (rank <=64) on attention/MLP projections,
    with at most two registered candidates. Advanced PEFT scaling, saved base
    modules, vocabulary changes and vision adapters are deliberately rejected.
    The medical task uses text only, even for image_text_to_text base models.
    """
    def __init__(self, model_name_or_path, *, actor_model, decision_model, artifacts,
                 native_model_class, device='cuda:0', dtype='bfloat16',
                 max_context_tokens=None, runtime_loader=None, decision_thinking=False,
                 tensor_parallel_size=8, runtime_root=None):
        super().__init__(model_name_or_path, actor_model=actor_model, decision_model=decision_model,
            artifacts=artifacts, native_model_class=native_model_class, device=device, dtype=dtype,
            max_context_tokens=max_context_tokens, runtime_loader=runtime_loader,
            decision_thinking=decision_thinking)
        if type(tensor_parallel_size) is not int or tensor_parallel_size < 1:
            raise ValueError('tensor_parallel_size must be a positive integer')
        if device not in ('cuda', 'cuda:0'):
            raise ValueError('SGLang requires device=cuda:0; select GPUs with CUDA_VISIBLE_DEVICES')
        self.tensor_parallel_size = tensor_parallel_size
        self.runtime_root = Path(runtime_root).expanduser().resolve() if runtime_root is not None else None
        self._limit = None
        self._engine_metadata = {}

    def _candidate(self, system):
        candidate = super()._candidate(system)
        if candidate is None:
            return None
        path = Path(candidate['path'])
        config = json.loads((path / 'adapter_config.json').read_text())
        rank, targets = config.get('r'), config.get('target_modules')
        if type(rank) is not int or not 1 <= rank <= _MAX_RANK:
            raise ValueError(f'SGLang requires LoRA rank in [1, {_MAX_RANK}]')
        if not isinstance(targets, list) or not targets or any(t not in _TARGETS for t in targets):
            raise ValueError(f'SGLang requires explicit standard LoRA target_modules from {_TARGETS}')
        for key in ('use_dora', 'use_rslora', 'fan_in_fan_out', 'modules_to_save',
                    'rank_pattern', 'alpha_pattern', 'layer_replication', 'target_parameters',
                    'trainable_token_indices', 'lora_bias', 'megatron_config'):
            if config.get(key):
                raise ValueError(f'SGLang medical runtime does not support PEFT {key}')
        added = path / 'added_tokens.json'
        if added.exists() and json.loads(added.read_text()):
            raise ValueError('SGLang medical runtime does not support vocabulary changes')
        return candidate

    def _load(self):
        if self._model is not None:
            return
        before = _files_identity(self.model_name_or_path)
        source = Path(self.model_name_or_path)
        if source.parent.name == 'snapshots' and source.name != self.actor_model.revision:
            raise ValueError('HF snapshot commit differs from the declared base revision')
        if self._base_sha is not None and before != self._base_sha:
            raise ValueError('Official base weights changed after revision binding')
        config = json.loads((source / 'config.json').read_text())
        text_config = config.get('text_config', config)
        limits = [text_config.get('max_position_embeddings'), self.max_context_tokens]
        if self.runtime_loader is not None:
            self._tokenizer, self._model = self.runtime_loader(self)
        else:
            from transformers import AutoTokenizer
            self._tokenizer = AutoTokenizer.from_pretrained(self.model_name_or_path,
                revision=self.actor_model.revision, local_files_only=True, trust_remote_code=False)
        try:
            limits.append(getattr(self._tokenizer, 'model_max_length', None))
            limits = [v for v in limits if type(v) is int and 0 < v < 10**12]
            if not limits:
                raise ValueError('No reliable context limit; supply max_context_tokens explicitly')
            self._limit = min(limits)
            if self._model is None:
                options = dict(model_path=self.model_name_or_path, tokenizer_path=self.model_name_or_path,
                    revision=self.actor_model.revision, tokenizer_revision=self.actor_model.revision,
                    trust_remote_code=False, dtype=self.dtype, tp_size=self.tensor_parallel_size,
                    dp_size=1, nnodes=1, base_gpu_id=0, gpu_id_step=1,
                    context_length=self._limit, language_only=True, enable_lora=True,
                    max_lora_rank=_MAX_RANK, lora_target_modules=list(_TARGETS),
                    lora_strict_loading=True, max_loras_per_batch=2, max_loaded_loras=2,
                    max_running_requests=1, max_total_tokens=self._limit, mem_fraction_static=0.8)
                self._model = _OwnedEngine(options, self.runtime_root)
            self._engine_metadata = getattr(self._model, 'runtime_metadata', {})
            self._base_sha = before
            for ref in (self.actor_model, self.decision_model):
                self.actual_weight_sha256[ref] = before
                self.weight_metadata[ref] = {'actual_weight_sha256': before, 'base_weight_sha256': before,
                    'model': asdict(ref), 'native_model_class': self.native_model_class, 'dtype': self.dtype,
                    'revision_binding': 'local_weight_content_pin',
                    'source_commit': source.name if source.parent.name == 'snapshots' else None,
                    'serving_backend': 'sglang', 'tensor_parallel_size': self.tensor_parallel_size,
                    'engine': self._engine_metadata}
        except BaseException:
            self.release_memory()
            raise

    def _ensure_adapter(self, ref):
        prepared = self._prepared[ref]
        if _files_identity(prepared['path'], adapter=True) != prepared['sha256']:
            raise ValueError('Prepared adapter weights changed; refusing stale revision')
        added = Path(prepared['path']) / 'added_tokens.json'
        if added.exists() and json.loads(added.read_text()):
            raise ValueError('SGLang medical runtime does not support vocabulary changes')
        if ref in self._loaded:
            return self._loaded[ref]
        name = 'rsi_' + _sha(encoded(asdict(ref)))
        try:
            if len(self._loaded) >= 2:
                oldest = next(iter(self._loaded))
                _lora_success(self._model.unload_lora_adapter(self._loaded[oldest]))
                del self._loaded[oldest]
            _lora_success(self._model.load_lora_adapter(name, prepared['path']))
            if _files_identity(prepared['path'], adapter=True) != prepared['sha256']:
                raise ValueError('Prepared adapter weights changed during native load')
        except BaseException:
            self.release_memory()  # Partial rank loads must never become serving routes.
            raise
        self._loaded[ref] = name
        combined = _sha(encoded({'base': self._base_sha, 'adapter': prepared['sha256']}))
        self.actual_weight_sha256[ref] = combined
        self.weight_metadata[ref] = {'model': asdict(ref), 'actual_weight_sha256': combined,
            'base_weight_sha256': self._base_sha, 'adapter_weight_sha256': prepared['sha256'],
            'adapter_path': prepared['path'], 'native_manifest': prepared['manifest'],
            'serving_backend': 'sglang', 'tensor_parallel_size': self.tensor_parallel_size,
            'engine': self._engine_metadata}
        return name

    def activate(self, system, *, idempotency_key):
        if not isinstance(idempotency_key, str) or not idempotency_key:
            raise ValueError('An activation idempotency key is required')
        with self._lock:
            manifest = encoded(asdict(system))
            if idempotency_key in self._activations and self._activations[idempotency_key] != manifest:
                raise ValueError('Activation key reused with a different system')
            result = self.prepare_inference(system)
            self._activations[idempotency_key] = manifest
            return result

    def generate(self, request):
        with self._lock:
            messages = _generation_messages(request)
            if request.role == 'decision_maker':
                if request.model != self.decision_model:
                    raise ValueError('Controller may only request its bound official base')
                candidate = False
            elif request.role == 'actor':
                candidate = request.model != self.actor_model
                if candidate and request.model not in self._prepared:
                    raise ValueError('Candidate must pass prepare_inference before generation')
            else:
                raise ValueError('Unknown generation role')
            self._load()
            adapter = self._ensure_adapter(request.model) if candidate else None
            thinking = request.role == 'decision_maker' and self.decision_thinking
            prompt = _render(self._tokenizer, messages,
                             generation=True, thinking=thinking)
            ids = self._tokenizer(prompt, add_special_tokens=False, truncation=False)['input_ids']
            if len(ids) + request.max_output_tokens > self._limit:
                raise ValueError(f'Context limit {self._limit}: prompt={len(ids)} + max_output_tokens='
                                 f'{request.max_output_tokens}; no prompt was truncated')
            sampling = {'max_new_tokens': request.max_output_tokens, 'temperature': 0.0}
            if thinking:
                sampling.update(temperature=0.6, top_p=0.95, top_k=20)
            # Native generate returns raw text, including </think>; no OpenAI
            # chat parser, json-mode, or schema constraint on reasoning tokens.
            try:
                result = self._model.generate(input_ids=list(ids), sampling_params=sampling,
                                              lora_path=adapter, stream=False)
            except BaseException:
                self.release_memory()
                raise
            self.generation_count += 1
            text, meta = result['text'], result['meta_info']
            if thinking:
                if '</think>' not in text:
                    raise ValueError('Decision thinking did not complete; no partial proposal will be executed')
                text = text.rsplit('</think>', 1)[1].strip()
                if not text:
                    raise ValueError('Decision thinking returned no final proposal')
            # completion_tokens already includes reasoning; never count only the
            # final JSON text or add reasoning_tokens a second time.
            return GenerationResponse(text, request.model, False,
                TokenUsage(meta['prompt_tokens'], meta['completion_tokens']),
                self.model_name_or_path + '#sha256=' + self.actual_weight_sha256[request.model],
                revision_verified=True)

    __call__ = generate

    def release_memory(self):
        """Block until all owned inference workers release CUDA, or raise.

        Prepared identities, activations and weight evidence survive training;
        no engine/adapter allocation survives a successful return.
        """
        with self._lock:
            if self._model is not None:
                self._model.shutdown()
            self._model = self._tokenizer = None
            self._loaded.clear()


if __name__ == '__main__':
    if len(sys.argv) != 3 or sys.argv[1] != '--worker':
        raise SystemExit('This module only exposes the private --worker entrypoint')
    _engine_worker(int(sys.argv[2]))
