"""Single-process Transformers/PEFT medical runtime; imports frameworks on use only.

The caller owns the allocated device, serializes training against inference, and
calls release_memory() before submitting a native training process. No downloads,
servers, environment mutations, or model-name guessing. runtime_loader(serving)
may inject (torch, tokenizer, base_model, PeftModelClass) for isolated tests.
"""
from contextlib import nullcontext
from dataclasses import asdict
import gc
import hashlib
import json
from pathlib import Path
import threading
from urllib.parse import unquote, urlsplit

from rsisys.loops.contracts.artifacts import ArtifactRef
from rsisys.loops.contracts.model_service import GenerationRequest, GenerationResponse, TokenUsage
from .task import LABELS, encoded


def _sha(data):
    return hashlib.sha256(data).hexdigest()


class _DecisionJSONStop:
    """Frame one final object; policy still validates schema and permissions."""
    def __init__(self, tokenizer, prompt_tokens, *, thinking):
        self.tokenizer, self.prompt_tokens, self.thinking = tokenizer, prompt_tokens, thinking

    def __call__(self, input_ids, scores=None, **kwargs):
        if '}' not in self.tokenizer.decode(input_ids[0, -1:], skip_special_tokens=True):
            return False
        text = self.tokenizer.decode(input_ids[0, self.prompt_tokens:], skip_special_tokens=True)
        if self.thinking:
            if '</think>' not in text:
                return False
            text = text.rsplit('</think>', 1)[1]
        try:
            return isinstance(json.loads(text.strip()), dict)
        except (ValueError, TypeError):
            return False


def _files_identity(path, *, adapter=False):
    """Digest filenames and bytes of all weight shards and loading configuration."""
    path = Path(path).resolve()
    names = ('adapter_config.json',) if adapter else ('config.json', 'generation_config.json')
    weights = sorted(p for p in path.iterdir() if p.is_file() and
                     (p.suffix == '.safetensors' or p.name.endswith('.bin')))
    if adapter:
        weights = [p for p in weights if p.name.startswith('adapter_model')]
    if not weights or not (path / names[0]).is_file():
        raise ValueError(f'Missing {"PEFT adapter" if adapter else "base"} config/weights: {path}')
    files = weights + [path / n for n in names if (path / n).is_file()]
    files += sorted(path.glob('*.index.json'))
    digest = hashlib.sha256()
    for file in sorted(set(files)):
        digest.update(encoded([file.name, file.stat().st_size]))
        with file.open('rb') as stream:
            for block in iter(lambda: stream.read(8 * 1024 * 1024), b''):
                digest.update(block)
    return digest.hexdigest()


def _messages(prompt, system):
    return ([{'role': 'system', 'content': system}] if system else []) + [
        {'role': 'user', 'content': prompt}]


def _generation_messages(request):
    if request.tools or request.response_schema is not None:
        raise NotImplementedError('Local medical generation does not implement native tools or response_schema')
    messages = request.chat_messages()
    if any(message['role'] == 'tool' or message.get('tool_calls') for message in messages):
        raise NotImplementedError('Local medical generation does not implement native tool history')
    return messages


def _render(tokenizer, messages, *, generation, thinking=False):
    if not tokenizer.chat_template:
        raise ValueError('An official chat_template is required; no plain-text fallback')
    return tokenizer.apply_chat_template(messages, tokenize=False,
        add_generation_prompt=generation, enable_thinking=thinking)


class _MedicalTokenizer:
    def __init__(self, path, *, max_length, padding, revision, tokenizer, system_prompt):
        if type(max_length) is not int or max_length < 2 or type(padding) is not bool:
            raise ValueError('max_length must be >=2 and padding must be boolean')
        self.path, self.max_length, self.padding = str(path), max_length, padding
        self.revision, self._tokenizer, self.system_prompt = revision, tokenizer, system_prompt

    @property
    def tokenizer(self):
        if self._tokenizer is None:
            from transformers import AutoTokenizer
            self._tokenizer = AutoTokenizer.from_pretrained(self.path, revision=self.revision,
                local_files_only=True, trust_remote_code=False)
        return self._tokenizer

    @property
    def metadata(self):
        tokenizer = self.tokenizer
        # Covers tokenizer vocabulary/config/template semantics, not just a path label.
        backend = getattr(tokenizer, 'backend_tokenizer', None)
        serialized = backend.to_str() if backend is not None else encoded(tokenizer.get_vocab()).decode()
        return {'format': 'medical-chat-completion-v1', 'model_name_or_path': self.path,
            'revision': self.revision, 'tokenizer_class': type(tokenizer).__name__,
            'tokenizer_sha256': _sha(serialized.encode()),
            'chat_template_sha256': _sha(encoded(tokenizer.chat_template)),
            'special_tokens': tokenizer.special_tokens_map,
            'max_length': self.max_length, 'padding': self.padding,
            'enable_thinking': False, 'completion_only': True, 'system_prompt': self.system_prompt}

    @property
    def conversion_id(self):
        return 'medical-chat-completion-v1:' + _sha(encoded(self.metadata))

    def _feature(self, row):
        tokenizer = self.tokenizer
        system = self.system_prompt
        if system is None:
            system = 'Return exactly one label: ' + ', '.join(LABELS[row.benchmark_id]) + '. No explanation.'
        messages = _messages(row.prompt, system)
        prefix = _render(tokenizer, messages, generation=True)
        full = _render(tokenizer, messages + [{'role': 'assistant', 'content': row.gold}], generation=False)
        if not full.startswith(prefix):
            raise ValueError('Chat template completion does not extend the inference prefix')
        ids = tokenizer(full, add_special_tokens=False, truncation=False)['input_ids']
        prompt = tokenizer(prefix, add_special_tokens=False, truncation=False)['input_ids']
        if ids[:len(prompt)] != prompt or len(ids) <= len(prompt):
            raise ValueError('Tokenizer completion boundary is unstable or has no supervised tokens')
        return {'input_ids': list(ids), 'labels': [-100] * len(prompt) + list(ids[len(prompt):]),
                'attention_mask': [1] * len(ids)}

    def filter_samples(self, rows):
        """Preflight before freezing train: exclude whole rows, retain audit records."""
        kept, excluded = [], []
        for row in rows:
            length = len(self._feature(row)['input_ids'])
            if length > self.max_length:
                excluded.append({'id': row.id, 'benchmark_id': row.benchmark_id,
                    'reason': 'sequence_too_long', 'length': length, 'max_length': self.max_length})
            else:
                kept.append(row)
        return tuple(kept), tuple(excluded)

    def __call__(self, rows):
        features = []
        for row in rows:
            feature = self._feature(row)
            length = len(feature['input_ids'])
            if length > self.max_length:
                raise ValueError(f'Sample {row.id}: {length} tokens exceeds max_length={self.max_length}; '
                                 'preflight filter_samples before freezing training data')
            if self.padding:
                pad = self.tokenizer.pad_token_id
                if pad is None:
                    pad = self.tokenizer.eos_token_id
                if pad is None:
                    raise ValueError('Padding requires pad_token_id or eos_token_id')
                count = self.max_length - length
                feature['input_ids'] += [pad] * count
                feature['labels'] += [-100] * count
                feature['attention_mask'] += [0] * count
            features.append(feature)
        return features


def tokenize_factory(model_name_or_path, *, max_length, padding=False, revision=None,
                     tokenizer=None, system_prompt=None):
    """Callable converter with .metadata, .conversion_id, .filter_samples(rows).

    padding=False supports variable row lengths with native collator batch size 1.
    Larger batches require fixed padding=True or an explicitly dynamic collator.
    No answer or prompt truncation is permitted.
    """
    return _MedicalTokenizer(model_name_or_path, max_length=max_length, padding=padding,
        revision=revision, tokenizer=tokenizer, system_prompt=system_prompt)


class LocalTransformerServing:
    def __init__(self, model_name_or_path, *, actor_model, decision_model, artifacts,
                 native_model_class, device='cuda:0', dtype='bfloat16',
                 max_context_tokens=None, runtime_loader=None, decision_thinking=False):
        if native_model_class not in ('causal_lm', 'image_text_to_text'):
            raise ValueError('native_model_class must be causal_lm or image_text_to_text')
        if (actor_model.model_id, actor_model.revision) != (decision_model.model_id, decision_model.revision):
            raise ValueError('Actor/controller must bind the same official base model_id and revision')
        if max_context_tokens is not None and (type(max_context_tokens) is not int or max_context_tokens < 1):
            raise ValueError('max_context_tokens must be a positive integer')
        if dtype not in ('bfloat16', 'float32', 'float16'):
            raise ValueError('dtype must be bfloat16, float32 or float16')
        if type(decision_thinking) is not bool:
            raise TypeError('decision_thinking must be boolean')
        self.decision_thinking = decision_thinking
        self.model_name_or_path = str(Path(model_name_or_path).expanduser().resolve())
        self.actor_model, self.decision_model, self.artifacts = actor_model, decision_model, artifacts
        self.native_model_class, self.device, self.dtype = native_model_class, device, dtype
        self.max_context_tokens, self.runtime_loader = max_context_tokens, runtime_loader
        self._lock = threading.RLock()
        self._torch = self._tokenizer = self._model = self._peft_class = None
        self._has_adapters = False
        self._loaded, self._prepared, self._activations = {}, {}, {}
        self.actual_weight_sha256 = {}
        self.weight_metadata = {}
        self.generation_count = 0
        self._base_sha = None

    def _load(self):
        if self._model is not None:
            return
        before = _files_identity(self.model_name_or_path)
        source = Path(self.model_name_or_path)
        if source.parent.name == 'snapshots' and source.name != self.actor_model.revision:
            raise ValueError('HF snapshot commit differs from the declared base revision')
        if self._base_sha is not None and before != self._base_sha:
            raise ValueError('Official base weights changed after revision binding')
        if self.runtime_loader is not None:
            runtime = self.runtime_loader(self)
        else:
            import torch
            from transformers import AutoTokenizer
            from peft import PeftModel
            if self.native_model_class == 'image_text_to_text':
                from transformers import AutoModelForImageTextToText as ModelAutoClass
            else:
                from transformers import AutoModelForCausalLM as ModelAutoClass
            tokenizer = AutoTokenizer.from_pretrained(self.model_name_or_path,
                revision=self.actor_model.revision, local_files_only=True, trust_remote_code=False)
            model = ModelAutoClass.from_pretrained(self.model_name_or_path,
                revision=self.actor_model.revision, local_files_only=True, trust_remote_code=False,
                dtype=getattr(torch, self.dtype), device_map=self.device, attn_implementation='sdpa').eval()
            runtime = torch, tokenizer, model, PeftModel
        self._torch, self._tokenizer, self._model, self._peft_class = runtime
        self._base_sha = before
        for ref in (self.actor_model, self.decision_model):
            self.actual_weight_sha256[ref] = before
            self.weight_metadata[ref] = {'actual_weight_sha256': before, 'base_weight_sha256': before,
                'model': asdict(ref), 'native_model_class': self.native_model_class,
                'dtype': self.dtype, 'revision_binding': 'local_weight_content_pin',
                'source_commit': source.name if source.parent.name == 'snapshots' else None}

    def _candidate(self, system):
        ref = system.model_for('actor')
        if ref is None:
            raise ValueError('Inference requires an explicit model')
        if ref == self.actor_model:
            manifest = system.components.get('model')
            if manifest is not None and manifest.kind == 'native_model_manifest':
                raise ValueError('PEFT candidate cannot reuse the official base revision')
            return None
        if (ref.backend_id, ref.model_id) != (self.actor_model.backend_id, self.actor_model.model_id):
            raise ValueError('Candidate must belong to the actor base/backend')
        manifest = system.components.get('model')
        if manifest is None or manifest.kind != 'native_model_manifest':
            raise ValueError('Candidate requires a managed native_model_manifest')
        native = ArtifactRef(**json.loads(self.artifacts.get(manifest))['native_ref'])
        parsed = urlsplit(native.uri)
        if native.kind != 'peft_adapter' or native.version != ref.revision:
            raise ValueError('PEFT manifest kind/revision differs from candidate model')
        if parsed.scheme != 'file' or parsed.netloc not in ('', 'localhost') or parsed.query or parsed.fragment:
            raise ValueError('PEFT adapter must be a local file directory')
        path = Path(unquote(parsed.path)).resolve()
        digest = _files_identity(path, adapter=True)
        config = json.loads((path / 'adapter_config.json').read_text())
        if config.get('peft_type') != 'LORA':
            raise ValueError('Only unmerged native LoRA adapters are supported')
        if config.get('bias', 'none') != 'none' or config.get('lora_bias', False):
            raise ValueError('Shared base isolation requires LoRA bias=none and lora_bias=false')
        base = config.get('base_model_name_or_path')
        if base not in (self.actor_model.model_id, self.model_name_or_path):
            raise ValueError('PEFT adapter names a different official base')
        if config.get('revision') not in (None, self.actor_model.revision):
            raise ValueError('PEFT adapter base revision mismatch')
        return {'path': str(path), 'sha256': digest, 'manifest': asdict(manifest)}

    def _ensure_adapter(self, ref):
        prepared = self._prepared[ref]
        path = prepared['path']
        if _files_identity(path, adapter=True) != prepared['sha256']:
            raise ValueError('Prepared adapter weights changed; refusing stale revision')
        if ref in self._loaded:
            return self._loaded[ref]
        name = 'rsi_' + _sha(encoded(asdict(ref)))
        # Pin before load; any failure discards the potentially mutated base.
        try:
            if not self._has_adapters:
                self._model = self._peft_class.from_pretrained(self._model, path,
                    adapter_name=name, is_trainable=False, local_files_only=True)
                self._has_adapters = True
            else:
                self._model.load_adapter(path, adapter_name=name, is_trainable=False,
                    local_files_only=True)
            self._model.eval()
        except Exception:
            self.release_memory()
            raise
        self._loaded[ref] = name
        combined = _sha(encoded({'base': self._base_sha, 'adapter': prepared['sha256']}))
        self.actual_weight_sha256[ref] = combined
        self.weight_metadata[ref] = {'model': asdict(ref), 'actual_weight_sha256': combined,
            'base_weight_sha256': self._base_sha, 'adapter_weight_sha256': prepared['sha256'],
            'adapter_path': path, 'native_manifest': prepared['manifest']}
        return name

    def prepare_inference(self, system):
        with self._lock:
            candidate = self._candidate(system)
            if candidate is not None:
                previous = self._prepared.get(system.model_for('actor'))
                if previous is not None and candidate != previous:
                    raise ValueError('Candidate revision already bound to different native weights/manifest')
                self._prepared[system.model_for('actor')] = candidate
            try:
                self._load()
                if candidate is not None:
                    self._ensure_adapter(system.model_for('actor'))
            except Exception:
                if candidate is not None and previous is None:
                    self._prepared.pop(system.model_for('actor'), None)
                raise
            return system

    def activate(self, system, *, idempotency_key):
        if any(name != 'actor' and (name != 'decision_maker' or ref != self.decision_model)
               for name, ref in system.models.items()):
            from rsisys.runtime.systems import SystemActivationRejected
            raise SystemActivationRejected('Meta model changes require a separate confirmed activation port')
        if not isinstance(idempotency_key, str) or not idempotency_key:
            raise ValueError('An activation idempotency key is required')
        with self._lock:
            manifest = encoded(asdict(system))
            if idempotency_key in self._activations and self._activations[idempotency_key] != manifest:
                raise ValueError('Activation key reused with a different system')
            self._activations[idempotency_key] = manifest
            return self.prepare_inference(system)

    def _context_limit(self):
        config = self._model.config
        text = getattr(config, 'text_config', config)
        limits = [getattr(text, 'max_position_embeddings', None),
                  getattr(config, 'max_position_embeddings', None),
                  getattr(self._tokenizer, 'model_max_length', None), self.max_context_tokens]
        limits = [v for v in limits if type(v) is int and 0 < v < 10**12]
        if not limits:
            raise ValueError('No reliable context limit; supply max_context_tokens explicitly')
        return min(limits)

    def generate(self, request: GenerationRequest) -> GenerationResponse:
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
            if adapter is not None:
                self._model.set_adapter(adapter)
            context = self._model.disable_adapter() if self._has_adapters and not candidate else nullcontext()
            thinking = request.role == 'decision_maker' and self.decision_thinking
            prompt = _render(self._tokenizer, messages,
                             generation=True, thinking=thinking)
            inputs = self._tokenizer(prompt, add_special_tokens=False, truncation=False, return_tensors='pt')
            count = inputs['input_ids'].shape[-1]
            limit = self._context_limit()
            if count + request.max_output_tokens > limit:
                raise ValueError(f'Context limit {limit}: prompt={count} + max_output_tokens='
                                 f'{request.max_output_tokens}; no prompt was truncated')
            inputs = inputs.to(self.device)
            pad = self._tokenizer.pad_token_id
            if pad is None:
                pad = self._tokenizer.eos_token_id
            with context, self._torch.inference_mode():
                sampling = {'do_sample': True, 'temperature': .6, 'top_p': .95, 'top_k': 20} if thinking else {'do_sample': False}
                if request.role == 'decision_maker':
                    sampling['stopping_criteria'] = [_DecisionJSONStop(self._tokenizer, count, thinking=thinking)]
                output = self._model.generate(**inputs, max_new_tokens=request.max_output_tokens,
                    **sampling, pad_token_id=pad, use_cache=True)
            tokens = output[0, count:]
            self.generation_count += 1
            text = self._tokenizer.decode(tokens, skip_special_tokens=True)
            if thinking:
                if '</think>' not in text:
                    raise ValueError('Decision thinking did not complete; no partial proposal will be executed')
                text = text.rsplit('</think>', 1)[1].strip()
                if not text:
                    raise ValueError('Decision thinking returned no final proposal')
            return GenerationResponse(text,
                request.model, False, TokenUsage(count, len(tokens)),
                self.model_name_or_path + '#sha256=' + self.actual_weight_sha256[request.model],
                revision_verified=True)

    __call__ = generate

    def release_memory(self):
        """Synchronous GPU release; the caller must avoid generation during training."""
        with self._lock:
            torch = self._torch
            self._model = self._tokenizer = self._peft_class = self._torch = None
            self._has_adapters = False
            self._loaded.clear()
            gc.collect()
            if torch is not None and str(self.device).startswith('cuda'):
                torch.cuda.synchronize(self.device)
                with torch.cuda.device(self.device):
                    torch.cuda.empty_cache()
