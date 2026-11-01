"""Bounded modelmate serving: existing HTTP base, process-local NPU LoRA.

This adapter never starts, stops, reloads, or registers weights with vLLM. The
caller serializes inference with training and calls release_memory() first.
Set ASCEND_RT_VISIBLE_DEVICES=13 before starting Python to expose physical 13
as npu:0; this module does not change the process environment. Frameworks are
imported only when a validated LoRA candidate needs the local base. An optional
runtime_loader(serving) returns the usual (torch, tokenizer, model, PeftModel)
tuple for framework-free tests. The independent researcher is wired by the
experiment, outside this actor adapter and outside Harness.
"""
from dataclasses import asdict, replace
import gc

from rsisys.adapters.serving import VLLMBackend
from rsisys.loops.contracts.model_service import BackendBinding
from .transformers_runtime import LocalTransformerServing


def _npu_runtime(serving):
    import os
    # Fail closed before importing a framework that might initialize a device.
    visible = os.environ.get('ASCEND_RT_VISIBLE_DEVICES', '')
    if not visible.isascii() or not visible.isdecimal():
        raise ValueError('ASCEND_RT_VISIBLE_DEVICES must select one explicit numeric device before Python starts')
    import torch
    import torch_npu  # noqa: F401 -- registers torch.npu; no CUDA transfer shim.
    from peft import PeftModel
    from transformers import AutoTokenizer
    if serving.native_model_class == 'image_text_to_text':
        from transformers import AutoModelForImageTextToText as ModelAutoClass
    else:
        from transformers import AutoModelForCausalLM as ModelAutoClass
    if torch.npu.device_count() != 1:
        raise ValueError('Ascend medical serving requires exactly one visible NPU')
    # Retain the allocator handle even if tokenizer/model construction fails.
    serving._torch = torch
    torch.npu.set_device(serving.device)
    tokenizer = AutoTokenizer.from_pretrained(serving.model_name_or_path,
        revision=serving.actor_model.revision, local_files_only=True, trust_remote_code=False)
    model = ModelAutoClass.from_pretrained(serving.model_name_or_path,
        revision=serving.actor_model.revision, local_files_only=True, trust_remote_code=False,
        dtype=getattr(torch, serving.dtype), device_map=serving.device,
        attn_implementation='eager')
    # Keep a handle for release if the explicit transfer fails partway through.
    serving._model = model
    model = model.to(serving.device).eval()
    return torch, tokenizer, model, PeftModel


class AscendMedicalServing(LocalTransformerServing):
    """Drop-in serving factory, with an 8192-token ceiling on both routes.

    Base/prompt requests use the existing Qwen3.5-9B alias. That alias does not
    attest a local revision or digest. Only prepared local LoRA responses claim
    the inherited content pin. No HTTP route other than chat/completions is used.
    """
    def __init__(self, model_name_or_path, *, actor_model, decision_model, artifacts,
                 native_model_class, device='npu:0', dtype='bfloat16',
                 max_context_tokens=None, runtime_loader=None, decision_thinking=False,
                 base_endpoint='http://127.0.0.1:8001/v1', base_alias='Qwen3.5-9B'):
        if device != 'npu:0':
            raise ValueError('Ascend medical serving requires npu:0; select one physical NPU via the environment')
        super().__init__(model_name_or_path, actor_model=actor_model, decision_model=decision_model,
            artifacts=artifacts, native_model_class=native_model_class, device=device, dtype=dtype,
            max_context_tokens=max_context_tokens, runtime_loader=runtime_loader or _npu_runtime,
            decision_thinking=decision_thinking)
        self.max_context_tokens = min(max_context_tokens or 8192, 8192)
        self._http = VLLMBackend(BackendBinding(actor_model.backend_id, 'chat_completions',
            endpoint=base_endpoint, json_mode=False), {actor_model: base_alias},
            generation_options={'temperature': 0, 'chat_template_kwargs': {'enable_thinking': False}})
        self.base_endpoint, self.base_alias = base_endpoint, base_alias
        self._http_metadata()

    def _http_metadata(self):
        for ref in (self.actor_model, self.decision_model):
            # Local candidate base bytes cannot attest the separately owned HTTP service.
            self.actual_weight_sha256.pop(ref, None)
            self.weight_metadata[ref] = {'model': asdict(ref),
                'serving_backend': 'vllm_http', 'source': self.base_endpoint,
                'provider_model': self.base_alias, 'revision_verified': False,
                'revision_binding': 'remote_alias_unverified',
                'cross_runtime_numerical_equivalence_verified': False,
                'max_context_tokens': self.max_context_tokens}

    def _load(self):
        try:
            super()._load()
        except BaseException:
            self.release_memory()
            raise
        finally:
            self._http_metadata()

    def _ensure_adapter(self, ref):
        if ref not in self._loaded and len(self._loaded) >= 2:
            # Bound resident adapters while retaining inherited revision/digest pins.
            self.release_memory()
            self._load()
        name = super()._ensure_adapter(ref)
        self.weight_metadata[ref].update(serving_backend='transformers_npu',
            device=self.device, revision_verified=True, source=self.model_name_or_path,
            dtype=self.dtype, attn_implementation='eager',
            cross_runtime_numerical_equivalence_verified=False)
        return name

    def prepare_inference(self, system):
        with self._lock:
            # Preserve all inherited identity/manifest checks even on the HTTP route.
            if self._candidate(system) is None:
                self._http_metadata()
                return system
            return super().prepare_inference(system)

    def generate(self, request):
        with self._lock:
            if request.role == 'actor':
                if request.model != self.actor_model:
                    return super().generate(request)
            elif request.role == 'decision_maker':
                if request.model != self.decision_model:
                    raise ValueError('Controller may only request its bound official base')
            else:
                raise ValueError('Unknown generation role')
            if request.max_output_tokens >= self.max_context_tokens:
                raise ValueError('Context limit requires space for the prompt; no prompt was truncated')
            if request.role == 'decision_maker' and self.decision_thinking:
                raise ValueError('Independent researcher thinking belongs in the experiment backend')
            # vLLM enforces its 8k prompt+completion limit without prompt truncation.
            response = self._http.generate(replace(request, model=self.actor_model))
            self.generation_count += 1
            self._http_metadata()
            return replace(response, model=request.model, revision_verified=False)

    def __call__(self, request):
        # A base-class __call__ alias would bypass this subclass's HTTP routing.
        return self.generate(request)

    def release_memory(self):
        """Synchronously drop owned local weights before the caller trains on NPU0."""
        with self._lock:
            torch = self._torch
            self._model = self._tokenizer = self._peft_class = self._torch = None
            self._has_adapters = False
            self._loaded.clear()
            gc.collect()
            if torch is not None:
                # Never touch CUDA or the separately owned HTTP service.
                torch.npu.synchronize(self.device)
                with torch.npu.device(self.device):
                    torch.npu.empty_cache()
